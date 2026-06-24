"""Collaborative-filtering table functions for DuckDB via VGI.

Each function consumes a *whole* interaction relation -- passed as a
``(SELECT user, item, value)`` subquery (positional ``Arg(0)``) -- and the
column roles + hyperparameters as NAMED args (``user := 'user_id'``,
``item := 'item_id'``, ``value := 'rating'``, ``n := 10``, ``factors := 50``).
Because ALS factorizes the entire user x item matrix at once, these are
buffering (Sink+Source) functions: they sink all input batches, then fit the
``implicit`` ALS model once in finalize.

    SELECT * FROM recommender.recommend_all((SELECT u, i, v FROM events),
        user := 'u', item := 'i', value := 'v', n := 10);
    SELECT * FROM recommender.similar_items((SELECT u, i, v FROM events),
        user := 'u', item := 'i', value := 'v', n := 5);
    SELECT * FROM recommender.recommend_for((SELECT u, i, v FROM events),
        user := 'u', item := 'i', value := 'v', target_user := 'alice', n := 5);

Implicit-feedback semantics: each row is positive evidence that the user
interacted with the item; ``value`` is the interaction strength (confidence) and
defaults to 1.0 when the value column is absent. See
``vgi_recommender.recommender`` for the math and full conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from . import recommender
from .buffering import ROWS_PER_TICK, DrainState, SinkBuffer, ipc_to_table, result_to_ipc
from .schema_utils import field as sfield


def _emit_next_slice(state: DrainState, out: OutputCollector, output_schema: pa.Schema) -> None:
    """Emit one bounded slice of the cursor's result, advancing the offset.

    Once ``state.started`` is set the full result lives in ``state.result_ipc``.
    Each call emits at most :data:`ROWS_PER_TICK` rows starting at ``state.offset``
    and advances ``offset``; when the cursor is drained it finishes the stream.
    Because ``state`` is wire-serialized between finalize ticks, this resumes from
    the persisted offset over the HTTP continuation boundary instead of restarting.

    Args:
        state: The finalize cursor (already computed; ``started`` is true).
        out: Sink for the output batch / stream completion.
        output_schema: The function's output schema (for the empty-result case).
    """
    table = ipc_to_table(state.result_ipc)
    total = table.num_rows
    if state.offset >= total:
        out.finish()
        return
    end = min(state.offset + ROWS_PER_TICK, total)
    chunk = table.slice(state.offset, end - state.offset).combine_chunks()
    batches = chunk.to_batches()
    out.emit(batches[0] if batches else pa.RecordBatch.from_pylist([], schema=output_schema))
    state.offset = end


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

_RECOMMEND_ALL_SCHEMA = pa.schema(
    [
        sfield("user", pa.string(), "User id (original, as supplied).", nullable=False),
        sfield("item", pa.string(), "Recommended item id (not previously interacted with)."),
        sfield("score", pa.float64(), "ALS recommendation score (higher = stronger)."),
        sfield("rank", pa.int32(), "Rank within this user's recommendations (1 = best)."),
    ]
)

_SIMILAR_ITEMS_SCHEMA = pa.schema(
    [
        sfield("item", pa.string(), "Item id (original, as supplied).", nullable=False),
        sfield("similar_item", pa.string(), "A similar item id."),
        sfield("similarity", pa.float64(), "Cosine similarity of ALS item factors (1 = identical)."),
        sfield("rank", pa.int32(), "Rank within this item's neighbours (1 = most similar)."),
    ]
)

_RECOMMEND_FOR_SCHEMA = pa.schema(
    [
        sfield("item", pa.string(), "Recommended item id (not previously interacted with)."),
        sfield("score", pa.float64(), "ALS recommendation score (higher = stronger)."),
        sfield("rank", pa.int32(), "Rank within the recommendations (1 = best)."),
    ]
)


# ---------------------------------------------------------------------------
# Argument dataclasses -- (SELECT ...) relation as Arg(0), roles/params as named
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class RecommendAllArgs:
    """Arguments for ``recommend_all`` (relation as ``Arg(0)``, roles/params named)."""

    data: Annotated[TableInput, Arg(0, doc="Relation of (user, item, value) interactions.")]
    user: Annotated[str, Arg("user", default="user", doc="User-id column.")]
    item: Annotated[str, Arg("item", default="item", doc="Item-id column.")]
    value: Annotated[str, Arg("value", default="value", doc="Interaction-strength column (defaults to 1.0 if absent).")]
    n: Annotated[int, Arg("n", default=10, doc="Recommendations per user.", ge=1)]
    factors: Annotated[int, Arg("factors", default=50, doc="ALS latent-factor count.", ge=1)]


@dataclass(slots=True, frozen=True)
class SimilarItemsArgs:
    """Arguments for ``similar_items`` (relation as ``Arg(0)``, roles/params named)."""

    data: Annotated[TableInput, Arg(0, doc="Relation of (user, item, value) interactions.")]
    user: Annotated[str, Arg("user", default="user", doc="User-id column.")]
    item: Annotated[str, Arg("item", default="item", doc="Item-id column.")]
    value: Annotated[str, Arg("value", default="value", doc="Interaction-strength column (defaults to 1.0 if absent).")]
    n: Annotated[int, Arg("n", default=10, doc="Similar items per item.", ge=1)]
    factors: Annotated[int, Arg("factors", default=50, doc="ALS latent-factor count.", ge=1)]


@dataclass(slots=True, frozen=True)
class RecommendForArgs:
    """Arguments for ``recommend_for`` (relation as ``Arg(0)``, roles/params named)."""

    data: Annotated[TableInput, Arg(0, doc="Relation of (user, item, value) interactions.")]
    user: Annotated[str, Arg("user", default="user", doc="User-id column.")]
    item: Annotated[str, Arg("item", default="item", doc="Item-id column.")]
    value: Annotated[str, Arg("value", default="value", doc="Interaction-strength column (defaults to 1.0 if absent).")]
    target_user: Annotated[str, Arg("target_user", doc="Id of the single user to recommend for.")]
    n: Annotated[int, Arg("n", default=10, doc="Number of recommendations.", ge=1)]
    factors: Annotated[int, Arg("factors", default=50, doc="ALS latent-factor count.", ge=1)]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


class RecommendAll(SinkBuffer[RecommendAllArgs, DrainState]):
    """Top-N ALS recommendations per user over a buffered interaction relation."""

    FunctionArguments: ClassVar[type] = RecommendAllArgs

    class Meta:
        """VGI function metadata (name, description, categories, examples)."""

        name = "recommend_all"
        description = (
            "Collaborative-filtering: fit implicit-feedback ALS, emit top-N recommended items "
            "per user (excluding already-seen items) as (user, item, score, rank). "
            "value is the interaction strength (confidence); defaults to 1.0 if absent."
        )
        categories = ["recommender", "collaborative-filtering"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM recommender.recommend_all((SELECT u, i, v FROM events), "
                    "user := 'u', item := 'i', value := 'v', n := 10) ORDER BY user, rank"
                ),
                description="Top-10 recommendations per user",
            )
        ]
        tags = {
            "vgi.columns_md": (
                "| Column | Type | Description |\n"
                "| --- | --- | --- |\n"
                "| `user` | VARCHAR | User id (original, as supplied). |\n"
                "| `item` | VARCHAR | Recommended item id (not previously interacted with). |\n"
                "| `score` | DOUBLE | ALS recommendation score (higher = stronger). |\n"
                "| `rank` | INTEGER | Rank within this user's recommendations (1 = best). |"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[RecommendAllArgs]) -> BindResponse:
        """Declare the output schema for this function."""
        return BindResponse(output_schema=_RECOMMEND_ALL_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[RecommendAllArgs]
    ) -> DrainState:
        """Start each finalize stream with a fresh offset cursor (computed on first tick)."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[RecommendAllArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Fit ALS once, then stream the result in bounded offset-cursor slices.

        Top-N per user is unbounded (n x #users), so the result can exceed one
        producer batch. The first tick computes it and stores it in the cursor;
        each tick (resumed from the wire-serialized cursor over HTTP) emits the
        next ``ROWS_PER_TICK`` slice and advances the offset until drained.

        Args:
            params: The buffering params (buffered input + parsed args).
            finalize_state_id: The finalize stream key.
            state: The offset cursor (computed once, then paged).
            out: Sink for the output batch / stream completion.
        """
        if not state.started:
            a = params.args
            df = cls.buffered_frame(params)
            result = recommender.recommend_all(df, user=a.user, item=a.item, value=a.value, n=a.n, factors=a.factors)
            batch = pa.RecordBatch.from_pydict(result, schema=params.output_schema)
            state.result_ipc = result_to_ipc(batch)
            state.started = True
            state.offset = 0
        _emit_next_slice(state, out, params.output_schema)


class SimilarItems(SinkBuffer[SimilarItemsArgs, DrainState]):
    """Top-N most similar items per item (cosine of ALS item factors)."""

    FunctionArguments: ClassVar[type] = SimilarItemsArgs

    class Meta:
        """VGI function metadata (name, description, categories, examples)."""

        name = "similar_items"
        description = (
            "Item-item collaborative filtering: top-N most similar items per item by cosine "
            "of the ALS item factors, as (item, similar_item, similarity, rank)."
        )
        categories = ["recommender", "collaborative-filtering"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM recommender.similar_items((SELECT u, i, v FROM events), "
                    "user := 'u', item := 'i', value := 'v', n := 5) ORDER BY item, rank"
                ),
                description="Top-5 similar items per item",
            )
        ]
        tags = {
            "vgi.columns_md": (
                "| Column | Type | Description |\n"
                "| --- | --- | --- |\n"
                "| `item` | VARCHAR | Item id (original, as supplied). |\n"
                "| `similar_item` | VARCHAR | A similar item id. |\n"
                "| `similarity` | DOUBLE | Cosine similarity of ALS item factors (1 = identical). |\n"
                "| `rank` | INTEGER | Rank within this item's neighbours (1 = most similar). |"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[SimilarItemsArgs]) -> BindResponse:
        """Declare the output schema for this function."""
        return BindResponse(output_schema=_SIMILAR_ITEMS_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[SimilarItemsArgs]
    ) -> DrainState:
        """Start each finalize stream with a fresh offset cursor (computed on first tick)."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[SimilarItemsArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Fit ALS once, then stream the result in bounded offset-cursor slices.

        Top-N per item is unbounded (n x #items), so the result can exceed one
        producer batch. The first tick computes it and stores it in the cursor;
        each tick (resumed from the wire-serialized cursor over HTTP) emits the
        next ``ROWS_PER_TICK`` slice and advances the offset until drained.

        Args:
            params: The buffering params (buffered input + parsed args).
            finalize_state_id: The finalize stream key.
            state: The offset cursor (computed once, then paged).
            out: Sink for the output batch / stream completion.
        """
        if not state.started:
            a = params.args
            df = cls.buffered_frame(params)
            result = recommender.similar_items(df, user=a.user, item=a.item, value=a.value, n=a.n, factors=a.factors)
            batch = pa.RecordBatch.from_pydict(result, schema=params.output_schema)
            state.result_ipc = result_to_ipc(batch)
            state.started = True
            state.offset = 0
        _emit_next_slice(state, out, params.output_schema)


class RecommendFor(SinkBuffer[RecommendForArgs, DrainState]):
    """Top-N ALS recommendations for one target user."""

    FunctionArguments: ClassVar[type] = RecommendForArgs

    class Meta:
        """VGI function metadata (name, description, categories, examples)."""

        name = "recommend_for"
        description = (
            "Top-N recommended items for ONE user (target_user, a named scalar arg), excluding "
            "already-seen items, as (item, score, rank). Empty if target_user is unknown."
        )
        categories = ["recommender", "collaborative-filtering"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM recommender.recommend_for((SELECT u, i, v FROM events), "
                    "user := 'u', item := 'i', value := 'v', target_user := 'alice', n := 5) "
                    "ORDER BY rank"
                ),
                description="Top-5 recommendations for one user",
            )
        ]
        tags = {
            "vgi.columns_md": (
                "| Column | Type | Description |\n"
                "| --- | --- | --- |\n"
                "| `item` | VARCHAR | Recommended item id (not previously interacted with). |\n"
                "| `score` | DOUBLE | ALS recommendation score (higher = stronger). |\n"
                "| `rank` | INTEGER | Rank within the recommendations (1 = best). |"
            ),
        }

    @classmethod
    def on_bind(cls, params: BindParams[RecommendForArgs]) -> BindResponse:
        """Declare the output schema for this function."""
        return BindResponse(output_schema=_RECOMMEND_FOR_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[RecommendForArgs]
    ) -> DrainState:
        """Start each finalize stream with a fresh offset cursor (computed on first tick)."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[RecommendForArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Fit ALS once, then stream the result in bounded offset-cursor slices.

        This result is bounded (<= n), but it uses the same offset-cursor path as
        the unbounded functions for uniformity: the first tick computes it into
        the cursor, then each tick (resumed from the wire-serialized cursor over
        HTTP) emits the next ``ROWS_PER_TICK`` slice until drained.

        Args:
            params: The buffering params (buffered input + parsed args).
            finalize_state_id: The finalize stream key.
            state: The offset cursor (computed once, then paged).
            out: Sink for the output batch / stream completion.
        """
        if not state.started:
            a = params.args
            df = cls.buffered_frame(params)
            result = recommender.recommend_for(
                df,
                user=a.user,
                item=a.item,
                value=a.value,
                target_user=a.target_user,
                n=a.n,
                factors=a.factors,
            )
            batch = pa.RecordBatch.from_pydict(result, schema=params.output_schema)
            state.result_ipc = result_to_ipc(batch)
            state.started = True
            state.offset = 0
        _emit_next_slice(state, out, params.output_schema)


TABLE_FUNCTIONS: list[type] = [RecommendAll, SimilarItems, RecommendFor]
