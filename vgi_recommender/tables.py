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
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import BindParams

from . import recommender
from .buffering import DrainState, SinkBuffer
from .schema_utils import field as sfield

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
    data: Annotated[TableInput, Arg(0, doc="Relation of (user, item, value) interactions.")]
    user: Annotated[str, Arg("user", default="user", doc="User-id column.")]
    item: Annotated[str, Arg("item", default="item", doc="Item-id column.")]
    value: Annotated[
        str, Arg("value", default="value", doc="Interaction-strength column (defaults to 1.0 if absent).")
    ]
    n: Annotated[int, Arg("n", default=10, doc="Recommendations per user.", ge=1)]
    factors: Annotated[int, Arg("factors", default=50, doc="ALS latent-factor count.", ge=1)]


@dataclass(slots=True, frozen=True)
class SimilarItemsArgs:
    data: Annotated[TableInput, Arg(0, doc="Relation of (user, item, value) interactions.")]
    user: Annotated[str, Arg("user", default="user", doc="User-id column.")]
    item: Annotated[str, Arg("item", default="item", doc="Item-id column.")]
    value: Annotated[
        str, Arg("value", default="value", doc="Interaction-strength column (defaults to 1.0 if absent).")
    ]
    n: Annotated[int, Arg("n", default=10, doc="Similar items per item.", ge=1)]
    factors: Annotated[int, Arg("factors", default=50, doc="ALS latent-factor count.", ge=1)]


@dataclass(slots=True, frozen=True)
class RecommendForArgs:
    data: Annotated[TableInput, Arg(0, doc="Relation of (user, item, value) interactions.")]
    user: Annotated[str, Arg("user", default="user", doc="User-id column.")]
    item: Annotated[str, Arg("item", default="item", doc="Item-id column.")]
    value: Annotated[
        str, Arg("value", default="value", doc="Interaction-strength column (defaults to 1.0 if absent).")
    ]
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

    @classmethod
    def on_bind(cls, params: BindParams[RecommendAllArgs]) -> BindResponse:
        return BindResponse(output_schema=_RECOMMEND_ALL_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[RecommendAllArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[RecommendAllArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True
        a = params.args
        df = cls.buffered_frame(params)
        result = recommender.recommend_all(
            df, user=a.user, item=a.item, value=a.value, n=a.n, factors=a.factors
        )
        out.emit(pa.RecordBatch.from_pydict(result, schema=params.output_schema))


class SimilarItems(SinkBuffer[SimilarItemsArgs, DrainState]):
    """Top-N most similar items per item (cosine of ALS item factors)."""

    FunctionArguments: ClassVar[type] = SimilarItemsArgs

    class Meta:
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

    @classmethod
    def on_bind(cls, params: BindParams[SimilarItemsArgs]) -> BindResponse:
        return BindResponse(output_schema=_SIMILAR_ITEMS_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[SimilarItemsArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[SimilarItemsArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True
        a = params.args
        df = cls.buffered_frame(params)
        result = recommender.similar_items(
            df, user=a.user, item=a.item, value=a.value, n=a.n, factors=a.factors
        )
        out.emit(pa.RecordBatch.from_pydict(result, schema=params.output_schema))


class RecommendFor(SinkBuffer[RecommendForArgs, DrainState]):
    """Top-N ALS recommendations for one target user."""

    FunctionArguments: ClassVar[type] = RecommendForArgs

    class Meta:
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

    @classmethod
    def on_bind(cls, params: BindParams[RecommendForArgs]) -> BindResponse:
        return BindResponse(output_schema=_RECOMMEND_FOR_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[RecommendForArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[RecommendForArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True
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
        out.emit(pa.RecordBatch.from_pydict(result, schema=params.output_schema))


TABLE_FUNCTIONS: list[type] = [RecommendAll, SimilarItems, RecommendFor]
