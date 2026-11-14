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

import json
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

# ---------------------------------------------------------------------------
# Per-object discovery/description metadata (vgi-lint strict profile)
# ---------------------------------------------------------------------------

# A small, self-contained planted-signal interaction relation used by the
# guaranteed-runnable examples. ``u1``/``u2`` interacted with {A, B, C};
# ``u3``/``u4`` with only {A, B}; so collaborative filtering recommends ``C`` to
# ``u3``/``u4``, and ``A``/``B`` are each other's nearest neighbours. Inlined as
# a VALUES subquery so each example runs without any pre-existing table.
_EVENTS_VALUES = (
    "(SELECT * FROM (VALUES "
    "('u1','A',1.0),('u1','B',1.0),('u1','C',1.0),"
    "('u2','A',1.0),('u2','B',1.0),('u2','C',1.0),"
    "('u3','A',1.0),('u3','B',1.0),"
    "('u4','A',1.0),('u4','B',1.0)"
    ") AS events(u, i, v))"
)


# Composed examples (one function's output feeding / joined with another's) run
# over a richer inline relation than the planted signal above: on a 3-item matrix
# the item similarities tie (C is equally similar to A and B), so "which watched
# title explains this recommendation" would be ambiguous. ``plays`` is a small
# viewing history -- (viewer, title, n_plays) -- with a sci-fi and a drama cluster
# bridged by fay; the role names deliberately differ from the defaults.
WATCH_HISTORY_CTE = (
    "WITH plays AS (SELECT * FROM (VALUES "
    "('ana','Alien',3.0),('ana','Dune',2.0),('ana','Matrix',4.0),"
    "('ben','Alien',1.0),('ben','Dune',5.0),('ben','Matrix',2.0),('ben','Blade Runner',3.0),"
    "('cal','Dune',2.0),('cal','Blade Runner',4.0),"
    "('dee','Notebook',5.0),('dee','Titanic',2.0),('dee','Amelie',3.0),"
    "('eve','Notebook',2.0),('eve','Titanic',4.0),"
    "('fay','Titanic',3.0),('fay','Amelie',2.0),('fay','Alien',1.0)"
    ") AS t(viewer, title, n_plays))"
)

_PLAYS_ROLES = "(SELECT viewer, title, n_plays FROM plays), user := 'viewer', item := 'title', value := 'n_plays'"

# recommend_all + similar_items: explain each viewer's top recommendation by the
# title they already watched whose ALS item factors are most similar to it.
EXPLAIN_RECOMMENDATIONS_SQL = (
    f"{WATCH_HISTORY_CTE}, "
    "recs AS (SELECT user AS viewer, item AS title FROM recommender.main.recommend_all("
    f"{_PLAYS_ROLES}, n := 1)), "
    "sims AS (SELECT item, similar_item, similarity FROM recommender.main.similar_items("
    f"{_PLAYS_ROLES}, n := 10)) "
    "SELECT r.viewer, r.title AS recommended, "
    "arg_max(s.similar_item, s.similarity) AS because_you_watched, "
    "round(max(s.similarity), 3) AS similarity "
    "FROM recs r JOIN plays p ON p.viewer = r.viewer "
    "JOIN sims s ON s.item = r.title AND s.similar_item = p.title "
    "GROUP BY r.viewer, r.title ORDER BY r.viewer"
)
EXPLAIN_RECOMMENDATIONS_DESCRIPTION = (
    "Explain each viewer's top recommendation (recommend_all) with the title they already "
    "watched that is most similar to it (similar_items), e.g. cal gets Matrix because of Dune."
)

# recommend_for + similar_items: a hybrid sanity check -- put one viewer's ALS
# recommendations next to an item-based score (summed similarity to the titles
# that viewer watched), so the two signals can be compared or blended.
HYBRID_SUPPORT_SQL = (
    f"{WATCH_HISTORY_CTE}, "
    "recs AS (SELECT item AS title, score, rank FROM recommender.main.recommend_for("
    f"{_PLAYS_ROLES}, target_user := 'eve', n := 3)), "
    "sims AS (SELECT item, similar_item, similarity FROM recommender.main.similar_items("
    f"{_PLAYS_ROLES}, n := 10)) "
    "SELECT r.rank, r.title, round(r.score, 3) AS als_score, "
    "round(sum(s.similarity), 3) AS item_similarity_support "
    "FROM recs r JOIN plays p ON p.viewer = 'eve' "
    "JOIN sims s ON s.item = r.title AND s.similar_item = p.title "
    "GROUP BY r.rank, r.title, r.score ORDER BY r.rank"
)
HYBRID_SUPPORT_DESCRIPTION = (
    "Score viewer eve's top-3 ALS recommendations (recommend_for) by their summed item-item "
    "similarity to the titles eve watched (similar_items) -- a second, item-based signal to "
    "compare or blend with the ALS ranking."
)


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
                    f"SELECT user, item, rank FROM recommender.main.recommend_all({_EVENTS_VALUES}, "
                    "user := 'u', item := 'i', value := 'v', n := 3) ORDER BY user, rank"
                ),
                description="Top-3 recommendations per user",
            ),
            FunctionExample(sql=EXPLAIN_RECOMMENDATIONS_SQL, description=EXPLAIN_RECOMMENDATIONS_DESCRIPTION),
        ]
        tags = {
            "vgi.title": "Recommend Items for All Users",
            "vgi.category": "user-recommendations",
            "vgi.doc_llm": (
                "# recommend_all\n\n"
                "Batch collaborative-filtering recommender. Consumes a whole relation of "
                "`(user, item, value)` interactions, fits an implicit-feedback Alternating "
                "Least Squares (ALS) model over the entire user x item matrix, and returns the "
                "**top-N recommended items for every user** that appears in the input.\n\n"
                "## When to use\n"
                "Use this when you want personalized recommendations for the full user base in "
                "one pass -- e.g. precomputing a 'recommended for you' shelf for every user. For "
                "a single user prefer `recommend_for`; for item-to-item neighbours prefer "
                "`similar_items`.\n\n"
                "## Inputs\n"
                "- `Arg(0)`: a `(SELECT user, item, value)` subquery -- the interaction relation.\n"
                "- `user`, `item`, `value`: names of the columns playing each role.\n"
                "- `n`: recommendations per user (default 10).\n"
                "- `factors`: ALS latent-factor count (default 50, clamped on tiny matrices).\n\n"
                "## Output\n"
                "Rows of `(user, item, score, rank)`; `score` is the ALS dot-product score and "
                "`rank` is 1-based within each user (1 = best).\n\n"
                "## Behavior & edge cases\n"
                "Items the user has already interacted with are excluded. `value` is the "
                "interaction confidence and defaults to 1.0 when the value column is absent. "
                "Results are deterministic (pinned random_state, single-threaded BLAS)."
            ),
            "vgi.doc_md": (
                "## recommend_all\n\n"
                "Top-N collaborative-filtering recommendations for **all** users, computed with "
                "implicit-feedback ALS over the supplied interaction relation.\n\n"
                "### Usage\n\n"
                "Pass the interaction relation as the positional `(user, item, value)` subquery "
                "and name the role columns plus `n` (recommendations per user). Each output row "
                "is `(user, item, score, rank)`; order by `user` then `rank` for a stable, "
                "per-user ranked list. A ready-to-run demonstration lives in this function's "
                "example queries.\n\n"
                "### Notes\n\n"
                "- One ALS model is fit over the entire matrix, so the whole relation is buffered "
                "before any rows are emitted.\n"
                "- Already-seen items are never recommended back to a user.\n"
                "- `value` (confidence) defaults to 1.0 if the column is missing; `factors` is "
                "auto-clamped on small matrices."
            ),
            "vgi.keywords": json.dumps(
                [
                    "recommend",
                    "recommendation",
                    "recommender",
                    "collaborative filtering",
                    "ALS",
                    "implicit feedback",
                    "top-N",
                    "personalization",
                    "batch recommendations",
                    "matrix factorization",
                ]
            ),
            "vgi.result_columns_schema": json.dumps(
                [
                    {"name": "user", "type": "VARCHAR", "description": "User id (original, as supplied)."},
                    {
                        "name": "item",
                        "type": "VARCHAR",
                        "description": "Recommended item id (not previously interacted with).",
                    },
                    {"name": "score", "type": "DOUBLE", "description": "ALS recommendation score (higher = stronger)."},
                    {
                        "name": "rank",
                        "type": "INTEGER",
                        "description": "Rank within this user's recommendations (1 = best).",
                    },
                ]
            ),
            "vgi.example_queries": json.dumps(
                [
                    {
                        "description": "Top-3 recommendations per user over an inline interaction relation.",
                        "sql": (
                            f"SELECT user, item, rank FROM recommender.main.recommend_all({_EVENTS_VALUES}, "
                            "user := 'u', item := 'i', value := 'v', n := 3) ORDER BY user, rank"
                        ),
                    },
                    {
                        "description": "Recommend novel item C to users u3 and u4 (planted CF signal).",
                        "sql": (
                            f"SELECT user, item FROM recommender.main.recommend_all({_EVENTS_VALUES}, "
                            "user := 'u', item := 'i', value := 'v', n := 1) "
                            "WHERE user IN ('u3', 'u4') ORDER BY user"
                        ),
                    },
                    {"description": EXPLAIN_RECOMMENDATIONS_DESCRIPTION, "sql": EXPLAIN_RECOMMENDATIONS_SQL},
                ]
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
                    f"SELECT item, similar_item, rank FROM recommender.main.similar_items({_EVENTS_VALUES}, "
                    "user := 'u', item := 'i', value := 'v', n := 2) ORDER BY item, rank"
                ),
                description="Top-2 similar items per item",
            ),
            FunctionExample(sql=EXPLAIN_RECOMMENDATIONS_SQL, description=EXPLAIN_RECOMMENDATIONS_DESCRIPTION),
        ]
        tags = {
            "vgi.title": "Find Similar Items by Co-Interaction",
            "vgi.category": "item-similarity",
            "vgi.doc_llm": (
                "# similar_items\n\n"
                "Item-to-item collaborative-filtering neighbour finder. Fits implicit-feedback ALS "
                "over the supplied `(user, item, value)` interaction relation, then for **every "
                "item** returns its top-N most similar items ranked by the cosine similarity of "
                "their ALS latent factors.\n\n"
                "## When to use\n"
                "Use this to power 'customers who liked X also liked Y', 'related products', or "
                "'more like this' surfaces -- recommendations that depend on an item, not a user. "
                "For per-user recommendations use `recommend_all` or `recommend_for`.\n\n"
                "## Inputs\n"
                "- `Arg(0)`: a `(SELECT user, item, value)` subquery -- the interaction relation.\n"
                "- `user`, `item`, `value`: names of the columns playing each role.\n"
                "- `n`: similar items per item (default 10).\n"
                "- `factors`: ALS latent-factor count (default 50, clamped on tiny matrices).\n\n"
                "## Output\n"
                "Rows of `(item, similar_item, similarity, rank)`; `similarity` is cosine of the "
                "ALS item factors (1.0 = identical) and `rank` is 1-based per item.\n\n"
                "## Behavior & edge cases\n"
                "An item is never listed as its own neighbour (the self-match at similarity 1.0 is "
                "dropped). `value` is interaction confidence and defaults to 1.0 when absent. "
                "Items co-interacted by the same users score as nearest neighbours. Deterministic "
                "(pinned random_state, single-threaded BLAS)."
            ),
            "vgi.doc_md": (
                "## similar_items\n\n"
                "Top-N most **similar items** for every item, by cosine similarity of "
                "implicit-feedback ALS item factors over the supplied interaction relation.\n\n"
                "### Usage\n\n"
                "Pass the interaction relation as the positional `(user, item, value)` subquery "
                "and name the role columns plus `n` (neighbours per item). Each output row is "
                "`(item, similar_item, similarity, rank)`; order by `item` then `rank` for a "
                "stable neighbour list. A ready-to-run demonstration lives in this function's "
                "example queries.\n\n"
                "### Notes\n\n"
                "- The query item itself is excluded from its own neighbour list.\n"
                "- One ALS model is fit over the whole matrix, so the relation is fully buffered "
                "before any rows are emitted.\n"
                "- `value` (confidence) defaults to 1.0 if the column is missing; `factors` is "
                "auto-clamped on small matrices."
            ),
            "vgi.keywords": json.dumps(
                [
                    "similar items",
                    "item similarity",
                    "related items",
                    "item-item",
                    "more like this",
                    "nearest neighbours",
                    "cosine similarity",
                    "collaborative filtering",
                    "ALS",
                    "recommendations",
                ]
            ),
            "vgi.result_columns_schema": json.dumps(
                [
                    {"name": "item", "type": "VARCHAR", "description": "Item id (original, as supplied)."},
                    {"name": "similar_item", "type": "VARCHAR", "description": "A similar item id."},
                    {
                        "name": "similarity",
                        "type": "DOUBLE",
                        "description": "Cosine similarity of ALS item factors (1 = identical).",
                    },
                    {
                        "name": "rank",
                        "type": "INTEGER",
                        "description": "Rank within this item's neighbours (1 = most similar).",
                    },
                ]
            ),
            "vgi.example_queries": json.dumps(
                [
                    {
                        "description": "Top-2 most similar items for every item in an inline relation.",
                        "sql": (
                            f"SELECT item, similar_item, rank FROM recommender.main.similar_items({_EVENTS_VALUES}, "
                            "user := 'u', item := 'i', value := 'v', n := 2) ORDER BY item, rank"
                        ),
                    },
                    {"description": EXPLAIN_RECOMMENDATIONS_DESCRIPTION, "sql": EXPLAIN_RECOMMENDATIONS_SQL},
                    {"description": HYBRID_SUPPORT_DESCRIPTION, "sql": HYBRID_SUPPORT_SQL},
                ]
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
            "Top-N recommended items for a single specified user, excluding items that user has "
            "already interacted with, as (item, score, rank). Returns no rows when the requested "
            "user is absent from the interaction relation."
        )
        categories = ["recommender", "collaborative-filtering"]
        examples = [
            FunctionExample(
                sql=(
                    f"SELECT item, rank FROM recommender.main.recommend_for({_EVENTS_VALUES}, "
                    "user := 'u', item := 'i', value := 'v', target_user := 'u3', n := 2) "
                    "ORDER BY rank"
                ),
                description="Top-2 recommendations for one user",
            ),
            FunctionExample(sql=HYBRID_SUPPORT_SQL, description=HYBRID_SUPPORT_DESCRIPTION),
        ]
        tags = {
            "vgi.title": "Recommend Items for One User",
            "vgi.category": "user-recommendations",
            "vgi.doc_llm": (
                "# recommend_for\n\n"
                "Single-user collaborative-filtering recommender. Fits implicit-feedback ALS over "
                "the supplied `(user, item, value)` interaction relation and returns the **top-N "
                "recommended items for one target user** (`target_user`, a named scalar argument), "
                "excluding items that user has already interacted with.\n\n"
                "## When to use\n"
                "Use this for an on-demand 'recommended for you' list for a specific user. For the "
                "whole user base in one pass use `recommend_all`; for item-to-item neighbours use "
                "`similar_items`.\n\n"
                "## Inputs\n"
                "- `Arg(0)`: a `(SELECT user, item, value)` subquery -- the interaction relation.\n"
                "- `user`, `item`, `value`: names of the columns playing each role.\n"
                "- `target_user`: the id of the single user to recommend for (named scalar).\n"
                "- `n`: number of recommendations (default 10).\n"
                "- `factors`: ALS latent-factor count (default 50, clamped on tiny matrices).\n\n"
                "## Output\n"
                "Rows of `(item, score, rank)`; `score` is the ALS dot-product score and `rank` is "
                "1-based (1 = best).\n\n"
                "## Behavior & edge cases\n"
                "An unknown `target_user` yields zero rows (not an error). Already-seen items are "
                "excluded. `value` is interaction confidence and defaults to 1.0 when absent. "
                "Deterministic (pinned random_state, single-threaded BLAS)."
            ),
            "vgi.doc_md": (
                "## recommend_for\n\n"
                "Top-N collaborative-filtering recommendations for a **single target user**, "
                "computed with implicit-feedback ALS over the supplied interaction relation.\n\n"
                "### Usage\n\n"
                "Pass the interaction relation as the positional `(user, item, value)` subquery, "
                "name the role columns, and set `target_user` to the id of the one user to score "
                "(plus `n`, the number of recommendations). Each output row is `(item, score, "
                "rank)`, ordered by `rank`. A ready-to-run demonstration lives in this function's "
                "example queries.\n\n"
                "### Notes\n\n"
                "- `target_user` is a named scalar argument naming the one user to score.\n"
                "- An unknown `target_user` returns an empty result rather than raising.\n"
                "- Already-seen items are never recommended; `value` (confidence) defaults to 1.0 "
                "if the column is missing."
            ),
            "vgi.keywords": json.dumps(
                [
                    "recommend for user",
                    "single user recommendations",
                    "personalized",
                    "recommend_for",
                    "top-N",
                    "collaborative filtering",
                    "ALS",
                    "implicit feedback",
                    "user recommendations",
                ]
            ),
            "vgi.result_columns_schema": json.dumps(
                [
                    {
                        "name": "item",
                        "type": "VARCHAR",
                        "description": "Recommended item id (not previously interacted with).",
                    },
                    {"name": "score", "type": "DOUBLE", "description": "ALS recommendation score (higher = stronger)."},
                    {"name": "rank", "type": "INTEGER", "description": "Rank within the recommendations (1 = best)."},
                ]
            ),
            "vgi.example_queries": json.dumps(
                [
                    {
                        "description": "Top-2 recommendations for a single target user (u3).",
                        "sql": (
                            f"SELECT item, rank FROM recommender.main.recommend_for({_EVENTS_VALUES}, "
                            "user := 'u', item := 'i', value := 'v', target_user := 'u3', n := 2) "
                            "ORDER BY rank"
                        ),
                    },
                    {"description": HYBRID_SUPPORT_DESCRIPTION, "sql": HYBRID_SUPPORT_SQL},
                ]
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
