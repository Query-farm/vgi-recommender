"""VGI worker exposing collaborative-filtering recommendations to DuckDB/SQL.

Assembles the recommender table functions in ``vgi_recommender`` into a single
``recommender`` catalog and provides the process entry point. The repo-root
``recommender_worker.py`` is a thin shim over this module for ``uv run``;
installed users get the ``vgi-recommender`` console script, which calls ``main``
here.

    ATTACH 'recommender' (TYPE vgi, LOCATION 'uv run recommender_worker.py');
    SELECT * FROM recommender.recommend_all((SELECT u, i, v FROM events),
        user := 'u', item := 'i', value := 'v', n := 10);
"""

from __future__ import annotations

import json
import sys

from vgi import Worker
from vgi.catalog import Catalog, Schema, View

from vgi_recommender.tables import TABLE_FUNCTIONS

_FUNCTIONS: list[type] = [*TABLE_FUNCTIONS]

_CATALOG_DESCRIPTION_LLM = (
    "Collaborative-filtering product/content recommendations over a relation of "
    "(user, item, value) interactions. Fits an implicit-feedback Alternating "
    "Least Squares (ALS) model and exposes three SQL table functions: "
    "recommend_all (top-N recommended items per user), similar_items (top-N "
    "item-item neighbours by cosine of the ALS item factors), and recommend_for "
    "(top-N recommendations for one target user). value is the interaction "
    "strength (confidence) and defaults to 1.0 when the value column is absent. "
    "Use for 'people who liked X also liked Y', personalized recommendations, and "
    "item similarity directly in SQL."
)

_CATALOG_DESCRIPTION_MD = (
    "# Recommender: Collaborative-Filtering Recommendations in SQL\n\n"
    "Generate personalized product and content recommendations directly in "
    "DuckDB SQL with implicit-feedback **Alternating Least Squares (ALS)** "
    "collaborative filtering -- no separate ML pipeline, notebook, or model "
    "server required.\n\n"
    "This VGI worker turns a single relation of `(user, item, value)` "
    "interactions -- clicks, purchases, plays, ratings, or any engagement "
    "signal -- into top-N recommendations, item-to-item similarity, and "
    "per-user suggestions. It is built for data engineers, analysts, and "
    "application developers who want a fast, reproducible recommender system "
    "they can call from a query instead of standing up a dedicated service. "
    "Because it runs as a DuckDB attachment, you can join recommendations "
    "straight back to your catalog tables, filter them, and ship them to "
    "downstream tables or dashboards in a single statement.\n\n"
    "Under the hood the worker fits an implicit-feedback ALS matrix-"
    "factorization model with the [`implicit`](https://github.com/benfred/"
    "implicit) library by Ben Frederickson -- a fast, Cython/BLAS-accelerated "
    "implementation of the classic ALS algorithm for implicit datasets. Each "
    "call buffers the whole interaction relation, learns latent user and item "
    "factors, and scores candidates by the dot product (recommendations) or "
    "the cosine of the item factors (similarity). Results are deterministic "
    "(pinned `random_state`, single-threaded BLAS), and `value` is treated as "
    "the interaction confidence, defaulting to `1.0` when the value column is "
    "omitted. See the [implicit documentation](https://benfred.github.io/"
    "implicit/) and the [collaborative filtering overview](https://en."
    "wikipedia.org/wiki/Collaborative_filtering) for background on the method.\n\n"
    "## Function surface\n\n"
    "The worker covers the three everyday collaborative-filtering workloads: "
    "personalized top-N recommendations for the whole user base at once, "
    "on-demand recommendations for a single target user, and item-to-item "
    'nearest-neighbour lookups (the classic *"people who liked X also liked '
    'Y"* / related-items surface).\n\n'
    "Every recommendation function takes the interaction relation as a positional "
    "`(user, item, value)` subquery (`Arg(0)`) and accepts the column "
    "roles and hyperparameters as named arguments "
    "(`user := ...`, `item := ...`, `value := ...`, `n := ...`, "
    "`factors := ...`). List the schema to see the exact object names, browse the "
    "discovery registry view to learn each function's argument and result surface, "
    "and read a function's own example queries for ready-to-run demonstrations.\n"
)

_SCHEMA_DESCRIPTION_LLM = (
    "Collaborative-filtering recommendation table functions: recommend_all "
    "(top-N items per user), similar_items (item-item neighbours), and "
    "recommend_for (top-N items for one user), all fitting implicit-feedback ALS "
    "over a (user, item, value) interaction relation."
)

_SCHEMA_DESCRIPTION_MD = (
    "# Recommendation Functions\n\n"
    "Collaborative-filtering recommendation table functions powered by "
    "implicit-feedback **Alternating Least Squares (ALS)**. Each function takes a "
    "single relation of `(user, item, value)` interactions as a subquery and fits "
    "a matrix-factorization model over the whole user x item matrix before "
    "emitting results.\n\n"
    "## Key concepts\n\n"
    "- **Interaction relation** -- one row per user/item interaction; `value` is "
    "the interaction confidence and defaults to `1.0` when omitted.\n"
    "- **Latent factors** -- ALS learns dense user and item vectors; scores are "
    "dot products (recommendations) or cosines (item similarity).\n"
    "- **Determinism** -- a pinned `random_state` and single-threaded BLAS make "
    "every call reproducible.\n\n"
    "## When to use\n\n"
    "Reach for these functions to generate personalized recommendations, build "
    "'people who liked X also liked Y' related-item surfaces, or score a single "
    "user on demand -- all directly in SQL, without a separate ML service or "
    "feature pipeline."
)

_SCHEMA_KEYWORDS = json.dumps(
    [
        "recommender",
        "recommendations",
        "collaborative filtering",
        "ALS",
        "implicit feedback",
        "recommend_all",
        "similar_items",
        "recommend_for",
        "similar items",
        "personalization",
        "matrix factorization",
        "top-N",
    ]
)

# A small inline planted-signal interaction relation (no pre-existing table needed)
# for the schema's representative example queries (VGI506). u1/u2 saw {A,B,C};
# u3/u4 saw only {A,B}, so C is recommended to u3/u4 and A/B are neighbours.
_EVENTS_VALUES = (
    "(SELECT * FROM (VALUES "
    "('u1','A',1.0),('u1','B',1.0),('u1','C',1.0),"
    "('u2','A',1.0),('u2','B',1.0),('u2','C',1.0),"
    "('u3','A',1.0),('u3','B',1.0),"
    "('u4','A',1.0),('u4','B',1.0)"
    ") AS events(u, i, v))"
)

_SCHEMA_EXAMPLE_QUERIES = (
    f"SELECT * FROM recommender.main.recommend_all({_EVENTS_VALUES}, "
    "user := 'u', item := 'i', value := 'v', n := 3) ORDER BY user, rank;\n"
    f"SELECT * FROM recommender.main.similar_items({_EVENTS_VALUES}, "
    "user := 'u', item := 'i', value := 'v', n := 2) ORDER BY item, rank;\n"
    f"SELECT * FROM recommender.main.recommend_for({_EVENTS_VALUES}, "
    "user := 'u', item := 'i', value := 'v', target_user := 'u3', n := 2) ORDER BY rank;"
)

# Agent-suitability suite (VGI152/VGI920). Each prompt fully specifies its own
# interaction data (there is no pre-existing table) so an analyst who has only
# read the catalog metadata can reproduce it; the grader-only reference_sql uses
# the planted-signal relation (u1/u2 saw {A,B,C}; u3/u4 saw {A,B}), which makes C
# the novel recommendation for u3/u4 and A/B mutual nearest neighbours. ALS is
# seeded (pinned random_state, single-threaded BLAS) so these references are
# deterministic. Comparisons are relaxed (ignore_column_names/unordered) because
# only the returned id set matters, not the analyst's column aliases or row order.
_AGENT_TEST_TASKS = json.dumps(
    [
        {
            "name": "recommend_for_single_user",
            "prompt": (
                "There is no pre-existing table. Consider these interactions: user u1 "
                "interacted with items A, B and C; u2 with A, B and C; u3 with A and B; "
                "u4 with A and B. Recommend the single best new item for user u3 -- one "
                "they have not already interacted with. Return just the item id."
            ),
            "reference_sql": (
                f"SELECT item FROM recommender.main.recommend_for({_EVENTS_VALUES}, "
                "user := 'u', item := 'i', value := 'v', target_user := 'u3', n := 1)"
            ),
            "success_criteria": "Returns exactly one row whose item id is C.",
            "ignore_column_names": True,
            "unordered": True,
        },
        {
            "name": "most_similar_item",
            "prompt": (
                "There is no pre-existing table. Given interactions where u1 and u2 each "
                "interacted with items A, B and C, and u3 and u4 each interacted with only "
                "A and B, find the item most similar to item A (excluding A itself). "
                "Return the similar item's id."
            ),
            "reference_sql": (
                f"SELECT similar_item FROM recommender.main.similar_items({_EVENTS_VALUES}, "
                "user := 'u', item := 'i', value := 'v', n := 1) WHERE item = 'A'"
            ),
            "success_criteria": "Returns exactly one row whose similar item id is B.",
            "ignore_column_names": True,
            "unordered": True,
        },
        {
            "name": "top_item_per_user",
            "prompt": (
                "There is no pre-existing table. Given interactions where u1 and u2 each "
                "interacted with items A, B and C, and u3 and u4 each interacted with only "
                "A and B, list the single top recommended new item for every user that has "
                "one, returning the user id alongside the recommended item id."
            ),
            "reference_sql": (
                f"SELECT user, item FROM recommender.main.recommend_all({_EVENTS_VALUES}, "
                "user := 'u', item := 'i', value := 'v', n := 1) ORDER BY user"
            ),
            "success_criteria": "Returns (u3, C) and (u4, C); u1/u2 have no new items to recommend.",
            "ignore_column_names": True,
            "unordered": True,
        },
        {
            "name": "discover_similarity_function",
            "prompt": (
                "This worker publishes a browsable discovery view that lists its recommendation "
                "functions, each tagged with a category. Using that view, find the name of the "
                "function whose category is item-to-item similarity. Return just the function name."
            ),
            "reference_sql": (
                "SELECT function_name FROM recommender.main.recommender_functions WHERE category = 'item-similarity'"
            ),
            "success_criteria": "Returns exactly one row whose function name is similar_items.",
            "ignore_column_names": True,
            "unordered": True,
        },
    ]
)

_CATALOG_TAGS = {
    "vgi.title": "Collaborative-Filtering Recommendations",
    "vgi.keywords": _SCHEMA_KEYWORDS,
    "vgi.doc_llm": _CATALOG_DESCRIPTION_LLM,
    "vgi.doc_md": _CATALOG_DESCRIPTION_MD,
    "vgi.agent_test_tasks": _AGENT_TEST_TASKS,
    "vgi.author": "Query.Farm",
    "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
    "vgi.license": "MIT",
    "vgi.support_contact": "https://github.com/Query-farm/vgi-recommender/issues",
    "vgi.support_policy_url": "https://github.com/Query-farm/vgi-recommender/blob/main/README.md",
}

_SCHEMA_CATEGORIES = json.dumps(
    [
        {
            "name": "user-recommendations",
            "description": (
                "Personalized top-N item recommendations for users -- for the whole "
                "user base at once or for a single target user."
            ),
        },
        {
            "name": "item-similarity",
            "description": (
                "Item-to-item neighbours by cosine of the ALS item factors -- "
                "'related items' / 'people who liked X also liked Y' surfaces."
            ),
        },
        {
            "name": "discovery",
            "description": (
                "Browsable registry/metadata objects that describe the worker's own "
                "recommendation functions (their arguments and result columns)."
            ),
        },
    ]
)

_SCHEMA_TAGS = {
    "vgi.title": "Recommender Functions (main)",
    "vgi.keywords": _SCHEMA_KEYWORDS,
    "vgi.doc_llm": _SCHEMA_DESCRIPTION_LLM,
    "vgi.doc_md": _SCHEMA_DESCRIPTION_MD,
    "vgi.example_queries": _SCHEMA_EXAMPLE_QUERIES,
    "vgi.categories": _SCHEMA_CATEGORIES,
    # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
    "domain": "machine-learning",
    "topic": "collaborative-filtering",
}

# A browsable, credential-free registry of the worker's table functions (VGI146).
# Backed by an inline VALUES scan so it needs no network, table, or model fit --
# an agent can query it to discover what functions exist, what they return, and
# which named arguments they take before ever constructing a call. Kept in lock-
# step with tables.py by hand (three rows, one per TABLE_FUNCTION).
_FUNCTIONS_VIEW_DEFINITION = (
    "SELECT function_name, category, purpose, key_arguments, output_columns FROM (VALUES "
    "('recommend_all', 'user-recommendations', "
    "'Top-N recommended items for every user in the interaction relation.', "
    "'user, item, value, n, factors', 'user, item, score, rank'), "
    "('recommend_for', 'user-recommendations', "
    "'Top-N recommended items for one target user (named target_user argument).', "
    "'user, item, value, target_user, n, factors', 'item, score, rank'), "
    "('similar_items', 'item-similarity', "
    "'Top-N most similar items per item by cosine of the ALS item factors.', "
    "'user, item, value, n, factors', 'item, similar_item, similarity, rank')"
    ") AS t(function_name, category, purpose, key_arguments, output_columns)"
)

_FUNCTIONS_VIEW_DOC_LLM = (
    "# recommender_functions\n\n"
    "A browsable registry of the three collaborative-filtering table functions this "
    "worker exposes, one row per function. Query it first to discover what is "
    "available: each row gives the function name, its category "
    "(user-recommendations vs item-similarity), a one-line purpose, the comma-"
    "separated list of named arguments it accepts, and the columns it returns. It is "
    "a static VALUES-backed view -- reading it fits no model and touches no network -- "
    "so it is safe to scan freely while planning a query.\n\n"
    "## When to use\n"
    "Use it to answer 'which function should I call and what does it return?' before "
    "constructing a `recommend_all` / `recommend_for` / `similar_items` call. The "
    "`key_arguments` and `output_columns` strings mirror the per-function "
    "documentation and the actual result schemas."
)

_FUNCTIONS_VIEW_DOC_MD = (
    "## recommender_functions\n\n"
    "A discovery view listing the worker's collaborative-filtering table functions, "
    "one row per function. Each row records the function name, its category, a short "
    "purpose, the named arguments it takes, and the columns it returns. Backed by an "
    "inline `VALUES` scan, so it reads instantly with no model fit and no network.\n\n"
    "### Columns\n\n"
    "- `function_name` -- the callable name (`recommend_all`, `recommend_for`, "
    "`similar_items`).\n"
    "- `category` -- `user-recommendations` or `item-similarity`.\n"
    "- `purpose` -- a one-line summary of what the function computes.\n"
    "- `key_arguments` -- the comma-separated named arguments the function accepts.\n"
    "- `output_columns` -- the comma-separated columns the function returns."
)

_RECOMMENDER_FUNCTIONS_VIEW = View(
    name="recommender_functions",
    definition=_FUNCTIONS_VIEW_DEFINITION,
    comment="Discovery registry of the worker's recommendation table functions (one row per function).",
    column_comments={
        "function_name": "Name of the table function (recommend_all, recommend_for, similar_items).",
        "category": "Function grouping: 'user-recommendations' or 'item-similarity'.",
        "purpose": "One-line summary of what the function computes.",
        "key_arguments": "Comma-separated named arguments the function accepts.",
        "output_columns": "Comma-separated columns the function returns.",
    },
    tags={
        "vgi.title": "Recommendation Functions Registry",
        "vgi.category": "discovery",
        # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
        "domain": "machine-learning",
        "topic": "collaborative-filtering",
        "vgi.doc_llm": _FUNCTIONS_VIEW_DOC_LLM,
        "vgi.doc_md": _FUNCTIONS_VIEW_DOC_MD,
        "vgi.keywords": json.dumps(
            [
                "functions",
                "registry",
                "discovery",
                "catalog",
                "recommend_all",
                "recommend_for",
                "similar_items",
                "capabilities",
            ]
        ),
        "vgi.example_queries": json.dumps(
            [
                {
                    "description": "List the per-user recommendation functions and what each returns.",
                    "sql": (
                        "SELECT function_name, output_columns "
                        "FROM recommender.main.recommender_functions "
                        "WHERE category = 'user-recommendations' ORDER BY function_name"
                    ),
                },
                {
                    "description": "Discover the named arguments accepted by the similar_items function.",
                    "sql": (
                        "SELECT key_arguments "
                        "FROM recommender.main.recommender_functions "
                        "WHERE function_name = 'similar_items'"
                    ),
                },
            ]
        ),
    },
)

_RECOMMENDER_CATALOG = Catalog(
    name="recommender",
    default_schema="main",
    comment="Collaborative-filtering recommendation worker (implicit-feedback ALS) for DuckDB via VGI.",
    tags=_CATALOG_TAGS,
    source_url="https://github.com/Query-farm/vgi-recommender",
    schemas=[
        Schema(
            name="main",
            comment="Recommendation table functions: recommend_all, similar_items, recommend_for.",
            tags=_SCHEMA_TAGS,
            views=[_RECOMMENDER_FUNCTIONS_VIEW],
            functions=list(_FUNCTIONS),
        ),
    ],
)


class RecommenderWorker(Worker):
    """Worker process hosting the ``recommender`` catalog."""

    catalog = _RECOMMENDER_CATALOG


def main() -> None:
    """Run the worker (stdio by default; pass ``--http`` for the HTTP server)."""
    RecommenderWorker.main()


def main_http() -> None:
    """Run the worker over HTTP (injects ``--http`` into the worker CLI)."""
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    RecommenderWorker.main()
