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

from vgi_recommender.tables import (
    EXPLAIN_RECOMMENDATIONS_DESCRIPTION,
    EXPLAIN_RECOMMENDATIONS_SQL,
    HYBRID_SUPPORT_DESCRIPTION,
    HYBRID_SUPPORT_SQL,
    TABLE_FUNCTIONS,
)

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
    "`factors := ...`). The relation's columns are the data; the named arguments just "
    "declare which column plays the user, item, and confidence role, so the same query "
    "works over any interaction table by pointing the roles at its columns.\n"
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

# sample_interactions + recommend_all: feed the built-in view to recommend_all,
# then join each recommendation back to the same view to report how many users
# already interacted with the recommended item (its current audience).
_RECOMMENDATION_AUDIENCE_SQL = (
    'WITH recs AS (SELECT "user", item FROM recommender.main.recommend_all('
    "(SELECT * FROM recommender.main.sample_interactions), n := 1)) "
    'SELECT r."user", r.item, count(DISTINCT s."user") AS users_already_interacted '
    "FROM recs r JOIN recommender.main.sample_interactions s ON s.item = r.item "
    'GROUP BY r."user", r.item ORDER BY r."user"'
)
_RECOMMENDATION_AUDIENCE_DESCRIPTION = (
    "Top recommendation per user from the built-in sample dataset (recommend_all over "
    "sample_interactions), joined back to the dataset to count how many users already "
    "interacted with the recommended item: u3 and u4 both get C, which 2 users already have."
)

_SCHEMA_EXAMPLE_QUERIES = json.dumps(
    [
        {
            "description": "Top-3 recommended items per user (recommend_all), ranked best-first.",
            "sql": (
                f"SELECT user, item, rank FROM recommender.main.recommend_all({_EVENTS_VALUES}, "
                "user := 'u', item := 'i', value := 'v', n := 3) ORDER BY user, rank"
            ),
        },
        {
            "description": "Top-2 nearest-neighbour items per item (similar_items) by ALS cosine.",
            "sql": (
                f"SELECT item, similar_item, rank FROM recommender.main.similar_items({_EVENTS_VALUES}, "
                "user := 'u', item := 'i', value := 'v', n := 2) ORDER BY item, rank"
            ),
        },
        {
            "description": "Top-2 recommendations for one target user (recommend_for), ranked best-first.",
            "sql": (
                f"SELECT item, rank FROM recommender.main.recommend_for({_EVENTS_VALUES}, "
                "user := 'u', item := 'i', value := 'v', target_user := 'u3', n := 2) ORDER BY rank"
            ),
        },
        {"description": EXPLAIN_RECOMMENDATIONS_DESCRIPTION, "sql": EXPLAIN_RECOMMENDATIONS_SQL},
        {"description": HYBRID_SUPPORT_DESCRIPTION, "sql": HYBRID_SUPPORT_SQL},
        {"description": _RECOMMENDATION_AUDIENCE_DESCRIPTION, "sql": _RECOMMENDATION_AUDIENCE_SQL},
    ]
)

# Agent-suitability suite (VGI152/VGI920). Each prompt fully specifies its own
# interaction data (there is no pre-existing table) so an analyst who has only
# read the catalog metadata can reproduce it. VGI416: the grader-only fields
# (reference_sql, success_criteria, ignore_column_names, unordered) live in the
# private vgi-agent-tests.yaml sidecar at the repo root. The references use
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
        },
        {
            "name": "most_similar_item",
            "prompt": (
                "There is no pre-existing table. Given interactions where u1 and u2 each "
                "interacted with items A, B and C, and u3 and u4 each interacted with only "
                "A and B, find the item most similar to item A (excluding A itself). "
                "Return the similar item's id."
            ),
        },
        {
            "name": "top_item_per_user",
            "prompt": (
                "There is no pre-existing table. Given interactions where u1 and u2 each "
                "interacted with items A, B and C, and u3 and u4 each interacted with only "
                "A and B, list the single top recommended new item for every user that has "
                "one, returning the user id alongside the recommended item id."
            ),
        },
        {
            "name": "browse_sample_interactions",
            "prompt": (
                "This worker ships a small built-in sample interaction dataset as a browsable "
                "relation. Using that dataset, report how many interactions each user has. "
                "Return the user id alongside the interaction count."
            ),
        },
        {
            "name": "explain_top_recommendation",
            "prompt": (
                "There is no pre-existing table. A shop's purchase history (customer, product), "
                "one row per purchase: c1 bought kettle, teapot and tea; c2 bought kettle, teapot "
                "and tea; c3 bought kettle and tea; c4 bought grinder, coffee and mug; c5 bought "
                "grinder and coffee; c6 bought coffee and mug; c7 bought tea and mug. Find customer "
                "c7's single top recommended new product, and explain it with the product c7 "
                "already bought whose item-item similarity to that recommendation is highest. "
                "Return two columns: the recommended product and the explaining product."
            ),
        },
    ]
)

_CATALOG_TAGS = {
    "vgi.title": "Item Recommender",
    "vgi.subtitle": "Recommend items and find similar items from user interaction data.",
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
            "name": "sample-data",
            "description": (
                "A small built-in (user, item, value) interaction dataset, browsable as "
                "a view, that shows the expected input shape and can be fed straight into "
                "the recommendation functions."
            ),
        },
    ]
)

_SCHEMA_TAGS = {
    "vgi.title": "Recommendation Functions",
    "vgi.keywords": _SCHEMA_KEYWORDS,
    "vgi.doc_llm": _SCHEMA_DESCRIPTION_LLM,
    "vgi.doc_md": _SCHEMA_DESCRIPTION_MD,
    "vgi.example_queries": _SCHEMA_EXAMPLE_QUERIES,
    "vgi.categories": _SCHEMA_CATEGORIES,
    # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
    "domain": "machine-learning",
    "topic": "collaborative-filtering",
}

# A small, browsable built-in interaction dataset (VGI146): a plain data view an
# agent can SELECT from to learn the (user, item, value) input shape before ever
# calling a function -- and feed straight into recommend_all / similar_items /
# recommend_for. Its columns are named exactly user/item/value so it works with the
# functions' DEFAULT role arguments. This is sample DATA, not a registry of the
# worker's own functions (which VGI327 forbids). The rows are the planted signal:
# u1/u2 saw {A,B,C}; u3/u4 saw only {A,B}, so C is the novel recommendation for
# u3/u4 and A/B are mutual nearest neighbours.
_SAMPLE_INTERACTIONS_DEFINITION = (
    'SELECT "user", "item", "value" FROM (VALUES '
    "('u1','A',1.0),('u1','B',1.0),('u1','C',1.0),"
    "('u2','A',1.0),('u2','B',1.0),('u2','C',1.0),"
    "('u3','A',1.0),('u3','B',1.0),"
    "('u4','A',1.0),('u4','B',1.0)"
    ') AS t("user", "item", "value")'
)

_SAMPLE_INTERACTIONS_DOC_LLM = (
    "# sample_interactions\n\n"
    "A small, static, built-in interaction dataset -- 10 `(user, item, value)` rows -- "
    "exposed as a browsable view so you can see the exact input shape the recommendation "
    "functions expect without supplying your own data. `SELECT` from it freely (it fits "
    "no model and touches no network), then feed it straight into `recommend_all`, "
    "`similar_items`, or `recommend_for` as their interaction relation. Its columns are "
    "named `user`, `item`, and `value`, which are exactly the functions' default role "
    "arguments, so `recommend_all((SELECT * FROM recommender.main.sample_interactions))` "
    "works with no role arguments at all.\n\n"
    "## Contents\n"
    "A deliberately rigged signal: users `u1`/`u2` interacted with items `A`, `B`, and "
    "`C`; `u3`/`u4` with only `A` and `B`. Collaborative filtering therefore recommends "
    "the unseen item `C` to `u3` and `u4`, and treats `A` and `B` as each other's nearest "
    "neighbours -- handy for a first, self-contained end-to-end call."
)

_SAMPLE_INTERACTIONS_DOC_MD = (
    "## sample_interactions\n\n"
    "A built-in, static `(user, item, value)` interaction dataset (10 rows) exposed as a "
    "browsable view. Use it to learn the input shape the recommendation functions expect "
    "and as a ready-made relation to try them on -- no data of your own required.\n\n"
    "### Columns\n\n"
    "- `user` -- interacting user id (`VARCHAR`).\n"
    "- `item` -- interacted item id (`VARCHAR`).\n"
    "- `value` -- interaction strength / confidence (`DOUBLE`, all `1.0` here).\n\n"
    "### Notes\n\n"
    "The column names match the functions' default role arguments, so the whole view can "
    "be passed as the interaction relation with no `user :=` / `item :=` / `value :=` "
    "arguments. The rows carry a planted signal: `C` is the novel recommendation for "
    "`u3`/`u4`, and `A`/`B` are mutual nearest neighbours."
)

_SAMPLE_INTERACTIONS_VIEW = View(
    name="sample_interactions",
    definition=_SAMPLE_INTERACTIONS_DEFINITION,
    comment="Built-in sample (user, item, value) interaction dataset to browse and feed to the functions.",
    column_comments={
        "user": "Interacting user id (original, as supplied).",
        "item": "Interacted item id (original, as supplied).",
        "value": "Interaction strength / confidence (all 1.0 in this sample).",
    },
    tags={
        "vgi.title": "Sample Interaction Dataset",
        "vgi.category": "sample-data",
        # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
        "domain": "machine-learning",
        "topic": "collaborative-filtering",
        "vgi.doc_llm": _SAMPLE_INTERACTIONS_DOC_LLM,
        "vgi.doc_md": _SAMPLE_INTERACTIONS_DOC_MD,
        "vgi.keywords": json.dumps(
            [
                "sample data",
                "example dataset",
                "interactions",
                "user item value",
                "getting started",
                "browsable",
                "demo",
                "collaborative filtering",
            ]
        ),
        "vgi.example_queries": json.dumps(
            [
                {
                    "description": "Count how many interactions each user has in the sample dataset.",
                    "sql": (
                        'SELECT "user", count(*) AS interactions '
                        "FROM recommender.main.sample_interactions GROUP BY 1 ORDER BY 1"
                    ),
                },
                {
                    "description": "Feed the sample dataset straight into recommend_all (default role args).",
                    "sql": (
                        'SELECT "user", item, rank FROM recommender.main.recommend_all('
                        "(SELECT * FROM recommender.main.sample_interactions), n := 3) "
                        "ORDER BY 1, rank"
                    ),
                },
                {"description": _RECOMMENDATION_AUDIENCE_DESCRIPTION, "sql": _RECOMMENDATION_AUDIENCE_SQL},
            ]
        ),
        "vgi.executable_examples": json.dumps(
            [
                {
                    "description": "Count interactions per user in the built-in sample dataset.",
                    "sql": (
                        'SELECT "user", count(*) AS interactions '
                        "FROM recommender.main.sample_interactions GROUP BY 1 ORDER BY 1"
                    ),
                    "expected_result": [
                        {"user": "u1", "interactions": 3},
                        {"user": "u2", "interactions": 3},
                        {"user": "u3", "interactions": 2},
                        {"user": "u4", "interactions": 2},
                    ],
                },
                {
                    "description": "Recommend the novel item C to users u3 and u4 from the sample dataset.",
                    "sql": (
                        'SELECT "user", item FROM recommender.main.recommend_all('
                        "(SELECT * FROM recommender.main.sample_interactions), n := 1) "
                        "WHERE \"user\" IN ('u3', 'u4') ORDER BY 1"
                    ),
                    "expected_result": [
                        {"user": "u3", "item": "C"},
                        {"user": "u4", "item": "C"},
                    ],
                },
                {
                    "description": _RECOMMENDATION_AUDIENCE_DESCRIPTION,
                    "sql": _RECOMMENDATION_AUDIENCE_SQL,
                    "expected_result": [
                        {"user": "u3", "item": "C", "users_already_interacted": 2},
                        {"user": "u4", "item": "C", "users_already_interacted": 2},
                    ],
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
            path=["main"],
            comment="Recommendation table functions: recommend_all, similar_items, recommend_for.",
            tags=_SCHEMA_TAGS,
            views=[_SAMPLE_INTERACTIONS_VIEW],
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
