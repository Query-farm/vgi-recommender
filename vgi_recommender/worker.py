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
from vgi.catalog import Catalog, Schema

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
    "Three SQL table functions cover the common recommender workloads:\n\n"
    "- `recommend_all` -- top-N recommended items for **every** user "
    "(personalized recommendations at scale).\n"
    "- `similar_items` -- top-N nearest item-item neighbours by cosine of the "
    "ALS item factors (the classic *\"people who liked X also liked Y\"* / "
    "related-items surface).\n"
    "- `recommend_for` -- top-N recommendations for a **single** target user.\n\n"
    "Every function takes the interaction relation as a positional "
    "`(SELECT user, item, value)` subquery (`Arg(0)`) and accepts the column "
    "roles and hyperparameters as named arguments "
    "(`user := ...`, `item := ...`, `value := ...`, `n := ...`, "
    "`factors := ...`). For example:\n\n"
    "```sql\n"
    "SELECT * FROM recommender.recommend_all(\n"
    "    (SELECT user_id, item_id, weight FROM events),\n"
    "    user := 'user_id', item := 'item_id', value := 'weight', n := 10)\n"
    "ORDER BY user, rank;\n"
    "```\n"
)

_SCHEMA_DESCRIPTION_LLM = (
    "Collaborative-filtering recommendation table functions: recommend_all "
    "(top-N items per user), similar_items (item-item neighbours), and "
    "recommend_for (top-N items for one user), all fitting implicit-feedback ALS "
    "over a (user, item, value) interaction relation."
)

_SCHEMA_DESCRIPTION_MD = (
    "Collaborative-filtering recommendation table functions powered by "
    "implicit-feedback ALS, operating over a (user, item, value) interaction "
    "relation passed as a single subquery. Contains `recommend_all` (top-N "
    "recommended items per user), `similar_items` (top-N item-item neighbours by "
    "cosine of the ALS item factors), and `recommend_for` (top-N recommendations "
    "for one target user). Use these for personalized recommendations, "
    "'people who liked X also liked Y', and item-similarity surfaces directly in SQL."
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

_CATALOG_TAGS = {
    "vgi.title": "Collaborative-Filtering Recommendations",
    "vgi.keywords": _SCHEMA_KEYWORDS,
    "vgi.doc_llm": _CATALOG_DESCRIPTION_LLM,
    "vgi.doc_md": _CATALOG_DESCRIPTION_MD,
    "vgi.author": "Query.Farm",
    "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
    "vgi.license": "MIT",
    "vgi.support_contact": "https://github.com/Query-farm/vgi-recommender/issues",
    "vgi.support_policy_url": "https://github.com/Query-farm/vgi-recommender/blob/main/README.md",
}

_SCHEMA_TAGS = {
    "vgi.title": "Recommender Functions (main)",
    "vgi.keywords": _SCHEMA_KEYWORDS,
    "vgi.doc_llm": _SCHEMA_DESCRIPTION_LLM,
    "vgi.doc_md": _SCHEMA_DESCRIPTION_MD,
    "vgi.example_queries": _SCHEMA_EXAMPLE_QUERIES,
    # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
    "domain": "machine-learning",
    "category": "recommender-systems",
    "topic": "collaborative-filtering",
}

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
