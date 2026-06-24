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
    "# recommender\n\n"
    "Collaborative-filtering recommendations (implicit-feedback ALS) for DuckDB "
    "via VGI, powered by [implicit](https://github.com/benfred/implicit).\n\n"
    "Each function consumes the whole interaction relation as a "
    "`(SELECT user, item, value)` subquery (positional `Arg(0)`) plus the column "
    "roles and hyperparameters as named args "
    "(`user := ...`, `item := ...`, `value := ...`, `n := ...`, "
    "`factors := ...`).\n\n"
    "Table functions: `recommend_all`, `similar_items`, `recommend_for`."
)

_SCHEMA_DESCRIPTION_LLM = (
    "Collaborative-filtering recommendation table functions: recommend_all "
    "(top-N items per user), similar_items (item-item neighbours), and "
    "recommend_for (top-N items for one user), all fitting implicit-feedback ALS "
    "over a (user, item, value) interaction relation."
)

_SCHEMA_DESCRIPTION_MD = (
    "Collaborative-filtering recommendation functions (implicit-feedback ALS) "
    "over a (user, item, value) interaction relation."
)

_CATALOG_TAGS = {
    "vgi.description_llm": _CATALOG_DESCRIPTION_LLM,
    "vgi.description_md": _CATALOG_DESCRIPTION_MD,
    "vgi.author": "Query.Farm",
    "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
    "vgi.license": "MIT",
    "vgi.support_contact": "https://github.com/Query-farm/vgi-recommender/issues",
    "vgi.support_policy_url": "https://github.com/Query-farm/vgi-recommender/blob/main/README.md",
}

_SCHEMA_TAGS = {
    "vgi.description_llm": _SCHEMA_DESCRIPTION_LLM,
    "vgi.description_md": _SCHEMA_DESCRIPTION_MD,
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
