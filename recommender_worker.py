# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python",
#     "implicit>=0.7",
#     "scipy",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
#
# [tool.uv.sources]
# vgi-python = { path = "../vgi-python" }
# ///
"""Stdio entry shim for the recommender VGI worker.

Lets the worker run straight from a source checkout (``uv run
recommender_worker.py``) and keeps ``import recommender_worker`` working for
tests. The implementation lives in ``vgi_recommender.worker``; installed users
invoke the ``vgi-recommender`` console script (which points at
``vgi_recommender.worker:main``).

    ATTACH 'recommender' (TYPE vgi, LOCATION 'uv run recommender_worker.py');
    SELECT * FROM recommender.recommend_all((SELECT u, i, v FROM events),
        user := 'u', item := 'i', value := 'v', n := 10);
"""

from vgi_recommender.worker import RecommenderWorker, main

__all__ = ["RecommenderWorker", "main"]

if __name__ == "__main__":
    main()
