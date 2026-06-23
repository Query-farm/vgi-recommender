"""Collaborative-filtering recommendations as a VGI worker for DuckDB/SQL.

The implementation is split so each concern stays focused:

- ``recommender`` -- pure ``implicit`` ALS logic (recommend_all, similar_items,
  recommend_for) over ``pandas`` frames; no Arrow or VGI dependency, directly
  unit-testable.
- ``buffering``   -- the single-bucket Sink+Source plumbing every function
  shares (buffer all interaction batches, then fit ALS once).
- ``tables``      -- the VGI ``TableBufferingFunction`` wrappers: the whole
  interaction relation in via ``(SELECT user, item, value)`` (``Arg(0)``),
  column roles + hyperparameters as named args.

``recommender_worker.py`` at the repo root assembles these into the
``recommender`` catalog and runs the worker over stdio (or HTTP).
"""

from __future__ import annotations

__version__ = "0.1.0"
