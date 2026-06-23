"""Pure collaborative-filtering logic over ``implicit`` ALS.

This module is the framework-free core: it takes a ``pandas.DataFrame`` (the
buffered ``(SELECT user, item, value)`` interaction relation) plus the column
roles, fits an implicit-feedback Alternating Least Squares (ALS) model, and
returns plain ``dict[str, list]`` column blocks ready to hand to pyarrow. No
VGI, no Arrow, no DuckDB here -- so every function is directly unit-testable.

Implicit-feedback ALS semantics
-------------------------------
We model *implicit feedback*: each ``(user, item, value)`` row is positive
evidence that ``user`` interacted with ``item``, and ``value`` is the
interaction *strength* / **confidence** (e.g. play count, purchase count,
rating). There are no explicit negatives -- absence of a row is treated as
"unknown", not "disliked". This is the Hu/Koren/Volinsky 2008 implicit-ALS
formulation that the `implicit` library implements: it factorizes the
user x item confidence matrix into user and item latent factors. A
recommendation score is the dot product of a user's factors with an item's
factors; item-item similarity is the cosine of their ALS item factors.

If a relation maps every interaction to the same value (or no value column is
given), all confidences are ``1.0`` -- pure "did/didn't interact" feedback.

Determinism
-----------
``implicit`` seeds its factor initialization from ``random_state`` and runs a
fixed number of ALS sweeps; we pin both ``random_state`` and ``num_threads=1``
(BLAS-level nondeterminism across threads is the main reproducibility hazard) so
identical input yields identical factors and therefore identical, reproducible
rankings.

``implicit`` is MIT-licensed; ``scipy`` is BSD; ``numpy``/``pandas`` are BSD.
"""

from __future__ import annotations

import os

# Pin BLAS thread counts *before* numpy/scipy/implicit import their backends, so
# the matrix math is single-threaded and reproducible. implicit also warns to
# stderr about thread/GPU/BLAS layout; that is harmless and stays on stderr.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Importing implicit is expensive (compiles/links its native ALS kernels and
# pulls in scipy); do it once at module import so the per-call path is cheap.
# The worker imports this module at startup, so the cost is paid before the
# first SQL call. Suppress implicit's import/runtime warnings (GPU not found,
# BLAS thread layout, etc.) -- they are advisory and would otherwise leak into
# the worker's stderr noisily.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from implicit.als import AlternatingLeastSquares

__all__ = [
    "RecommenderError",
    "recommend_all",
    "recommend_for",
    "similar_items",
]

# Default ALS hyperparameters. Iterations/regularization are fixed here (not
# user-facing) for reproducible, well-converged small-matrix results.
_ITERATIONS = 30
_REGULARIZATION = 0.01
_RANDOM_STATE = 42


class RecommenderError(ValueError):
    """Raised for user-facing input problems (missing columns / empty relation).

    A plain, explicit error so the worker surfaces a clear message to SQL
    instead of crashing with an opaque pandas/scipy/implicit traceback.
    """


def _require_columns(df: pd.DataFrame, required: dict[str, str]) -> None:
    """Validate that each required role maps to a present column.

    Args:
        df: The interaction relation.
        required: Mapping of role name (e.g. ``"user"``) to the column name the
            caller passed for that role.

    Raises:
        RecommenderError: If any named column is absent from the relation.
    """
    have = set(df.columns)
    missing = {role: col for role, col in required.items() if col not in have}
    if missing:
        detail = ", ".join(f"{role} := '{col}'" for role, col in missing.items())
        raise RecommenderError(
            f"missing required column(s): {detail}; "
            f"input relation has columns: {', '.join(map(str, df.columns))}"
        )


class _Indexed:
    """A fitted ALS model plus the id<->index maps it was trained on.

    Holds everything the three public operations need: the integer-indexed
    user/item id maps (so we can translate back to the original string ids), the
    confidence CSR matrix (user x item), and the fitted ALS model.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        user: str,
        item: str,
        value: str | None,
        factors: int,
    ) -> None:
        # Original ids as strings (we always emit the caller's ids back).
        users = df[user].astype(str)
        items = df[item].astype(str)

        if value is not None and value in df.columns:
            conf = pd.to_numeric(df[value], errors="coerce")
            if conf.isna().any():
                raise RecommenderError(
                    f"value column '{value}' must be numeric, but contains "
                    f"non-numeric values (dtype {df[value].dtype})"
                )
            conf = conf.astype(float).to_numpy()
        else:
            # No value column -> pure implicit "did interact" feedback (all 1.0).
            conf = np.ones(len(df), dtype=float)

        # Stable, sorted category ordering -> deterministic index assignment.
        self.user_ids: list[str] = sorted(users.unique().tolist())
        self.item_ids: list[str] = sorted(items.unique().tolist())
        self._user_index = {u: i for i, u in enumerate(self.user_ids)}
        self._item_index = {it: i for i, it in enumerate(self.item_ids)}

        u_idx = users.map(self._user_index).to_numpy()
        i_idx = items.map(self._item_index).to_numpy()

        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        # Sum confidences for duplicate (user, item) pairs.
        self.matrix = sp.csr_matrix(
            (conf, (u_idx, i_idx)),
            shape=(n_users, n_items),
        )
        self.matrix.sum_duplicates()

        # factors must be < min dimension for a meaningful factorization; clamp
        # so tiny test matrices still fit without imploding.
        eff_factors = max(1, min(factors, max(1, min(n_users, n_items) - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = AlternatingLeastSquares(
                factors=eff_factors,
                iterations=_ITERATIONS,
                regularization=_REGULARIZATION,
                random_state=_RANDOM_STATE,
                num_threads=1,
            )
            self.model.fit(self.matrix, show_progress=False)

    def user_row(self, idx: int) -> sp.csr_matrix:
        """The single-row CSR slice of interactions for one user index."""
        return self.matrix[idx]

    def seen_items(self, idx: int) -> set[int]:
        """Item indices the given user has already interacted with."""
        return set(self.matrix[idx].indices.tolist())


def _collect_user_recs(fitted: _Indexed, u_idx: int, n: int) -> list[tuple[str, float]]:
    """Top-``n`` novel recommendations for one user as ``(item_id, score)``.

    Explicitly drops already-seen items and de-duplicates, since ``implicit``'s
    ``recommend`` pads the result up to ``N`` with repeated / already-liked items
    (some scored ``-inf``, some a finite ``0.0``) when fewer than ``N`` genuinely
    novel items remain.
    """
    # Over-fetch: ask for n + (#seen) so n novel items survive the seen filter.
    seen = fitted.seen_items(u_idx)
    want = n + len(seen)
    ids, scores = fitted.model.recommend(
        u_idx,
        fitted.user_row(u_idx),
        N=want,
        filter_already_liked_items=True,
    )
    recs: list[tuple[str, float]] = []
    emitted: set[int] = set()
    for it_idx, score in zip(ids.tolist(), scores.tolist(), strict=False):
        if it_idx in seen or it_idx in emitted:
            continue
        if not np.isfinite(score):
            continue
        emitted.add(it_idx)
        recs.append((fitted.item_ids[it_idx], float(score)))
        if len(recs) >= n:
            break
    return recs


def _fit(
    df: pd.DataFrame,
    *,
    user: str,
    item: str,
    value: str | None,
    factors: int,
) -> _Indexed:
    """Validate input, then fit the indexed ALS model."""
    _require_columns(df, {"user": user, "item": item})
    if len(df) == 0:
        raise RecommenderError("recommender requires a non-empty interaction relation")
    return _Indexed(df, user=user, item=item, value=value, factors=factors)


def recommend_all(
    df: pd.DataFrame,
    *,
    user: str,
    item: str,
    value: str | None = None,
    n: int = 10,
    factors: int = 50,
) -> dict[str, list]:
    """Top-N recommended items for every user (excluding already-seen items).

    Fits implicit-feedback ALS on the interaction matrix and, for each user,
    emits the ``n`` highest-scoring items the user has **not** already
    interacted with, ranked ``1..n``.

    Args:
        df: Interaction relation; must contain ``user`` and ``item`` columns.
        user: Name of the user-id column.
        item: Name of the item-id column.
        value: Optional interaction-strength (confidence) column; defaults to
            ``1.0`` for every interaction when absent.
        n: Number of recommendations per user.
        factors: ALS latent-factor dimensionality.

    Returns:
        Column block with keys ``user`` (str), ``item`` (str), ``score``
        (float), ``rank`` (int). Rows are grouped by user and ordered by score
        descending (rank 1 = best). Already-seen items are excluded.

    Raises:
        RecommenderError: On missing columns or empty input.
    """
    fitted = _fit(df, user=user, item=item, value=value, factors=factors)
    out_user: list[str] = []
    out_item: list[str] = []
    out_score: list[float] = []
    out_rank: list[int] = []

    for u_idx, u_id in enumerate(fitted.user_ids):
        for rank, (it_id, score) in enumerate(_collect_user_recs(fitted, u_idx, n), start=1):
            out_user.append(u_id)
            out_item.append(it_id)
            out_score.append(score)
            out_rank.append(rank)

    return {"user": out_user, "item": out_item, "score": out_score, "rank": out_rank}


def similar_items(
    df: pd.DataFrame,
    *,
    user: str,
    item: str,
    value: str | None = None,
    n: int = 10,
    factors: int = 50,
) -> dict[str, list]:
    """Top-N most similar items for every item (cosine of ALS item factors).

    Args:
        df: Interaction relation; must contain ``user`` and ``item`` columns.
        user: Name of the user-id column.
        item: Name of the item-id column.
        value: Optional interaction-strength column (see :func:`recommend_all`).
        n: Number of similar items per item (the item itself is excluded).
        factors: ALS latent-factor dimensionality.

    Returns:
        Column block with keys ``item`` (str), ``similar_item`` (str),
        ``similarity`` (float, cosine in factor space), ``rank`` (int). Grouped
        by item, ordered by similarity descending.

    Raises:
        RecommenderError: On missing columns or empty input.
    """
    fitted = _fit(df, user=user, item=item, value=value, factors=factors)
    out_item: list[str] = []
    out_similar: list[str] = []
    out_sim: list[float] = []
    out_rank: list[int] = []

    for it_idx, it_id in enumerate(fitted.item_ids):
        # Ask for N+1: implicit includes the item itself (similarity 1.0); drop it.
        ids, sims = fitted.model.similar_items(it_idx, N=n + 1)
        rank = 0
        for other_idx, sim in zip(ids.tolist(), sims.tolist(), strict=False):
            if other_idx == it_idx:
                continue
            if not np.isfinite(sim):
                continue
            rank += 1
            if rank > n:
                break
            out_item.append(it_id)
            out_similar.append(fitted.item_ids[other_idx])
            out_sim.append(float(sim))
            out_rank.append(rank)

    return {
        "item": out_item,
        "similar_item": out_similar,
        "similarity": out_sim,
        "rank": out_rank,
    }


def recommend_for(
    df: pd.DataFrame,
    *,
    user: str,
    item: str,
    value: str | None = None,
    target_user: str,
    n: int = 10,
    factors: int = 50,
) -> dict[str, list]:
    """Top-N recommended items for ONE user (``target_user``), excluding seen.

    Args:
        df: Interaction relation; must contain ``user`` and ``item`` columns.
        user: Name of the user-id column.
        item: Name of the item-id column.
        value: Optional interaction-strength column (see :func:`recommend_all`).
        target_user: The id of the single user to recommend for.
        n: Number of recommendations.
        factors: ALS latent-factor dimensionality.

    Returns:
        Column block with keys ``item`` (str), ``score`` (float), ``rank``
        (int), ordered by score descending. Empty (zero rows) when
        ``target_user`` is unknown.

    Raises:
        RecommenderError: On missing columns or empty input.
    """
    fitted = _fit(df, user=user, item=item, value=value, factors=factors)
    target = str(target_user)
    if target not in fitted._user_index:
        # Cold / unknown user: no interaction history -> no recommendations.
        return {"item": [], "score": [], "rank": []}

    u_idx = fitted._user_index[target]
    out_item: list[str] = []
    out_score: list[float] = []
    out_rank: list[int] = []
    for rank, (it_id, score) in enumerate(_collect_user_recs(fitted, u_idx, n), start=1):
        out_item.append(it_id)
        out_score.append(score)
        out_rank.append(rank)

    return {"item": out_item, "score": out_score, "rank": out_rank}
