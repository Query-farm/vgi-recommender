"""Unit tests for the pure collaborative-filtering logic.

These test the framework-free ``vgi_recommender.recommender`` functions directly
on a small constructed matrix with a planted signal (see ``tests/data.py``):
structure (top-N, descending scores, rank 1..n, already-seen excluded) plus the
planted recommendation, item-item similarity, single-user recs, and the error
edges. A fixed ALS seed makes every assertion reproducible.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vgi_recommender import recommender
from vgi_recommender.recommender import RecommenderError

from .data import planted_frame

# --------------------------------------------------------------------------
# recommend_all — structure + planted signal
# --------------------------------------------------------------------------


def test_recommend_all_structure() -> None:
    df = planted_frame()
    out = recommender.recommend_all(df, user="user", item="item", n=3, factors=8)

    # Group by user; within each group rank is 1..k and scores are descending.
    by_user: dict[str, list[tuple[int, float, str]]] = {}
    for u, it, sc, rk in zip(out["user"], out["item"], out["score"], out["rank"], strict=True):
        by_user.setdefault(u, []).append((rk, sc, it))

    for _u, rows in by_user.items():
        rows_sorted = sorted(rows)
        ranks = [r for r, _, _ in rows_sorted]
        assert ranks == list(range(1, len(ranks) + 1))  # rank 1..k contiguous
        scores = [s for _, s, _ in rows_sorted]
        assert all(a >= b - 1e-9 for a, b in zip(scores, scores[1:], strict=False))


def test_recommend_all_excludes_already_seen() -> None:
    df = planted_frame()
    out = recommender.recommend_all(df, user="user", item="item", n=10, factors=8)
    seen = {(u, i) for u, i in zip(df["user"], df["item"], strict=True)}
    recommended = set(zip(out["user"], out["item"], strict=True))
    # No recommended (user, item) pair may be one the user already interacted with.
    assert seen.isdisjoint(recommended)


def test_recommend_all_planted_signal_recommends_C() -> None:
    # u3 and u4 interacted with {A, B} exactly like u1/u2 who also have C.
    # Collaborative filtering should therefore surface C for u3 and u4.
    df = planted_frame()
    out = recommender.recommend_all(df, user="user", item="item", n=3, factors=8)
    recs = {(u, i) for u, i in zip(out["user"], out["item"], strict=True)}
    assert ("u3", "C") in recs
    assert ("u4", "C") in recs


def test_recommend_all_is_deterministic() -> None:
    df = planted_frame()
    a = recommender.recommend_all(df, user="user", item="item", n=3, factors=8)
    b = recommender.recommend_all(df, user="user", item="item", n=3, factors=8)
    assert a == b  # fixed seed + single-threaded BLAS -> identical results


# --------------------------------------------------------------------------
# similar_items
# --------------------------------------------------------------------------


def test_similar_items_structure_and_excludes_self() -> None:
    df = planted_frame()
    out = recommender.similar_items(df, user="user", item="item", n=2, factors=8)
    # An item is never its own "similar_item".
    assert all(i != s for i, s in zip(out["item"], out["similar_item"], strict=True))
    # Per-item ranks are contiguous 1..k and similarities descending.
    by_item: dict[str, list[tuple[int, float]]] = {}
    for it, _s, sim, rk in zip(out["item"], out["similar_item"], out["similarity"], out["rank"], strict=True):
        by_item.setdefault(it, []).append((rk, sim))
    for rows in by_item.values():
        rows_sorted = sorted(rows)
        assert [r for r, _ in rows_sorted] == list(range(1, len(rows_sorted) + 1))
        sims = [s for _, s in rows_sorted]
        assert all(a >= b - 1e-9 for a, b in zip(sims, sims[1:], strict=False))


def test_similar_items_planted_A_and_B_are_neighbours() -> None:
    # A and B are co-purchased by exactly the same users -> top neighbours.
    df = planted_frame()
    out = recommender.similar_items(df, user="user", item="item", n=1, factors=8)
    top = {it: s for it, s, rk in zip(out["item"], out["similar_item"], out["rank"], strict=True) if rk == 1}
    assert top["A"] == "B"
    assert top["B"] == "A"


# --------------------------------------------------------------------------
# recommend_for
# --------------------------------------------------------------------------


def test_recommend_for_single_user_planted() -> None:
    df = planted_frame()
    out = recommender.recommend_for(df, user="user", item="item", target_user="u3", n=3, factors=8)
    assert out["rank"] == list(range(1, len(out["item"]) + 1))
    assert "C" in out["item"]  # planted: u3 should get C
    # u3 already has A, B -> never recommended back.
    assert "A" not in out["item"] and "B" not in out["item"]


def test_recommend_for_unknown_user_is_empty() -> None:
    df = planted_frame()
    out = recommender.recommend_for(df, user="user", item="item", target_user="nobody", n=3, factors=8)
    assert out == {"item": [], "score": [], "rank": []}


# --------------------------------------------------------------------------
# value / confidence column
# --------------------------------------------------------------------------


def test_value_column_all_ones_matches_no_value() -> None:
    # A value column of all 1.0 is equivalent to no value column.
    no_val = recommender.recommend_all(planted_frame(), user="user", item="item", n=3, factors=8)
    with_val = recommender.recommend_all(
        planted_frame(with_value=True), user="user", item="item", value="value", n=3, factors=8
    )
    assert set(zip(no_val["user"], no_val["item"], strict=True)) == set(
        zip(with_val["user"], with_val["item"], strict=True)
    )


def test_missing_value_column_falls_back_to_ones() -> None:
    # Naming a value column that isn't present is tolerated (all-ones fallback),
    # since `value` has a default name and may simply not be selected.
    df = planted_frame()
    out = recommender.recommend_all(df, user="user", item="item", value="value", n=3, factors=8)
    assert ("u3", "C") in set(zip(out["user"], out["item"], strict=True))


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------


def test_missing_user_column_errors() -> None:
    df = planted_frame()
    with pytest.raises(RecommenderError, match="missing required column"):
        recommender.recommend_all(df, user="nope", item="item", n=3, factors=8)


def test_empty_relation_errors() -> None:
    df = pd.DataFrame({"user": pd.Series([], dtype=str), "item": pd.Series([], dtype=str)})
    with pytest.raises(RecommenderError, match="non-empty"):
        recommender.recommend_all(df, user="user", item="item", n=3, factors=8)


def test_non_numeric_value_errors() -> None:
    df = planted_frame()
    df["value"] = "not-a-number"
    with pytest.raises(RecommenderError, match="must be numeric"):
        recommender.recommend_all(df, user="user", item="item", value="value", n=3, factors=8)


def test_cold_item_user_handled() -> None:
    # u5 only touched item D (off the main cluster). It must not crash and must
    # not get D recommended back; any recs it gets are from the other items.
    df = planted_frame()
    out = recommender.recommend_for(df, user="user", item="item", target_user="u5", n=5, factors=8)
    assert "D" not in out["item"]
