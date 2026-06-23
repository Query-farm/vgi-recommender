"""Table-function tests via the in-process buffering harness.

Drive each recommender function through the real bind -> process(sink) ->
combine -> finalize lifecycle (no subprocess), checking the emitted Arrow result
and that the named column-role / hyperparameter args resolve correctly.
"""

from __future__ import annotations

import pyarrow as pa

from vgi_recommender.tables import RecommendAll, RecommendFor, SimilarItems

from .data import planted_frame
from .harness import run_buffering


def _arrow(with_value: bool = False) -> pa.Table:
    return pa.Table.from_pandas(planted_frame(with_value=with_value), preserve_index=False)


def test_recommend_all_function_planted() -> None:
    out = run_buffering(
        RecommendAll,
        _arrow(),
        named={"user": "user", "item": "item", "n": 3, "factors": 8},
    )
    d = out.to_pydict()
    assert out.schema.names == ["user", "item", "score", "rank"]
    assert pa.types.is_int32(out.schema.field("rank").type)
    recs = set(zip(d["user"], d["item"], strict=True))
    assert ("u3", "C") in recs  # planted signal survives the wire/schema path
    assert ("u4", "C") in recs


def test_recommend_all_excludes_seen_via_function() -> None:
    out = run_buffering(RecommendAll, _arrow(), named={"user": "user", "item": "item", "n": 10, "factors": 8})
    d = out.to_pydict()
    # u3/u4 already have A,B -> never recommended back.
    for u, it in zip(d["user"], d["item"], strict=True):
        if u in ("u3", "u4"):
            assert it not in ("A", "B")


def test_similar_items_function() -> None:
    out = run_buffering(SimilarItems, _arrow(), named={"user": "user", "item": "item", "n": 1, "factors": 8})
    d = out.to_pydict()
    assert out.schema.names == ["item", "similar_item", "similarity", "rank"]
    top = {it: s for it, s, rk in zip(d["item"], d["similar_item"], d["rank"], strict=True) if rk == 1}
    assert top["A"] == "B" and top["B"] == "A"


def test_recommend_for_function() -> None:
    out = run_buffering(
        RecommendFor,
        _arrow(with_value=True),
        named={"user": "user", "item": "item", "value": "value", "target_user": "u3", "n": 3, "factors": 8},
    )
    d = out.to_pydict()
    assert out.schema.names == ["item", "score", "rank"]
    assert "C" in d["item"]
    assert d["rank"] == list(range(1, len(d["item"]) + 1))


def test_recommend_for_unknown_user_empty() -> None:
    out = run_buffering(
        RecommendFor,
        _arrow(),
        named={"user": "user", "item": "item", "target_user": "ghost", "n": 3, "factors": 8},
    )
    assert out.num_rows == 0


def test_missing_column_raises() -> None:
    import pytest

    with pytest.raises(Exception, match="missing required column"):
        run_buffering(RecommendAll, _arrow(), named={"user": "nope", "item": "item", "n": 3, "factors": 8})
