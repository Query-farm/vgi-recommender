"""Table-function tests via the in-process buffering harness.

Drive each recommender function through the real bind -> process(sink) ->
combine -> finalize lifecycle (no subprocess), checking the emitted Arrow result
and that the named column-role / hyperparameter args resolve correctly.
"""

from __future__ import annotations

import pyarrow as pa

import vgi_recommender.buffering as buffering
import vgi_recommender.tables as tables
from vgi_recommender.tables import RecommendAll, RecommendFor, SimilarItems

from .data import cohort_frame, planted_frame
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


class TestCursorSurvivesContinuation:
    """Regression for the HTTP-continuation bug: finalize must page its result
    through a wire-serializable offset cursor.

    Over the stateless HTTP transport the framework round-trips the finalize state
    between ticks and emits at most one producer batch per response. A
    position-less "emit everything then finish" finalize restarts from row 0 on
    every resume and never terminates once the result exceeds one batch. Driving
    finalize with ``serialize_state=True`` re-deserializes the state between every
    tick, so it reproduces that round-trip in-process.

    These tests fail on the old ``DrainState{done}`` code (single emit, no
    paging / does not survive serialization) and pass on the offset cursor.
    """

    def _rows(self, table: pa.Table) -> list[tuple[object, ...]]:
        cols = [table.column(name).to_pylist() for name in table.schema.names]
        return list(zip(*cols, strict=True))

    def test_recommend_all_pages_and_survives_serialization(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # A cohort large enough that top-N per user spans many ROWS_PER_TICK
        # slices: 40 users each missing 2 held-out items -> ~80 result rows; with
        # ROWS_PER_TICK small the cursor must page across many ticks.
        monkeypatch.setattr(buffering, "ROWS_PER_TICK", 4)
        monkeypatch.setattr(tables, "ROWS_PER_TICK", 4)
        table = pa.Table.from_pandas(cohort_frame(40, 20), preserve_index=False)
        named = {"user": "user", "item": "item", "n": 6, "factors": 8}

        plain = run_buffering(RecommendAll, table, named=named)
        # The body below would HANG/overrun the guard on the old code; the guard
        # turns the non-termination into a RuntimeError instead of a hang.
        paged = run_buffering(RecommendAll, table, named=named, serialize_state=True)

        assert plain.num_rows > buffering.ROWS_PER_TICK  # genuinely multi-slice
        plain_rows = self._rows(plain)
        paged_rows = self._rows(paged)
        # (1) identical content AND order across the continuation boundary.
        assert paged_rows == plain_rows
        # (2) no row emitted twice.
        assert len(paged_rows) == len(set(paged_rows))

    def test_recommend_all_termination_and_slice_bound(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # Assert the cursor actually paged: with ROWS_PER_TICK small, finalize
        # must emit MORE than one batch, each bounded. The old single-emit code
        # produces exactly one (oversized) batch and fails this.
        monkeypatch.setattr(buffering, "ROWS_PER_TICK", 4)
        monkeypatch.setattr(tables, "ROWS_PER_TICK", 4)
        from .harness import _Collector  # noqa: PLC0415

        table = pa.Table.from_pandas(cohort_frame(40, 20), preserve_index=False)
        named = {"user": "user", "item": "item", "n": 6, "factors": 8}

        # Run the finalize lifecycle directly to inspect per-tick emit sizes.
        out = run_buffering(RecommendAll, table, named=named, serialize_state=True)
        assert out.num_rows > 4

        # Re-drive once more capturing individual batches.
        collector = _Collector()
        from vgi.arguments import Arguments
        from vgi.function_storage import BoundStorage, FunctionStorageSqlite
        from vgi.invocation import FunctionType
        from vgi.protocol import BindRequest, InitRequest
        from vgi.table_buffering_function import TableBufferingParams

        args = Arguments(positional=(), named={k: pa.scalar(v) for k, v in named.items()})
        bind_req = BindRequest(
            function_name=RecommendAll.Meta.name,
            arguments=args,
            function_type=FunctionType.TABLE_BUFFERING,
            input_schema=table.schema,
        )
        bind_resp = RecommendAll.bind(bind_req)
        init_req = InitRequest(bind_call=bind_req, output_schema=bind_resp.output_schema)
        init_resp = RecommendAll.global_init(init_req)
        storage = BoundStorage(FunctionStorageSqlite(":memory:"), init_resp.execution_id)
        parsed = RecommendAll._parse_arguments(RecommendAll.FunctionArguments, args)
        params = TableBufferingParams(
            args=parsed,
            init_call=init_req,
            init_response=init_resp,
            output_schema=bind_resp.output_schema,
            settings={},
            secrets={},
            storage=storage,
            execution_id=init_resp.execution_id,
            attach_id=b"",
            transaction_id=None,
            function_name=RecommendAll.Meta.name,
        )
        for batch in table.to_batches():
            RecommendAll.process(batch, params)
        fids = RecommendAll.combine([init_resp.execution_id], params)
        fid = fids[0]
        state = RecommendAll.initial_finalize_state(fid, params)
        ticks = 0
        while not collector.finished and ticks < 10_000:
            RecommendAll.finalize(params, fid, state, collector)
            ticks += 1
            if not collector.finished:
                state = type(state).deserialize_from_bytes(state.serialize_to_bytes())
        assert collector.finished
        # Cursor paged: more than one batch, each within the slice bound.
        assert len(collector.batches) > 1
        for b in collector.batches:
            assert b.num_rows <= buffering.ROWS_PER_TICK
