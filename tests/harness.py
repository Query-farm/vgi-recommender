"""In-process driver for the recommender buffering (Sink+Source) functions.

Runs a ``TableBufferingFunction`` through its real bind -> init -> process(sink)
-> combine -> finalize lifecycle without spawning a worker process, so unit
tests stay fast and debuggable while still exercising the framework's argument
parsing, storage round-trip, and output schema.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from vgi.arguments import Arguments
from vgi.function_storage import BoundStorage, FunctionStorageSqlite
from vgi.invocation import FunctionType
from vgi.protocol import BindRequest, InitRequest
from vgi.table_buffering_function import TableBufferingParams


class _Collector:
    """Captures emitted batches from a finalize stream."""

    def __init__(self) -> None:
        self.batches: list[pa.RecordBatch] = []
        self.finished = False

    def emit(self, batch: pa.RecordBatch, *_a: Any, **_kw: Any) -> None:
        self.batches.append(batch)

    def finish(self) -> None:
        self.finished = True

    def client_log(self, *_a: Any, **_kw: Any) -> None:
        pass


def run_buffering(
    func_cls: type,
    table: pa.Table,
    *,
    named: dict[str, Any] | None = None,
) -> pa.Table:
    """Drive a recommender buffering function over a whole input ``table``.

    Args:
        func_cls: The ``TableBufferingFunction`` subclass to run.
        table: The interaction relation (the ``(SELECT ...)`` data) as an Arrow
            table.
        named: Named args (column roles + hyperparameters), e.g.
            ``{"user": "u", "item": "i", "n": 3}``.

    Returns:
        The emitted result as a single Arrow table (the function's output).
    """
    input_schema = table.schema
    args = Arguments(
        positional=(),
        named={k: pa.scalar(v) for k, v in (named or {}).items()},
    )

    bind_req = BindRequest(
        function_name=func_cls.Meta.name,
        arguments=args,
        function_type=FunctionType.TABLE_BUFFERING,
        input_schema=input_schema,
    )
    bind_resp = func_cls.bind(bind_req)

    init_req = InitRequest(bind_call=bind_req, output_schema=bind_resp.output_schema)
    init_resp = func_cls.global_init(init_req)
    execution_id = init_resp.execution_id

    storage = BoundStorage(FunctionStorageSqlite(":memory:"), execution_id)
    parsed_args = func_cls._parse_arguments(func_cls.FunctionArguments, args)

    def make_params() -> TableBufferingParams:
        return TableBufferingParams(
            args=parsed_args,
            init_call=init_req,
            init_response=init_resp,
            output_schema=bind_resp.output_schema,
            settings={},
            secrets={},
            storage=storage,
            execution_id=execution_id,
            attach_id=b"",
            transaction_id=None,
            function_name=func_cls.Meta.name,
        )

    # Sink phase: one process() call per input batch.
    state_ids: list[bytes] = []
    for batch in table.to_batches():
        state_ids.append(func_cls.process(batch, make_params()))

    # Combine phase.
    finalize_ids = func_cls.combine(state_ids, make_params())

    # Source phase: drain each finalize stream.
    out = _Collector()
    for fid in finalize_ids:
        params = make_params()
        state = func_cls.initial_finalize_state(fid, params)
        while not out.finished:
            func_cls.finalize(params, fid, state, out)

    return pa.Table.from_batches(out.batches, schema=bind_resp.output_schema)
