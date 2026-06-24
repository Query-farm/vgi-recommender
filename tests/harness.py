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


# Cap on finalize ticks when draining with state re-serialization. A correct
# offset cursor drains in ceil(total / ROWS_PER_TICK) + 1 ticks; the old
# position-less "emit all then finish" finalize restarts from row 0 on every
# re-deserialized state and never terminates, so it overruns this guard.
_MAX_TICKS = 10_000


def run_buffering(
    func_cls: type,
    table: pa.Table,
    *,
    named: dict[str, Any] | None = None,
    serialize_state: bool = False,
) -> pa.Table:
    """Drive a recommender buffering function over a whole input ``table``.

    Args:
        func_cls: The ``TableBufferingFunction`` subclass to run.
        table: The interaction relation (the ``(SELECT ...)`` data) as an Arrow
            table.
        named: Named args (column roles + hyperparameters), e.g.
            ``{"user": "u", "item": "i", "n": 3}``.
        serialize_state: If true, wire-serialize and re-deserialize the finalize
            state between every ``finalize`` tick (mimicking the HTTP
            continuation token round-trip). A correct offset cursor survives this
            and terminates; a position-less cursor restarts from row 0 and
            overruns the ``_MAX_TICKS`` guard (raising ``RuntimeError``).

    Returns:
        The emitted result as a single Arrow table (the function's output).

    Raises:
        RuntimeError: If finalize does not terminate within ``_MAX_TICKS`` ticks
            (i.e. the cursor does not survive state re-serialization).
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
        ticks = 0
        while not out.finished:
            func_cls.finalize(params, fid, state, out)
            ticks += 1
            if ticks > _MAX_TICKS:
                raise RuntimeError(
                    f"{func_cls.Meta.name}.finalize did not terminate within {_MAX_TICKS} ticks "
                    "(finalize cursor does not survive state re-serialization)"
                )
            if serialize_state and not out.finished:
                # Round-trip the finalize state through the wire exactly as the
                # stateless HTTP continuation token does between ticks.
                state = type(state).deserialize_from_bytes(state.serialize_to_bytes())

    return pa.Table.from_batches(out.batches, schema=bind_resp.output_schema)
