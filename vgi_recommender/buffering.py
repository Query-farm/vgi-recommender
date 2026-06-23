"""Shared plumbing for the table-buffering recommender functions.

Every recommender function (recommend_all, similar_items, recommend_for) must
see the *whole* interaction relation before it can produce any output: ALS
factorizes the entire user x item matrix at once. They are therefore
``TableBufferingFunction`` (Sink+Source) functions. The sink phase serializes
each input batch to execution-scoped storage; finalize reassembles the full
table and fits the model once.

This module holds the single-bucket sink/combine implementation (``SinkBuffer``)
plus the Arrow (de)serialization and a ``pandas`` assembly helper, so each
function only writes its ``finalize`` logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyarrow as pa
from vgi.table_buffering_function import TableBufferingFunction, TableBufferingParams
from vgi_rpc import ArrowSerializableDataclass

_DATA_KEY = b"input_batches"


@dataclass(kw_only=True)
class DrainState(ArrowSerializableDataclass):
    """Per-finalize-stream cursor: emit the single result batch once, then finish."""

    done: bool = False


def serialize_batch(batch: pa.RecordBatch) -> bytes:
    """Serialize one RecordBatch to a self-describing Arrow IPC stream."""
    sink = pa.BufferOutputStream()
    # pyarrow's stubs leave new_stream untyped; the call is sound.
    with pa.ipc.new_stream(sink, batch.schema) as writer:  # type: ignore[no-untyped-call]
        writer.write_batch(batch)
    result: bytes = sink.getvalue().to_pybytes()
    return result


def deserialize_batches(value: bytes) -> list[pa.RecordBatch]:
    """Inverse of :func:`serialize_batch` for one stored blob."""
    # pyarrow's stubs leave open_stream untyped; the call is sound.
    reader = pa.ipc.open_stream(pa.BufferReader(value))  # type: ignore[no-untyped-call]
    batches: list[pa.RecordBatch] = reader.read_all().to_batches()
    return batches


def input_schema_of(params: Any) -> pa.Schema:
    """Input schema from a process/finalize params object."""
    schema = params.init_call.bind_call.input_schema
    assert schema is not None
    return schema


class SinkBuffer[TArgs, TState](TableBufferingFunction[TArgs, TState]):
    """Single-bucket sink/combine: buffer every input batch under one key.

    Subclasses implement ``on_bind``, ``initial_finalize_state``, and
    ``finalize`` (calling ``buffered_frame(params)`` to get the full input as a
    ``pandas.DataFrame``).
    """

    @classmethod
    def process(cls, batch: pa.RecordBatch, params: TableBufferingParams[TArgs]) -> bytes:
        """Sink one input batch under the single buffering key.

        Args:
            batch: One input record batch.
            params: The buffering params (storage + execution id).

        Returns:
            The execution id, used as this stream's combine key.
        """
        if batch.num_rows:
            params.storage.state_append(_DATA_KEY, b"", serialize_batch(batch))
        return params.execution_id

    @classmethod
    def combine(cls, state_ids: list[bytes], params: TableBufferingParams[TArgs]) -> list[bytes]:
        """Collapse every sink state into the single finalize bucket.

        Args:
            state_ids: The per-process state ids to combine.
            params: The buffering params (carries the execution id).

        Returns:
            A one-element list with the single finalize key.
        """
        return [params.execution_id]

    @classmethod
    def buffered_frame(cls, params: TableBufferingParams[TArgs]) -> pd.DataFrame:
        """Reassemble all sunk batches into a single pandas DataFrame.

        Returns an empty (zero-row) frame -- with the right column names -- when
        no rows were sunk, so finalize can apply uniform empty-input handling.
        """
        input_schema = input_schema_of(params)
        batches: list[pa.RecordBatch] = []
        for _sid, value in params.storage.state_log_scan(_DATA_KEY, b""):
            batches.extend(deserialize_batches(value))
        if not batches:
            return pa.Table.from_batches([], schema=input_schema).to_pandas()
        return pa.Table.from_batches(batches, schema=input_schema).to_pandas()
