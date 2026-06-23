"""End-to-end tests driving recommender_worker.py as a real subprocess.

These spawn the worker via ``vgi.client.Client`` and invoke each function
through the real ``table_buffering_function`` RPC path -- exactly how DuckDB
drives a buffering function after ``ATTACH`` -- exercising bind, the sink
process RPC per batch, combine, and the finalize source stream over the wire.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from vgi import Arguments
from vgi.client import Client, ClientError

from .data import planted_frame

_WORKER = str(Path(__file__).resolve().parent.parent / "recommender_worker.py")


@pytest.fixture(scope="module")
def client() -> Iterator[Client]:
    with Client(f"{sys.executable} {_WORKER}", worker_limit=1) as c:
        yield c


def _run(client: Client, name: str, table: pa.Table, **named: Any) -> pa.Table:
    batches = list(
        client.table_buffering_function(
            function_name=name,
            input=iter(table.to_batches()),
            arguments=Arguments(named={k: pa.scalar(v) for k, v in named.items()}),
        )
    )
    return pa.Table.from_batches(batches)


def _arrow(with_value: bool = False) -> pa.Table:
    return pa.Table.from_pandas(planted_frame(with_value=with_value), preserve_index=False)


def test_recommend_all_e2e(client: Client) -> None:
    out = _run(client, "recommend_all", _arrow(with_value=True), user="user", item="item", value="value", n=3)
    d = out.to_pydict()
    recs = set(zip(d["user"], d["item"], strict=True))
    assert ("u3", "C") in recs  # planted signal over the real RPC path
    assert ("u4", "C") in recs


def test_similar_items_e2e(client: Client) -> None:
    out = _run(client, "similar_items", _arrow(), user="user", item="item", n=1)
    d = out.to_pydict()
    top = {it: s for it, s, rk in zip(d["item"], d["similar_item"], d["rank"], strict=True) if rk == 1}
    assert top["A"] == "B" and top["B"] == "A"


def test_recommend_for_e2e(client: Client) -> None:
    out = _run(client, "recommend_for", _arrow(), user="user", item="item", target_user="u3", n=3)
    d = out.to_pydict()
    assert "C" in d["item"]
    assert "A" not in d["item"] and "B" not in d["item"]


def test_missing_column_errors_e2e(client: Client) -> None:
    with pytest.raises(ClientError):
        _run(client, "recommend_all", _arrow(), user="nope", item="item", n=3)
