"""Shared Arrow-schema helper for the recommender worker.

Keeps column-comment plumbing in one place so every function exposes
consistent, documented output schemas to DuckDB.
"""

from __future__ import annotations

import pyarrow as pa


def field(
    name: str,
    type: pa.DataType,
    comment: str,
    *,
    nullable: bool = True,
) -> pa.Field:
    """Build a ``pa.Field`` carrying a column comment in its metadata.

    The ``comment`` metadata key is the framework's transport for column
    comments -- DuckDB surfaces it via ``duckdb_columns()`` and ``DESCRIBE``.

    Args:
        name: Column name.
        type: Arrow data type.
        comment: Human-readable column comment.
        nullable: Whether the column is nullable.

    Returns:
        A pyarrow Field with the comment attached as metadata.
    """
    return pa.field(
        name,
        type,
        nullable=nullable,
        metadata={b"comment": comment.encode("utf-8")},
    )
