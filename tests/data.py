"""Small constructed interaction matrix with a predictable, planted signal.

The data is engineered so collaborative filtering has an unambiguous answer:

- Users ``u1`` and ``u2`` interacted with items ``A, B, C``.
- Users ``u3`` and ``u4`` interacted with only ``A, B``.

Because ``u3``/``u4`` look just like ``u1``/``u2`` minus ``C``, ALS should
recommend ``C`` to ``u3`` and ``u4`` (the planted signal). Items ``A`` and ``B``
are co-purchased by exactly the same users, so they are each other's nearest
neighbour under ``similar_items``. ``u5`` only touched the otherwise-isolated
item ``D`` (a cold-ish item), giving us a user/item pair off the main cluster.
"""

from __future__ import annotations

import pandas as pd

# (user, item) interactions; value defaults to 1.0 (implicit "did interact").
_INTERACTIONS: list[tuple[str, str]] = [
    ("u1", "A"),
    ("u1", "B"),
    ("u1", "C"),
    ("u2", "A"),
    ("u2", "B"),
    ("u2", "C"),
    ("u3", "A"),
    ("u3", "B"),
    ("u4", "A"),
    ("u4", "B"),
    ("u5", "D"),
]


def planted_frame(*, with_value: bool = False) -> pd.DataFrame:
    """Return the planted interaction relation as a DataFrame.

    Args:
        with_value: If true, include a numeric ``value`` (confidence) column;
            otherwise return only ``user`` / ``item``.
    """
    df = pd.DataFrame(_INTERACTIONS, columns=["user", "item"])
    if with_value:
        df["value"] = 1.0
    return df


def cohort_frame(n_users: int, n_items: int, *, held_out: int = 2) -> pd.DataFrame:
    """Return an ``n_users`` x ``n_items`` interaction relation with held-out items.

    Every user interacts with every item EXCEPT a small per-user ``held_out`` set,
    so collaborative filtering has novel items to recommend back. Because users
    overlap heavily, ALS recommends each user's held-out items -- giving roughly
    ``held_out`` result rows per user. With a large cohort the total result spans
    many ``ROWS_PER_TICK`` slices, exercising the HTTP-continuation offset cursor.

    Args:
        n_users: Number of distinct users (``u000``..).
        n_items: Number of distinct items (``i000``..).
        held_out: Items each user has NOT interacted with (its candidate recs).
    """
    rows: list[tuple[str, str]] = []
    for u in range(n_users):
        held = {(u * held_out + k) % n_items for k in range(held_out)}
        for i in range(n_items):
            if i not in held:
                rows.append((f"u{u:03d}", f"i{i:03d}"))
    return pd.DataFrame(rows, columns=["user", "item"])
