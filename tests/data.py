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
