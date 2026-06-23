<p align="center">
  <img src="https://raw.githubusercontent.com/Query-farm/vgi/main/docs/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# Collaborative-Filtering Recommendations (ALS) in DuckDB

> **vgi-recommender** · a [Query.Farm](https://query.farm) VGI worker · powered by implicit

A [VGI](https://query.farm) worker that brings **collaborative-filtering
recommendations** to DuckDB/SQL: top-N recommendations per user, item-item
similarity, and recommendations for a single user — backed by
[implicit](https://github.com/benfred/implicit) (MIT) implicit-feedback
Alternating Least Squares (ALS).

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'recommender' (TYPE vgi, LOCATION 'uv run recommender_worker.py');

-- Top-10 recommended items per user (already-seen items excluded)
SELECT * FROM recommender.recommend_all((SELECT u, i, v FROM events),
                                        user := 'u', item := 'i', value := 'v', n := 10)
ORDER BY user, rank;

-- Top-5 most similar items per item (cosine of ALS item factors)
SELECT * FROM recommender.similar_items((SELECT u, i, v FROM events),
                                        user := 'u', item := 'i', value := 'v', n := 5)
ORDER BY item, rank;

-- Top-5 recommendations for ONE user
SELECT * FROM recommender.recommend_for((SELECT u, i, v FROM events),
                                        user := 'u', item := 'i', value := 'v',
                                        target_user := 'alice', n := 5)
ORDER BY rank;
```

## Data flow: one relation in, a result set out

Every function is a **table function** that consumes a *whole interaction
relation* — passed as a single `(SELECT user, item, value)` subquery (the
positional argument) — and emits a result set. The roles of the columns inside
that relation, plus the hyperparameters, are passed as **named arguments**:

| named arg | meaning |
|-----------|---------|
| `user := 'col'`   | the user-id column |
| `item := 'col'`   | the item-id column |
| `value := 'col'`  | the interaction strength (confidence); defaults to `1.0` if the column is absent |
| `n := 10`         | number of results per user/item |
| `factors := 50`   | ALS latent-factor dimensionality |
| `target_user := <id>` | (`recommend_for` only) the single user to recommend for — a named **scalar** arg |

The relation *is* the data; the named args just say which column plays which
role. Because ALS factorizes the **entire** user × item matrix at once, these
are buffering (Sink+Source) functions — they buffer all input batches, then fit
the model once. String user/item ids are mapped to integer indices internally;
the original string ids are always emitted back.

## Implicit-feedback ALS semantics

This models **implicit feedback**: each `(user, item, value)` row is positive
evidence that the user interacted with the item, and `value` is the interaction
*strength* / **confidence** (play count, purchase count, rating). There are no
explicit negatives — absence of a row is "unknown", not "disliked". This is the
Hu/Koren/Volinsky (2008) implicit-ALS formulation that `implicit` implements: it
factorizes the user × item confidence matrix into latent user and item factors.

- A **recommendation score** is the dot product of a user's factors with an
  item's factors.
- **Item-item similarity** is the cosine of two items' ALS factors.
- If no `value` column is supplied (or every value is the same), all confidences
  are `1.0` — pure "did/didn't interact" feedback.

## Determinism

ALS is seeded (`random_state=42`) and runs a fixed number of sweeps; the worker
pins `num_threads=1` and the BLAS thread counts so identical input yields
identical factors and therefore reproducible rankings. Combined with `ORDER BY`
in queries, results are stable across runs.

## Functions

| function | returns |
|----------|---------|
| `recommend_all(rel, user, item, value, n, factors)` | `(user, item, score, rank)` — top-N per user, already-seen excluded |
| `similar_items(rel, user, item, value, n, factors)` | `(item, similar_item, similarity, rank)` — top-N per item |
| `recommend_for(rel, user, item, value, target_user, n, factors)` | `(item, score, rank)` — top-N for one user |

## Robustness

Missing `user`/`item` columns, an empty relation, or a non-numeric `value`
column all surface a **clear error** rather than crashing the worker. An unknown
`target_user`, a user with no interactions, or a cold item are handled
gracefully (empty / no self-recommendation).

## Development

```sh
uv sync --extra dev
uv run --no-sync pytest -q                 # unit + in-proc tables + Client RPC E2E
make test-sql                              # haybarn-unittest SQL E2E (authoritative)
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_recommender/
```

## Licensing

This worker is MIT. `implicit` is **MIT**; `scipy` is **BSD**; `numpy` and
`pandas` are **BSD** — all permissive, no copyleft obligations.

---

## Authorship & License

Written by [Query.Farm](https://query.farm).

Copyright 2026 Query Farm LLC - https://query.farm

