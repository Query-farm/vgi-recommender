# CLAUDE.md — vgi-recommender

Contributor/agent notes. User-facing docs live in `README.md`; this is the
"how it's built and where the sharp edges are" companion. Sibling style/tooling
to `vgi-conform` / `vgi-survival` (structure) and `vgi-scikit-learn` /
`vgi-survival` (the whole-relation buffering data-flow).

## What this is

A [VGI](https://query.farm) worker exposing **collaborative-filtering
recommendations** to DuckDB/SQL via [implicit](https://github.com/benfred/implicit)
(MIT) implicit-feedback ALS: top-N per user, item-item similarity, single-user
recs. `recommender_worker.py` assembles every function into one `recommender`
catalog (single `main` schema) over stdio.

## Layout

```
recommender_worker.py    repo-root stdio entry shim; PEP 723 inline deps; main()
vgi_recommender/
  recommender.py         pure implicit-ALS logic over pandas frames; no Arrow/VGI; unit-testable
  buffering.py           SinkBuffer (single-bucket sink/combine) + Arrow<->pandas plumbing
  tables.py              the three TableBufferingFunction wrappers + output schemas + arg classes
  schema_utils.py        pa.Field comment / column-doc helper
  worker.py              assembles the catalog; main() / main_http()
tests/                   pytest: test_recommender (pure), test_tables (in-proc harness),
                         test_client (Client RPC); data.py = planted-signal matrix
test/sql/*.test          haybarn-unittest sqllogictest — authoritative E2E
Makefile                 test / test-unit / test-sql / lint
```

To add a function: implement the math in `recommender.py` (pure, takes a pandas
frame + role kwargs, returns a `dict[str, list]`, raises `RecommenderError` on
bad input), add a `pa.schema` + `@dataclass` args class + a `SinkBuffer`
subclass in `tables.py`, append it to `TABLE_FUNCTIONS`.

## THE core convention (read first): one relation in, named args

These are **table functions**, not scalars. Each takes the whole interaction
relation as a single `(SELECT user, item, value)` subquery — `Arg(0)`, typed
`TableInput` — and the column **roles** plus hyperparameters as NAMED args
(`user := 'u'`, `item := 'i'`, `value := 'v'`, `n := 10`, `factors := 50`).
`recommend_for` additionally takes `target_user := <id>` — a named **scalar**
arg, the one user to recommend for. The relation's columns *are* the data; the
named args just name which column plays which role. This mirrors
`vgi-survival`'s `kaplan_meier(..., duration := 't', event := 'e')`.

ALS factorizes the **whole** user × item matrix at once, so every function is a
`TableBufferingFunction` (Sink+Source):

- `process(batch)` — sink each input batch to execution-scoped `BoundStorage`.
- `combine(state_ids)` — collapse to a single finalize key (one bucket).
- `finalize(...)` — reassemble the full table (`buffered_frame()` → pandas),
  fit the ALS model once, emit one result batch, then `out.finish()`.

`SinkBuffer` in `buffering.py` implements `process`/`combine`/`buffered_frame`;
each function only writes `on_bind` (its output schema) + `finalize`. A
`DrainState(done: bool)` cursor makes finalize emit exactly once.

## Sharp edges (learned the hard way)

1. **`haybarn-unittest` silently SKIPS `require vgi`.** Under haybarn the
   extension isn't autoloaded for `require`, so a `.test` using `require vgi` is
   SKIPPED, not run. Use an explicit `statement ok` / `LOAD vgi;` (the `.test`
   here does).
2. **Determinism is load-bearing.** ALS is randomized. We pin `random_state=42`,
   `num_threads=1`, and `OPENBLAS/MKL/OMP_NUM_THREADS=1` (set *before* numpy
   imports its backend, at the top of `recommender.py`). Without single-threaded
   BLAS the factors — and thus the rankings — drift run to run and the planted
   assertions go flaky. Ids are sorted before indexing so index assignment is
   stable too.
3. **`recommend()` pads with already-liked items scored `-inf`.** With
   `filter_already_liked_items=True`, implicit still *returns* N entries; if
   fewer than N novel items remain it fills the tail with already-liked items at
   score `-3.4e38` (`-inf`). We drop non-finite scores, which is also how
   already-seen exclusion is enforced. `similar_items` always includes the item
   itself at similarity 1.0 — we request `N+1` and skip the self-match.
4. **`value` is confidence, and its column may be absent.** `value` has a default
   arg name (`'value'`); if that column isn't in the relation we fall back to
   all-ones implicit feedback rather than erroring. A *present* but non-numeric
   value column raises a clear `RecommenderError`.
5. **`factors` is clamped on tiny matrices.** A factor count ≥ min(#users,
   #items) is meaningless and can blow up; `_Indexed` clamps it to
   `min(factors, min(n_users, n_items) - 1)` so the small test matrices fit.
6. **Cold / unknown entities don't crash.** Unknown `target_user` → zero rows; a
   user/item off the main cluster still factorizes (and is never recommended its
   own seen items).
7. **The unit suite can pass while the RPC path is broken.** `test_recommender.py`
   calls pure functions; only `test_tables.py` (in-proc bind→process→finalize),
   `test_client.py` (real `vgi.client.Client` subprocess), and `test/sql/*.test`
   exercise the framework/wire. **Run the SQL suite** — it's authoritative.

## Planted-signal validation

`tests/data.py` builds a deliberately rigged matrix: `u1`/`u2` interacted with
`{A, B, C}`; `u3`/`u4` with only `{A, B}`. Collaborative filtering must then
recommend `C` to `u3` and `u4` (asserted in pure, in-proc, Client, and SQL
tests). `A` and `B` are co-purchased by exactly the same users, so they are each
other's top `similar_items` neighbour. `u5`/`D` sit off the cluster as a
cold-ish edge case.

## Licensing

`implicit` is **MIT**; `scipy`/`numpy`/`pandas` are **BSD** — all permissive, no
copyleft. The worker's own code is MIT. No vendoring, no patched deps.

## Testing

```sh
uv sync --extra dev
uv run --no-sync pytest -q     # pure logic + in-proc tables + Client RPC E2E
make test-sql                  # haybarn-unittest over test/sql/*  (authoritative)
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_recommender/
```

`make test-sql` sets `VGI_RECOMMENDER_WORKER="uv run --python 3.13
recommender_worker.py"`, puts `~/.local/bin` on PATH, and runs `haybarn-unittest
--test-dir . "test/sql/*"`. Install the runner once with
`uv tool install haybarn-unittest`.
```
