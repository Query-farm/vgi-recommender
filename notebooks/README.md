# Item Recommender example

[01 · How does a larger recommendation slate change exposure?](01_audit_recommendation_slates.ipynb)
uses the worker’s built-in sample to establish a novelty ceiling, then audits
recommendations over an authored eight-learner, eight-module matrix. It compares
single-user and all-user APIs, keeps exhausted and unknown users distinct, and
shows item-neighbor similarity separately from user recommendation scores.
[Open in Colab](https://colab.research.google.com/github/Query-farm/vgi-recommender/blob/main/notebooks/01_audit_recommendation_slates.ipynb).

Choose **Runtime → Run all** in Colab. The short bootstrap installs only the shared
base package, preserving compatible preloaded libraries, and downloads immutable
SHA256-verified presentation code. For local execution install
[uv](https://docs.astral.sh/uv/) and Git, then run from `notebooks/`:

```sh
uv sync --locked
uv run --locked vgi-notebooks
uv run --locked vgi-notebooks 01_audit_recommendation_slates.ipynb --parameters '{"SLATE_SIZE":3}' --output-dir artifacts/slate-three
```

For Jupyter select `.venv/bin/python`. The locked `[runner]` extra is local tooling;
Colab does not install that extra. Parameter overrides require a separate output
directory, unknown parameters fail, and a failed run cannot replace the successful
default. If old setup replaced kernel libraries, open this version in a fresh runtime.

## Portable release

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'recommender' (TYPE vgi,
  LOCATION 'uvx --python 3.13 --from git+https://github.com/Query-farm/vgi-recommender@v0.4.0 vgi-recommender');
```

This short command was verified through Haybarn 2.0.0rc11 using the release’s own
dependency declarations, without the older example’s transitive overrides. It
resolved vgi-python 0.51.1, vgi-rpc 0.53.1, PyArrow 26.0.1, implicit 0.7.3, NumPy
2.5.4, SciPy 1.18.1 and pandas 3.1.1 in a separate cached Python 3.13 worker
environment. The local notebook lock does not lock this worker environment.

Initial Git/package/extension/companion retrieval requires internet and compatible
numerical wheels. The runtime was tested on macOS arm64; other platforms were not
executed. No local source checkout, worker `.venv`, credentials, service or uploaded
dataset is required. SQL persistent secrets are isolated by shared connection setup.
Each function buffers the whole relation, trains its own model, then returns rows.
The example materializes each call before DataFrame capture, avoiding a second fit
when the SQL display machinery inspects results.

The [shared v0.3.24 standard](https://github.com/Query-farm/vgi-example-notebooks/blob/v0.3.24/docs/standard.md)
provides the catalog-derived title/subtitle, branding, isolated connection, SVG
output and runner protections. Keep `charts/` beside the saved notebook for GitHub
image links; interactive runs retain native SVG. Review PNGs, logs and separate
experiments remain ignored. Hosted Colab was not manually operated; the validation
uses a fresh compatible kernel with eight libraries preloaded before bootstrap.

## Coverage and release boundary

Complete actual release and current catalogs contain **four public logical
surfaces**: `recommend_all`, `recommend_for`, `similar_items` and the
`sample_interactions` view. All four have meaningful SQL execution, results and
interpretation in this notebook. Catalog exports, full argument/output schemas,
reviewed coverage mappings and hashed execution receipts live in `coverage.json`
and `coverage/`. The actual sample view’s `value` column is `DECIMAL(2,1)`;
metadata prose describes it as DOUBLE. The snapshot retains the actual SQL type.

The notebook executes **v0.4.0**, source `ea8f44a987cecb5a4ff584fb4bae36f77849ca7b`.
Current public source at `d910f9d42221ab0ac3de8b0743988594764a88bd` has identical
interfaces, but adds a cap and deduplication to `similar_items` when requests
exceed the number of items. The release can emit padded duplicate neighbors in
that case. This notebook requests three neighbors among eight items, below that
boundary, and checks uniqueness. Compatible current coverage does not claim
identical behavior for oversized requests or a separately executed current-main
notebook. Other source differences are composed examples and metadata.

Worker source and root dependencies remain untouched. Existing green
[source CI](https://github.com/Query-farm/vgi-recommender/actions/runs/43582186464)
is reused; no new source rebuild, full test repetition, container publication or
worker release is required for this notebook update.

## What the experiment means

The built-in ten-row, four-user, three-item dataset is deliberately planted.
Users u1/u2 have seen every item; u3/u4 have only C left. The two recommendations
therefore equal the eligible-candidate ceiling. That verifies input shape and
novelty handling, not predictive skill. The larger matrix remains authored SQL:
interaction counts are positive confidence strengths, not ratings. An absent
interaction is unknown, not disliked. Titles are display labels, not model features.

The model uses seed 42, 30 sweeps, regularization 0.01 and one numerical thread.
The notebook sets OPENBLAS/MKL/OMP thread variables before launching the child and
restores their previous values at cleanup. It uses three factors for the 8×8
matrix and two for the 4×3 sample, both below the worker’s tiny-matrix clamp.
Determinism supports within-runtime comparison; floating-point values may differ
with future numerical libraries or architectures.

`SLATE_SIZE` changes only the maximum number of requested novel recommendations,
from **2 to 3** (allowed integers 1–4). Seven learners can gain one slot, while
L08 has seen every module and stays at zero. The unknown learner NEW also returns
zero, for a different reason. Exposure counts describe proposed slates, not delivered
impressions, clicks or learning outcomes. A larger list can include negative
factor affinity and is not evidence of improved recommendations.

The two panels keep interaction counts and ordinal ranks on separate fixed scales;
blank is not zero, and gray × marks previously seen modules. Item similarities
are factor cosines, while user recommendation scores are factor dot products.
Neither is a probability. Rounded equal cosines need not imply exact ties.
The three-neighbor query for C stays fixed across slate-size experiments.

Validation includes the default and a separate slate-three run, four reviewed
figures, 14 SQL checks, three built-in sample assertions, runner preservation guards
and a fresh preloaded-kernel bootstrap. No held-out labels or predictive evaluation
are supplied; the example audits API behavior and proposed exposure.
