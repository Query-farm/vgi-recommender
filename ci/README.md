# CI: the vgi-recommender worker integration suite

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs the unit tests
and this repo's sqllogictest suite (`test/sql/*.test`) against the vgi-recommender
VGI worker through the **real DuckDB `vgi` extension** on every push / PR.

## How it works (no C++ build)

CI drives a **prebuilt** standalone `haybarn-unittest` and installs the
**signed** `vgi` extension from the Haybarn community channel:

1. **Install the worker** — `uv sync --frozen` installs the package and its
   `vgi-recommender` console script into the venv; the .test files ATTACH it via
   `.venv/bin/vgi-recommender`.
2. **Download the runner** — the matching `haybarn_unittest-*` asset per platform.
3. **Preprocess** — [`preprocess-require.awk`](preprocess-require.awk) injects a
   signed `INSTALL vgi FROM community;` before each bare `LOAD vgi;`.
4. **Run** — [`run-integration.sh`](run-integration.sh) stages the preprocessed
   tree, points `VGI_RECOMMENDER_WORKER` at the console script, and runs the suite.

## Run it locally

```bash
uv sync --python 3.13
HAYBARN_UNITTEST=/path/to/haybarn-unittest \
VGI_RECOMMENDER_WORKER="$PWD/.venv/bin/vgi-recommender" \
  ci/run-integration.sh
```
