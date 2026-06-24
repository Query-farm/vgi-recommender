# CI: the vgi-recommender worker integration suite

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs the unit tests
and this repo's sqllogictest suite (`test/sql/*.test`) against the vgi-recommender
VGI worker through the **real DuckDB `vgi` extension** on every push / PR.

## How it works (no C++ build)

CI drives a **prebuilt** standalone `haybarn-unittest` and installs the
**signed** `vgi` extension from the Haybarn community channel:

1. **Install the worker** — `uv sync --frozen --extra http` installs the package
   and its deps (including waitress, via the `http` extra, so the worker can
   serve over HTTP) into the venv.
2. **Download the runner** — the matching `haybarn_unittest-*` asset per platform.
3. **Preprocess** — [`preprocess-require.awk`](preprocess-require.awk) injects a
   signed `INSTALL vgi FROM community;` before each bare `LOAD vgi;`.
4. **Run** — [`run-integration.sh`](run-integration.sh) stages the preprocessed
   tree, resolves `VGI_RECOMMENDER_WORKER` from `$WORKER_CMD` per `$TRANSPORT`,
   and runs the suite.

## Three transports

The SAME suite runs over all three VGI transports — a CI matrix
(`transport: [subprocess, http, unix]` × `os: [ubuntu, macos]`). The vgi
extension selects the transport from the ATTACH `LOCATION` string that
`run-integration.sh` builds:

- **subprocess** (default) — `LOCATION` is the bare stdio command
  (`uv run recommender_worker.py`); the extension spawns the worker per query
  and talks Arrow IPC over stdin/stdout.
- **http** — the script boots the worker with `--http --port 0 --port-file <f>`,
  waits for the port file, and sets `LOCATION='http://127.0.0.1:<port>'`. The
  vgi HTTP transport rides on DuckDB's `httpfs`, so the script injects a signed
  `INSTALL httpfs FROM core; LOAD httpfs;` into each staged file (without it the
  ATTACH errors with "VGI HTTP transport requires the httpfs extension", and the
  sqllogictest runner's skip-list would silently SKIP the whole suite — a fake
  pass). Needs the `http` extra (waitress).
- **unix** — the script boots the worker with `--unix <sock>`, waits for the
  socket, and sets `LOCATION='unix://<sock>'` (AF_UNIX launcher).

For http/unix the script boots the worker out-of-band and trap-kills it on exit.

## Run it locally

```bash
uv sync --python 3.13 --extra http
export PATH="$HOME/.local/bin:$PATH"   # haybarn-unittest

HAYBARN_UNITTEST=$(which haybarn-unittest) TRANSPORT=subprocess ci/run-integration.sh
HAYBARN_UNITTEST=$(which haybarn-unittest) TRANSPORT=http       ci/run-integration.sh
HAYBARN_UNITTEST=$(which haybarn-unittest) TRANSPORT=unix       ci/run-integration.sh
```

Each must end with `All tests passed (N>0 ...)`. (The http leg reports two extra
assertions — the injected `httpfs` INSTALL/LOAD — and is otherwise identical.)
`WORKER_CMD` defaults to `uv run --python 3.13 <repo>/recommender_worker.py`.
