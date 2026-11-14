# Recommendation notebooks

[How does a larger recommendation slate change exposure?](01_audit_recommendation_slates.ipynb)
uses an authored 8-learner × 8-module corpus. Real ALS worker calls generate
novel recommendations and item similarities; SQL checks their identities,
ranks, exclusions and score consistency. This is no predictive-accuracy study.

## Setup and execution

Install uv and Git. This entire directory runs outside the source checkout.

```sh
cd notebooks
uv sync --locked
uv run python execute.py
uv run python -m ipykernel install --user --name vgi-recommender-notebooks --display-name 'Python (Recommender notebooks)'
uv run --with jupyterlab jupyter lab
```

Select that kernel interactively. The runner uses a temporary kernel and this
directory as cwd, replacing a notebook atomically only after all cells succeed.
The locked notebook environment pins published Haybarn **2.0.0rc11**.

Visible SQL attaches public [worker tag v0.4.0](https://github.com/Query-farm/vgi-recommender/tree/v0.4.0)
through uvx, with Python 3.13, SDK 0.51.0 / RPC 0.53.1, implicit 0.7.3,
NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1 and PyArrow 26.0.1. No local worker binary or
source environment is discovered. First launch needs internet for Git/packages
and the extension. Subsequent analytical work uses only authored SQL inputs.
To upgrade, edit the tag and dependency pins and repeat both experiments/checks.
The tag's similar-items padding behavior for n at least the item count is outside
this notebook: it requests three neighbors from eight items. Current source has
a later fix for that case; source checks and the tested release are distinguished.

The worker fixes ALS to 30 sweeps, seed 42, regularization 0.01 and one model
thread. The notebook temporarily sets BLAS thread variables to 1 before worker
launch and restores the caller's values at cleanup. Three factors fit the 8×8
input without clamping. Floating-point backends can still differ by platform.

## Experiment

```sh
uv run python execute.py --parameters '{"SLATE_SIZE":3}' --output-dir artifacts/three-slots
```

Valid slate sizes: integers 1–4. Overrides need a separate destination; successful
defaults are preserved on failure. Copies include branding, docs, helper, runner
and environment files. Compare recommendations and proposed item exposure, with
input and model fixed. Reset to 2. Scores are model affinities, not probabilities;
missing interactions are unknown, not negative ratings. L08 has no novel candidates;
the separate unknown-user call has no learned user factors. These are distinct
zero-row outcomes, not unavailable scores to replace with zeros.

## Validation

The pinned v0.4.0 notebook passes with 14 code cells, 8 visible SQL cells,
2 charts and 14 invariants: default 3.9 seconds and detached notebooks-only
three-slot experiment 4.0 seconds. All four charts were reviewed, with a fixed
rank color scale across experiments. Returned slots rise from 14 to 21 across
seven learners; L08 and unknown NEW still return zero. The authored-input
diagnostic and item-neighbor displays are identical between runs. Three
rejected-run guards verify preservation of successful defaults.

Current unchanged source checks (revision 1f41e64a54fcb004699b8f365fc2bba74d0e3fbf)
passed separately: 27 pytest tests; 22 subprocess, 24 HTTP and 22 Unix SQL
assertions; Ruff lint/format including the notebook, mypy on 7 files, and isolated
pydoclint. Source metadata scores 100/100 at L2 with zero findings. These source
gates include later changes than the pinned release; the saved notebook and
experiment directly validate the release features they use. No source or root
lockfile changes were made.
