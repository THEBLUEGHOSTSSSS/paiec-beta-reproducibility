# Reproduce the fixed Beta diagnostic

Use Python 3.11 or newer. The recorded local run used Python 3.13 and PyArrow 24.0.0.
The candidate itself needs only Python's standard library.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest test_reproduce -v
```

Obtain the eight Parquet files at the exact URLs in `data_sources.json`, subject
to the upstream dataset terms. Put each file under `data/<dataset>/<filename>`;
the two dataset directories are `real_webagents` and `researchcodebench`.
The manifest preserves original relative provenance paths, but `--data-root`
points directly to the directory containing these two dataset directories.
The pinned measurement-db revision is
`54074453da7b9911be0f48e4c560be547bc84d21`.

```sh
python reproduce_beta.py --data-root data --verify-inputs
python reproduce_beta.py --data-root data --output results
```

Output must be new or empty. All eight hashes are checked before computation.
The script checks primary/foreign keys, binary outcomes, benchmark metadata,
visible identities and the fixed 57-pair inventory. It uses no reference scores
as inputs. `reference_summary.json` is supplied for result comparison only.

The output contains `summary.json`, 102,600 metric rows in `metrics.csv`, and
5,700 split records in `splits.jsonl`. Seeds are fixed at 0–99. Within each
pair, distinct items are split in half; repeated responses remain in one pool.
A nested acquisition selects one response per item. Budgets are 0, 1, 3, 7,
15 and 31. Brier scores average evaluation responses within each pair;
log-budget trapezoidal ALC weights are 0.1, 0.2, 0.2, 0.2, 0.2 and 0.1.
Pair ALCs receive equal weight. The actual `model.predict` is called for Beta;
constant and empirical-frequency comparators are elementary formulas.

Do not publish output split identities or raw tables as part of this repository.
This command performs CPU replay only: no model fitting, API prediction, new
seeds, hyperparameter search or confirmation-set access.

This repository does not reproduce semantic encoder training, later transfer
experiments, or platform-side hidden evaluation. Those historical development
studies must not be represented as reproduced by the Beta command.
