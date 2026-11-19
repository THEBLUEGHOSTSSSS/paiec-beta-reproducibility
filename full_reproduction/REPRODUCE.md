# Reproduce the fixed studies

Use Python 3.11 or later and install `requirements.txt` in a separate environment.
The execution receipt records the versions actually tested. Allow approximately
2 CPU threads, 3 GiB RAM and 1 GiB for reference inputs and numerical outputs;
fresh encoding additionally needs the upstream encoder assets.

1. Obtain the ten benchmark directories listed in `input_manifest.json` from
   [measurement-db at revision 54074453da7b9911be0f48e4c560be547bc84d21](https://huggingface.co/datasets/aims-foundations/measurement-db/tree/54074453da7b9911be0f48e4c560be547bc84d21).
   Each directory needs `benchmarks.parquet`, `items.parquet`, `subjects.parquet`
   and `responses.parquet`. Exact per-file URLs and SHA256 values are in the
   manifest. Access may require the dataset owner's approval. Obtain access
   directly; the code does not read credentials or download files.
2. Download `reference-20261031.zip` from the repository's
   [fixed reproduction release](https://github.com/THEBLUEGHOSTSSSS/paiec-beta-reproducibility/releases/tag/reproduction-20261031).
   The archive is a derived numerical reference, not raw data or a newly fitted
   result. It contains no original-workspace dependency.
3. Set ROOT to an absolute working directory containing the ZIP and the data subdirectory. From this code directory run the commands below, using fresh output paths.

```sh
python build_reference.py --bundle-archive "$ROOT/reference-20261031.zip" --output "$ROOT/reference"
python reproduce.py --mode verify --data-root "$ROOT/data" --reference-root "$ROOT/reference" --output "$ROOT/verify-output"
python reproduce.py --mode replay --data-root "$ROOT/data" --reference-root "$ROOT/reference" --output "$ROOT/replay-output"
python reproduce.py --mode train --data-root "$ROOT/data" --reference-root "$ROOT/reference" --output "$ROOT/fresh-fit-output"
```

The archive extractor verifies every member, rejects unlisted paths, links,
duplicates and size/hash mismatches, and refuses nonempty outputs. The replay
checks all forty raw hashes and all reference hashes before numerical work.
It also rejects overlapping input/output directories, path traversal and
symlinked path components. No original project checkout, private guard tree,
operational ledger or historical completion flag is needed.

Replay recomputes 39,690 semantic scoring cells and the 140,400 / 70,200 /
105,300 score rows for the proxy / cross-subject / mean studies. Those three
studies share 1,950 trajectories and 670,165 acquisition decisions. Their
original candidate/control contrasts and negative gates remain unchanged.
The five-head full-data export is a descriptive inference artifact, not a
held-out score.

Training fits fifty source-excluded semantic heads, five full-data heads,
twenty-six cross-subject models and seventy-four mean-auxiliary models. This
includes four actual excluded-source poison refits. Fixed seeds, 256 updates,
batch size 512, bilinear rank 8, AdamW learning rate .01, weight decay .03 and
gradient clipping at 5 are retained. Each fit is reserved before execution;
failure stops the run. No automatic retry, seed selection or retuning exists.
The original semantic fit/scope ceilings remain; other fits retain their
resource bounds. New numerical differences are reported rather than tuned away.

To rebuild embeddings, obtain the seven pinned assets listed under `encoder`
in `input_manifest.json` from
[multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3), then run:

```sh
python reproduce.py --mode train --data-root "$ROOT/data" --reference-root "$ROOT/reference" --encoder-root "$ROOT/e5-assets" --output "$ROOT/fresh-encoding-output"
python assemble_export.py --heads-root "$ROOT/replay-output/full_export" --encoder-root "$ROOT/e5-assets" --output "$ROOT/inference-export"
```

The November 19 CPU run rebuilt all 9,389 prefix vectors and 1,565 tail windows. Maximum absolute differences from the archived vectors were 2.87e-7 and zero. All 155 fresh-encoding fits and all five scoring scopes completed. Each scope exceeded at least one original tolerance. The three transfer scopes retained their discrete decision gates, with maximum row-level Brier difference 2.96e-06. The parent archive includes a separate check of the fitted files and score rows. All 155 fits had previously completed with archived embeddings. Hosted execution remains unverified.

`replay_report.json`, `training_summary.json`, score tables and negative-contrast
files distinguish executed checks from unexecuted routes. They are development
reproduction evidence, not organizer acceptance or new-source confirmation.
