# Five fixed semantic and transfer experiments

This package reproduces the development studies accompanying the PAIEC Beta
submission. It implements semantic stability across ten sources and five heads,
the five-head full-development export, the structural proxy factorial,
intact-versus-shuffled cross-subject training, and the three-seed mean auxiliary.
It recomputes model forwards, purchased observations, Brier curves, aggregates and
the original decision gates. Saved summaries are comparison anchors.

Use [REPRODUCE.md](REPRODUCE.md) for complete commands. Obtain the forty pinned
measurement-db tables directly from the official dataset under its access
conditions. The separately downloadable reference archive contains fitted
parameters, derived vectors, fixed schedules, predictions and comparison
anchors. Raw question text, tokenized question text and individual observed
outcomes are excluded. The code reconstructs the needed text and labels from
the user's local, hash-checked official data.

`replay` uses the saved parameters and original embeddings; `train` performs the
fixed 155 fits with those embeddings. `train --encoder-root ...` additionally
rebuilds the embeddings using pinned local E5 assets. These are different
validation levels. Cache-backed replay and 155 fits with archived embeddings completed. The fresh-encoding run rebuilt all vectors and completed all 155 fits. Five scopes exceeded their original reference tolerances. All three transfer studies retained their discrete gate outcomes. The organizer-hosted runtime was not rerun.

The released execution receipts record exact counts, numeric differences,
original source hashes and dependency versions. All reported studies use the
same development sources and preserve their negative findings. They do not
establish performance on new sources. The full-development export has no
held-out evaluation. The actual competition candidate remains the parameter-free
Beta predictor in the repository root.

Project code is MIT licensed. Dataset access and upstream assets retain their
own conditions; see [PROVENANCE.md](PROVENANCE.md). These programs never publish,
submit or access the network.
