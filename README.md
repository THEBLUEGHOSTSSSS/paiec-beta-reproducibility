# Beta smoothing for PAIEC

Exact standard-library Beta(1,1) predictor associated with Codabench submission
989987, plus a standalone replay of the original two-benchmark diagnostic.
The predictor returns `(k + 1) / (n + 2)` for matching visible subject and
benchmark identities. It has no trained parameters, network access or cross-call state.

The frozen study covers 57 eligible subject-benchmark pairs and seeds 0–99:

| Predictor | Mean Brier ALC |
| --- | ---: |
| Constant 0.5 | 0.250000000000 |
| Empirical frequency | 0.256955631773 |
| Beta(1,1) | 0.219791599516 |

These are offline development results. Seeds reuse the same two sources;
they do not establish generalization to independent benchmarks. Hosted evaluation
uses its own samples and is not reproduced by this script.

See [REPRODUCE.md](REPRODUCE.md) for data acquisition and commands, and
[PROVENANCE.md](PROVENANCE.md) for scope and licensing. Raw data are not redistributed.
Code and documentation are MIT licensed. PyArrow and downloaded data retain
their respective upstream licenses.

This is a preparation release. Local deterministic checks are recorded in
`CHECKS.json`; required independent scientific review remains incomplete.
No final competition submission or organizer acceptance is claimed.

## Full development reproduction

The [five-scope package](full_reproduction/README.md) covers semantic stability, full-development export, structural proxy, cross-subject controls and mean auxiliary studies. Download the reference archive and code from [the fixed reproduction release](https://github.com/THEBLUEGHOSTSSSS/paiec-beta-reproducibility/releases/tag/reproduction-20261031), then follow [the commands](full_reproduction/REPRODUCE.md).

**Local validation passed; independent model review remains incomplete.** See [release status](full_reproduction/RELEASE_STATUS.md). These studies retain their negative findings and do not establish new-source generalization.
