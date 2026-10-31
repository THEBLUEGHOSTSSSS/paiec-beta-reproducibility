# Provenance and rights scope

- `model.py` is the project's fixed Beta implementation. It is byte-identical
  to the predictor in `submission.zip`; the archive SHA-256 is
  `b5fc15fee6c5af12b2cbaa79ac4828ac72a8765a2d7ee18891ac674994724e03`.
- `submission.zip` preserves the submitted two-member archive without
  modification. The repository MIT license also covers this project's code
  and README inside that archive; the archive is preserved for byte identity.
- `reproduce_beta.py` adapts this project's original independent raw-data audit
  and its visible-input validation. Its candidate execution path is newly
  connected to the exact submitted predictor. It includes no source copied
  from the official empirical-frequency implementation, pretrained encoder,
  model weights or official report template.
- `test_reproduce.py` adapts this project's adversarial audit fixtures.
- `reference_summary.json` contains aggregate and seed-level summaries, not
  raw outcomes, response IDs, credentials or internal execution records.
- `data_sources.json` identifies the original public inputs. No data license
  is granted by this repository. Obtain the inputs from their original host
  and follow the source-specific license information in their metadata.
- PyArrow is an external dependency, not bundled software. See the
  [Apache Arrow project](https://arrow.apache.org/) for its license.

Lishuo Guo is the project author. AI coding and research assistants supported
implementation, verification and documentation. The public preparation does
not replace independent scientific review or grant organizer approval.
