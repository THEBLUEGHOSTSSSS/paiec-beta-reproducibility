# Provenance and licenses

The project-authored source, adapters and documentation are released under MIT.
`source_provenance.json` records original scientific source hashes, copied or
adapted functions and packaged hashes. `package_manifest.json` binds the code
and metadata. No third-party baseline implementation is incorporated.

Raw inputs remain external. `input_manifest.json` pins measurement-db revision
`54074453da7b9911be0f48e4c560be547bc84d21`, ten allowed development benchmarks and
forty individual file hashes. Download/access rights come from the upstream
dataset, not this project's MIT license. No raw response table or question text
is redistributed. Confirmation reserve data are not used.

The E5 encoder is pinned to revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`. Its
[model card](https://huggingface.co/intfloat/multilingual-e5-small/blob/614241f622f53c4eeff9890bdc4f31cfecc418b3/README.md)
declares MIT. Upstream tokenizer/model assets are obtained separately and retain
that license. The release includes project-computed embeddings and fitted heads,
not upstream weights or tokenized question text.

The reference archive contains original fitted parameters, derived features,
fixed splits/panels/schedules, predictions, aggregate scores and hash commitments.
Its public projection removes raw text, tokenized text, observed-label lists and
per-item outcome arrays. Text is reconstructed from approved local raw tables;
split outcomes and mean targets are recomputed and compared with their original
hash commitments. Unused outcome arrays are removed from prediction anchors.
Model parameters, prediction arrays, controls and numerical gates are unchanged.

For each reference, `sha256` binds the released bytes. `original_source_sha256`
binds the historical source named by the relative provenance path; those paths
are provenance only and are never opened by the reproduction program.
`public_projection.json` records every changed or omitted reference. Original
artifacts and their full hashes are retained locally.

All studies repeat public development data. Three mean-training seeds remain
distinct from the old five-head reference, intact pairing is not promoted over
its shuffled control, and the failed structure/mean gates remain failures.
No new-source generalization, private-test reconstruction or fresh hosted
validation is claimed.
