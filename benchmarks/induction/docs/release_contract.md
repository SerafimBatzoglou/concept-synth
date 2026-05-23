# INDUCTION Release Contract

This document defines the canonical public INDUCTION release layout for
Concept Synth.

## Canonical artifact files

Paths are relative to `benchmarks/induction/`.

- `data/induction_fullobs_v1.yaml.gz`
- `data/induction_ci_v1.yaml.gz`
- `data/induction_ec_v1.yaml.gz`
- `predictions/induction_predictions_v1.jsonl.gz`
- `eval/induction_eval_cache_v1.jsonl`
- `eval/induction_eval_cache_v1.meta.json`
- `release_manifest.json`

These files are the public source of truth for the `induction-v1.0` release.

## Separation of concerns

- Benchmark files contain benchmark instances only.
- Released model outputs live in the predictions file.
- Evaluation scores live in the frozen eval cache.
- Prompt template source of truth lives under
  `src/concept_synth/induction/prompts/`.
- The public package renders prompts and evaluates predictions; it does not
  include model-running clients.
- The public generator creates new schema-compatible instances. It is not a
  promise to reproduce the exact internal calibrated release construction.

## Compatibility expectations

- Benchmark rows use schema `induction_benchmark_record_v1`.
- Prediction rows use schema `induction_prediction_record_v1`.
- Eval rows use schema `induction_eval_v1`.
- Future releases may add fields, but should not silently remove or rename the
  canonical files for an existing tag.

## Release hygiene

- Public metadata must not contain internal absolute paths.
- Public benchmark data must not contain API keys, local batch jobs, or hidden
  EC full-world predicates.
- Release tags are immutable once published.
- File hashes and sizes must be recorded in `release_manifest.json`.
- Tests and CI must pass before publishing a new release tag.

## Licensing

Unless explicitly noted otherwise, all benchmark data, predictions, prompts,
cached outputs, scripts, and code in this repository are released under the MIT
License in the repository root.
