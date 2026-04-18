# ABD Release Contract

This document defines the canonical public ABD release layout for Concept
Synth.

## Canonical artifact files

Paths are relative to `benchmarks/abduction/`.

- `data/abd_instances_v1.yaml.gz`
- `data/abd_holdouts_v1.jsonl.gz`
- `predictions/abd_predictions_v1.jsonl.gz`
- `eval/abd_eval_cache_v1.jsonl`
- `eval/abd_eval_cache_v1.meta.json`
- `release_manifest.json`

These files are the public source of truth for the `abduction-v1.1` release.

## Separation of concerns

- The benchmark file contains benchmark instances only.
- Holdout worlds live in a separate sidecar file.
- Released model outputs live in the predictions file.
- Evaluation scores live in the frozen eval cache.
- Prompt template source of truth lives under
  `src/concept_synth/abduction/prompts/`.

## Compatibility expectations

- Benchmark rows use schema `abd_benchmark_record_v1`.
- Holdout rows use schema `abd_holdout_record_v1`.
- Prediction rows use schema `abd_prediction_record_v1`.
- Eval rows use schema `abd_eval_v1`.
- Future releases may add fields, but should not silently remove or rename the
  canonical files for an existing tag.

## Release hygiene

- Public metadata must not contain internal absolute paths.
- Release tags are immutable once published.
- File hashes and sizes must be recorded in `release_manifest.json`.
- Tests and CI must pass before publishing a new release tag.

## Licensing

Unless explicitly noted otherwise, all benchmark data, predictions, prompts,
cached outputs, scripts, and code in this repository are released under the MIT
License in the repository root.
