# Provenance and Scope

## Included in the public artifact

| Release path | Source in the research workspace | Purpose |
| --- | --- | --- |
| `data/induction_fullobs_v1.yaml.gz` | sanitized export from `datasets/v1/ad_benchmark_v1.yaml` | Canonical FullObs instances only |
| `data/induction_ci_v1.yaml.gz` | sanitized export from `datasets/v1/c_benchmark_v1.yaml` | Canonical CI instances only |
| `data/induction_ec_v1.yaml.gz` | sanitized export from `datasets/v1/e_benchmark_v1b.yaml` | Canonical EC instances only |
| `predictions/induction_predictions_v1.jsonl.gz` | extracted from embedded `llmResults` in the three source benchmark files | Released model predictions |
| `eval/induction_eval_cache_v1.jsonl` | extracted from embedded evaluation objects where present, with missing rows scored by the public evaluator | Frozen evaluation cache |
| `eval/induction_eval_cache_v1.meta.json` | public metadata generated during export | Frozen evaluation provenance without internal paths |
| `release_manifest.json` | generated in the public repo | Hashes, sizes, counts, and explicit release scope |
| `prompts/examples/*` | rendered by the public prompt builder | Example prompts for the three tasks |
| `analysis/make_tables.py` | public summary script | Table regeneration against the public eval-cache schema |
| `../../src/concept_synth/induction/` | public extraction from induction prompt/evaluator logic | Prompt rendering and exact evaluation |
| `../../src/concept_synth/induction/generator.py` | new public generator | Deterministic generation of new schema-compatible instances |
| `../../src/concept_synth/sexpr_parser.py`, `../../src/concept_synth/fol/`, `../../src/concept_synth/metrics.py` | shared public logic utilities | Formula parsing, ASTs, metrics, and finite-model evaluation |
| `../../src/concept_synth/e_completion_z3.py`, `../../src/concept_synth/fo_grounding_z3.py` | shared Z3 grounding/evaluation support | Exact EC completion checking |

## Original internal pipeline

These were the main internal code paths used for the INDUCTION run:

- Prompt rendering and model execution:
  `src/concept_synth/benchmark_runner.py`
- Prompt template files:
  `src/concept_synth/prompts/ad_scenario_task.txt`
  `src/concept_synth/prompts/ad_scenario_suffix.txt`
  `src/concept_synth/prompts/c_scenario_task.txt`
  `src/concept_synth/prompts/e_scenario_task.txt`
  `src/concept_synth/prompts/e_scenario_suffix.txt`
- Evaluation pipeline:
  `src/concept_synth/evaluate_results.py`
- EC completion checker:
  `src/concept_synth/e_completion_z3.py`
- Batch inference runners:
  `src/concept_synth/batch_runner.py`
  `src/concept_synth/run_parallel_batches.py`

## Intentionally omitted

The public artifact omits the following on purpose:

- provider-specific live inference plumbing and batch-submission code
- API keys, environment files, and local runtime state
- large raw result directories and local batch-job artifacts
- manuscript source files, review drafts, and LaTeX build products
- unrelated causal-reasoning code and results
- hidden EC `fullPredicates` fields from benchmark records

The public release is centered on benchmark data, prompt generation,
schema-compatible instance generation, prediction records, exact evaluation,
cached results, and reproducibility scripts.

The public generator is intentionally smaller than the internal ICML benchmark
construction pipeline. It creates valid new FullObs, CI, and EC instances with
the released schema, but it does not reproduce the private calibration,
version-space diagnostics, or downselection process used to choose the
canonical 775 released instances.

## Canonicalization notes

- Benchmark files contain problem records only. Embedded `llmResults` were
  split into `predictions/` and `eval/`.
- Prediction rows follow the ABD convention and keep useful model-record
  details, including timestamps, raw responses, extracted formulas,
  descriptions, and provider metadata when present.
- Evaluation objects are kept in the frozen eval cache rather than in the
  prediction file.
- The frozen eval cache has one row per released prediction. The
  `computedEvalRows` field in `eval/induction_eval_cache_v1.meta.json` records
  rows scored by the public evaluator because the source model record did not
  contain an embedded evaluation.
- EC benchmark records omit hidden full-world predicates; the public evaluator
  uses observed facts and unknown-atom sets.
- File hashes and counts are recorded in `release_manifest.json`.
- All files in this public benchmark release are covered by the MIT License in
  the repository root.
