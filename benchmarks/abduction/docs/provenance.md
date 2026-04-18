# Provenance and Scope

## Included in the public artifact

| Release path | Source in the research workspace | Purpose |
| --- | --- | --- |
| `data/abd_instances_v1.yaml.gz` | normalized export from `results/abduction/abd_combined_v1.yaml` | Canonical benchmark instances only |
| `data/abd_holdouts_v1.jsonl.gz` | normalized export from `results/abduction/abd_combined_v1.yaml.holdout_k5_seed0_delta12.jsonl` | Canonical holdout worlds |
| `predictions/abd_predictions_v1.jsonl.gz` | extracted from embedded `llmResults` in `results/abduction/abd_combined_v1.yaml` | Canonical released predictions |
| `eval/abd_eval_cache_v1.jsonl` | sanitized export of `results/abduction/abd_combined_v1_eval_cache.jsonl` | Frozen evaluation cache for the released predictions |
| `eval/abd_eval_cache_v1.meta.json` | sanitized public metadata derived from the original eval pass | Frozen evaluation provenance without internal paths |
| `release_manifest.json` | generated in the public repo | Hashes, sizes, counts, and explicit release scope |
| `prompts/examples/*` | `results/abduction/paper/prompts/prompt_*_example.txt` | Appendix prompt examples |
| `analysis/make_tables.py` | adapted from `scripts/make_tables.py` | Table regeneration against the public eval-cache schema |
| `../../src/concept_synth/abduction/abd_b1_prompt.py` | `src/concept_synth/abd_b1_prompt.py` | Public prompt builder |
| `../../src/concept_synth/abduction/evaluate_abd_b1.py` | `src/concept_synth/evaluate_abd_b1.py` | Public ABD evaluator core |
| `../../src/concept_synth/abduction/abd_b1_z3_checker.py` | `src/concept_synth/abd_b1_z3_checker.py` | Public Z3 validity/cost checker |
| `../../src/concept_synth/sexpr_parser.py` and `../../src/concept_synth/fo_grounding_z3.py` | `src/concept_synth/sexpr_parser.py` and `src/concept_synth/fo_grounding_z3.py` | Public parser and grounding support |
| `../../src/concept_synth/abduction/cli.py` | new public wrapper | Public prompt/evaluation CLI |
| `scripts/rebuild_eval_cache.sh` | new public wrapper | Rebuild eval cache from released predictions |
| `scripts/reproduce_tables.sh` | new public wrapper | Local regeneration entrypoint |

## Original internal pipeline

These were the main internal code paths used for the published ABD run:

- Prompt rendering:
  `src/concept_synth/abd_b1_prompt.py`
- Prompt template files:
  `src/concept_synth/abduction/prompts/abd_full_scenario_task.txt`
  `src/concept_synth/abduction/prompts/abd_full_scenario_suffix.txt`
  `src/concept_synth/abduction/prompts/abd_partial_scenario_task.txt`
  `src/concept_synth/abduction/prompts/abd_partial_scenario_suffix.txt`
  `src/concept_synth/abduction/prompts/abd_skeptical_scenario_task.txt`
  `src/concept_synth/abduction/prompts/abd_skeptical_scenario_suffix.txt`
- Batch inference runner:
  `src/concept_synth/batch_runner.py`
- Evaluation pipeline:
  `src/concept_synth/evaluate_abd_b1_batch.py`
- Holdout materialization:
  `src/concept_synth/materialize_abd_b1_holdouts.py`
- Z3 checker:
  `src/concept_synth/abd_b1_z3_checker.py`
- Paper table generation:
  `scripts/make_tables.py`

## Intentionally omitted

The public artifact omits the following on purpose:

- the rest of the `concept_synth` monorepo
- unrelated projects such as `causal_reasoning`
- older ABD pilots, smoke runs, and intermediate benchmark variants
- exported ZIP artifacts from internal review cycles
- provider-specific live inference plumbing needed to submit new API jobs
- manuscript source files and compiled paper artifacts

The goal here is a clean public release that directly supports the published
artifact claim in the paper: benchmark files, prompts, cached outputs,
evaluation scripts, runnable checker code, and instructions to reproduce the
reported tables.

## Canonicalization notes

- The old mixed benchmark bundle is split into instances, holdouts, predictions,
  and eval artifacts.
- Prompt templates are sourced from `src/concept_synth/abduction/prompts/`; the
  benchmark tree keeps examples only.
- The frozen eval metadata is rewritten to remove internal absolute paths and
  stale command lines.
- The canonical release files are recorded with hashes and counts in
  `release_manifest.json`.
- All files in this public benchmark release are covered by the MIT License in
  the repository root.
