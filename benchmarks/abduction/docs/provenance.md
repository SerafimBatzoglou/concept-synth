# Provenance and Scope

## Included in the public artifact

| Release path | Source in the research workspace | Purpose |
| --- | --- | --- |
| `data/abd_combined_v1.yaml.gz` | `results/abduction/abd_combined_v1.yaml` | Released benchmark bundle |
| `data/abd_combined_v1.yaml.holdout_k5_seed0_delta12.jsonl` | `results/abduction/abd_combined_v1.yaml.holdout_k5_seed0_delta12.jsonl` | Pre-generated holdout sidecar |
| `eval/abd_combined_v1_eval_cache.jsonl` | `results/abduction/abd_combined_v1_eval_cache.jsonl` | Frozen evaluation cache |
| `eval/abd_combined_v1_eval_cache.jsonl.meta.json` | `results/abduction/abd_combined_v1_eval_cache.jsonl.meta.json` | Original evaluation provenance |
| `prompts/templates/*` | `src/concept_synth/prompts/abd_*` and `results/abduction/paper/prompts/system_prompt.txt` | Prompt templates used by the ABD prompting pipeline |
| `prompts/examples/*` | `results/abduction/paper/prompts/prompt_*_example.txt` | Appendix prompt examples |
| `analysis/make_tables.py` | adapted from `scripts/make_tables.py` | Standalone table regeneration |
| `analysis/eval_cache_reader.py` | adapted from `src/concept_synth/eval_cache.py` | Standalone JSONL cache reader |
| `scripts/reproduce_tables.sh` | new release-local wrapper | Local regeneration entrypoint |

## Original internal pipeline

These were the main internal code paths used for the published ABD run:

- Prompt rendering:
  `src/concept_synth/abd_b1_prompt.py`
- Prompt template files:
  `src/concept_synth/prompts/abd_full_scenario_task.txt`
  `src/concept_synth/prompts/abd_full_scenario_suffix.txt`
  `src/concept_synth/prompts/abd_partial_scenario_task.txt`
  `src/concept_synth/prompts/abd_partial_scenario_suffix.txt`
  `src/concept_synth/prompts/abd_skeptical_scenario_task.txt`
  `src/concept_synth/prompts/abd_skeptical_scenario_suffix.txt`
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
evaluation scripts, and instructions to reproduce the reported tables.
