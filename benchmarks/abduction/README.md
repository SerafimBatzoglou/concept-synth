# ABD Benchmark Release

Public artifact repository for the ABD KR 2026 paper:
`ABD: Default--Exception Abduction in Finite First-Order Worlds`.

This is the first benchmark release inside the broader `concept-synth`
repository.

Paper: [arXiv:2602.18843](https://arxiv.org/abs/2602.18843)

Contact: serafim.batzoglou@gmail.com

This release is scoped to the published artifact surface:

- the released ABD benchmark bundle
- the frozen evaluation cache used for the reported tables
- the prompt templates and example prompts
- standalone evaluation scripts to regenerate those tables from the frozen cache

The repo is intentionally not a copy of the full `concept_synth` monorepo.
Unrelated projects, intermediate benchmarks, provider-specific inference code,
and manuscript artifacts are excluded.

## Quick start

From the repository root, regenerate the reported tables from the frozen cache
with:

```bash
./benchmarks/abduction/scripts/reproduce_tables.sh
```

This writes LaTeX/CSV outputs to `benchmarks/abduction/generated_tables/`.

All paths below are relative to `benchmarks/abduction/`.

## Repository layout

- `data/abd_combined_v1.yaml.gz`
  Exact released benchmark bundle, compressed for Git hosting. Decompress with
  `gzip -dk data/abd_combined_v1.yaml.gz` if you want the raw YAML.
- `data/abd_combined_v1.yaml.holdout_k5_seed0_delta12.jsonl`
  Pre-generated holdout sidecar paired with the frozen evaluation run.
- `eval/abd_combined_v1_eval_cache.jsonl`
  Frozen per-instance evaluation cache used to generate the published tables.
- `eval/abd_combined_v1_eval_cache.jsonl.meta.json`
  Original run metadata preserved from the internal evaluation pass.
- `prompts/templates/`
  System prompt plus the ABD-Full / ABD-Partial / ABD-Skeptical prompt templates.
- `prompts/examples/`
  One concrete generated prompt per scenario, matching the appendix examples.
- `analysis/`
  Standalone table-regeneration scripts adapted from the internal codebase.
- `generated_tables/`
  Local output directory created by the regeneration script; not committed.
- `docs/provenance.md`
  Source mapping, scope, and omitted items.

## Reproduction target

The reproducibility target in this repo is the reported ABD table set.
The table generator reads the frozen JSONL cache and emits the same LaTeX/CSV
table files used in the paper workflow.

## Notes

- This public repo does not ship the paper PDF or pre-generated paper tables.
- The public repo ships the frozen artifact needed to reproduce the reported
  tables locally. It does not rerun live model inference.
