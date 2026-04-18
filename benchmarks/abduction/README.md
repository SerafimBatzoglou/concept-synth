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
- a standalone installable ABD package with prompt rendering, evaluator CLI,
  checker, and parser support
- standalone table-regeneration scripts

The repo is intentionally not a copy of the full `concept_synth` monorepo.
Unrelated projects, intermediate benchmarks, provider-specific inference code,
and manuscript artifacts are excluded.

## Quick start

Create an isolated environment and install the public package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

From the repository root, regenerate the reported tables from the frozen cache:

```bash
./benchmarks/abduction/scripts/reproduce_tables.sh
```

This writes LaTeX/CSV outputs to `benchmarks/abduction/generated_tables/`.

Render a released prompt:

```bash
concept-synth-abd-build-prompt --instance-id ABD_FULL_TH10_000
```

Rebuild an eval cache from the embedded released model outputs:

```bash
./benchmarks/abduction/scripts/rebuild_eval_cache.sh
```

This writes a fresh JSONL cache to
`benchmarks/abduction/generated_eval/abd_combined_v1_eval_cache.jsonl`.

Regenerate tables from that fresh cache:

```bash
./benchmarks/abduction/scripts/reproduce_tables.sh \
  benchmarks/abduction/generated_eval/abd_combined_v1_eval_cache.jsonl
```

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
- `../../src/concept_synth/`
  Standalone public Python package for prompt rendering, parsing, checking,
  and eval-cache generation.
- `generated_tables/`
  Local output directory created by the regeneration script; not committed.
- `generated_eval/`
  Local eval-cache output directory used by the rebuild script; not committed.
- `docs/provenance.md`
  Source mapping, scope, and omitted items.

## Reproduction target

The reproducibility target in this repo is the reported ABD table set.
The table generator reads the frozen JSONL cache and emits the same LaTeX/CSV
table files used in the paper workflow.

The repo also ships enough code to rerun the released checker on:

- embedded benchmark outputs shipped in the YAML bundle
- external prediction JSONL files with `instanceId`, `model`, and either
  `extractedFormula` or `response`

The evaluator writes the same `abd_eval_v1` JSONL schema consumed by
`analysis/make_tables.py`.

## Notes

- This public repo does not ship the paper PDF or pre-generated paper tables.
- The public repo ships the frozen artifact needed to reproduce the reported
  tables locally. It does not rerun live model inference.
- Provider-specific API runners from the internal monorepo are still excluded.
