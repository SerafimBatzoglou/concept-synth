# Concept Synth

Public benchmark releases for Concept Synth / Concept Synthesis.

Contact: serafim.batzoglou@gmail.com

This repository is meant to grow into a broader benchmark repo covering
multiple task families, including:

- Abduction
- Induction
- Causal Reasoning

The first public release is the ABD KR 2026 benchmark artifact:

- [`benchmarks/abduction/`](./benchmarks/abduction/)

Paper link: [arXiv:2602.18843](https://arxiv.org/abs/2602.18843)

The ABD release includes the benchmark bundle, prompt templates, frozen
evaluation cache, table-regeneration scripts, and a standalone installable
Python package with:

- prompt rendering
- evaluator CLI
- Z3 checker and parser support

Task-family-specific code lives in subpackages such as
`concept_synth.abduction`, leaving room for future releases under the broader
`concept_synth` namespace.

Install the runnable package from the repo root with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then, for example:

```bash
concept-synth-abd-build-prompt --instance-id ABD_FULL_TH10_000
./benchmarks/abduction/scripts/rebuild_eval_cache.sh --limit 5
./benchmarks/abduction/scripts/reproduce_tables.sh benchmarks/abduction/generated_eval/abd_combined_v1_eval_cache.jsonl
```
