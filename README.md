# Concept Synth

Public benchmark releases for Concept Synth / Concept Synthesis.

Contact: serafim.batzoglou@gmail.com

This repository is intended to grow into a broader benchmark collection across
multiple task families, including Abduction, Induction, and Causal Reasoning.

The first public release is the ABD benchmark:

- [`benchmarks/abduction/`](benchmarks/abduction/)

Paper: [arXiv:2602.18843](https://arxiv.org/abs/2602.18843)
Summary: [gist.science paper summary](https://gist.science/paper/2602.18843)

The ABD release ships canonical benchmark instances, matched holdout worlds,
released model predictions, a frozen evaluation cache, prompt examples, and a
runnable `concept_synth.abduction` package with the prompt builder, evaluator
CLI, checker, and parser support.

Task-family-specific code lives in subpackages such as
`concept_synth.abduction`, leaving room for future releases under the broader
`concept_synth` namespace.

All benchmark data, prompts, cached outputs, scripts, and code in this
repository are released under the MIT License unless noted otherwise.

Install from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Then, for example:

```bash
concept-synth-abd-build-prompt --instance-id ABD_FULL_TH10_000
./benchmarks/abduction/scripts/rebuild_eval_cache.sh --limit 5
./benchmarks/abduction/scripts/reproduce_tables.sh
pytest -q
```
