# INDUCTION: Finite-Structure Concept Synthesis in First-Order Logic

**INDUCTION is a solver-checkable benchmark for synthesizing first-order
concept definitions from finite relational worlds.** Each instance gives small
finite structures over a fixed vocabulary and asks for one formula
`phi(x)` whose extension satisfies the task-specific semantic criterion.

Paper: `INDUCTION: Finite-Structure Concept Synthesis in First-Order Logic`
(ICML 2026, spotlight)

Contact: serafim.batzoglou@gmail.com

## Project Overview

INDUCTION studies whether prompted models can infer compact first-order rules
from extensional evidence. All tasks use the same output language:

- unary predicates: `P(x)`, `Q(x)`
- binary predicates: `R(x,y)`, `S(x,y)`
- equality: `=`
- connectives: `not`, `and`, `or`
- quantifiers: `forall`, `exists`

The required output is a single S-expression formula with exactly one free
variable, `x`. Users can render the released prompts with the public tooling,
but model calls are intentionally out of scope: use your own inference
infrastructure and write prediction rows in the released schema.

The current release contains 775 benchmark instances:

- 375 FullObs instances
- 200 CI instances
- 200 EC instances

It also contains 11,413 released model prediction rows and 11,413 matching
eval-cache rows across 15 model identifiers, including full 775-instance
coverage for Gemini 3.5 Flash.

## Tasks

### FullObs

FullObs is fully observed concept induction under a closed-world assumption.
A prediction is valid when `phi(a)` exactly matches the target label `T(a)` for
every object in every input world.

### CI

CI is contrastive induction. YES worlds use the same exact-match criterion as
FullObs. In each contrastive NO world, the formula must fail to match the
target extension for at least one object.

### EC

EC is existential-completion induction under partial observation. Some
predicate atoms are unknown. A prediction is valid if, for each input world,
there exists a world-local completion of the unknown atoms under which `phi`
matches all target labels.

## Evaluation

The public evaluator reports:

- parse validity and the normalized formula
- task validity
- AST size and quantifier depth
- gold-relative AST deltas
- budgeted indicators at gold+0, gold+10, and gold+25
- task-specific diagnostics for CI and EC

FullObs and CI are checked by exact finite-model evaluation. EC uses exact
Z3-based completion checking.

## Repository Contents

Paths below are relative to `benchmarks/induction/` unless noted.

- `data/induction_fullobs_v1.yaml.gz`
  Canonical FullObs benchmark instances.
- `data/induction_ci_v1.yaml.gz`
  Canonical CI benchmark instances.
- `data/induction_ec_v1.yaml.gz`
  Canonical EC benchmark instances.
- `predictions/induction_predictions_v1.jsonl.gz`
  Released model prediction records. These keep model-response fields such as
  timestamps, raw responses, parsed formulas, descriptions, and provider
  metadata when present.
- `eval/induction_eval_cache_v1.jsonl`
  Frozen per-prediction evaluation cache for the released predictions.
- `eval/induction_eval_cache_v1.meta.json`
  Sanitized metadata for the frozen evaluation cache.
- `release_manifest.json`
  Hashes, sizes, counts, and release-level provenance for canonical files.
- `schemas/`
  JSON Schemas for benchmark, prediction, and eval-cache rows.
- `prompts/examples/`
  One rendered prompt example per task.
- `analysis/`
  Table-regeneration code for the released eval cache.
- `scripts/rebuild_eval_cache.sh`
  Rebuild an eval cache from released or external predictions.
- `scripts/reproduce_tables.sh`
  Regenerate compact CSV and LaTeX tables from an eval cache.
- `scripts/generate_instances.sh`
  Generate new public-schema INDUCTION instances with the clean public
  generator.
- `docs/provenance.md`
  Source mapping, release scope, and omitted items.
- `docs/release_contract.md`
  Canonical artifact layout and compatibility expectations.
- `docs/release_checklist.md`
  Pre-release checklist for future benchmark updates.
- `docs/dataset_schema.md`
  Human-readable schema notes.
- `../../src/concept_synth/induction/`
  INDUCTION-specific package with prompt rendering, evaluator CLI, and
  source-of-truth prompt templates.
- `../../src/concept_synth/`
  Shared parser, formula AST, metrics, finite-model, and Z3 grounding support.

Local output directories created during reproduction:

- `generated_eval/`
  Fresh eval caches written by the rebuild script; not committed.
- `generated_tables/`
  LaTeX and CSV tables generated locally; not committed.

<!-- BEGIN CHALLENGE64 ROUND1 -->
## Challenge64 Round-1 Results

The additive Challenge64 release contains normalized **pre-symbolic Round-1** formulas for configurations with more than 40 evaluable responses. It preserves the public prediction and eval-cache schemas while redacting raw model responses and reasoning traces.

- `data/induction_fullobs_challenge64_v1.yaml.gz`
- `predictions/induction_challenge64_round1_predictions_v1.jsonl.gz`
- `eval/induction_challenge64_round1_eval_cache_v1.jsonl`
- `eval/induction_challenge64_round1_holdout_eval_cache_v1.jsonl`
- `docs/challenge64_round1_results.md`
- `docs/challenge64_round1_provenance.md`
- `analysis/make_challenge64_round1_table.py`

The fixed generated-holdout diagnostic is post-selection only and is not part of the benchmark task or solver input.
<!-- END CHALLENGE64 ROUND1 -->

## Reproducing the Release

All released benchmark data, predictions, prompts, cached outputs, scripts, and
code in this repository are covered by the MIT License in the repository root.

### Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Render a Released Prompt

```bash
concept-synth-induction-build-prompt \
  --dataset benchmarks/induction/data/induction_fullobs_v1.yaml.gz \
  --instance-id simple_001
```

### Generate New Benchmark Instances

The public generator creates deterministic FullObs, CI, and EC instances in
the same benchmark-record schema as the released data. It is a clean generator
for new instances, not the full internal calibration pipeline used to select
the ICML release set.

```bash
concept-synth-induction-generate \
  --task all \
  --n 5 \
  --seed 20260516 \
  --validate \
  --output benchmarks/induction/generated_instances/induction_generated_sample.yaml
```

Then render or evaluate generated records with the same public tools:

```bash
concept-synth-induction-build-prompt \
  --dataset benchmarks/induction/generated_instances/induction_generated_sample.yaml \
  --index 0
```

### Regenerate Tables from the Frozen Cache

```bash
./benchmarks/induction/scripts/reproduce_tables.sh
```

This writes local outputs to `benchmarks/induction/generated_tables/`.

### Rebuild an Eval Cache from Released Predictions

```bash
./benchmarks/induction/scripts/rebuild_eval_cache.sh --limit 5 --no-diagnostics
```

This reads the released benchmark instances and prediction file and writes a
fresh cache to:

`benchmarks/induction/generated_eval/induction_eval_cache_v1.jsonl`

The full rebuild evaluates all released outputs with exact checking and is
slower than regenerating tables from the frozen cache.

### Score an External Predictions File

External prediction JSONL rows should include `instanceId`, `model`, and either
`extractedFormula`, `response`, or `rawResponse`.

```bash
concept-synth-induction-evaluate \
  --dataset benchmarks/induction/data/induction_ci_v1.yaml.gz \
  --predictions /path/to/predictions.jsonl \
  --output benchmarks/induction/generated_eval/custom_eval_cache.jsonl
```

The evaluator writes the same `induction_eval_v1` JSONL schema consumed by
`analysis/make_tables.py`.

### Prompt Templates

The prompt template source of truth lives in:

- [`src/concept_synth/induction/prompts/`](../../src/concept_synth/induction/prompts/)

The benchmark tree keeps prompt examples only.

### Regenerate Tables from a Fresh Cache

```bash
./benchmarks/induction/scripts/reproduce_tables.sh \
  benchmarks/induction/generated_eval/induction_eval_cache_v1.jsonl
```

### Validate the Release Locally

```bash
pytest -q
```

## Citation

```bibtex
@inproceedings{batzoglou2026induction,
  title     = {{INDUCTION}: Finite-Structure Concept Synthesis in First-Order Logic},
  author    = {Batzoglou, Serafim},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```

## Notes

- The canonical benchmark files do not embed model outputs.
- The canonical prediction file does not embed evaluation scores.
- The frozen eval metadata is sanitized to remove internal workspace paths.
- The frozen eval cache has one row per released prediction. Most rows come
  from embedded research-workspace evaluations; rows without embedded
  evaluations were scored with the public evaluator and counted in
  `eval/induction_eval_cache_v1.meta.json`.
- Model-running clients, API wrappers, batch-submission code, and local result
  dumps are intentionally omitted from the public release.
- The public generator is intended for creating new schema-compatible
  instances; it does not reproduce the internal calibrated downselection that
  produced the released benchmark set.
