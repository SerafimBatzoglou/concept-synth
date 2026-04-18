# ABD: Default-Exception Abduction in Finite First-Order Worlds

**ABD is a solver-checkable benchmark for synthesizing exception rules in
small finite first-order worlds.** Each instance asks a model to infer a
single abnormality definition that repairs a default theory across multiple
worlds, with exact evaluation of validity, parsimony, and holdout
generalization.

Paper: [arXiv:2602.18843](https://arxiv.org/abs/2602.18843)
Summary: [gist.science paper summary](https://gist.science/paper/2602.18843)

Contact: serafim.batzoglou@gmail.com

## Project Overview

ABD studies default-exception abduction in a controlled relational setting.
Each benchmark instance provides:

1. a fixed first-order background theory with an abnormality predicate `Ab(x)`,
   and
2. multiple small finite relational worlds that instantiate the theory.

The task is to synthesize one shared first-order formula `alpha(x)` defining
when an entity is abnormal. This formula is substituted for `Ab(x)` throughout
the theory, and the repaired theory is then checked against all worlds in the
instance. A good solution must do two things at once:

- repair the theory correctly across the full prompt set of worlds
- keep exceptions sparse, marking as few domain elements abnormal as possible

Because all worlds are finite, ABD supports exact SMT-based evaluation of both
correctness and cost. This makes it possible to measure not only whether a
model finds a valid repair rule, but also how parsimonious that rule is and
whether it transfers to matched holdout worlds.

The current release contains 600 benchmark instances spanning three observation
regimes and seven default theories.

## Worlds and Tasks

### Finite Relational Worlds

ABD worlds are finite relational structures over the signature:

- unary predicates: `P(x)`, `Q(x)`
- binary predicates: `R(x,y)`, `S(x,y)`
- equality: `=`

Each world has a small finite domain and explicit predicate interpretations. In
the partially observed settings, some ground atoms are marked as unknown rather
than fixed true or false.

### What the Model Must Produce

For each instance, the model receives:

- the background theory
- the set of allowed and forbidden predicates for the hypothesis language
- multiple prompt worlds
- a requirement to output a single formula `alpha(x)`

The output is not a separate rule per world, and it is not a label for each
object. Instead, the model must synthesize one shared abnormality rule that
works jointly across all prompt worlds in the instance.

Formally, the repaired theory is obtained by substituting `alpha` for `Ab`:
`Ab := alpha`.

ABD therefore evaluates multi-world concept synthesis for theory repair, rather
than single-world classification or fact prediction.

## Observation Regimes

ABD includes three task regimes, which share the same interface but differ in
how missing information is treated.

### ABD-Full

All relevant facts are observed under a closed-world assumption. A prediction
is valid if the repaired theory is satisfiable in every prompt world.

### ABD-Partial

Some atoms are unknown. A prediction is valid if, for each prompt world, some
completion of the unknown atoms satisfies the repaired theory. Cost is computed
in the best case over satisfying completions.

### ABD-Skeptical

Some atoms are unknown. A prediction is valid only if, for each prompt world,
every completion of the unknown atoms satisfies the repaired theory. Cost is
computed in the worst case over completions.

These three regimes let ABD separate ordinary repair from reasoning under
uncertainty, and distinguish optimistic from robust exception policies.

## Evaluation

ABD reports three main classes of metrics.

### 1. Validity

Does the synthesized rule actually repair the theory across the full prompt
set?

### 2. Parsimony

Among valid rules, how many domain elements are marked abnormal? ABD reports
cost-relative metrics such as gap above a solver-computed lower bound and gap
relative to a planted generator reference.

### 3. Generalization

Does a rule that works on the prompt worlds also work on matched holdout worlds
drawn from the same generator family?

This combination makes ABD useful for studying not only solver-checkable
correctness, but also overfitting, case-splitting, and brittle exception
rules.

## Why ABD?

ABD is designed to make abduction evaluation:

- formal: the task is defined over explicit finite worlds, not ambiguous
  natural-language descriptions
- exact: validity and cost are computed mechanically with SMT
- multi-world: one rule must satisfy several worlds jointly
- cost-sensitive: binary correctness is not enough; sparse repairs matter
- generalization-aware: holdout worlds expose brittle rules that only fit the
  prompt set

## Repository Contents

Paths below are relative to `benchmarks/abduction/` unless noted.

- `data/abd_instances_v1.yaml.gz`
  Canonical benchmark instances only.
- `data/abd_holdouts_v1.jsonl.gz`
  Canonical holdout worlds.
- `predictions/abd_predictions_v1.jsonl.gz`
  Released model predictions.
- `eval/abd_eval_cache_v1.jsonl`
  Frozen per-instance evaluation cache for the released predictions.
- `eval/abd_eval_cache_v1.meta.json`
  Sanitized metadata for the frozen evaluation cache.
- `release_manifest.json`
  Hashes, sizes, counts, and release-level provenance for the canonical files.
- `schemas/`
  JSON Schemas for benchmark, holdout, prediction, and eval rows.
- `prompts/examples/`
  One concrete prompt example per scenario.
- `analysis/`
  Table-regeneration code for the released eval cache.
- `scripts/rebuild_eval_cache.sh`
  Rebuild an eval cache from the released prediction file.
- `scripts/reproduce_tables.sh`
  Regenerate the reported tables from a selected eval cache.
- `docs/provenance.md`
  Source mapping, release scope, and omitted items.
- `docs/release_contract.md`
  Canonical artifact layout and compatibility expectations.
- `docs/release_checklist.md`
  Pre-release checklist for future benchmark updates.
- `../../src/concept_synth/abduction/`
  ABD-specific package with the prompt builder, evaluator CLI, checker,
  eval-cache utilities, parser support, and source-of-truth prompt templates.
- `../../src/concept_synth/`
  Shared package-level utilities and generic logic support used by ABD and
  future benchmark families.

Local output directories created during reproduction:

- `generated_eval/`
  Fresh eval caches written by the rebuild script; not committed.
- `generated_tables/`
  LaTeX and CSV tables generated locally; not committed.

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

### Regenerate the Reported Tables from the Frozen Cache

```bash
./benchmarks/abduction/scripts/reproduce_tables.sh
```

This writes local outputs to `benchmarks/abduction/generated_tables/`.

### Render a Released Prompt

```bash
concept-synth-abd-build-prompt --instance-id ABD_FULL_TH10_000
```

### Rebuild an Eval Cache from the Released Predictions

```bash
./benchmarks/abduction/scripts/rebuild_eval_cache.sh
```

This reads the released benchmark instances, holdouts, and predictions and
writes a fresh cache to:

`benchmarks/abduction/generated_eval/abd_eval_cache_v1.jsonl`

The full rebuild evaluates the released outputs with exact Z3 checking, so it
is materially slower than regenerating tables from the frozen cache.

### Score an External Predictions File

External prediction JSONL rows should include `instanceId`, `model`, and either
`extractedFormula` or `response`. Evaluate them with:

```bash
concept-synth-abd-evaluate \
  --predictions /path/to/predictions.jsonl \
  --output benchmarks/abduction/generated_eval/custom_eval_cache.jsonl
```

The evaluator writes the same `abd_eval_v1` JSONL schema consumed by
`analysis/make_tables.py`.

### Prompt Templates

The prompt template source of truth lives in:

- [`src/concept_synth/abduction/prompts/`](../../src/concept_synth/abduction/prompts/)

The benchmark tree keeps prompt examples only.

### Regenerate Tables from a Fresh Cache

```bash
./benchmarks/abduction/scripts/reproduce_tables.sh \
  benchmarks/abduction/generated_eval/abd_eval_cache_v1.jsonl
```

### Validate the Release Locally

```bash
pytest -q
```

### Notes

- The canonical benchmark file does not embed model outputs.
- The canonical predictions file does not embed evaluation scores.
- The frozen eval metadata is sanitized to remove internal workspace paths.
- Two partial instances have zero holdout worlds by design; this is recorded in
  `release_manifest.json`.
