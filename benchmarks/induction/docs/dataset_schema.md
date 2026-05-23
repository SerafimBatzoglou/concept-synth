# Dataset Schema Notes

INDUCTION benchmark files are gzip-compressed YAML lists. Each row has:

- `schemaVersion`: `induction_benchmark_record_v1`
- `instanceId`: canonical instance id
- `task`: one of `FullObs`, `CI`, or `EC`
- `scenario`: legacy internal scenario code, one of `AD`, `C`, or `E`
- `problem`: machine-readable benchmark instance
- `problemDescription`: generator metadata and the planted reference concept

## Problem

The `problem` object contains:

- `instanceId`
- `scenario`
- `signature.predicates`, with predicate names and arities
- `backgroundAxioms`, usually empty for INDUCTION v1
- `worlds`
- `task`, which names the relevant world ids for the task

Each world contains:

- `worldId`
- `domain`
- `predicates`, represented as observed true/false extensions
- `targetExtension.T_true` and `targetExtension.T_false`
- `unknownAtoms` for EC worlds only

EC public records intentionally do not include hidden `fullPredicates`.

## Task Fields

FullObs and EC use:

- `trainWorldIds`
- `testWorldIds`

CI uses:

- `yesWorldIds`
- `noWorldIds`

## Prediction Rows

Prediction rows are JSONL and use schema
`induction_prediction_record_v1`. Rows should include:

- `instanceId`
- `task`
- `scenario`
- `model`
- `extractedFormula`, `response`, or `rawResponse`

Released rows also keep useful model-record fields such as timestamps, raw
responses, parsed descriptions, parse errors, token/provider metadata, and
other non-secret run metadata when present.

## Eval Cache Rows

Eval-cache rows are JSONL and use schema `induction_eval_v1`. They contain the
normalized prediction, validity, parse status, AST metrics, gold-relative
budget indicators, and task-specific diagnostics.

## Generated Rows

Rows produced by `concept-synth-induction-generate` use the same
`induction_benchmark_record_v1` schema as the released benchmark files. Their
`problemDescription.benchmark_name` is `induction_public_generator`, and band
fields such as `ad_band`, `c_band`, and `e_band` are set to `generated`.
