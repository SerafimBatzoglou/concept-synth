# GPT-6 Sol — completed Challenge100 results

Both max and xhigh passes are terminal and verified. All 100 tasks have parser-evaluable selected responses, so high and medium were skipped on empty residual. These are final results.

| Set | Evaluable | Correct | Correct among evaluable | Frozen holdout |
|---|---:|---:|---:|---:|
| Challenge100 | 100/100 | 57/100 | 57.00% | 35/53 (66.04%) |
| Challenge64 | 64/64 | 42/64 | 65.63% | 24/41 (58.54%) |
| New36 | 36/36 | 15/36 | 41.67% | 11/12 (91.67%) |

## Protocol and request accounting

Direct Round 1 only, exact model `gpt-6-sol`, OpenAI Responses Batch API (`/v1/responses`, 24-hour window), standard/default reasoning mode, `max_output_tokens=128000` shared reasoning/visible-output budget, JSON-object output, temperature omitted, unchanged original prompts, one response per task per physical pass, zero client retries, tools or symbolic work. Only effort and the exact non-evaluable residual task set changed.

| Pass | Effort | Submitted | Evaluable | Correct | Residual |
|---|---|---:|---:|---:|---:|
| 1 | max | 100 | 50 | 48 | 50 |
| 2 | xhigh | 50 | 50 | 9 | 0 |
| 3 | high | 0 (skipped) | — | — | 0 |
| 4 | medium | 0 (skipped) | — | — | 0 |

All 150 submitted requests have terminal records across two batches. Initial max returned 50 HTTP 200 completed nonempty responses, 49 status-0 empty-body failure records without detailed errors, and one HTTP 503 server-is-overloaded response. Xhigh returned 50 HTTP 200 completed nonempty responses, with zero provider failures or caps.

Total: 101 confirmed dispatched/attempted and 49 attempt-unknown; 100 confirmed HTTP 200 acceptances and 50 acceptance-unknown (49 status-0 records plus the HTTP 503). HTTP 503 confirms dispatch, not generation acceptance or rejection. Zero automatic retries. Status checks, retrievals, merges and evaluations are not generation calls.

The first parser-evaluable direct response wins in launch order, never by correctness, AST size or holdout. All 43 evaluable incorrect answers are excluded from later residuals. Selected responses: 50 max and 50 xhigh. Missing/error/canceled/expired/empty/capped/refusal/incomplete responses are ineligible even if nonempty. Only assistant output text is extracted; reasoning blocks are excluded.

## Syntax, holdout and complexity

One xhigh response used the established conservative parser normalization, not symbolic repair. Strict original syntax yields **99 evaluable / 56 correct out of 100**; Challenge64 **63 evaluable / 41 correct**; New36 unchanged. Leaderboards use the established standard-parser convention.

Frozen generated-IID holdouts are reported only for train-correct formulas and never used for selection. Four train-correct tasks lack worlds (one Challenge64, three New36). The nine newly train-correct xhigh formulas passed **0/9** available holdouts. Correct-formula AST mean/median: Challenge100 **73.35 / 18**, Challenge64 **93.05 / 18**, New36 **18.2 / 16**.

Known physical usage: **584,685 input + 4,667,846 output = 5,252,531 total tokens**, including **4,656,306 reasoning tokens within output**. Selected-response usage is also **5,252,531 total**. Usage for 50 initial failure records is **unknown, not zero**: equality of known totals does not establish complete physical cost or billing.

The [sanitized verified report](../eval/gpt6_sol_challenge100_round1_report.json) includes per-task formulas and original syntax, categories/ten New36 slices, holdouts, physical-versus-selected usage, and source report hashes. Stored scores were copied without rescoring. Raw responses, reasoning traces, credentials, provider identifiers and private paths are excluded.
