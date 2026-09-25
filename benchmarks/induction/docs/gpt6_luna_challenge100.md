# GPT-6 Luna — final Challenge100 results

All four authorized passes (MAX → XHIGH → HIGH → MEDIUM) are terminal, processed exactly once and independently verified. This final result replaces the provisional two-pass snapshot. No pending passes or non-evaluable tasks remain; no fifth pass is authorized.

| Set | Evaluable | Correct | Correct among evaluable | Frozen holdout |
|---|---:|---:|---:|---:|
| Challenge100 | 100/100 | 18/100 | 18.00% | 12/17 (70.59%) |
| Challenge64 | 64/64 | 18/64 | 28.12% | 12/17 (70.59%) |
| New36 | 36/36 | 0/36 | 0.00% | N/A |

## Protocol and selection

Direct Round 1 only, exact model `gpt-6-luna`, OpenAI Responses Batch API, standard/default mode, `max_output_tokens=128000` shared reasoning/visible-output budget, JSON-object output, temperature omitted, unchanged original prompts, one response per task per pass, zero client retries, tools or symbolic work.

| Pass | Submitted / terminal | Evaluable | Correct | Residual | Selected |
|---|---:|---:|---:|---:|---:|
| MAX | 100 | 18 | 18 | 82 | 18 |
| XHIGH | 82 | 68 | 0 | 14 | 68 |
| HIGH | 14 | 9 | 0 | 5 | 9 |
| MEDIUM | 5 | 5 | 0 | 0 | 5 |

Selection is the first parser-evaluable direct response in launch order, never correctness, AST size, mismatch count or holdout. Evaluable incorrect responses are excluded from later residuals. All 100 final selections are evaluable, with no fallback selections. Error, incomplete, capped, refusal, empty, missing, canceled or expired responses are ineligible even if nonempty; reasoning text is never extracted as an answer.

MAX returned 18 completed nonempty eligible responses, 67 empty incomplete responses that reached the output-token cap, and 15 status-0 empty-body failures. XHIGH returned 70 HTTP 200 completed nonempty eligible responses (68 evaluable incorrect, two parse-invalid), plus 12 status-0 empty-body failures. HIGH returned nine HTTP 200 completed nonempty eligible responses, all evaluable incorrect, plus two HTTP 503 `server_is_overloaded` failures and three status-0 empty-body failures. MEDIUM returned five HTTP 200 completed nonempty eligible responses, all evaluable incorrect. The three residual passes had zero caps or refusals. No detailed errors were supplied for status-0 records; their causes are not inferred.

## Physical calls and token costs

The following costs cover every physical pass, including unselected responses. Each pass is one batch, with zero retries. Confirmed HTTP 200 acceptance does not imply eligibility (MAX includes 67 capped responses). A 503 confirms dispatch, not generation acceptance; status 0 leaves both unknown. Internal provider attempts are not exposed.

| Pass | Attempt confirmed / unknown | Acceptance confirmed / unknown | Known input | Known output | Known total | Reasoning within output | Unknown usage records |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAX | 85 / 15 | 85 / 15 | 496,646 | 9,887,712 | 10,384,358 | 9,886,358 | 15 |
| XHIGH | 70 / 12 | 70 / 12 | 426,617 | 3,176,753 | 3,603,370 | 3,172,306 | 12 |
| HIGH | 11 / 3 | 9 / 5 | 50,478 | 197,439 | 247,917 | 196,863 | 5 |
| MEDIUM | 5 / 0 | 5 / 0 | 32,686 | 44,077 | 76,763 | 43,833 | 0 |
| Total | 171 / 30 | 169 / 32 | 1,006,427 | 13,305,981 | 14,312,408 | 13,299,360 | 32 |

There are 201 submitted/terminal records across four batches. Selected-response usage is separately **584,685 input + 4,602,961 output = 5,187,646 total tokens**, including 4,596,673 reasoning tokens within output. It is not the physical campaign cost. Usage for 32 failure records is **unknown, not zero**; these known token totals are not complete billing. Monetary cost is unknown.

## Syntax, holdout and audit

Strict original syntax yields **83 evaluable / 10 correct** overall (Challenge64 49/10; New36 34/0). Seventeen selected responses used existing conservative parser normalization; MEDIUM required none. There were no parser changes or symbolic repairs. Leaderboards retain the standard-parser convention.

Frozen generated-IID holdouts are reported only for train-correct formulas, never for selection. Twelve of 17 available holdouts passed; one train-correct task lacks worlds. Correct-formula AST mean/median: **21.44 / 16.5** overall and on Challenge64; New36 has no correct formulas.

The [sanitized final report](../eval/gpt6_luna_challenge100_round1_report.json) includes task formulas, original syntax, all four source report hashes, subset/category/ten New36 slices, holdouts and physical-versus-selected usage. All 100 ordered task/frontier records, candidate provenance and stored scores were independently compared read-only and copied without rescoring. Each physical pass was retrieved, normalized and evaluated exactly once. All component reports and private databases are preserved unchanged. Raw responses, reasoning traces, credentials, provider identifiers and private paths are excluded.
