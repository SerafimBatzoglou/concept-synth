# GPT-6 Luna — provisional Challenge100 results

This is a verified **provisional MAX+XHIGH snapshot**, not a completed campaign. HIGH has been submitted for the 14 remaining non-evaluable tasks; MEDIUM is conditional on the later residual. Neither later pass is included in the scores or usage below. The public leaderboards label this row **GPT-6 Luna (provisional)**.

| Set | Evaluable | Correct | Correct among evaluable | Frozen holdout |
|---|---:|---:|---:|---:|
| Challenge100 | 86/100 | 18/100 | 20.93% | 12/17 (70.59%) |
| Challenge64 | 56/64 | 18/64 | 32.14% | 12/17 (70.59%) |
| New36 | 30/36 | 0/36 | 0.00% | N/A |

## Protocol and snapshot boundaries

Direct Round 1 only, exact model `gpt-6-luna`, OpenAI Responses Batch API, standard/default mode, `max_output_tokens=128000` shared reasoning/visible-output budget, JSON-object output, temperature omitted, unchanged original prompts, one response per task per pass, zero client retries, tools or symbolic work.

| Pass | Effort | Submitted | Evaluable | Correct | Residual |
|---|---|---:|---:|---:|---:|
| 1 | max | 100 | 18 | 18 | 82 |
| 2 | xhigh | 82 | 68 | 0 | 14 |
| 3 | high | 14 (in flight; excluded) | pending | pending | pending |
| 4 | medium | not yet submitted | pending/conditional | pending/conditional | pending |

MAX returned 18 completed nonempty eligible responses, 67 empty incomplete responses that reached the output-token cap, and 15 status-0 empty-body failures. XHIGH returned 70 HTTP 200 completed nonempty eligible responses (68 evaluable incorrect, 2 parse-invalid), plus 12 status-0 empty-body failures. XHIGH had zero caps or refusals. No detailed errors were supplied for the status-0 records; their causes are not inferred.

The published snapshot includes 182 terminal request records across two batches: 155 confirmed attempts and HTTP 200 acceptances, plus 27 attempt/acceptance-unknown records; zero retries. HTTP 200 does not establish eligibility: MAX's 67 capped responses remain excluded. Active HIGH requests are not included in these snapshot totals.

Selection is the first parser-evaluable direct response in launch order, never correctness, AST size, or holdout. There are 18 evaluable MAX selections, 68 evaluable XHIGH selections, and 14 earliest-response MAX fallbacks. All 68 evaluable incorrect answers are excluded from subsequent residuals. Error, incomplete, capped, refusal, empty, missing, canceled or expired responses are ineligible even if nonempty; reasoning text is never extracted as an answer.

## Syntax, holdout and usage

Strict original syntax yields **70 evaluable / 10 correct** overall (Challenge64 42/10; New36 28/0). Sixteen selected responses used existing conservative parser normalization; there were no parser changes or symbolic repairs. Leaderboards retain the standard-parser convention.

Frozen generated-IID holdouts are reported only for train-correct formulas, never for selection. Twelve of 17 available holdouts passed; one train-correct task lacks worlds. Correct-formula AST mean/median: **21.44 / 16.5** overall and on Challenge64; New36 has no correct formulas.

Known physical MAX+XHIGH usage is **923,263 input + 13,064,465 output = 13,987,728 total tokens**, including **13,058,664 reasoning tokens within output**. Selected-response usage is **6,070,544 total tokens**. Usage for 27 failure records is **unknown, not zero**. Active HIGH usage is excluded; these numbers are not complete billing.

The [sanitized verified snapshot](../eval/gpt6_luna_challenge100_round1_report.json) includes task formulas, original syntax, source report hashes, subset/category/ten New36 slices, holdouts and physical-versus-selected usage. All 100 ordered task/frontier records and stored candidate scores were independently verified and copied without rescoring. Raw responses, reasoning traces, credentials, provider identifiers and private paths are excluded.
