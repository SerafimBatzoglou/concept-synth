# Grok 4.7 — completed Challenge100 results

All five physical runs are terminal and verified, including the explicitly authorized two-task resubmission. This replaces the earlier three-pass interim snapshot.

| Set | Evaluable | Correct | Correct among evaluable | Frozen holdout |
|---|---:|---:|---:|---:|
| Challenge100 | 100/100 | 24/100 | 24/100 (24.00%) | 14/23 (60.87%) |
| Challenge64 | 64/64 | 22/64 | 22/64 (34.38%) | 12/21 (57.14%) |
| New36 | 36/36 | 2/36 | 2/36 (5.56%) | 2/2 (100.00%) |

## Protocol and request accounting

Direct Round 1 only, xAI `grok-4.7`, no explicit output-token cap, temperature 0.6, JSON output, up to 20 parallel workers, one request per task per physical pass, zero automatic retries, no tools and no symbolic work. Residuals include only tasks without a parser-evaluable response, never evaluable incorrect tasks. The first parser-evaluable response wins in launch order, never by correctness or holdout.

| Physical pass | Effort | Submitted / accepted HTTP 200 | Evaluable | Correct | Cumulative residual |
|---|---|---:|---:|---:|---:|
| 1 | xhigh | 100 / 100 | 69 | 19 | 31 |
| 2 | high | 31 / 31 | 22 | 3 | 9 |
| 3 | medium | 9 / 9 | 7 | 1 | 2 |
| 4 | medium | 2 / 2 | 0 | 0 | 2 |
| 5 (authorized resubmission) | medium | 2 / 2 | 2 | 1 | 0 |

All **144 submitted generation calls** were accepted with HTTP 200 and are included, including the two original final-medium streams that subsequently ended with ConnectionError and no usable output. The user explicitly authorized two additional requests for `hard_036` and `hard_051` while the original requests were unresolved; this was a manual resubmission in a separate physical run, not an automatic retry. The originals were retained and counted, not canceled or silently replaced. Evaluations, merges and partial snapshots are not generation calls.

Selected responses: 69 from pass 1, 22 from pass 2, 7 from pass 3, zero from pass 4, and 2 from the resubmission. Missing, provider-error, timeout, empty, output-capped and parse-invalid responses count incorrect and non-evaluable. One train-correct task lacks generated holdout worlds and is excluded from the holdout denominator. The additional train-correct resubmission formula did not pass its generated holdout.

## Syntax and complexity

The established conservative parser normalized two responses from the initial pass. These are disclosed, not symbolic repair. Strict original syntax gives **98 evaluable / 22 correct out of 100**, and **62 evaluable / 20 correct on Challenge64**. New36 is unchanged. The leaderboard follows the standard-parser convention.

Correct-formula AST mean/median: Challenge100 **59 / 17**, Challenge64 **62.91 / 17.5**, New36 **16 / 16**. Per-task formulas, original syntax, categories/slices, frozen holdout, physical-pass and selected-response token statistics, and provenance are in the [sanitized verified report](../eval/grok_4_7_challenge100_round1_report.json). This publication copies verified stored scores without rescoring. Raw responses, reasoning traces, credentials, internal IDs and private paths are excluded.
