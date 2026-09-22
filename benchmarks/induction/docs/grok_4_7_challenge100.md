# Grok 4.7 — interim Challenge100 results

This is the verified **three-pass snapshot**, not the final four-pass campaign. The additional medium pass on `hard_036` and `hard_051` was still pending at this snapshot and is excluded. Its outcomes will be published separately after verification.

| Set | Evaluable | Correct | Correct among evaluable | Frozen holdout |
|---|---:|---:|---:|---:|
| Challenge100 | 98/100 | 23/100 | 23/98 (23.47%) | 14/22 (63.64%) |
| Challenge64 | 62/64 | 21/64 | 21/62 (33.87%) | 12/20 (60.00%) |
| New36 | 36/36 | 2/36 | 2/36 (5.56%) | 2/2 (100.00%) |

## Protocol and request accounting

Direct Round 1 only, xAI `grok-4.7`, no explicit output-token cap, temperature 0.6, JSON output, up to 20 parallel workers, one request per task per physical pass, zero automatic retries, no tools and no symbolic work. Residuals include only tasks without a parser-evaluable response, never evaluable incorrect tasks. The first parser-evaluable response wins in pass order; if none parses, the earliest response is retained as a non-evaluable fallback.

| Pass | Effort | Submitted / accepted HTTP 200 | Evaluable | Correct | Residual |
|---|---|---:|---:|---:|---:|
| 1 | xhigh | 100 / 100 | 69 | 19 | 31 |
| 2 | high | 31 / 31 | 22 | 3 | 9 |
| 3 | medium | 9 / 9 | 7 | 1 | 2 |

The published snapshot contains **140 accepted generation calls**. Two further requests were started for the pending final medium pass: **142 started campaign-wide**, but their acceptance, results and usage are not yet included. HTTP 200 establishes acceptance even if a stream subsequently fails. Evaluations, merges and earlier partial snapshots are not generation calls.

Selected evaluable responses: 69 from xhigh, 22 from high and 7 from medium. Two non-evaluable fallback records retain initial-pass provenance. Missing, provider-error, timeout, empty, output-capped and parse-invalid responses count incorrect and non-evaluable. One train-correct task lacks generated holdout worlds and is excluded from the holdout denominator.

## Syntax and complexity

The established conservative parser normalized two submitted responses from the initial pass. These are disclosed, not symbolic repair. Under strict original syntax the snapshot is **96 evaluable / 21 correct out of 100**, and **60 evaluable / 19 correct on Challenge64**. New36 is unchanged. The leaderboard follows the established standard-parser convention.

Correct-formula AST mean/median: Challenge100 **60.57 / 17**, Challenge64 **64.81 / 17**, New36 **16 / 16**. Detailed per-task formulas, original submitted syntax, category/slice metrics, frozen holdout, physical-pass and selected-response token statistics, and component provenance are in the [sanitized verified report](../eval/grok_4_7_challenge100_round1_report.json). All figures were copied from frozen verified scores; this publication did not rescore any run. Raw responses, reasoning traces, credentials, internal IDs and private paths are excluded.
