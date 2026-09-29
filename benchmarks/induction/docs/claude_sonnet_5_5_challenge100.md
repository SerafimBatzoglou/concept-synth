# Claude Sonnet 5.5 — completed Challenge100 results

All four authorized passes are terminal and independently verified. The residual is empty.

| Set | Evaluable | Correct | Correct among evaluable | Frozen holdout |
|---|---:|---:|---:|---:|
| Challenge100 | 100/100 | 44/100 | 44.00% | 37/42 (88.10%) |
| Challenge64 | 64/64 | 32/64 | 50.00% | 26/31 (83.87%) |
| New36 | 36/36 | 12/36 | 33.33% | 11/11 (100%) |

## Protocol and physical usage

Direct Round 1 only, Anthropic `claude-sonnet-5-5`, Message Batches, adaptive thinking, `max_tokens=300000` shared thinking/output budget, `output-300k-2026-03-24` beta, temperature omitted, original JSON-instructed prompts, one response per task per pass. Only effort and the exact non-evaluable subset change. No tools, symbolic work, fallback models or client retries.

| Effort | Submitted / confirmed executed | Evaluable | Correct | Empty caps | Input tokens | Output tokens | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAX | 100 / 100 | 31 | 31 | 69 | 667,158 | 24,786,807 | 25,453,965 |
| XHIGH | 69 / 69 | 12 | 10 | 57 | 471,782 | 19,816,613 | 20,288,395 |
| HIGH | 57 / 57 | 54 | 3 | 3 | 394,848 | 11,316,059 | 11,710,907 |
| MEDIUM | 3 / 3 | 3 | 0 | 0 | 22,170 | 111,109 | 133,279 |
| Total | 229 / 229 | — | — | 129 | 1,555,958 | 56,030,588 | 57,586,546 |

All 229 requests have terminal provider-succeeded results across four batches: 229 confirmed attempted and accepted, zero unknown acceptance or attempts, zero refusals/provider errors, zero retries. This is not a claim of individual-message HTTP 200 responses; batch-level HTTP status is distinct. Provider success does not imply eligible output or correctness. Status, retrieval, evaluation and combination are not generations.

Physical usage is **57,586,546 tokens**. Selected-response usage is **667,158 input + 17,330,588 output = 17,997,746 tokens**, not physical cost. Output includes thinking; separate reasoning usage is **unknown**, not zero. Monetary cost is not supplied and remains unknown.

## Selection, syntax and audit

The first eligible parser-evaluable response in MAX → XHIGH → HIGH → MEDIUM order wins, never by correctness, AST size, mismatch or holdout. Evaluable incorrect responses are excluded from later residuals. Selected responses: 31 MAX, 12 XHIGH, 54 HIGH, 3 MEDIUM. There are no fallbacks or remaining non-evaluable tasks. No fifth pass was run.

Errors, refusals, empty, capped, incomplete/paused, missing, canceled and expired responses are ineligible even if nonempty. Thinking blocks are never extracted as formulas. Strict original syntax matches standard-parser results: 100 evaluable / 44 correct; zero parser normalizations and zero symbolic candidates.

Frozen generated-IID holdouts are evaluated only for train-correct formulas and never used for selection. Two correct tasks lack worlds (one Challenge64 and one New36); the available denominator is 42. Correct-formula AST mean / median: Challenge100 18.954545 / 17; Challenge64 19.59375 / 18; New36 17.25 / 16.

The [sanitized report](../eval/claude_sonnet_5_5_challenge100_round1_report.json) includes all 100 source-ordered task formulas, stored train/holdout outcomes, categories and all ten New36 slices, per-pass accounting and source hashes. Both Challenge64 public caches and the full100 report copy verified stored scores without rescoring. Independent audits checked selected fields, formulas, candidate metadata, stored scores, prompt provenance, 100 frontiers and preserved source hashes. Raw responses, thinking traces, credentials, private paths and provider identifiers are excluded.
