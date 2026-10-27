# GPT-6.1 Terra — Challenge100, holdouts v2

Verified final results, October 26, 2026 (Pacific time). Stored formulas and scores
are copied from the completed campaign and independently audited. Publication
made no model calls and did not re-evaluate answers. No symbolic search or repair.

| Set | Tasks | Evaluable | Strict | Training correct | Holdout correct | Correct AST mean / median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Challenge100 | 100 | 100 | 99 | 67 | 33 | 144.72 / 23 |
| Challenge64 | 64 | 64 | 63 | 53 | 23 | 134.89 / 38 |
| New36 | 36 | 36 | 36 | 14 | 10 | 181.93 / 16 |

Leaderboard holdout percentages use all tasks: **33/100 (33.0%)** and
**23/64 (35.9%)**. All 67 training-correct answers were tested on five frozen
holdout-v2 worlds; 33 passed all five. One training-incorrect answer required
conservative normalization; the other 99 parsed strictly. No non-evaluable
problems remain. Holdout outcomes never influence retries or selection.

## Protocol and selection

- Direct OpenAI Responses API, `gpt-6.1-terra`, `max_output_tokens=128000`
  including reasoning. Original messages and JSON answer contract retained;
  streaming, no tools, previous responses, sampling overrides, automatic HTTP
  retries or client timeout; `store=false`.
- All 100 originals used MAX reasoning. Only an audited non-evaluable outcome
  qualified for one XHIGH retry, then one HIGH retry if still non-evaluable.
  Actual requests: **100 MAX + 59 XHIGH + 0 HIGH**.
- Up to 50 concurrent local workers, with eligible retries prioritized as
  capacity became available rather than waiting for the original round.
- Flex first, with standard-tier fallback permitted only for explicit
  pre-acceptance Flex-unavailability rejection. All 159 accepted requests
  reported Flex; no standard fallback or rejected extra delivery occurred.
- Select the first eligible parser-evaluable response in actual attempt order,
  even when incorrect or normalized. Never choose by training/holdout success
  or AST size. Evaluable answers were never retried.

## New36 slices

All 36 selected answers were strictly evaluable.

| Slice | Tasks | Training correct | Holdout correct |
|---|---:|---:|---:|
| Binary guard incoming lift, qd2 | 5 | 4 | 3 |
| Compact EF lift, qd2 | 4 | 3 | 3 |
| Compact response, qd3 | 3 | 0 | 0 |
| Dual-role cross lift, qd2 | 4 | 0 | 0 |
| Filtered dual-role lift, qd2 | 4 | 0 | 0 |
| Filtered EF lift, qd2 | 4 | 0 | 0 |
| Guardless cross-scope lift, qd2 | 3 | 3 | 1 |
| Guardless same-relation scope lift, qd2 | 3 | 3 | 2 |
| Nested containment, qd3 | 3 | 0 | 0 |
| Universal nested response, qd3 | 3 | 1 | 1 |

## Physical accounting and uncertainty

159 physical requests and 159 HTTP POSTs, each with a unique accepted ID:
158 confirmed terminal outcomes and one locally closed disconnected stream.
No local requests or queued work remain; the monitor is paused. Evaluation
components and polls are not extra generations.

Known usage across the 158 terminal outcomes: **930,613 input + 11,959,948 output
= 12,890,561 total tokens**. Output includes **11,939,572 reasoning tokens once**
plus 20,376 visible tokens. No cached input was reported. The leaderboard rounds
known output to 12.0M; it is a lower bound on total campaign usage.

The original MAX request for `hard_081` ended with `RemoteProtocolError` after
HTTP 200 and acceptance. Its XHIGH retry produced an evaluable answer. Original
remote completion and final usage/billing remain **unknown, not zero** and are
excluded from known totals. Freeing the local slot does not establish remote
cancellation; possible remote overlap cannot be excluded. No cost or invoice
amount is asserted.

## Stored evidence

- [Sanitized stored scores, formulas, histories and hashes](../eval/terra61_challenge100_holdout_v2.json)
- [Predictions with responses and reasoning redacted](../predictions/terra61_challenge100_predictions_v2.jsonl.gz)
- [Current leaderboards](leaderboard.md) and [holdout-v2 details](challenge_holdouts_v2.md)
- [Release manifest](../holdout_v2_evaluation_release_manifest.json)

Only allowlisted scores, formulas and accounting fields are public. No raw
responses/reasoning, credentials, provider/internal IDs or local paths are
released. All previous 52 Challenge64 configurations, 33 Challenge100
configurations and historical v1 artifacts are preserved.
