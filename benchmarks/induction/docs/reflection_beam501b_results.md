# Reflection Beam 501B — Challenge100, holdouts v2

Verified stored results, October 28, 2026 (Pacific time). Publication copies
stored formulas and scores without model calls, rescoring, or symbolic repair.
All authorized requests have ended. **One task remains an explicit exclusion**:
`hard_038` received HTTP529 without an accepted ID and was not retried.
This release does not claim that its remote status has been resolved.

| Set | Tasks | Evaluable | Strict | Training correct | Holdout correct | Correct AST mean / median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Challenge100 | 100 | 99 | 92 | 11 | 8 | 37 / 16 |
| Challenge64 | 64 | 63 | 58 | 11 | 8 | 37 / 16 |
| New36 | 36 | 36 | 34 | 0 | 0 | — |

Leaderboard holdout rates retain all benchmark tasks in the denominator:
**8/100 (8.0%)** and **8/64 (12.5%)**. All 11 training-correct formulas were
evaluated on five frozen holdout-v2 worlds; eight passed all five. Seven
training-incorrect answers needed conservative normalization; 92 parsed strictly.

## Protocol and selection

- OpenRouter `reflection/beam-1.0-501b-a23b`, model version
  `reflection/beam-1.0-501b-a23b-20261016`, pinned to `nebius/fp4`, no fallback.
- MAX reasoning and `max_tokens=131072`, including reasoning, for every attempt.
  Original messages and JSON answer contract, streaming, no tools, prior outputs,
  sampling overrides, automatic HTTP retries, or client timeout.
- Ten random pilot tasks, then the remaining 90. Initially ten local workers;
  after a provider-overload interruption, the 42 never-started originals were
  adopted under the same duplicate guards with a 50-worker limit.
- Up to three same-settings residual attempts were authorized only after an
  audited non-evaluable outcome. Actual requests: **100 originals + 4 first
  residuals**. Two recovered pilot disconnects; two recovered token-capped
  answers. No second or third residual was needed.
- First eligible parser-evaluable answer in actual attempt order is retained,
  including incorrect or normalized answers. No selection by correctness,
  holdout outcome, or AST size. Evaluable answers are never retried. If no
  answer is evaluable, retain the original exclusion.

## New36 slices

All 36 tasks are evaluable; no training or holdout passes in any slice.

| Slice | Tasks | Strict | Training correct | Holdout correct |
|---|---:|---:|---:|---:|
| Binary guard incoming lift, qd2 | 5 | 5 | 0 | 0 |
| Compact EF lift, qd2 | 4 | 4 | 0 | 0 |
| Compact response, qd3 | 3 | 3 | 0 | 0 |
| Dual-role cross lift, qd2 | 4 | 3 | 0 | 0 |
| Filtered dual-role lift, qd2 | 4 | 3 | 0 | 0 |
| Filtered EF lift, qd2 | 4 | 4 | 0 | 0 |
| Guardless cross-scope lift, qd2 | 3 | 3 | 0 | 0 |
| Guardless same-relation scope lift, qd2 | 3 | 3 | 0 | 0 |
| Nested containment, qd3 | 3 | 3 | 0 | 0 |
| Universal nested response, qd3 | 3 | 3 | 0 | 0 |

## Physical accounting and uncertainty

104 physical HTTP submissions: 103 unique accepted IDs, 101 confirmed-terminal
outcomes, and three outcomes with unknown remote completion and final usage.
Snapshots, adoption, evaluations, and polls are not extra requests. No local
requests or authorized queued work remain; the monitor is paused.

Known usage: **602,368 input + 8,568,426 output = 9,170,794 total tokens**.
Output includes **8,556,065 reasoning tokens once**, plus 12,361 visible tokens;
69,760 cached input tokens were reported. The token table rounds known output
to 8.6M; this is a lower bound. Known provider-reported cost is **$8.705066**,
not an invoice and excluding the three unknown-usage outcomes.

- `hard_038`: HTTP529, Nebius worker overloaded, no accepted ID. Original
  exclusion retained; no replacement authorized by the never-started42 request.
- Original `hard_097` and `fullobs_v2_compact_ef_lift_qd2_001`: HTTP200 streams
  ended with `RemoteProtocolError`. Both received evaluable first residuals.
  Original remote completion and usage/billing remain unknown, not zero.

Freeing local capacity does not establish remote cancellation. Possible remote
overlap cannot be excluded. The unresolved task and historical uncertainties
remain explicit in this authorized publication.

## Stored evidence

- [Sanitized scores, formulas, histories and hashes](../eval/reflection_beam501b_challenge100_holdout_v2.json)
- [Predictions with responses and reasoning redacted](../predictions/reflection_beam501b_challenge100_predictions_v2.jsonl.gz)
- [Current leaderboards](leaderboard.md) and [release manifest](../holdout_v2_evaluation_release_manifest.json)

All prior 53 Challenge64 configurations, 34 Challenge100 configurations and
historical v1 artifacts are preserved. Only allowlisted fields are published;
no raw responses/reasoning, credentials, internal IDs, or local paths.
