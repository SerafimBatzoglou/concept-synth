# Claude Haiku 5.5 — Challenge100, holdouts v2

Verified final results, October 26, 2026. Stored scores and formulas are copied
from the completed campaign and independently audited. Publication made no model
calls and did not re-evaluate answers. No symbolic search or repair was used.

| Set | Tasks | Evaluable (all strict) | Training correct | Holdout correct | Correct AST mean / median |
|---|---:|---:|---:|---:|---:|
| Challenge100 | 100 | 99 | 16 | 15 | 16.375 / 16 |
| Challenge64 | 64 | 63 | 15 | 14 | 16.4 / 16 |
| New36 | 36 | 36 | 1 | 1 | 16 / 16 |

Leaderboard holdout percentages use all tasks: **15/100 (15.0%)** and
**14/64 (21.9%)**. All16 training-correct answers were evaluated on five frozen
holdout-v2 worlds;15 passed all five. No normalization was needed. Holdouts never
influence prompts, retry eligibility or response selection.

## Protocol and selection

- Direct Anthropic `claude-haiku-5-5`, adaptive thinking, `max_tokens=128000`
  including thinking. Original prompts, JSON contract and streaming unchanged;
  no tools, previous responses, sampling overrides or automatic HTTP retries;
  no client timeout.
- Random pilot10 (seed20261025), then its exact remaining90-task complement.
  Pilot:10 MAX originals,10 XHIGH retries,9 HIGH retries and2 MEDIUM retries.
  The pilot produced10 strictly evaluable answers and2 training/holdout successes.
- Remaining90:90 XHIGH originals,79 HIGH retries,34 MEDIUM retries. Only an
  audited non-evaluable outcome qualified for the next lower effort. Evaluable
  answers, even incorrect, never retried. Pilot successes and all prior scores
  remain unchanged.
- At most50 simultaneous local jobs for the remaining90, with retries starting
  as capacity became available rather than waiting for a round. Ten pilot workers.
- Selection is the first eligible parser-evaluable answer in actual attempt
  order, never the most correct answer. Original exclusion retained if none.

`extreme_logic_024` remains non-evaluable: XHIGH and HIGH reached the token cap;
its MEDIUM attempt ended with an `api_error` after HTTP200. No further request was
authorized or launched. It receives no leaderboard credit.

## New36 slices

All36 selected answers were strictly evaluable.

| Slice | Tasks | Training correct | Holdout correct |
|---|---:|---:|---:|
| Binary guard incoming lift, qd2 | 5 | 1 | 1 |
| Compact EF lift, qd2 | 4 | 0 | 0 |
| Compact response, qd3 | 3 | 0 | 0 |
| Dual-role cross lift, qd2 | 4 | 0 | 0 |
| Filtered dual-role lift, qd2 | 4 | 0 | 0 |
| Filtered EF lift, qd2 | 4 | 0 | 0 |
| Guardless cross-scope lift, qd2 | 3 | 0 | 0 |
| Guardless same-relation scope lift, qd2 | 3 | 0 | 0 |
| Nested containment, qd3 | 3 | 0 | 0 |
| Universal nested response, qd3 | 3 | 0 | 0 |

## Physical accounting and uncertainty

**234 physical submissions**:31 pilot plus203 remaining90 submissions. All234
had unique accepted IDs. There were228 confirmed terminal outcomes and6 locally
closed streams with unknown remote completion. No local jobs or queues remain;
the monitor is paused. Snapshots/evaluation/polls are not extra model calls.

The228 terminal requests reported **1,522,100 input +23,323,724 output =24,845,824
total tokens**. Output already includes thinking once; the thinking/visible split
is unknown. The leaderboard's23.3M output figure excludes incomplete streams and
is marked as incomplete in its accounting source.

Six HTTP200 streams ended with `api_error`: five HIGH attempts later received
evaluable MEDIUM answers; the sixth is the unresolved MEDIUM outcome above.
Their observed counters were36,846 input and46 output tokens, but **final usage
and billing are unknown**, not zero. These incomplete counters are preserved
separately as `reported_stream_usage`, not included as complete known usage.
The sum of all observed counters is24,882,716 tokens and is not a final bill.
Releasing local slots does not establish remote cancellation; remote overlap
cannot be excluded. Pricing/cost estimates are unavailable.

## Stored evidence

- [Sanitized stored scores, formulas, histories and hashes](../eval/haiku55_challenge100_holdout_v2.json)
- [Predictions with responses and thinking redacted](../predictions/haiku55_challenge100_predictions_v2.jsonl.gz)
- [Current leaderboards](leaderboard.md) and [holdout-v2 version details](challenge_holdouts_v2.md)
- [Release manifest](../holdout_v2_evaluation_release_manifest.json)

Only allowlisted score, formula and accounting fields are public. No raw
responses/thinking, credentials, provider/internal IDs or local paths are released.
All previous51 Challenge64 configurations,32 Challenge100 configurations and
historical v1 artifacts are preserved.
