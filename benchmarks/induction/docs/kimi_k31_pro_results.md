# Kimi K3.1 Pro — Challenge100, holdouts v2

Verified final results, October 23, 2026. These are direct model answers before
symbolic processing, with scores copied from the completed, independently audited
campaign. Publication made no model calls and did not re-evaluate answers.

| Set | Tasks | Evaluable | Strict evaluable | Training correct | Holdout correct | Correct AST mean / median |
|---|---:|---:|---:|---:|---:|---:|
| Challenge100 | 100 | 99 | 98 | 34 | 21 | 45.44 / 18 |
| Challenge64 | 64 | 63 | 62 | 30 | 17 | 49.30 / 20.5 |
| New36 | 36 | 36 | 36 | 4 | 4 | 16.50 / 16 |

Leaderboard holdout percentages use all tasks: 21/100 (21.0%) and 17/64 (26.6%).
All 34 training-correct answers were evaluated on five frozen v2 holdout worlds;
21 passed all five. Holdouts never influence prompting or answer selection.
All 34 training-correct answers were strict. The one conservatively normalized
answer (`hard_035`: remove two trailing close parentheses and collapse a unary
Boolean expression) was training-incorrect. No symbolic repair/search was used.

## Protocol and selection

- Direct Moonshot `kimi-k3.1-pro`, `reasoning_effort=max` and
  `max_completion_tokens=500000`, including thinking tokens.
- Ten uniformly random pilot tasks (seed 20261023), then their exact 90-task
  complement. Ten pilot workers; at most 50 simultaneous local jobs thereafter.
- Original prompts, JSON response contract and streaming settings unchanged.
  No tools, previous responses, sampling overrides or automatic HTTP retries;
  no client timeout.
- Only closed, accepted, HTTP-200 disconnected streams qualified for transport
  recovery. The authorized bound was three retries per problem. In practice all
  38 affected tasks returned evaluable answers on the first retry. No second or
  third retries were needed. Terminal parser failures and evaluable answers,
  including incorrect answers, were not retried.
- Selection is the first eligible parser-evaluable response in actual attempt
  order, never the most correct answer. If none is evaluable, retain the original
  exclusion. All pilot answers and evaluable originals are preserved exactly.

`extreme_logic_009` is the sole residual non-evaluable task: HTTP 200, normal
`stop`, nonempty final content, but not parser-evaluable. It was neither capped
nor refused and did not qualify for a transport retry.

## New36 slices

All 36 responses were strictly evaluable.

| Slice | Tasks | Training correct | Holdout correct |
|---|---:|---:|---:|
| Binary guard incoming lift, qd2 | 5 | 2 | 2 |
| Compact EF lift, qd2 | 4 | 0 | 0 |
| Compact response, qd3 | 3 | 0 | 0 |
| Dual-role cross lift, qd2 | 4 | 0 | 0 |
| Filtered dual-role lift, qd2 | 4 | 0 | 0 |
| Filtered EF lift, qd2 | 4 | 0 | 0 |
| Guardless cross-scope lift, qd2 | 3 | 0 | 0 |
| Guardless same-relation scope lift, qd2 | 3 | 1 | 1 |
| Nested containment, qd3 | 3 | 0 | 0 |
| Universal nested response, qd3 | 3 | 1 | 1 |

## Physical accounting and uncertainty

There were **138 physical requests**: 100 originals and 38 first transport retries,
all with distinct accepted IDs. They produced 100 confirmed terminal outcomes and
38 locally closed outcomes with unknown remote completion. Snapshot creation,
evaluation and combination are not additional generation requests.

Known usage covers 100 requests: 594,862 input tokens plus 11,539,644 completion
tokens = 12,134,506 total. Completion already includes 11,530,392 reasoning tokens
and 9,252 visible tokens; do not add reasoning again. There are 79,616 reported
cached-input tokens, with cache usage unknown for 76 requests. Input/completion
usage is unknown for the 38 interrupted originals, so totals are incomplete.
Pricing and billing are unknown, not zero or an invoice estimate.

Releasing a local worker after disconnection does not establish provider
cancellation or zero billing. Remote overlap cannot be excluded. The original
failures recorded `RemoteProtocolError`; the exact provider/proxy/network cause
was not established. No local requests or authorized queues remain.

## Stored evidence

- [Sanitized stored scores, formulas, per-task histories and source hashes](../eval/kimi_k31_pro_challenge100_holdout_v2.json)
- [Predictions, with raw responses and thinking redacted](../predictions/kimi_k31_pro_challenge100_predictions_v2.jsonl.gz)
- [Current leaderboards](leaderboard.md) and [holdout-v2 version details](challenge_holdouts_v2.md)
- [Release manifest](../holdout_v2_evaluation_release_manifest.json)

Only allowlisted scalar usage and stored score fields are released. Private raw
responses/reasoning, provider IDs, internal candidate/prompt IDs, credentials and
local paths are excluded. Previous 50 Challenge64 configurations, 31 Challenge100
configurations, and all historical v1 artifacts remain unchanged.
