# Gemini Hydrogen — Challenge100, holdouts v2

| Set | Strictly evaluable | Training correct | Holdout correct | Correct AST mean / median |
|---|---:|---:|---:|---:|
| Challenge100 | 95/100 | 73/100 | 52/100 | 40.8 / 18 |
| Challenge64 | 61/64 | 53/64 | 32/64 | 49.7 / 18 |
| New36 | 34/36 | 20/36 | 20/36 | 17.4 / 16 |

Each problem received one original request to the direct Gemini API
(`gemini-4-hydrogen`, XHIGH). Thinking and visible output shared a cap of
1,048,576 tokens minus the counted input. The random pilot of 10 was followed by
the remaining 90, initially with 10 workers, then up to 40. No tools, retries,
normalization, or symbolic repair were used. Only training-correct formulas
were evaluated on the five frozen holdout worlds.

The pilot contributed 7 holdout passes and the remaining 90 contributed 45.
Five requests returned incomplete at their output caps: `extreme_logic_024`,
`extreme_logic_003`, `hard_067`, `fullobs_v2_dual_role_cross_lift_qd2_003`,
and `fullobs_v2_compact_response_qd3_007`. These count in the leaderboard
denominators; none was a lost request.

All 100 requests reported usage: 307,231,841 input tokens (73,017,629 cached)
and 51,025,623 output tokens (51,014,104 thinking and 11,519 visible).
The [per-task results](../eval/gemini_hydrogen_challenge100_holdout_v2.json)
and [predictions](../predictions/gemini_hydrogen_challenge100_predictions_v2.jsonl.gz)
copy the verified scores and extracted formulas. Raw responses and reasoning are not published.
