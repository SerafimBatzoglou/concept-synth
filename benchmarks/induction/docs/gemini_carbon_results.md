# Gemini Carbon — Challenge100, holdouts v2

| Set | Strictly evaluable | Training correct | Holdout correct | Correct AST mean / median |
|---|---:|---:|---:|---:|
| Challenge100 | 100/100 | 82/100 | 70/100 | 26.8 / 18 |
| Challenge64 | 64/64 | 57/64 | 47/64 | 30.8 / 18 |
| New36 | 36/36 | 25/36 | 23/36 | 17.6 / 16 |

The direct Gemini API ran `gemini-4-carbon` with HIGH thinking. Each task's output
cap was the 1,048,576-token window minus its provider-counted input. The run used
10 random pilot tasks followed by the remaining 90, with up to 50 workers. The
frozen results use the first evaluable attempt in submission order available at
the cutoff, regardless of correctness. No answer needed normalization or symbolic
repair. Only training-correct formulas were evaluated on the five holdout worlds.

**Accounting cutoff: November 11, 2026, 23:11 UTC.** There were 120 submissions
including retries for provider failures and 11 explicitly authorized replacements
for long-running requests. Nine superseded requests still lacked terminal results
at the cutoff; they were not assumed cancelled. None of the 100 problems lacks a
selected answer. Known usage across 108 submissions was 27,158,517 output tokens,
including 27,150,473 reasoning tokens counted once. The other 12 submissions lacked
final token usage. This is a lower bound, not a billing total.

The [per-task report](../eval/gemini_carbon_challenge100_holdout_v2.json) and
[predictions](../predictions/gemini_carbon_challenge100_predictions_v2.jsonl.gz)
copy stored formulas and scores; publishing required no model or scoring calls.
Raw responses, private reasoning, and provider request IDs are excluded.
