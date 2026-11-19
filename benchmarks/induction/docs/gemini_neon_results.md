# Gemini 4 Neon — Challenge100, holdouts v2

| Set | Strictly evaluable | Training correct | Holdout correct | Correct AST mean / median |
|---|---:|---:|---:|---:|
| Challenge100 | 100/100 | 77/100 | **73/100** | 17.3 / 17 |
| Challenge64 | 64/64 | 51/64 | **48/64** | 17.1 / 18 |
| New36 | 36/36 | 26/36 | **25/36** | 17.7 / 16 |

The direct Google Gemini Interactions API ran `gemini-4-neon` on Flex. The
initial ten tasks matched the GPT-6.2 Astra/Sol pilot, and the continuation
covered the remaining 90 with up to 90 workers. The output cap stayed at
128,000 tokens, including thinking. No tools, normalization, or symbolic repair
were used.

The run started at XHIGH. Capped non-evaluable answers could step down to HIGH
and MEDIUM, with one MEDIUM repeat. Confirmed provider errors could retry at
the same effort. Two separately authorized final passes covered 25 remaining
non-evaluable tasks at MEDIUM and then 23 at LOW. All 23 LOW answers were
strictly evaluable, but none passed training.

For each task, the first parser-evaluable answer in attempt order was selected
regardless of correctness. Only training-correct answers were evaluated on the
five frozen holdout worlds. The holdout scores above use all tasks in each
set as the denominator.

All **262 accepted Flex requests** (100 originals and 162 retries) reached a
terminal state and reported usage. They used 37,769,958 input tokens, of which
1,682,965 were cached, and 27,340,377 output tokens, of which 27,333,734 were
thinking tokens. Output totals include capped responses and retries.

The [per-task scores](../eval/gemini_neon_challenge100_holdout_v2.json) and
[extracted predictions](../predictions/gemini_neon_challenge100_predictions_v2.jsonl.gz)
copy verified stored results. Publication made no model calls and did not
rescore any answer. Raw responses, private reasoning, and provider IDs are excluded.
