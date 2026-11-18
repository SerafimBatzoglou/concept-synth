# GPT-6.2 Astra and Sol — Challenge100, holdouts v2

| Model | Set | Evaluable (strict) | Training correct | Holdout correct | Correct AST mean / median |
|---|---|---:|---:|---:|---:|
| GPT-6.2 Astra | Challenge100 | 100/100 (100 strict) | 100/100 | **97/100** | 18.0 / 16 |
| GPT-6.2 Astra | Challenge64 | 64/64 (64 strict) | 64/64 | 62/64 | 16.7 / 18 |
| GPT-6.2 Astra | New36 | 36/36 (36 strict) | 36/36 | 35/36 | 20.2 / 16 |
| GPT-6.2 Sol | Challenge100 | 100/100 (99 strict) | 98/100 | **73/100** | 27.6 / 18 |
| GPT-6.2 Sol | Challenge64 | 64/64 (64 strict) | 62/64 | 39/64 | 33.1 / 18 |
| GPT-6.2 Sol | New36 | 36/36 (35 strict) | 36/36 | 34/36 | 18.2 / 16 |

Both models used the direct OpenAI Responses API on Standard, with `max` reasoning
and a 128,000-token cap inclusive of reasoning. They received the same ten randomly
selected pilot problems (seed 20261117) and then the same remaining 90, with a shared
60-worker limit for the continuation. No tools or symbolic repair were used.

The first parser-evaluable answer in attempt order was selected, regardless of
correctness. Terminal non-evaluable responses could receive up to two additional
same-settings attempts. Astra needed none. Five Sol originals hit the token cap;
all five became evaluable on their first retry. One selected Sol answer required
conservative normalization. Only training-correct answers were tested on the five
frozen holdout worlds.

| Model | Accepted requests | Input tokens (cached) | Output tokens (reasoning included) |
|---|---:|---:|---:|
| GPT-6.2 Astra | 100 | 584,685 (0) | 1,521,265 |
| GPT-6.2 Sol | 105 | 615,200 (30,500) | 5,446,141 |

Usage includes capped originals and retries. Four rejected pilot Flex submissions
(two per model) did not become generations. All accepted requests reached a
terminal state and reported usage.

Per-task scores: [Astra](../eval/gpt62_astra_challenge100_holdout_v2.json)
· [Sol](../eval/gpt62_sol_challenge100_holdout_v2.json).
Extracted predictions: [Astra](../predictions/gpt62_astra_challenge100_predictions_v2.jsonl.gz)
· [Sol](../predictions/gpt62_sol_challenge100_predictions_v2.jsonl.gz).
Publishing copied verified stored formulas and scores without rescoring or provider calls.
Raw responses, private reasoning, and provider request IDs are excluded.
