# INDUCTION Challenge Leaderboards

Challenge100 is the ordered union of the frozen Challenge64 benchmark and the disjoint New36 component. All 20 Challenge100 models appear in the Challenge64 table, alongside 20 additional models with Challenge64 results. Each table is ranked independently by correct answers on its own task set, so model order differs.

Missing, provider-error, empty, output-limit-incomplete, and parse-invalid responses count as incorrect. A multi-formula response is evaluable if any submitted formula parses and correct if any submitted formula is train-valid. Residual cascades use parser-evaluable priority only, never correctness or holdout outcomes.

## Challenge100

Rows are ranked by Challenge100 Correct, then Evaluable coverage, then model name.

| Model | Evaluable | Correct | Holdout Correct<br>(among train-correct) | Formula Complexity<br>(AST mean/median) |
|---|---:|---:|---:|---:|
| GPT-6 Astra | 94/100 | 94/100 (94.0%) | 91.9% (79/86) | 22.0 / 16.0 |
| GPT-5.6 Sol | 99/100 | 43/100 (43.0%) | 56.4% (22/39) | 125.0 / 18.0 |
| Fable 5.1 | 66/100 | 33/100 (33.0%) | 93.8% (30/32) | 18.1 / 16.0 |
| Fable 5 | 96/100 | 31/100 (31.0%) | 63.3% (19/30) | 46.1 / 18.0 |
| GPT-5.6 Terra | 98/100 | 25/100 (25.0%) | 70.8% (17/24) | 66.6 / 18.0 |
| Grok 4.6 | 98/100 | 25/100 (25.0%) | 70.8% (17/24) | 72.2 / 17.0 |
| Claude Opus 5 | 97/100 | 24/100 (24.0%) | 87.0% (20/23) | 18.8 / 16.0 |
| Muse Spark 1.3 | 93/100 | 23/100 (23.0%) | 68.2% (15/22) | 47.6 / 17.0 |
| Muse Spark 1.1 | 86/100 | 21/100 (21.0%) | 70.0% (14/20) | 49.8 / 17.0 |
| GPT-5.6 Luna | 97/100 | 16/100 (16.0%) | 50.0% (7/14) | 102.8 / 18.0 |
| DeepSeek V4 Pro 0813 | 98/100 | 15/100 (15.0%) | 35.7% (5/14) | 130.9 / 77.0 |
| Ox Alpha | 87/100 | 12/100 (12.0%) | 72.7% (8/11) | 25.5 / 16.0 |
| Gemini 3.7 Flash | 100/100 | 11/100 (11.0%) | 90.0% (9/10) | 16.4 / 15.0 |
| Grok 4.5 | 100/100 | 11/100 (11.0%) | 80.0% (8/10) | 20.1 / 15.0 |
| Muse Spark 1.2 | 92/100 | 11/100 (11.0%) | 70.0% (7/10) | 29.1 / 17.0 |
| Gemini 3.8 Flash | 100/100 | 9/100 (9.0%) | 100.0% (8/8) | 15.6 / 16.0 |
| Gemini 3.5 Flash | 98/100 | 7/100 (7.0%) | 85.7% (6/7) | 16.6 / 15.0 |
| DeepSeek V4 Flash | 97/100 | 6/100 (6.0%) | 20.0% (1/5) | 77.8 / 48.0 |
| DeepSeek V4 Pro | 94/100 | 6/100 (6.0%) | 66.7% (4/6) | 31.5 / 16.0 |
| Gemini 3.6 Flash | 93/100 | 5/100 (5.0%) | 75.0% (3/4) | 16.6 / 16.0 |

Challenge100 formula complexity covers all train-correct direct formulas across its 100 tasks. Its generated-IID holdout diagnostic combines the frozen Challenge64 and New36 sidecars and reports only train-correct responses whose task has generated holdout worlds.

## Challenge64 projection

Rows are ranked by Challenge64 train-set Correct, then Evaluable coverage, then model name. Holdout is a post-selection diagnostic and is never used for prompting or selection.

| Model | Evaluable | Correct | Holdout Correct<br>(among train-correct) | Formula Complexity<br>(AST mean/median) |
|---|---:|---:|---:|---:|
| GPT-6 Astra | 63/64 | 63/64 (98.4%) | 88.7% (55/62) | 20.6 / 18.0 |
| GPT-5.6 Sol | 63/64 | 37/64 (57.8%) | 52.8% (19/36) | 143.0 / 18.0 |
| Fable 5 | 63/64 | 27/64 (42.2%) | 57.7% (15/26) | 50.3 / 18.0 |
| Qwen 3.8 Max | 63/64 | 27/64 (42.2%) | 53.8% (14/26) | 82.5 / 18.0 |
| GPT-5.6 Terra | 62/64 | 24/64 (37.5%) | 69.6% (16/23) | 68.8 / 18.0 |
| Grok 4.6 | 62/64 | 23/64 (35.9%) | 68.2% (15/22) | 76.9 / 17.0 |
| Muse Spark 1.3 | 59/64 | 23/64 (35.9%) | 68.2% (15/22) | 47.6 / 17.0 |
| Fable 5.1 | 47/64 | 22/64 (34.4%) | 90.5% (19/21) | 17.6 / 16.0 |
| Claude Opus 5 | 62/64 | 21/64 (32.8%) | 85.0% (17/20) | 19.2 / 16.0 |
| Muse Spark 1.1 | 56/64 | 21/64 (32.8%) | 70.0% (14/20) | 49.8 / 17.0 |
| DeepSeek V4 Pro 0813 | 64/64 | 15/64 (23.4%) | 35.7% (5/14) | 130.9 / 77.0 |
| GPT-5.6 Luna | 61/64 | 15/64 (23.4%) | 50.0% (7/14) | 108.9 / 18.0 |
| Grok 4 | 59/64 | 13/64 (20.3%) | 91.7% (11/12) | 17.7 / 16.0 |
| GPT-5.4 | 64/64 | 12/64 (18.8%) | 81.8% (9/11) | 22.8 / 16.0 |
| Ox Alpha | 54/64 | 12/64 (18.8%) | 72.7% (8/11) | 25.5 / 16.0 |
| Gemini 3.7 Flash | 64/64 | 11/64 (17.2%) | 90.0% (9/10) | 16.4 / 15.0 |
| Grok 4.5 | 64/64 | 11/64 (17.2%) | 80.0% (8/10) | 20.1 / 15.0 |
| Muse Spark 1.2 | 59/64 | 11/64 (17.2%) | 70.0% (7/10) | 29.1 / 17.0 |
| GPT-5.2 | 64/64 | 9/64 (14.1%) | 25.0% (2/8) | 85.1 / 95.0 |
| Gemini 3.8 Flash | 64/64 | 9/64 (14.1%) | 100.0% (8/8) | 15.6 / 16.0 |
| Kimi K3 | 64/64 | 7/64 (10.9%) | 71.4% (5/7) | 22.9 / 17.0 |
| Gemini 3.5 Flash | 63/64 | 7/64 (10.9%) | 85.7% (6/7) | 16.6 / 15.0 |
| Grok 4.1 Fast | 64/64 | 6/64 (9.4%) | 100.0% (5/5) | 14.8 / 14.5 |
| DeepSeek V4 Pro | 63/64 | 6/64 (9.4%) | 66.7% (4/6) | 31.5 / 16.0 |
| DeepSeek V4 Flash | 62/64 | 6/64 (9.4%) | 20.0% (1/5) | 77.8 / 48.0 |
| Claude Opus 4.6 | 64/64 | 5/64 (7.8%) | 100.0% (4/4) | 14.8 / 15.0 |
| Gemini 3.6 Flash | 64/64 | 5/64 (7.8%) | 75.0% (3/4) | 16.6 / 16.0 |
| Kimi K2.7 Code | 64/64 | 5/64 (7.8%) | 75.0% (3/4) | 36.0 / 15.0 |
| Claude Opus 4.8 | 64/64 | 4/64 (6.2%) | 100.0% (3/3) | 15.2 / 15.5 |
| Gemini 3 Pro Preview | 64/64 | 3/64 (4.7%) | 33.3% (1/3) | 31.0 / 23.0 |
| Kimi K2.6 | 62/64 | 3/64 (4.7%) | 100.0% (3/3) | 15.3 / 15.0 |
| Claude Sonnet 5 | 47/64 | 3/64 (4.7%) | 100.0% (2/2) | 14.3 / 14.0 |
| Grok 4.3 | 63/64 | 2/64 (3.1%) | 100.0% (2/2) | 15.5 / 15.5 |
| Gemini 3.1 Pro | 64/64 | 1/64 (1.6%) | N/A | 12.0 / 12.0 |
| DeepSeek Reasoner | 63/64 | 1/64 (1.6%) | 100.0% (1/1) | 15.0 / 15.0 |
| Claude Opus 4.5 | 64/64 | 0/64 (0.0%) | N/A | N/A |
| GPT-4o | 64/64 | 0/64 (0.0%) | N/A | N/A |
| Hermes 4 | 64/64 | 0/64 (0.0%) | N/A | N/A |
| Qwen 3.7 Max | 59/64 | 0/64 (0.0%) | N/A | N/A |
| Qwen 3.5 | 43/64 | 0/64 (0.0%) | N/A | N/A |

Formula complexity reports AST mean/median over train-correct direct formulas.
