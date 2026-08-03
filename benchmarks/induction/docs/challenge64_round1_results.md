# INDUCTION Challenge64 Leaderboard: Round 1 (Pre-Symbolic)

Each configuration contributes one direct Round-1 formula per task. The release contains no symbolic repair or simplification outputs. Rows are ranked by train-set Correct, then Evaluable coverage, then model name.

| Model | Evaluable | Correct | Holdout Correct<br>(among train-correct) | Formula Complexity<br>(AST mean/median) |
|---|---:|---:|---:|---:|
| GPT-5.6 Sol | 63/64 | 37/64 (57.8%) | 52.8% (19/36) | 143.0 / 18.0 |
| Qwen 3.8 Max | 63/64 | 27/64 (42.2%) | 53.8% (14/26) | 82.5 / 18.0 |
| Fable 5 | 59/64 | 26/64 (40.6%) | 60.0% (15/25) | 49.7 / 18.0 |
| GPT-5.6 Terra | 62/64 | 24/64 (37.5%) | 69.6% (16/23) | 68.8 / 18.0 |
| Claude Opus 5 | 62/64 | 21/64 (32.8%) | 85.0% (17/20) | 19.2 / 16.0 |
| Muse Spark 1.1 | 53/64 | 21/64 (32.8%) | 70.0% (14/20) | 49.8 / 17.0 |
| GPT-5.6 Luna | 61/64 | 15/64 (23.4%) | 50.0% (7/14) | 108.9 / 18.0 |
| Grok 4 | 59/64 | 13/64 (20.3%) | 91.7% (11/12) | 17.7 / 16.0 |
| GPT-5.4 | 64/64 | 12/64 (18.8%) | 81.8% (9/11) | 22.8 / 16.0 |
| Grok 4.5 | 64/64 | 11/64 (17.2%) | 80.0% (8/10) | 20.1 / 15.0 |
| GPT-5.2 | 64/64 | 9/64 (14.1%) | 25.0% (2/8) | 85.1 / 95.0 |
| Kimi K3 | 64/64 | 7/64 (10.9%) | 71.4% (5/7) | 22.9 / 17.0 |
| Gemini 3.5 Flash | 63/64 | 7/64 (10.9%) | 85.7% (6/7) | 16.6 / 15.0 |
| Grok 4.1 Fast | 64/64 | 6/64 (9.4%) | 100.0% (5/5) | 14.8 / 14.5 |
| DeepSeek V4 Pro | 63/64 | 6/64 (9.4%) | 66.7% (4/6) | 31.5 / 16.0 |
| Claude Opus 4.6 | 64/64 | 5/64 (7.8%) | 100.0% (4/4) | 14.8 / 15.0 |
| Gemini 3.6 Flash | 64/64 | 5/64 (7.8%) | 75.0% (3/4) | 16.6 / 16.0 |
| Kimi K2.7 Code | 64/64 | 5/64 (7.8%) | 75.0% (3/4) | 36.0 / 15.0 |
| Claude Opus 4.8 | 64/64 | 4/64 (6.2%) | 100.0% (3/3) | 15.2 / 15.5 |
| DeepSeek V4 Flash | 64/64 | 3/64 (4.7%) | 50.0% (1/2) | 22.3 / 18.0 |
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

Evaluable: parser-valid formula under the exact FullObs evaluator. Correct: train-world exact-match validity, with the fixed 64-task denominator. Holdout Correct: conditional exact-match validity among train-correct formulas with generated holdout worlds available. Formula complexity summarizes train-correct direct formulas.

The fixed holdout sidecar contains five generated worlds where generation succeeded (63/64 tasks); it is only a post-selection reporting diagnostic. It was not used for model prompting, candidate selection, or symbolic search.
