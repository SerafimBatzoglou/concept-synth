# INDUCTION Challenge64: Round-1 Pre-Symbolic Results

Each configuration contributes one direct Round-1 formula per task. The release contains no symbolic repair or simplification outputs.

| Model | Evaluable | Correct | Holdout Correct | Formula Complexity (AST mean/median) |
|---|---:|---:|---:|---:|
| Claude Opus 4.5 | 64/64 | 0/64 (0.0%) | N/A | 12.7 / 11.5 |
| Claude Opus 4.6 | 64/64 | 5/64 (7.8%) | 100.0% (4/4) | 19.3 / 18.5 |
| DeepSeek Reasoner | 63/64 | 1/64 (1.6%) | 100.0% (1/1) | 14.1 / 13.0 |
| Gemini 3 Pro Preview | 64/64 | 3/64 (4.7%) | 33.3% (1/3) | 16.4 / 10.0 |
| Gemini 3.1 Pro | 64/64 | 1/64 (1.6%) | N/A | 9.0 / 9.0 |
| GPT-4o | 64/64 | 0/64 (0.0%) | N/A | 7.1 / 5.0 |
| GPT-5.2 | 64/64 | 9/64 (14.1%) | 25.0% (2/8) | 42.3 / 21.5 |
| GPT-5.4 | 64/64 | 12/64 (18.8%) | 81.8% (9/11) | 24.1 / 14.0 |
| Grok 4 | 59/64 | 13/64 (20.3%) | 91.7% (11/12) | 12.7 / 9.0 |
| Grok 4.1 Fast | 64/64 | 6/64 (9.4%) | 100.0% (5/5) | 11.2 / 10.0 |
| Hermes 4 | 64/64 | 0/64 (0.0%) | N/A | 9.4 / 8.5 |
| Qwen 3.5 | 43/64 | 0/64 (0.0%) | N/A | 12.0 / 9.0 |
| DeepSeek V4 Pro | 63/64 | 6/64 (9.4%) | 66.7% (4/6) | 19.9 / 16.0 |
| Gemini 3.5 Flash | 63/64 | 7/64 (10.9%) | 85.7% (6/7) | 17.4 / 16.0 |
| Grok 4.3 | 63/64 | 2/64 (3.1%) | 100.0% (2/2) | 9.2 / 9.0 |
| Kimi K2.6 | 62/64 | 3/64 (4.7%) | 100.0% (3/3) | 16.9 / 13.5 |
| Kimi K2.7 Code | 64/64 | 5/64 (7.8%) | 75.0% (3/4) | 67.6 / 22.0 |
| Kimi K3 | 64/64 | 7/64 (10.9%) | 71.4% (5/7) | 16.6 / 15.5 |
| Claude Opus 4.8 | 64/64 | 4/64 (6.2%) | 100.0% (3/3) | 10.4 / 9.0 |
| Claude Sonnet 5 | 47/64 | 3/64 (4.7%) | 100.0% (2/2) | 15.2 / 12.0 |
| Qwen 3.7 Max | 59/64 | 0/64 (0.0%) | N/A | 9.7 / 9.0 |
| GPT-5.6 Sol | 63/64 | 37/64 (57.8%) | 52.8% (19/36) | 96.7 / 20.0 |
| GPT-5.6 Terra | 62/64 | 24/64 (37.5%) | 69.6% (16/23) | 82.9 / 18.5 |
| GPT-5.6 Luna | 61/64 | 15/64 (23.4%) | 50.0% (7/14) | 151.2 / 21.0 |
| Grok 4.5 | 64/64 | 11/64 (17.2%) | 80.0% (8/10) | 14.6 / 13.0 |
| Fable 5 | 59/64 | 26/64 (40.6%) | 60.0% (15/25) | 48.0 / 22.0 |
| Muse Spark 1.1 | 53/64 | 21/64 (32.8%) | 70.0% (14/20) | 38.8 / 17.0 |

Evaluable: parser-valid formula under the exact FullObs evaluator. Correct: train-world exact-match validity, with the fixed 64-task denominator. Holdout Correct: conditional exact-match validity among train-correct formulas with generated holdout worlds available. Formula complexity summarizes all evaluable direct formulas.

The fixed holdout sidecar contains five generated worlds where generation succeeded (63/64 tasks); it is only a post-selection reporting diagnostic. It was not used for model prompting, candidate selection, or symbolic search.
