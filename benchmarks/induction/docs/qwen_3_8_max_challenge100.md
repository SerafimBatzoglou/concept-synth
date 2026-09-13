# Qwen 3.8 Max: final Challenge100 direct cascade

Direct Round-1 cascade: frozen Qwen 3.8 Max Preview Challenge64 plus New36 and residuals; three released qwen3.8-max-0902 low-effort passes (22, 13, 19 tasks), 65,536 output tokens, eight workers; parser-evaluability-only response selection.

| Subset | Evaluable | Correct | Correct / evaluable |
|---|---:|---:|---:|
| Challenge64 | 63/64 | 27/64 | 27/63 |
| New36 | 20/36 | 0/36 | 0/20 |
| Challenge100 | 83/100 | 27/100 | 27/83 |

The frozen Challenge64 projection is unchanged. Generated-IID holdout is 14/26 among train-correct formulas with available worlds (one unavailable); correct AST mean/median is 82.48/18. All five newly evaluable tasks are New36 and incorrect. Seventeen tasks remain non-evaluable. No further pass is included.

## Request accounting and effort provenance

228 scoped HTTP attempts: 133 accepted responses and 95 rejected requests (87 quota HTTP 429, eight authentication HTTP 401). Historical 166-attempt reporting incorrectly called all attempts accepted; this release distinguishes them. No automatic retries. The 64-task frozen projection excludes unrelated retries and later rounds.

Historical Preview requests sent the maximum thinking budget of 262,144 even where requested-effort metadata said medium/low. The three released 0902 passes used explicit low (4,096 reasoning tokens), 65,536 maximum output tokens, and eight workers. The mixed-snapshot cascade is not a homogeneous single-setting run.

The final 19-task pass produced 12 output-cap, five parse-invalid, two evaluable incorrect, and zero correct responses. Provider-completed and parser-evaluable are distinct.

[Full task results, categories/slices, pass metrics, token usage, and provenance](../eval/qwen_3_8_max_challenge100_round1_report.json).
