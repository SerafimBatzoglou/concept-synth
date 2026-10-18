# DeepSeek 4.1 Flash-1017 — final verified results

Holdouts v2: five worlds per task; only the 14 changed tasks were re-evaluated.

| Set | Evaluable | Strict parse | Train correct | Holdout correct |
|---|---:|---:|---:|---:|
| Challenge100 | 100/100 | 96/100 | 21/100 | 9/100 |
| Challenge64 projection | 64/64 | 61/64 | 21/64 | 9/64 |
| New36 | 36/36 | 35/36 | 0/36 | 0/36 |

Holdout scoring is conditional on training correctness: all 21 train-correct
answers were tested on five v2 worlds. The other 79 have no measured holdout
outcome, not a measured failure, and receive no verified credit. Four selected
answers required the established parser normalization; 20 training passes were
strict as submitted. Correct-formula AST mean/median: 118.9 / 40.

## Protocol and attempts

All 100 logical first trials used MAX thinking. Local connection failures led to
25 human-authorized same-MAX transport recoveries, still logical first trials.
One confirmed parser-non-evaluable response received the authorized one-notch
lower HIGH fallback. Selection uses the first parser-evaluable response in actual
launch order, never training or holdout correctness. Evaluable incorrect answers
were not retried. Final selection: 99 MAX answers and one HIGH answer; this is a
mixed-effort result, not MAX-only.

There were 126 physical submissions: 100 originals, 25 MAX recoveries and one
HIGH fallback. The initial 75 originals used a 384,000-token cap, mistakenly
described at launch as the maximum. The remaining 25 originals, all 25 recoveries
and the HIGH fallback used the corrected 393,216-token maximum. Messages,
model and response schema were preserved. The shared scheduler limit was 25,
with no tools, no prior answers in prompts, no timeout and zero automatic retries.

101 requests have confirmed terminal results. The other 25 original streams
were disconnected or administratively abandoned; their final provider status
and usage remain unknown. Administrative slot release was not provider
cancellation, and remote overlap with waived requests cannot be excluded.
There are no locally pending requests or authorized queued runs.

Known physical usage: 596,296 input tokens and 12,526,535 completion tokens,
including 12,503,662 reasoning tokens exactly once. Cached input: 184,960;
visible output: 22,873. Estimated known cost: $7.58–$15.16, not an invoice,
excluding the 25 unknown-usage requests. Polls, snapshots and administrative
envelopes are not additional submissions.

## Stored-score release

The final v1 result was 8 holdout passes among 20 available evaluations, with
one train-correct answer missing holdouts. V2 supplies all five worlds and has
9 passes among 21 evaluations. Only 3 of the 14 changed tasks were train-correct
and holdout-scored; 11 were skipped. The other 86 tasks and all training scores
were copied exactly. No inference or scoring occurred during publication.

- [Final report, task scores, attempt histories and ten New36 slices](../eval/deepseek_flash1017_challenge100_holdout_v2.json)
- [Sanitized formulas](../predictions/deepseek_flash1017_challenge100_predictions_v2.jsonl.gz)
- [Current Challenge64 training cache](../eval/induction_challenge64_round1_eval_cache_v2.jsonl)
- [Current Challenge64 holdout cache](../eval/induction_challenge64_round1_holdout_eval_cache_v2.jsonl)
- [Release manifest and verification hashes](../holdout_v2_evaluation_release_manifest.json)

Raw provider payloads, reasoning and private paths are excluded. All existing
models and historical v1 scores remain unchanged.
