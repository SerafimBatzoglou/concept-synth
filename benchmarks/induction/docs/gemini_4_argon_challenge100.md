# Gemini 4 Argon: verified mixed-effort Challenge100 campaign

Final results, October 13, 2026 (Pacific time). This is **not a high-only result**.

| Scope | Evaluable | Train correct | Verified holdout correct | Available holdouts among train-correct | Missing holdout worlds among train-correct |
|---|---:|---:|---:|---:|---:|
| Challenge100 | 100/100 | 57/100 | 37/100 | 54 | 3 |
| Challenge64 | 64/64 | 42/64 | 24/64 | 41 | 1 |
| New36 | 36/36 | 15/36 | 13/36 | 13 | 2 |

The leaderboard uses all-problem denominators for holdout success. Conditional
holdout success is 37/54; missing outcomes remain unknown, not asserted failures.
All selected answers are strict/parser-evaluable. No syntax normalization,
symbolic repair, simplification, or score-based replacement was used. AST size
among train-correct answers has mean 45.5263 and median 18.

## Protocol and selection

The exact model was `gemini-4-argon`. All requests used original frozen benchmark
messages, JSON output, no tools or prior answers, background stored interactions,
no timeout, and zero automatic transport/generation retries. Per-task maximum
output tokens were 1,048,576 minus provider-counted prompt input and remained
unchanged across effort levels. One shared scheduler enforced at most 50 workers.

1. High effort: 100 original requests and 28 one-time same-setting retries for
   confirmed non-evaluable failures. The immutable high-stage baseline was
   84 evaluable, 57 train-correct, and 37 holdout-correct.
2. Medium effort: one human-authorized attempt on each of the 16 residual
   non-evaluable tasks. Fifteen were evaluable but train-incorrect and were
   retained; `extreme_context_022` failed with `INVALID_ARGUMENT`.
3. Low effort: one authorized fallback on that sole remaining non-evaluable task.
   Its answer was evaluable but train-incorrect. No additional retry followed.

There were 145 unique accepted physical requests: 128 high, 16 medium, and 1 low.
All are terminal and evaluated. Selected answers comprise 84 high, 15 medium,
and 1 low. Selection takes the first eligible parser-evaluable response in actual
per-task launch order; train correctness, formula complexity, and holdout results
never influence selection. Completed incorrect answers are never retried.
Canonical attempt numbers are preserved, including the task that skipped a
same-high retry and went directly from its original incomplete to medium.

Failed, incomplete, empty, refused, capped, canceled, and otherwise ineligible
outcomes are excluded from answer selection even if they contain text. One
original incomplete exactly matched its output budget but had no provider reason;
it remains classified as reason-unknown, not a verified cap. Its later medium
answer is evaluable. The rejected batch preflight accepted no batch job and is
not counted as 90 generations. All 100 final answers are now evaluable, with
43 train-incorrect answers retained.

## Stored-score export and accounting

Every physical outcome was evaluated once. Final snapshots, selected candidate
fields, 100 TRAIN scores and 54 HOLDOUT scores were copied without reparsing,
rescoring, or recomputing holdouts. Independent SQL/source-hash checks verified
the combined ledger and exact preservation of all 84 high-stage selected answers.
Frozen holdout worlds were used only after train correctness, never as model input.

Physical usage across all 145 requests: 262,449,163 input tokens (35,099,096 cached),
57,789,684 combined output tokens (57,777,205 reasoning plus 12,479 visible).
Provider totals also contain a separate 205,381 tool-use-token category despite
no tools being requested. Its cause and billing are unknown; it is neither added
to output nor priced. Absent tool fields are not asserted to be zero.

At the recorded estimate rates ($2/M uncached input, $0.10/M cached input,
$10/M combined output), arithmetic usage cost is $1,036.1069; subtracting the
provider's explicit $0.67084 waiver gives **$1,035.4360 estimated**, not an invoice.
Other failed-request billing notices charge only successfully generated tokens;
their adjustments and separate tool charges remain unknown. Selected-answer
usage is reported separately. Polls, snapshots, copied ledgers, and overlapping
checkpoints are not extra generations or extra cumulative usage.

- [Sanitized final report, per-task scores and attempt history](../eval/gemini_4_argon_challenge100_round1_report.json)
- [All 100 selected predictions (gzip)](../predictions/gemini_4_argon_challenge100_predictions.jsonl.gz)
- [Challenge100 and Challenge64 leaderboards](leaderboard.md)

Raw responses, raw reasoning, provider interaction identifiers, credentials,
internal candidate/prompt identifiers, and private paths are excluded. Public
report hashes identify the audited immutable sources without exposing their paths.
