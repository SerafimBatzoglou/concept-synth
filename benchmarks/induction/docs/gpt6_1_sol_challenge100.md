# GPT-6.1 Sol — final Challenge100 results

The regular-API campaign completed on September 29, 2026. Every local request is terminal, and the final 100-task ledger was independently verified by copying stored scores without rescoring.

| Set | Evaluable | Train correct | Holdout correct / all problems | Available holdouts passed | Strict evaluable / correct |
|---|---:|---:|---:|---:|---:|
| Challenge100 | 98/100 | 90/100 | 62/100 (62.0%) | 62/84 | 90 / 88 |
| Challenge64 | 63/64 | 61/64 | 42/64 (65.625%) | 42/60 | 61 / 60 |
| New36 | 35/36 | 29/36 | 20/36 (55.5556%) | 20/24 | 29 / 28 |

Six train-correct tasks lack generated holdout worlds (one Challenge64, five New36). Missing outcomes remain unknown, not asserted failures; they receive no verified-success credit in the full-denominator leaderboard. Holdouts are fixed generated-IID sidecars, used only for post-selection reporting on train-correct formulas.

## Protocol and selection

Exact model `gpt-6.1-sol`, regular OpenAI Responses API (`/v1/responses`), standard/default reasoning mode, `max_output_tokens=128000` shared by reasoning and visible output, JSON-object output, temperature omitted. Original frozen prompts/messages/schema were retained. Each task-attempt requested one direct Round-1 response (`max_rounds=1`, `n_candidates=1`). There were no tools, symbolic candidates, repairs, streaming, background transport, or model substitutions. Total local concurrency peaked at 20, with no duplicate active task.

The original effort was MAX. Policy v3 allowed three cumulative additional error retries and, separately, three explicit token-cap retries. Error retries one and two kept the failed effort; the third additional error retry lowered it one notch, charging only the error budget. The realized sequence was **original MAX → error retry 1 MAX → error retry 2 MAX → error retry 3 XHIGH**. There were no explicit token-cap outcomes or cap-triggered retries. Cohort number is not retry ordinal. The SDK made zero automatic retries; future error retries used explicit `timeout=None`, with no local watchdog/deadline. The original client had a 3600-second timeout. Gateway errors can still occur without a local timeout.

Selection was the **first eligible parser-evaluable answer in actual per-task launch order**, never train correctness, formula size, or holdout success. If none was evaluable, the original MAX record was retained as a non-evaluable fallback. All eight evaluable but train-incorrect answers were excluded from further retries. Errors, refusals, empty/capped/incomplete/missing responses were ineligible. Only assistant output text was processed, never reasoning text.

Two tasks stopped at the three-error-retry limit: `extreme_context_010` and `fullobs_v2_nested_containment_qd3_006`. Both ended with HTTP 524 after four physical attempts. No fourth error retry or cap-budget borrowing occurred. HTTP 524 indicates an API/gateway error, not token exhaustion; thinking time as a cause was not established.

## Syntax and complexity

Leaderboards retain the established conservative-parser convention. Eight selected answers used existing parser normalization; no parser/scorer change or symbolic repair was made. Strict as-submitted results are shown separately above. Two normalized answers are train-correct; six are train-incorrect.

Correct-formula AST mean / median: Challenge100 **61.88 / 18**, Challenge64 **45.67 / 18**, New36 **95.97 / 16**. The machine-readable report includes every category and all ten New36 slices, original submitted formula strings, normalized stored formulas and stored train/holdout score fields.

## Physical attempts and unknown usage

| Stage | Effort | Physical calls | HTTP 200 / evaluable | Train correct | API errors |
|---|---|---:|---:|---:|---:|
| Original | MAX | 100 | 55 | 55 | 45 |
| First additional error retry | MAX | 45 | 10 | 10 | 35 |
| Second additional error retry (cohorts 002–004) | MAX | 35 | 9 | 9 | 26 |
| Third additional error retry (cohorts 005–011) | XHIGH | 26 | 24 | 16 | 2 |
| Total | | 206 | 98 | 90 | 108 |

All 206 attempts were started, confirmed dispatched, and locally terminal across 12 physical components. HTTP statuses were 98×200, 96×524, 6×502, 4×500, and 2×520. HTTP 200 confirms acceptance; acceptance, remote outcomes, usage and billing for the 108 failed requests remain **unknown**, not zero or confirmed rejection. Local terminal status does not establish remote completion.

The original 100 calls comprise 98 new requests and two reused probes, counted once, not 102. Immutable evaluation snapshots are not physical passes. An earlier Batch submission failed model-support validation; it is recorded separately, not as 100 completed generations, and no billing inference is made.

Known physical usage is **570,320 input + 4,134,385 output = 4,704,705 total tokens**, with **4,111,197 reasoning tokens included within output**. Selected-response known usage coincidentally equals these totals and is reported separately. Neither total is complete billing: failed usage and all actual dollar charges remain unknown. Status checks, evaluation, copying and publication add no model generations.

## Evidence and privacy

The [sanitized final report](../eval/gpt6_1_sol_challenge100_round1_report.json) preserves stored task results, formulas, per-task retry histories/budgets, component summaries, strict syntax, holdouts, physical-versus-selected accounting, and source/completion/audit hashes. The final private ledger contains 100 frontiers, 98 candidates and 182 stored train/holdout scores; all were independently checked against their selected sources, with 2,958 preserved source-file hashes. Publication copies those verified scores; it does not re-evaluate formulas or recompute holdouts.

Raw responses, reasoning traces, credentials, provider identifiers, request/prompt identifiers, private paths and private databases are not published. Existing models and benchmark data are unchanged. See the [combined leaderboards](leaderboard.md) and [Challenge64 table](challenge64_round1_results.md).
