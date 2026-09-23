# Claude Opus 5.5 — completed Challenge100 results

All four authorized physical passes are terminal and verified. No pending batch or additional residual pass is included.

| Set | Evaluable | Correct | Correct among evaluable | Frozen holdout |
|---|---:|---:|---:|---:|
| Challenge100 | 91/100 | 65/100 | 65/91 (71.43%) | 50/60 (83.33%) |
| Challenge64 | 59/64 | 46/64 | 46/59 (77.97%) | 35/45 (77.78%) |
| New36 | 32/36 | 19/36 | 19/32 (59.38%) | 15/15 (100%) |

## Protocol and request accounting

Direct Round 1 only, Anthropic `claude-opus-5-5`, Message Batches API, adaptive thinking, `max_tokens=300000` shared thinking/output budget, `output-300k-2026-03-24` beta, temperature omitted, unchanged JSON-instructed prompts, one response per task per physical pass, zero client retries, tools or symbolic work. Only effort and the exact non-evaluable residual task set change.

| Pass | Effort | Submitted / confirmed executed | Evaluable | Correct | Empty caps | Empty refusals | Residual |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | max | 100 / 100 | 36 | 36 | 33 | 31 | 64 |
| 2 | xhigh | 64 / 64 | 15 | 15 | 31 | 18 | 49 |
| 3 | high | 49 / 49 | 18 | 10 | 23 | 8 | 31 |
| 4 | medium | 31 / 31 | 22 | 4 | 0 | 9 | 9 |

All **244 requests** have terminal provider-succeeded batch results across four batch submissions. This confirms execution/acceptance but is not a claim of 244 individual HTTP 200 responses: the provider does not expose per-request HTTP statuses. Provider success does not imply eligible output or benchmark correctness. The 87 caps and 66 refusals are ineligible; generic report provider_errors includes refusals, not transport errors. No automatic retries. Status checks, retrievals, merges and scoring are not generation calls.

The first parser-evaluable direct response wins in launch order, never by correctness, AST size or holdout. Evaluable incorrect responses are excluded from later residuals. Selected evaluable responses: 36 max, 15 xhigh, 18 high, 22 medium; nine non-evaluable initial fallbacks remain. Their selected status may differ from the final physical response (all nine final-medium residuals are refusals). Missing/error/canceled/expired/empty/capped/refusal/incomplete responses are excluded from candidate extraction even if content exists. Thinking blocks are never extracted as formulas. No fifth pass was run.

## Syntax, holdout and complexity

One high-pass response used the established conservative parser normalization, not symbolic repair. Strict original syntax yields **90 evaluable / 64 correct out of 100**; Challenge64 **58 evaluable / 45 correct**; New36 unchanged. Leaderboards use the established standard-parser convention.

Frozen generated-IID holdouts are reported only for train-correct formulas and never used for selection. Five train-correct tasks lack worlds (one Challenge64, four New36), so the holdout denominator is 60. Correct-formula AST mean/median: Challenge100 **23.2 / 17**, Challenge64 **26.02 / 18**, New36 **16.37 / 16**.

Physical usage: **1,636,960 input tokens including cache + 50,622,790 output tokens including thinking = 52,259,750 total**. Selected-response usage: **18,718,136 total**. Separate reasoning-token counts are unknown, not zero.

The [sanitized verified report](../eval/claude_opus_5_5_challenge100_round1_report.json) includes per-task formulas, original syntax, categories/ten slices, holdouts, physical-versus-selected usage and component report hashes. Scores were copied from verified stored evaluations without rescoring. Raw responses, thinking traces, credentials, provider identifiers and private paths are excluded.
