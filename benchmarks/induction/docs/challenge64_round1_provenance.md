# Challenge64 Round-1 Release Provenance

This release is an additive Challenge64 artifact. It does not alter the immutable INDUCTION v1.0 benchmark data, predictions, or frozen evaluation cache.

## Selection Policy

For each model configuration and task, the source registry fixes a component order. The release selects the earliest parser-evaluable direct Round-1 response from that order. A later component is considered only when earlier components have no evaluable direct response. The selector never examines train validity, mismatch count, AST size, symbolic candidates, or holdout outcomes.

## Privacy and Scope

Released prediction records contain normalized formulas and exact verifier outputs only. Raw responses, reasoning traces, API job identifiers, credentials, timestamps, internal candidate identifiers, and local filesystem paths are excluded. Every retained result is pre-symbolic Round 1.

## Included Configurations

| Model | Source class | Round-1 protocol |
|---|---|---|
| Claude Opus 4.5 | Released v1.0 ledger | released v1.0 Round-1 result |
| Claude Opus 4.6 | Released v1.0 ledger | released v1.0 Round-1 result |
| DeepSeek Reasoner | Released v1.0 ledger | released v1.0 Round-1 result |
| Gemini 3 Pro Preview | Released v1.0 ledger | released v1.0 Round-1 result |
| Gemini 3.1 Pro | Released v1.0 ledger | released v1.0 Round-1 result |
| GPT-4o | Released v1.0 ledger | released v1.0 Round-1 result |
| GPT-5.2 | Released v1.0 ledger | released v1.0 Round-1 result |
| GPT-5.4 | Released v1.0 ledger | released v1.0 Round-1 result |
| Grok 4 | Released v1.0 ledger | released v1.0 Round-1 result |
| Grok 4.1 Fast | Released v1.0 ledger | released v1.0 Round-1 result |
| Hermes 4 | Released v1.0 ledger | released v1.0 Round-1 result |
| Qwen 3.5 | Released v1.0 ledger | released v1.0 Round-1 result |
| DeepSeek V4 Pro | Current pipeline Round-1 ledger | single-formula pipeline Round 1, maximum thinking |
| DeepSeek V4 Pro 0813 | Current pipeline Round-1 ledger | single-formula Round 1, maximum thinking, 384K maximum output tokens, strict initial-to-residual cascade |
| DeepSeek v4 Flash | Current pipeline Round-1 ledger | single-formula pipeline Round 1, maximum thinking, max output 384K, 32-worker serial dispatch |
| Gemini 3.5 Flash | Current pipeline Round-1 ledger | 3-formula pipeline Round 1, high thinking |
| Gemini 3.6 Flash | Current pipeline Round-1 ledger | single-formula pipeline Round 1, high thinking |
| Gemini 3.7 Flash | Current pipeline Round-1 ledger | multi-formula response-set Round 1, high thinking, 65K maximum output tokens; response evaluable if any formula parses and correct if any formula is train-valid |
| Grok 4.3 | Current pipeline Round-1 ledger | single-formula pipeline Round 1, high thinking |
| Kimi K2.6 | Current pipeline Round-1 ledger | 3-formula pipeline Round 1, thinking enabled |
| Kimi K2.7 Code | Current pipeline Round-1 ledger | single-formula pipeline Round 1, high thinking |
| Kimi K3 | Current pipeline Round-1 ledger | single-formula pipeline Round 1, high thinking |
| Claude Opus 4.8 | Current pipeline Round-1 ledger | Round 1, xhigh thinking |
| Claude Opus 5 | Current pipeline Round-1 ledger | Round 1 direct effort cascade, xhigh with high/medium/low fallback, 128K maximum output tokens |
| Claude Sonnet 5 | Current pipeline Round-1 ledger | Round 1, high thinking |
| Qwen 3.7 Max | Current pipeline Round-1 ledger | Round 1, serial provider calls |
| Qwen 3.8 Max | Current pipeline Round-1 ledger | single-formula Round 1, maximum thinking, 64K maximum output tokens |
| GPT-6 Astra | Current pipeline Round-1 ledger | Interim direct-only Round 1, xhigh coverage with high-effort recovery of the fixed random-10 non-evaluable residual, 128K maximum output tokens; remaining residual batches pending |
| GPT-5.6 Sol | Current pipeline Round-1 ledger | single-formula Round 1, xhigh thinking |
| GPT-5.6 Terra | Current pipeline Round-1 ledger | single-formula Round 1, xhigh thinking |
| GPT-5.6 Luna | Current pipeline Round-1 ledger | single-formula Round 1, xhigh thinking |
| Grok 4.5 | Current pipeline Round-1 ledger | single-formula Round 1, high thinking |
| Grok 4.6 | Current pipeline Round-1 ledger | single-formula Round 1, maximum requested thinking mapped to provider high, 64K maximum output tokens, 32-worker serial dispatch |
| Fable 5 | Current pipeline Round-1 ledger | Round 1, medium thinking with low-effort non-evaluable recovery, including the Challenge100 residual2 pass |
| Fable 5.1 | Current pipeline Round-1 ledger | Direct-only Round 1, high-to-medium-to-low-to-low parser-evaluable residual cascade, 128K maximum output tokens |
| Muse Spark 1.1 | Current pipeline Round-1 ledger | Round 1, xhigh primary with low/high recovery and a final minimal-effort Challenge100 residual pass |
| Muse Spark 1.3 | Current pipeline Round-1 ledger | Direct-only Round 1, xhigh-to-high-to-medium-to-low-to-high-to-medium-to-low-to-low parser-evaluable residual cascade with contributor handoff and contributor residual passes, 131K maximum output tokens |
| Ox Alpha | Current pipeline Round-1 ledger | Direct-only Round 1 through OpenRouter, initial high effort followed by parser-evaluable high/low residual recovery, 131K maximum output tokens |
| Muse Spark 1.2 | Current pipeline Round-1 ledger | Round 1, xhigh thinking, 131K maximum output tokens, with non-evaluable retry |

The generated-holdout sidecar is documented in the companion holdout eval cache. It contains no gold formula and was generated from the task generator after the Challenge64 set was fixed.
