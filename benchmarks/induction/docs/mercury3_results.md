# Inception Mercury 3 — final Challenge100 results

Released October 19, 2026. This is the completed, bounded retry campaign, not a
single-request baseline. Results were copied from independently audited stored
scores; publication made no model calls and did not re-evaluate formulas.
All earlier model rows and historical v1 artifacts are preserved.

**Holdouts v2:** five frozen worlds per task, used from the start of this campaign.
Only train-correct selected formulas receive holdout evaluation. Holdouts never
enter prompts or response selection. Unscored outcomes are unknown, not measured
holdout failures, and receive no leaderboard credit.

| Set | Evaluable | Strict evaluable | Train correct | Holdout correct / all tasks |
|---|---:|---:|---:|---:|
| Challenge100 | 96/100 | 95/100 | 9/100 | 5/100 (5.0%) |
| Challenge64 projection | 60/64 | 59/64 | 9/64 | 5/64 (7.8%) |
| New36 | 36/36 | 36/36 | 0/36 | 0/36 (0.0%) |

All nine training successes are strict and have complete holdout evaluations;
five pass all five worlds. The one conservatively normalized response is
training-incorrect. Correct-formula AST size is 88.1 mean / 18 median. Every one
of the ten New36 slices has full evaluable coverage and zero training or holdout
successes; per-slice counts are in the stored report.

## Request and selection protocol

Model: `inception/mercury-3` (`inception/mercury-3-20261019`), through OpenRouter
with Inception-only routing. HIGH is the highest supported reasoning effort;
MEDIUM is one notch lower. Every request used a 131,072-token maximum, the same
original messages and JSON contract, no tools, no prior responses, no sampling
overrides, no timeout, and no automatic HTTP retries. The pilot used ten workers;
subsequent requests shared a maximum of forty live local workers, with retries
eligible as capacity became available.

| Physical attempt | Effort | Eligibility | Requests | Newly evaluable |
|---|---|---|---:|---:|
| 1 | HIGH | Original 100 tasks, including pilot ten | 100 | 45 |
| 2 | HIGH | Non-evaluable after attempt 1 | 55 | 26 |
| 3 | MEDIUM | Still non-evaluable after attempt 2 | 29 | 4 |
| 4 | HIGH | Separately authorized residual25 pass | 25 | 17 |
| 5 | MEDIUM | Still non-evaluable after attempt 4 | 8 | 3 |
| 6 | MEDIUM | Separately authorized residual5 pass | 5 | 1 |

Each outcome was evaluated once and independently audited before retry eligibility
was established. Evaluable answers were never retried, including incorrect,
normalized, or holdout-incorrect answers. Selection uses the first eligible
parser-evaluable response in actual launch order, **never correctness**. The
original exclusion is retained if none is evaluable. The final pass preserved
all 95 previously evaluable selections exactly; the final selection contains 88
HIGH and eight MEDIUM evaluable answers. No symbolic search, repair, or
simplification is included.

Four tasks remain non-evaluable: `hard_081` and `hard_026` ended locally with
unknown remote status in their final attempts; `hard_086` and `hard_035` reached
the token limit. No more requests are pending locally, and monitoring is paused.

## Accounting and uncertainty

There were 222 physical submissions: 194 unique accepted requests and 28 HTTP
429 outcomes. Of these, 142 have confirmed remote-terminal outcomes and 80 have
unknown remote completion/billing. Locally closed requests freed worker slots;
this does not confirm provider cancellation, and possible remote overlap cannot
be excluded. Snapshot components and polling are not counted as new requests.

Known usage across **all attempts**, including unsuccessful ones:

- Input: 649,627 tokens; cached input: 64,478.
- Completion: 11,584,662 tokens, already including 11,567,748 reasoning tokens
  and 16,914 visible tokens. Total input plus completion: 12,234,289.
- Known provider-reported cost: $4.11379138. This is incomplete, not an invoice.
- Usage is unreported for 108 submissions (80 remote-unknown outcomes and 28
  HTTP 429 outcomes); unknown usage is not zero.

Selected-answer known usage is reported separately: 571,887 input and 9,418,279
completion tokens, with three selected exclusions having unknown usage. The
leaderboard token column uses all-attempt known completion usage (11.6 million),
not selected-answer usage, and never adds reasoning a second time.

## Public artifacts

- [Stored report, per-task scores, retry history and all ten slices](../eval/mercury3_challenge100_holdout_v2.json)
- [Sanitized predictions](../predictions/mercury3_challenge100_predictions_v2.jsonl.gz)
- [Challenge100 and Challenge64 leaderboards](leaderboard.md)
- [Challenge64 standalone table](challenge64_round1_results.md)
- [Versioned release manifest](../holdout_v2_evaluation_release_manifest.json)

Predictions contain the selected formulas, not raw answer text or reasoning.
Provider response payloads, request identifiers, private paths and credentials
are omitted. Public tests check stored-score consistency, first-evaluable
selection, bounded attempts, accounting, privacy and preservation of the prior
49 configurations without re-running benchmark evaluation.
