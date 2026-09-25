# Challenge100 Round-1 provenance

Challenge100 is the ordered union of the released Challenge64 dataset and the disjoint New36 component. The release publishes compact benchmark records and aggregate direct Round-1 model results; internal provider requests, raw reasoning traces, credentials, and private pipeline paths are excluded.

Residual components are selected solely by parser-evaluable priority in the declared effort order. Correctness and fixed holdout outcomes are reporting signals only and are never used for selection. Multi-formula responses are evaluated as response sets: any parseable formula makes the response evaluable, and any train-valid formula makes it correct. Fixed generated-IID holdout sidecars cover both Challenge64 and New36; they were created after train-only selection and are reporting diagnostics only.

## Qwen 3.8 Max update

The final mixed Preview/released0902 cascade is documented in [the Qwen audit](qwen_3_8_max_challenge100.md), including the historical effort-payload discrepancy and corrected attempted-versus-accepted request accounting. Its Challenge64 projection remains unchanged; five additional New36 responses are evaluable but incorrect.

## Grok 4.7 completed update

The verified five-physical-pass campaign is documented in [the Grok audit](grok_4_7_challenge100.md). All 144 calls, including two connection-error originals and the authorized resubmission, are included. Standard-parser and strict original-syntax metrics are both disclosed.

## Claude Opus 5.5 completed update

The verified max/xhigh/high/medium campaign is documented in [the Opus 5.5 audit](claude_opus_5_5_challenge100.md). All 244 requests across four batches are included. Stored scores were copied without rescoring; strict original syntax and standard-parser results are both disclosed.

## GPT-6 Sol completed update

The verified max/xhigh campaign is documented in [the Sol audit](gpt6_sol_challenge100.md). High and medium were skipped on empty residual. All 150 terminal request records across two batches are included, with unknown dispatch, acceptance and token usage explicitly retained. Stored scores were copied without rescoring.

## GPT-6 Luna final update

The verified four-pass MAX/XHIGH/HIGH/MEDIUM cascade replaces the provisional snapshot. See [the Luna audit](gpt6_luna_challenge100.md). All 201 terminal records across four batches are accounted for: 171 confirmed attempts plus 30 unknown, 169 confirmed acceptances plus 32 unknown, zero retries. Final scores are 18/100 (100 evaluable), 18/64 (64 evaluable) and 0/36 (36 evaluable). All physical token costs are separate from selected-response usage; missing usage remains unknown. Stored scores were copied without rescoring, and other model records are unchanged.
