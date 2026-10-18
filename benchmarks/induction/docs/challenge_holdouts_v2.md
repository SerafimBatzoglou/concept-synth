# Challenge100 / Challenge64 holdout-complete v2

This is an additive **evaluation-data version**, `challenge-holdouts-v2`.
The 100 problems, original order, gold formulas, training worlds, and solver
prompts are unchanged. Challenge64 is the exact first-64 projection, not a
separately sampled benchmark.

| Component | Preserved v1 worlds | Added worlds | v2 worlds | Tasks with exactly five |
|---|---:|---:|---:|---:|
| Challenge64 | 313 | 7 | 320 | 64/64 |
| New36 | 136 | 44 | 180 | 36/36 |
| Challenge100 | 449 | 51 | 500 | 100/100 |

## Download and use

All paths below are relative to `benchmarks/induction/data/challenge_holdouts_v2/`:

- [Challenge100 solver inputs](../data/challenge_holdouts_v2/induction_fullobs_challenge100_v2.yaml.gz)
- [Challenge100 holdout worlds](../data/challenge_holdouts_v2/challenge100_holdouts_v2.jsonl)
- [Challenge64 solver inputs](../data/challenge_holdouts_v2/induction_fullobs_challenge64_v2.yaml.gz)
- [Challenge64 holdout worlds](../data/challenge_holdouts_v2/challenge64_holdouts_v2.jsonl)
- [Checksums and version manifest](../data/challenge_holdouts_v2/manifest.json)
- [Generation trace](../data/challenge_holdouts_v2/generation_trace.json)
- [Frozen policy](../data/challenge_holdouts_v2/policy.json)

**The benchmark version is the input dataset plus its holdout sidecar and
manifest.** The v2 solver-input gzip files are byte-identical to v1. Holdout
worlds are deliberately separate; never append them to model prompts. Join
sidecar `task_id` to dataset `instanceId` for offline evaluation only. Predicate
extensions use the existing public world schema and closed-world semantics.

## Why v1 was incomplete

The legacy generator made at most 50 proposals for each requested world and
accepted only target-positive fractions between 0.15 and 0.85. An exhausted
slot was silently omitted. Eight tasks had zero worlds; six had two to four.

The regular sampler gives every node two outgoing R edges and two outgoing S
edges. For `hard_070`, whose gold requires every R-successor to have an
R-successor, this makes every object positive. No number of regular proposals
can produce a balanced world for this task. Several other gold formulas have
very low acceptance rates under the regular sampler.

## Model-blind completion policy

The policy and seeds were fixed before any v2 model-response evaluation.
All 449 legacy world objects are preserved exactly, including labels and IDs.
The original legacy inputs are included as
[`challenge100_legacy_holdouts_v1.jsonl`](../data/challenge100_legacy_holdouts_v1.jsonl).

For each incomplete task:

1. Sample up to 1,000 regular-degree proposals, using a deterministic per-task
   seed and the original training domain sizes. Skip this impossible stage for
   `hard_070`, as declared in the policy.
2. If still short, sample up to 10,000 variable-degree proposals. Independently
   choose the R and S edge probabilities from `{0.1, 0.2, 0.35, 0.5, 0.65}`;
   then independently sample directed edges, allowing self-edges and nodes
   with no outgoing edges. Keep the original unary sampling rule.
3. Accept proposals in generation order only if their gold target is balanced
   and their structure is not a duplicate of a training or retained holdout
   world. Stop at exactly five. Fail the build if any task remains short.

Twenty-five additions came from the regular sampler; twenty-six used the
fallback. No model responses, model correctness, or counterexample-driven
selection were used. There were **zero model API calls**. Structural duplicate
checks compare named finite structures, not graph-isomorphism classes.

This is a versioned completion policy, **not an unchanged IID sample from the
legacy sampler**. Both the target-balance rejection rule and the variable-degree
fallback are part of the declared distribution. Existing 86 complete tasks
were not resampled. Preserving v1 worlds means this is not a fresh independent
test set for models or research already adapted to v1 holdout results.

## Reproduce and validate

From the repository root, with the public package dependencies installed:

```bash
PYTHONPATH=src python -m concept_synth.induction.complete_holdouts \
  --dataset benchmarks/induction/data/induction_fullobs_challenge100_v1.yaml.gz \
  --projection benchmarks/induction/data/induction_fullobs_challenge64_v1.yaml.gz \
  --legacy benchmarks/induction/data/challenge100_legacy_holdouts_v1.jsonl \
  --policy benchmarks/induction/docs/holdout_v2_policy.json \
  --output benchmarks/induction/generated_instances/challenge_holdouts_v2_reproduction

PYTHONPATH=src pytest -q tests/test_complete_holdouts.py
```

The output directory must not already exist. Python 3.12.2 produced the frozen
artifacts; a second clean build reproduced all eight bundle files byte-for-byte.
The validation checks five worlds per task, exact preservation of legacy
worlds, disjointness from training, unique named structures/world IDs, target
balance, and exact gold labels on all 500 worlds. An independent benchmark
scorer also verified all 100 gold formulas on the completed sets. Tests check
the exact Challenge64 projection and prompt equality for all 100 problems.

## Evaluation migration

The [current leaderboards](leaderboard.md) use **holdouts v2**. Only the 14
changed tasks were re-evaluated offline for the 48 completed model configurations
(29 with Challenge100 coverage): 133 train-correct responses scored, 330 entries
skipped under the existing eligibility rule. This adds 59 newly measurable
holdout successes; no previous pass was lost. The other 86 tasks were not rescored.

Original v1 predictions, scores, registries and active campaign inputs are
unchanged. [Historical v1 tables](leaderboard_holdout_v1.md) remain available.
The unfinished Flash-1017 campaign is excluded from this release.

Versioned evidence:

- [Affected-task evaluation rows](../eval/challenge_holdouts_v2_affected_results.jsonl)
- [Per-model v1-to-v2 totals](../eval/challenge_holdouts_v2_model_summary.json)
- [Stored-score verification](holdout_v2_evaluation_verification.json)
- [Evaluation release manifest](../holdout_v2_evaluation_release_manifest.json)

Regenerate the current tables without rescoring using
`python benchmarks/induction/analysis/make_holdout_v2_leaderboards.py`.

Saved model responses can be re-evaluated offline: keep the existing selected
response/formula and parsing policy fixed, then score against the frozen v2
holdouts. Do not select a different response because it scores better. Preserve
v1 results and record v1-to-v2 changes separately. No new inference is needed.
Existing train-correct gating and full-100/full-64 denominators should remain
explicit; evaluating additional train-incorrect formulas diagnostically is
separate from changing leaderboard credit. Five-world holdout availability is
now a dataset fact and must not be conflated with whether a model was evaluated.

Holdout-world completion does not authorize new model calls, changes to ongoing
run settings, or republication of unfinished campaign results.
