# INDUCTION Release Checklist

1. Refresh or verify the canonical benchmark artifacts:
   - `data/induction_fullobs_v1.yaml.gz`
   - `data/induction_ci_v1.yaml.gz`
   - `data/induction_ec_v1.yaml.gz`
   - `predictions/induction_predictions_v1.jsonl.gz`
   - `eval/induction_eval_cache_v1.jsonl`
   - `eval/induction_eval_cache_v1.meta.json`
   - `release_manifest.json`
2. Confirm benchmark files do not contain `llmResults`, `rawResponse`,
   `evaluation`, or EC `fullPredicates` fields.
3. Confirm prediction records keep useful model-output fields, including
   timestamps and raw responses when present, and do not contain evaluation
   scores.
4. Confirm the frozen eval metadata contains no internal absolute paths.
5. Validate the release in a fresh virtual environment, or run the local checks
   without changing the active editable install:
   - `PYTHONPATH=src pytest -q`
   - `PYTHONPATH=src python -m concept_synth.induction.cli generate --task all --n 1 --seed 20260516 --validate --output /tmp/induction_generated_sample.yaml`
   - `./benchmarks/induction/scripts/rebuild_eval_cache.sh --limit 1 --no-diagnostics`
   - `./benchmarks/induction/scripts/reproduce_tables.sh`
6. Check that `CITATION.cff`, `LICENSE`, and the benchmark README match the
   intended public release.
7. Commit the release state on `main`.
8. Publish a new immutable tag such as `induction-v1.0`.
9. Create the matching GitHub release and attach release notes.
