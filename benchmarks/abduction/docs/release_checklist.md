# ABD Release Checklist

1. Refresh or verify the canonical benchmark artifacts:
   - `data/abd_instances_v1.yaml.gz`
   - `data/abd_holdouts_v1.jsonl.gz`
   - `predictions/abd_predictions_v1.jsonl.gz`
   - `eval/abd_eval_cache_v1.jsonl`
   - `eval/abd_eval_cache_v1.meta.json`
   - `release_manifest.json`
2. Confirm the frozen metadata contains no internal absolute paths.
3. Run:
   - `pip install -e ".[dev]"`
   - `pytest -q`
   - `./benchmarks/abduction/scripts/rebuild_eval_cache.sh --limit 1 --overwrite`
   - `./benchmarks/abduction/scripts/reproduce_tables.sh`
4. Check that `CITATION.cff`, `LICENSE`, and the benchmark README match the
   intended public release.
5. Commit the release state on `main`.
6. Publish a new immutable tag such as `abduction-v1.2`.
7. Create the matching GitHub release and attach the release notes.
