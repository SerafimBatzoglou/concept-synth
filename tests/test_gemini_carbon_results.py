"""Check the published score snapshot without calling the provider or scorer."""
import gzip
import hashlib
import json
from pathlib import Path

B = Path(__file__).resolve().parents[1] / 'benchmarks/induction'


def test_carbon_selection_scores_and_accounting():
    report = json.loads((B / 'eval/gemini_carbon_challenge100_holdout_v2.json').read_bytes())
    tasks = report['tasks']
    assert len(tasks) == len({t['task_id'] for t in tasks}) == 100
    assert all(t['evaluable'] and t['strict_evaluable'] and not t['normalizations'] for t in tasks)
    for subset, expected in [('challenge64', (64, 57, 47)), ('new36', (36, 25, 23))]:
        group = [t for t in tasks if t['subset'] == subset]
        assert (len(group), sum(t['train_correct'] for t in group),
                sum(t['holdout_v2']['holdout_correct'] is True for t in group)) == expected
    for t in tasks:
        selected = next(h for h in t['physical_history'] if h['evaluable'])
        assert t['physical_attempt'] == selected['physical_attempt']
        assert t['train_correct'] == bool(t['stored_train_score']['train_valid'])
        assert t['train_correct'] == (t['stored_holdout_v2_score'] is not None)
        if t['train_correct']:
            assert t['holdout_v2']['holdout_world_count'] == 5
            assert t['holdout_v2']['holdout_correct'] == bool(t['stored_holdout_v2_score']['train_valid'])
    hs = [h for t in tasks for h in t['physical_history']]
    assert len(hs) == report['usage']['physical_requests'] == 120
    assert sum(h['outcome'] == 'pending_at_cutoff' for h in hs) == report['pending_at_cutoff'] == 9
    outputs = [h['usage']['output_tokens'] for h in hs if h['usage']['output_tokens'] is not None]
    assert len(outputs) == 108 and sum(outputs) == 27158517


def test_carbon_public_artifacts_and_additive_caches():
    r = json.loads((B / 'eval/gemini_carbon_challenge100_holdout_v2.json').read_bytes())
    with gzip.open(B / 'predictions/gemini_carbon_challenge100_predictions_v2.jsonl.gz', 'rt') as f:
        preds = [json.loads(s) for s in f]
    assert {v['instanceId']: v['extractedFormula'] for v in preds} == {t['task_id']: t['formula'] for t in r['tasks']}
    assert all(v['rawResponse'] is v['response'] is v['thinking'] is None for v in preds)
    for key in ('/Users/', 'response_id', 'prompt_id', 'candidate_id', 'api_call_id', 'reasoning_content', 'API_KEY'):
        assert key not in json.dumps([r, preds])
    for path in (B/'eval').glob('induction_challenge64_round1*eval_cache_v2.jsonl'):
        lines = path.read_bytes().splitlines(keepends=True)
        before = r['prior_release_preservation']['file_sha256_before_append']
        assert hashlib.sha256(b''.join(lines[:3456])).hexdigest() == before[str(path.relative_to(B.parents[1]))]
        added = [json.loads(line) for line in lines if json.loads(line)['model_id'] == 'gemini-4-carbon']
        assert len(added) == 64 and sum(t['valid'] for t in added) == (47 if 'holdout' in path.name else 57)
    table = (B/'docs/leaderboard.md').read_text()
    assert '| Gemini Carbon | 70.0% (70/100) | 82/100' in table
    assert '| Gemini Carbon | 73.4% (47/64) | 57/64' in table
