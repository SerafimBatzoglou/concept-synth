"""Verify the one-attempt Hydrogen release using only public artifacts."""
import gzip
import hashlib
import json
from pathlib import Path

B = Path(__file__).resolve().parents[1] / 'benchmarks/induction'


def test_hydrogen_report_and_predictions():
    r = json.loads((B/'eval/gemini_hydrogen_challenge100_holdout_v2.json').read_bytes())
    tasks = r['tasks']
    assert len(tasks) == len({t['task_id'] for t in tasks}) == 100
    with gzip.open(B/'predictions/gemini_hydrogen_challenge100_predictions_v2.jsonl.gz','rt') as f:
        preds = [json.loads(s) for s in f]
    assert len(preds) == 100
    assert {p['instanceId']:p['extractedFormula'] for p in preds} == {t['task_id']:t['formula'] for t in tasks}
    assert all(p['rawResponse'] is p['response'] is p['thinking'] is None for p in preds)
    for key in ('/Users/', 'response_id', 'prompt_id', 'candidate_id', 'api_call_id', 'reasoning_content', 'API_KEY'):
        assert key not in json.dumps([r, preds])
    for subset, expected in [('challenge64', (64,61,53,32)), ('new36',(36,34,20,20))]:
        group = [t for t in tasks if t['subset']==subset]
        assert (len(group), sum(t['strict_evaluable'] for t in group), sum(t['train_correct'] for t in group),
                sum(t['holdout_v2']['holdout_correct'] is True for t in group)) == expected
    for t in tasks:
        h, = t['physical_history']
        assert h['physical_attempt'] == 1 and h['effort'] == 'xhigh' and h['remote_terminal']
        assert t['strict_evaluable'] == t['evaluable'] and not t['normalizations']
        if t['evaluable']:
            assert h['outcome'] == 'completed' and t['formula']
            assert bool(t['stored_train_score']['train_valid']) == t['train_correct']
        else:
            assert h['outcome'] == 'incomplete' and h['usage']['output_tokens'] == h['max_tokens']
            assert t['formula'] is t['stored_train_score'] is None and not t['train_correct']
        assert bool(t['stored_holdout_v2_score']) == t['train_correct']
        if t['train_correct']:
            assert t['holdout_v2']['holdout_world_count'] == 5
            assert bool(t['stored_holdout_v2_score']['train_valid']) == t['holdout_v2']['holdout_correct']
    assert r['usage']['physical_requests'] == 100 and r['pending_at_cutoff'] == 0
    assert r['usage']['known_totals']['output_tokens'] == sum(t['physical_history'][0]['usage']['output_tokens'] for t in tasks) == 51025623


def test_hydrogen_caches_and_leaderboards():
    r = json.loads((B/'eval/gemini_hydrogen_challenge100_holdout_v2.json').read_bytes())
    for path in (B/'eval').glob('induction_challenge64_round1*eval_cache_v2.jsonl'):
        lines = path.read_bytes().splitlines(keepends=True)
        prior = r['prior_release_preservation']['file_sha256_before_append']
        assert hashlib.sha256(b''.join(lines[:3520])).hexdigest() == prior[str(path.relative_to(B.parents[1]))]
        added = [json.loads(line) for line in lines[3520:3584]]
        assert len(added) == 64 and {s['model_id'] for s in added} == {'gemini-4-hydrogen'}
        assert sum(s['valid'] for s in added) == (32 if 'holdout' in path.name else 53)
    table = (B/'docs/leaderboard.md').read_text()
    assert '| Gemini Hydrogen | 52.0% (52/100) | 73/100' in table
    assert '| Gemini Hydrogen | 50.0% (32/64) | 53/64' in table
