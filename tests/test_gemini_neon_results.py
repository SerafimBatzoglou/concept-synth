"""Verify the final release, redaction, selection order, and additive score projection."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

B = Path(__file__).resolve().parents[1] / 'benchmarks/induction'


def report():
    return json.loads((B/'eval/gemini_neon_challenge100_holdout_v2.json').read_bytes())


def test_neon_attempt_history_and_stored_scores():
    r = report()
    ts = r['tasks']
    assert len(ts)==len({t['task_id'] for t in ts})==100
    assert all(t['evaluable'] and t['strict_evaluable'] and not t['normalizations'] for t in ts)
    for subset,evaluable,train,holdout in [('challenge64',64,51,48),('new36',36,26,25)]:
        rows = [t for t in ts if t['subset']==subset]
        assert (len(rows),sum(t['train_correct'] for t in rows),
                sum(t['holdout_v2']['holdout_correct'] is True for t in rows)) == (evaluable,train,holdout)
    for t in ts:
        h = t['physical_history']
        assert [x['physical_attempt'] for x in h]==list(range(1, len(h)+1))
        assert next(x['physical_attempt'] for x in h if x['evaluable'])==t['physical_attempt']==len(h)
        assert all(x['service_tier']=='flex' and x['remote_terminal'] and x['max_tokens']==128000 for x in h)
        assert bool(t['stored_train_score']['train_valid'])==t['train_correct']
        assert bool(t['stored_holdout_v2_score'])==t['train_correct']
        if t['train_correct']:
            assert t['holdout_v2']['holdout_world_count']==5
            assert bool(t['stored_holdout_v2_score']['train_valid'])==t['holdout_v2']['holdout_correct']
    low = [t for t in ts if t['selected_effort']=='low']
    assert len(low)==23 and not any(t['train_correct'] for t in low)
    hs = [x for t in ts for x in t['physical_history']]
    u = r['usage']
    assert Counter(x['effort'] for x in hs)=={'xhigh':102,'high':50,'medium':87,'low':23}
    assert len(hs)==u['accepted_generations']==u['http_deliveries']==262
    assert u['reported_tiers']=={'flex':262} and not u['unknown_usage_counts']
    assert u['known_totals']['output_tokens']==sum(x['usage']['output_tokens'] for x in hs)==27340377


def test_neon_public_projection_and_privacy():
    r = report()
    by_id = {t['task_id']:t for t in r['tasks']}
    with gzip.open(B/'predictions/gemini_neon_challenge100_predictions_v2.jsonl.gz','rt') as f:
        ps = [json.loads(x) for x in f]
    assert len(ps)==100 and all(p['rawResponse'] is p['response'] is p['thinking'] is None for p in ps)
    assert {p['instanceId']:p['extractedFormula'] for p in ps}=={k:v['formula'] for k,v in by_id.items()}
    for private in ('/Users/', 'response_id', 'prompt_id', 'candidate_id', 'api_call_id', 'reasoning_content', 'API_KEY'):
        assert private not in json.dumps([r,ps])
    prefix = r['prior_release_preservation']['file_sha256_before_append']
    for p in (B/'eval').glob('induction_challenge64_round1*eval_cache_v2.jsonl'):
        lines = p.read_bytes().splitlines(keepends=True)
        assert hashlib.sha256(b''.join(lines[:3712])).hexdigest()==prefix[str(p.relative_to(B.parents[1]))]
        ours = [json.loads(x) for x in lines[3712:3776]]
        assert len(ours)==64 and {x['model_id'] for x in ours}=={'gemini-4-neon'}
        assert sum(x['valid'] for x in ours)==(48 if 'holdout' in p.name else 51)
        assert all(x['prediction']['formula']==by_id[x['instance_id']]['formula'] for x in ours)
    table = (B/'docs/leaderboard.md').read_text()
    assert '| Gemini 4 Neon | 73.0% (73/100) | 77/100' in table
    assert '| Gemini 4 Neon | 75.0% (48/64) | 51/64' in table
