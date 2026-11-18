"""Verify the redacted release, retry accounting, and additive cache projection."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import pytest

B = Path(__file__).resolve().parents[1] / 'benchmarks/induction'


@pytest.mark.parametrize('model,strict,train,holdout,attempts,output,c64,new36', [
    ('astra', 100, 100, 97, 100, 1521265, (64, 62), (36, 35)),
    ('sol', 99, 98, 73, 105, 5446141, (62, 39), (36, 34)),
])
def test_published_scores_and_usage(model, strict, train, holdout, attempts, output, c64, new36):
    r = json.loads((B/f'eval/gpt62_{model}_challenge100_holdout_v2.json').read_bytes())
    tasks = r['tasks']
    by_id = {t['task_id']:t for t in tasks}
    assert len(tasks) == len(by_id) == 100 and all(t['evaluable'] for t in tasks)
    assert (sum(t['strict_evaluable'] for t in tasks), sum(t['train_correct'] for t in tasks),
            sum(t['holdout_v2']['holdout_correct'] is True for t in tasks)) == (strict,train,holdout)
    for key, expected in [('challenge64',c64), ('new36',new36)]:
        s = r['by_subset'][key]
        assert (s['train_correct'],s['holdout_correct']) == expected
    all_attempts = []
    for t in tasks:
        hist = t['physical_history']
        all_attempts.extend(hist)
        assert 1 <= len(hist) <= 2 and t['physical_attempt'] == len(hist)
        assert hist[-1]['evaluable'] and all(h['remote_terminal'] and h['effort']=='max' for h in hist)
        if len(hist) == 2:
            assert hist[0]['outcome']=='incomplete' and not hist[0]['evaluable']
            assert hist[0]['usage']['output_tokens']==hist[0]['max_tokens']==128000
        assert bool(t['stored_train_score']['train_valid']) == t['train_correct']
        assert bool(t['stored_holdout_v2_score']) == t['train_correct']
        if t['train_correct']:
            assert t['holdout_v2']['holdout_world_count'] == 5
            assert bool(t['stored_holdout_v2_score']['train_valid']) == t['holdout_v2']['holdout_correct']
    counts = Counter(h['physical_attempt'] for h in all_attempts)
    assert counts[1] == 100 and counts[2] == attempts - 100 and set(counts) <= {1, 2}
    assert r['usage']['accepted_generations'] == attempts and r['pending_at_cutoff'] == 0
    assert r['usage']['known_totals']['output_tokens'] == sum(h['usage']['output_tokens'] for h in all_attempts) == output
    with gzip.open(B/f'predictions/gpt62_{model}_challenge100_predictions_v2.jsonl.gz','rt') as f:
        predictions = [json.loads(line) for line in f]
    assert len(predictions)==100
    assert {p['instanceId']:p['extractedFormula'] for p in predictions} == {t['task_id']:t['formula'] for t in tasks}
    assert all(p['rawResponse'] is p['response'] is p['thinking'] is None for p in predictions)
    for private in ('/Users/', 'response_id', 'prompt_id', 'candidate_id', 'api_call_id', 'reasoning_content', 'API_KEY'):
        assert private not in json.dumps([r,predictions])
    for path in (B/'eval').glob('induction_challenge64_round1*eval_cache_v2.jsonl'):
        entries = path.read_bytes().splitlines(keepends=True)
        pinned = r['prior_release_preservation']['file_sha256_before_append'][str(path.relative_to(B.parents[1]))]
        assert hashlib.sha256(b''.join(entries[:3584])).hexdigest() == pinned
        rows = [json.loads(e) for e in entries[3584:]]
        ours = [x for x in rows if x['model_id']==f'gpt-6.2-{model}']
        assert len(ours)==64 and sum(x['valid'] for x in ours)==c64['holdout' in path.name]
        assert all(x['prediction']['formula']==by_id[x['instance_id']]['formula'] for x in ours)
    leaderboard = (B/'docs/leaderboard.md').read_text()
    assert f'| GPT-6.2 {model.title()} | {holdout:.1f}% ({holdout}/100) | {train}/100' in leaderboard
    assert f'| GPT-6.2 {model.title()} | {c64[1]/64:.1%} ({c64[1]}/64) | {c64[0]}/64' in leaderboard
