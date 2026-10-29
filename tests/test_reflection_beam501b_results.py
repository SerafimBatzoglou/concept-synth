"""Frozen Beam evidence, privacy and additive release; no scoring or model calls."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import jsonschema
import yaml

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'benchmarks/induction'
MID='reflection/beam-1.0-501b-a23b'
def load(p):return json.loads(p.read_text())
def report():return load(B/'eval/reflection_beam501b_challenge100_holdout_v2.json')
def rows(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return [json.loads(x) for x in f if x.strip()]

def test_selected_scores_and_exclusion():
    r=report();assert r['final_verified'] and r['publication_with_unresolved_exclusion']
    assert len(r['tasks'])==len({t['task_id'] for t in r['tasks']})==100
    for k,n in [('evaluable',99),('strict_evaluable',92),('train_correct',11),('strict_correct',11),('holdout_correct',8),('holdout_available',11),('normalizations',7)]:assert r['totals'][k]==n
    assert r['totals']['correct_ast_mean']==37 and r['totals']['correct_ast_median']==16
    assert [t['task_id'] for t in r['tasks'] if not t['evaluable']]==['hard_038']
    for subset,expected in [('challenge64',(64,63,58,11,8)),('new36',(36,36,34,0,0))]:
        ts=[t for t in r['tasks'] if t['subset']==subset]
        assert (len(ts),sum(t['evaluable'] for t in ts),sum(t['strict_evaluable'] for t in ts),sum(t['train_correct'] for t in ts),sum(t['holdout_v2']['holdout_correct'] is True for t in ts))==expected
    for t in r['tasks']:
        assert t['evaluable']==(t['stored_train_score'] is not None)==(t['formula'] is not None)
        assert t['train_correct']==t['holdout_v2']['holdout_available']==(t['stored_holdout_v2_score'] is not None)
        if t['train_correct']:assert t['holdout_v2']['holdout_world_count']==5
        else:assert t['holdout_v2']['holdout_correct'] is None
    assert len(r['new36_slices'])==10
    for name,s in r['new36_slices'].items():
        ts=[t for t in r['tasks'] if t['task_id'].rsplit('_',1)[0]==name]
        assert len(ts)==s['count']==s['evaluable']
        assert sum(t['strict_evaluable'] for t in ts)==s['strict_evaluable']
        assert s['train_correct']==s['holdout_correct']==0

def test_bounded_attempts_unknowns_and_usage():
    r=report();hs=[h for t in r['tasks'] for h in t['physical_history']]
    assert len(hs)==104 and Counter(h['physical_attempt'] for h in hs)=={1:100,2:4}
    assert sum(h['remote_terminal'] for h in hs)==101
    assert sum(h['final_usage_unknown'] for h in hs)==3
    assert all(h['effort']=='max' and h['max_tokens']==131072 for h in hs)
    for t in r['tasks']:
        hist=t['physical_history'];assert hist==sorted(hist,key=lambda h:h['started_at'])
        assert [h['physical_attempt'] for h in hist]==list(range(1,len(hist)+1))
        assert all(not h['evaluable'] for h in hist[:-1])
        chosen=next((h for h in hist if h['evaluable']),hist[0])
        assert t['physical_attempt']==chosen['physical_attempt'] and t['usage']==chosen['usage']
        for h in hist:
            assert h['physical_attempt']==h['logical_trial'] and h['recovery_ordinal']==0
            if not h['remote_terminal']:assert all(v is None for v in h['usage'].values())
    excluded=next(t for t in r['tasks'] if t['task_id']=='hard_038')
    assert len(excluded['physical_history'])==1 and excluded['physical_history'][0]['http_status']==529
    for k,total in r['usage']['known_totals'].items():
        vals=[h['usage'][k] for h in hs if h['usage'][k] is not None]
        expected=sum(vals) if vals else None
        if 'cost' in k and expected is not None:
            assert math.isclose(total,expected,rel_tol=0,abs_tol=1e-12)
        else:
            assert total==expected
        assert r['usage']['unknown_counts'][k]==sum(h['usage'][k] is None for h in hs)
    u=r['usage']['known_totals']
    assert (u['input_tokens'],u['output_tokens'],u['total_tokens'])==(602368,8568426,9170794)
    assert u['reasoning_tokens_within_output']+u['visible_tokens']==u['output_tokens']
    assert r['usage']['unknown_counts']['output_tokens']==3
    token=next(t for t in load(B/'eval/challenge100_output_token_usage_v2.json')['models'] if t['model_id']==MID)
    assert (token['output_tokens'],token['physical_attempts'],token['unreported_responses_lower_bound'])==(8568426,104,3)

def test_additive_cache_and_stored_scores():
    r=report();tasks={t['task_id']:t for t in r['tasks']}
    validator=jsonschema.Draft202012Validator(load(B/'schemas/eval_cache_record.schema.json'))
    for name,hold in [('induction_challenge64_round1_eval_cache_v2.jsonl',False),('induction_challenge64_round1_holdout_eval_cache_v2.jsonl',True)]:
        p=B/'eval'/name;lines=p.read_bytes().splitlines(keepends=True)
        assert len(lines)==3456
        assert hashlib.sha256(b''.join(lines[:3392])).hexdigest()==r['prior_release_preservation']['file_sha256_before_append'][str(p.relative_to(ROOT))]
        for line in lines[3392:]:
            row=json.loads(line);validator.validate(row);assert row['model_id']==MID
            t=tasks[row['instance_id']];s=t['stored_holdout_v2_score' if hold else 'stored_train_score']
            assert row['prediction']['formula']==t['formula']
            assert row['valid']==(t['holdout_v2']['holdout_correct'] is True if hold else t['train_correct'])
            assert row['completed']==(t['train_correct'] if hold else t['evaluable'])
            if s:
                for k in ('status','mismatch_count','false_positives','false_negatives'):assert row['evaluation'][k]==s[k]

def test_privacy_manifest_and_leaderboards():
    r=report();ps=rows(B/'predictions/reflection_beam501b_challenge100_predictions_v2.jsonl.gz')
    validator=jsonschema.Draft202012Validator(load(B/'schemas/prediction_record.schema.json'))
    assert len(ps)==100 and sum(p['success'] for p in ps)==99
    for p in ps:
        validator.validate(p);assert p['rawResponse'] is p['response'] is p['thinking'] is None
        assert not {'train_correct','holdout_correct','strict_correct'} & p['metadata'].keys()
    for bad in ('/Users/','/tmp/','pipeline.sqlite','candidate_id','response_id','prompt_id','api_call_id','provider_usage','reasoning_content','raw_text','API_KEY','Bearer '):assert bad not in json.dumps([r,ps])
    for n,count in [(100,35),(64,54)]:
        reg=yaml.safe_load((B/f'docs/challenge{n}_round1_model_registry_holdout_v2.yaml').read_text())
        assert len(reg['models'])==count and reg['models'][-1]['id']==MID
    table=(B/'docs/leaderboard.md').read_text()
    assert '| Reflection Beam 501B | 8.0% (8/100) | 11/100 (11.0%) | 99/100 | 37.0 / 16.0 | 8.6 |' in table
    assert '| Reflection Beam 501B | 12.5% (8/64) | 11/64 (17.2%) | 63/64 | 37.0 / 16.0 |' in table
    assert '[Protocol and accounting](reflection_beam501b_results.md)' in table
    assert r['evaluation']['publication_scoring_calls']==r['evaluation']['publication_model_calls']==0
