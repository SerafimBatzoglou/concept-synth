"""Validate stored release evidence; never call models or rescore formulas."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import jsonschema
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
B = ROOT/'benchmarks/induction'
MID = 'claude-haiku-5-5'

def load(p): return json.loads(p.read_text())
def rows(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:
        return [json.loads(s) for s in f if s.strip()]
def report(): return load(B/'eval/haiku55_challenge100_holdout_v2.json')

def test_final_scores_strictness_and_holdout_gate():
    r=report(); assert r['final_verified'] and len(r['tasks'])==100
    assert len({t['task_id'] for t in r['tasks']})==100
    for key,n in [('count',100),('evaluable',99),('strict_evaluable',99),('train_correct',16),
                  ('strict_correct',16),('holdout_available',16),('holdout_correct',15),
                  ('normalizations',0),('train_correct_missing_holdout',0)]:
        assert r['totals'][key]==n
    for subset,expected in [('challenge64',(64,63,63,15,14)),('new36',(36,36,36,1,1))]:
        ts=[t for t in r['tasks'] if t['subset']==subset]
        assert (len(ts),sum(t['evaluable'] for t in ts),sum(t['strict_evaluable'] for t in ts),
                sum(t['train_correct'] for t in ts),sum(t['holdout_v2']['holdout_correct'] is True for t in ts))==expected
    ast=[]
    for t in r['tasks']:
        assert t['evaluable']==(t['stored_train_score'] is not None)==(t['formula'] is not None)
        assert t['train_correct']==t['holdout_v2']['holdout_available']==(t['stored_holdout_v2_score'] is not None)
        if t['train_correct']:
            assert t['strict_correct'] and t['holdout_v2']['holdout_world_count']==5
            ast.append(t['stored_train_score']['ast_size'])
        else: assert t['holdout_v2']['holdout_correct'] is None
    assert mean(ast)==r['totals']['correct_ast_mean']==pytest.approx(16.375)
    assert median(ast)==r['totals']['correct_ast_median']==16
    normalized=[t for t in r['tasks'] if t['normalizations']]
    assert not normalized
    assert len(r['new36_slices'])==10
    for key,s in r['new36_slices'].items():
        ts=[t for t in r['tasks'] if t['task_id'].rsplit('_',1)[0]==key]
        assert len(ts)==s['count']==s['evaluable']==s['strict_evaluable']
        assert sum(t['train_correct'] for t in ts)==s['train_correct']
        assert sum(t['holdout_v2']['holdout_correct'] is True for t in ts)==s['holdout_correct']

def test_bounded_effort_sequence_selection_and_unknown_usage():
    r=report();histories=[h for t in r['tasks'] for h in t['physical_history']]
    assert len(histories)==234
    assert Counter(h['physical_attempt'] for h in histories)=={1:100,2:89,3:43,4:2}
    assert Counter(h['effort'] for h in histories)=={'max':10,'xhigh':100,'high':88,'medium':36}
    assert sum(h['remote_terminal'] for h in histories)==228
    assert sum(h['final_usage_unknown'] for h in histories)==6
    for t in r['tasks']:
        hs=t['physical_history'];assert hs==sorted(hs,key=lambda h:h['started_at'])
        assert [h['physical_attempt'] for h in hs]==list(range(1,len(hs)+1))
        efforts=['max','xhigh','high','medium'] if hs[0]['pilot'] else ['xhigh','high','medium']
        assert [h['effort'] for h in hs]==efforts[:len(hs)]
        assert all(not h['evaluable'] for h in hs[:-1])
        assert all(h['max_tokens']==128000 and h['recovery_ordinal']==0 and h['logical_trial']==h['physical_attempt'] for h in hs)
        chosen=next((h for h in hs if h['evaluable']),hs[0])
        assert t['physical_attempt']==chosen['physical_attempt']
        assert t['selected_effort']==chosen['effort'] and t['usage']==chosen['usage']
        for h in hs:
            if not h['remote_terminal']:
                assert h['stream_error_type']=='api_error' and h['http_status']==200
                assert all(v is None for v in h['usage'].values())
    assert [t['task_id'] for t in r['tasks'] if not t['evaluable']]==['extreme_logic_024']
    for k,total in r['usage']['known_totals'].items():
        vs=[h['usage'][k] for h in histories if h['usage'][k] is not None]
        assert total==(sum(vs) if vs else None)
        assert r['usage']['unknown_counts'][k]==sum(h['usage'][k] is None for h in histories)
    assert r['usage']['known_totals']['input_tokens']==1522100
    assert r['usage']['known_totals']['output_tokens']==23323724
    assert r['usage']['known_totals']['total_tokens']==24845824
    assert r['usage']['unknown_counts']['output_tokens']==6
    assert r['usage']['known_totals']['reasoning_tokens_within_output'] is None
    assert r['usage']['known_totals']['estimated_cost_usd_min'] is None
    assert r['reported_stream_usage']['known_totals']['output_tokens']==23323770
    token=next(t for t in load(B/'eval/challenge100_output_token_usage_v2.json')['models'] if t['model_id']==MID)
    assert token['output_tokens']==23323724 and token['physical_attempts']==234
    assert token['unreported_responses_lower_bound']==6 and token['reasoning_included_once']

def test_cache_prefix_preserved_and_scores_copied():
    r=report(); before=r['prior_release_preservation']['file_sha256_before_append']
    tasks={t['task_id']:t for t in r['tasks']}
    validate=jsonschema.Draft202012Validator(load(B/'schemas/eval_cache_record.schema.json')).validate
    for name,hold in [('induction_challenge64_round1_eval_cache_v2.jsonl',False),
                      ('induction_challenge64_round1_holdout_eval_cache_v2.jsonl',True)]:
        p=B/'eval'/name; lines=p.read_bytes().splitlines(keepends=True)
        assert len(lines)==3456 and hashlib.sha256(b''.join(lines[:3264])).hexdigest()==before[str(p.relative_to(ROOT))]
        added=[json.loads(x) for x in lines[3264:3328]]; assert len({t['instance_id'] for t in added})==64
        for row in added:
            validate(row);assert row['model_id']==MID
            t=tasks[row['instance_id']];s=t['stored_holdout_v2_score' if hold else 'stored_train_score']
            assert row['prediction']['formula']==t['formula']
            assert row['valid']==(t['holdout_v2']['holdout_correct'] is True if hold else t['train_correct'])
            assert row['completed']==(t['train_correct'] if hold else t['evaluable'])
            assert row['parse_ok']==bool(s and s['parse_ok'])
            if s:
                for k in ('status','mismatch_count','false_positives','false_negatives'):assert row['evaluation'][k]==s[k]
                if not hold:
                    for k in ('ast_size','quantifier_depth','equality_count'):assert row['prediction'][k]==s[k]

def test_privacy_predictions_and_leaderboards():
    r=report();ps=rows(B/'predictions/haiku55_challenge100_predictions_v2.jsonl.gz')
    assert len(ps)==100 and sum(p['success'] for p in ps)==99
    ts={t['task_id']:t for t in r['tasks']}
    validate=jsonschema.Draft202012Validator(load(B/'schemas/prediction_record.schema.json')).validate
    for p in ps:
        validate(p);assert p['rawResponse'] is p['response'] is p['thinking'] is None
        assert p['extractedFormula']==ts[p['instanceId']]['formula']
        assert not {'strict_correct','train_correct','holdout_correct'}.intersection(p['metadata'])
    for bad in ('/Users/','/tmp/','pipeline.sqlite','candidate_id','response_id','prompt_id','api_call_id',
                'provider_usage','reasoning_content','raw_text','API_KEY','Bearer '):assert bad not in json.dumps([r,ps])
    for n,count in [(100,35),(64,54)]:
        reg=yaml.safe_load((B/f'docs/challenge{n}_round1_model_registry_holdout_v2.yaml').read_text())
        assert len(reg['models'])==count and reg['models'][-3]['id']==MID
    table=(B/'docs/leaderboard.md').read_text()
    assert '| Claude Haiku 5.5 | 15.0% (15/100) | 16/100 (16.0%) | 99/100 | 16.4 / 16.0 | 23.3 |' in table
    assert '| Claude Haiku 5.5 | 21.9% (14/64) | 15/64 (23.4%) | 63/64 | 16.4 / 16.0 |' in table
    assert '[Protocol and accounting](haiku55_results.md)' in table
    assert r['evaluation']['publication_scoring_calls']==r['evaluation']['publication_model_calls']==0
