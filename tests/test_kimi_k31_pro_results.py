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
MID = 'kimi-k3.1-pro'

def load(p): return json.loads(p.read_text())
def rows(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:
        return [json.loads(s) for s in f if s.strip()]
def report(): return load(B/'eval/kimi_k31_pro_challenge100_holdout_v2.json')

def test_final_scores_strictness_and_holdout_gate():
    r=report(); assert r['final_verified'] and len(r['tasks'])==100
    assert len({t['task_id'] for t in r['tasks']})==100
    for key,n in [('count',100),('evaluable',99),('strict_evaluable',98),('train_correct',34),
                  ('strict_correct',34),('holdout_available',34),('holdout_correct',21),
                  ('normalizations',1),('train_correct_missing_holdout',0)]:
        assert r['totals'][key]==n
    for subset,expected in [('challenge64',(64,63,62,30,17)),('new36',(36,36,36,4,4))]:
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
    assert mean(ast)==r['totals']['correct_ast_mean']==pytest.approx(45.44117647058823)
    assert median(ast)==r['totals']['correct_ast_median']==18
    normalized=[t for t in r['tasks'] if t['normalizations']]
    assert len(normalized)==1 and normalized[0]['task_id']=='hard_035'
    assert not normalized[0]['train_correct'] and not normalized[0]['strict_evaluable']
    assert len(r['new36_slices'])==10
    for key,s in r['new36_slices'].items():
        ts=[t for t in r['tasks'] if t['task_id'].rsplit('_',1)[0]==key]
        assert len(ts)==s['count']==s['evaluable']==s['strict_evaluable']
        assert sum(t['train_correct'] for t in ts)==s['train_correct']
        assert sum(t['holdout_v2']['holdout_correct'] is True for t in ts)==s['holdout_correct']

def test_transport_only_retries_and_original_preservation():
    r=report(); hs=[h for t in r['tasks'] for h in t['physical_history']]
    assert len(hs)==138 and Counter(h['physical_attempt'] for h in hs)=={1:100,2:38}
    assert sum(h['remote_terminal'] for h in hs)==100
    assert sum(not h['remote_terminal'] for h in hs)==38
    assert r['physical_requests']=={'total':138,'by_attempt':{'1':100,'2':38},'unique_accepted':138,
        'confirmed_remote_terminal':100,'remote_unknown':38,'live_local':0,'queued':0}
    for t in r['tasks']:
        hs=t['physical_history']; assert hs==sorted(hs,key=lambda h:h['started_at'])
        assert [h['physical_attempt'] for h in hs]==list(range(1,len(hs)+1))
        chosen=next((h for h in hs if h['evaluable']),hs[0])
        assert t['physical_attempt']==chosen['physical_attempt'] and t['usage']==chosen['usage']
        for h in hs:
            assert h['effort']=='max' and h['max_completion_tokens']==500000
            assert h['logical_trial']==1 and h['recovery_ordinal']==h['physical_attempt']-1
        if len(hs)>1:
            assert not hs[0]['remote_terminal'] and not hs[0]['evaluable']
            assert hs[0]['http_status']==200 and hs[0]['finish_reason'] is None
            assert hs[0]['error_type']=='RemoteProtocolError'
            assert hs[1]['evaluable'] and hs[1]['remote_terminal']
    residual=[t for t in r['tasks'] if not t['evaluable']]
    assert [t['task_id'] for t in residual]==r['residual_non_evaluable']==['extreme_logic_009']
    assert len(residual[0]['physical_history'])==1
    assert residual[0]['physical_history'][0]['remote_terminal']
    assert residual[0]['physical_history'][0]['finish_reason']=='stop'
    assert r['evaluation']['pilot10_and_evaluable_originals_preserved']

def test_usage_reasoning_once_and_unknowns():
    r=report(); hs=[h['usage'] for t in r['tasks'] for h in t['physical_history']]
    assert len(hs)==r['usage']['physical_requests']==138
    for k,total in r['usage']['known_totals'].items():
        values=[u[k] for u in hs if u[k] is not None]
        assert sum(u[k] is None for u in hs)==r['usage']['unknown_counts'][k]
        assert total==(sum(values) if values else None)
    k=r['usage']['known_totals']
    assert k['input_tokens']==594862
    assert k['output_tokens']==11539644==k['reasoning_tokens_within_output']+k['visible_tokens']
    assert k['total_tokens']==12134506==k['input_tokens']+k['output_tokens']
    assert r['usage']['unknown_counts']['output_tokens']==38
    assert k['estimated_cost_usd_min'] is k['estimated_cost_usd_max'] is None
    token=next(t for t in load(B/'eval/challenge100_output_token_usage_v2.json')['models'] if t['model_id']==MID)
    assert token['output_tokens']==11539644 and token['physical_attempts']==138
    assert token['unreported_responses_lower_bound']==38 and token['reasoning_included_once']

def test_cache_prefix_preserved_and_scores_copied():
    r=report(); before=r['prior_release_preservation']['file_sha256_before_append']
    tasks={t['task_id']:t for t in r['tasks']}
    validate=jsonschema.Draft202012Validator(load(B/'schemas/eval_cache_record.schema.json')).validate
    for name,hold in [('induction_challenge64_round1_eval_cache_v2.jsonl',False),
                      ('induction_challenge64_round1_holdout_eval_cache_v2.jsonl',True)]:
        p=B/'eval'/name; lines=p.read_bytes().splitlines(keepends=True)
        assert len(lines)==3264 and hashlib.sha256(b''.join(lines[:3200])).hexdigest()==before[str(p.relative_to(ROOT))]
        added=[json.loads(x) for x in lines[3200:]]; assert len({t['instance_id'] for t in added})==64
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
    r=report();ps=rows(B/'predictions/kimi_k31_pro_challenge100_predictions_v2.jsonl.gz')
    assert len(ps)==100 and sum(p['success'] for p in ps)==99
    ts={t['task_id']:t for t in r['tasks']}
    validate=jsonschema.Draft202012Validator(load(B/'schemas/prediction_record.schema.json')).validate
    for p in ps:
        validate(p);assert p['rawResponse'] is p['response'] is p['thinking'] is None
        assert p['extractedFormula']==ts[p['instanceId']]['formula']
        assert not {'strict_correct','train_correct','holdout_correct'}.intersection(p['metadata'])
    for bad in ('/Users/','/tmp/','pipeline.sqlite','candidate_id','response_id','prompt_id','api_call_id',
                'provider_usage','reasoning_content','raw_text','API_KEY','Bearer '):assert bad not in json.dumps([r,ps])
    for n,count in [(100,32),(64,51)]:
        reg=yaml.safe_load((B/f'docs/challenge{n}_round1_model_registry_holdout_v2.yaml').read_text())
        assert len(reg['models'])==count and reg['models'][-1]['id']==MID
    table=(B/'docs/leaderboard.md').read_text()
    assert '| Kimi K3.1 Pro | 21.0% (21/100) | 34/100 (34.0%) | 99/100 | 45.4 / 18.0 | 11.5 |' in table
    assert '| Kimi K3.1 Pro | 26.6% (17/64) | 30/64 (46.9%) | 63/64 | 49.3 / 20.5 |' in table
    assert '[Protocol and accounting](kimi_k31_pro_results.md)' in table
    assert r['evaluation']['publication_scoring_calls']==r['evaluation']['publication_model_calls']==0
