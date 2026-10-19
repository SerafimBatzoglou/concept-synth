"""Verify final stored Flash-1017 results without parsing or rescoring answers."""
import gzip
import json
from pathlib import Path

import jsonschema
import yaml

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'benchmarks/induction'
MID='deepseek-v4.1-flash-1017'
def load(p):return json.loads(p.read_text())
def rows(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return [json.loads(x) for x in f if x.strip()]

def test_final_results_and_attempt_accounting():
    r=load(B/'eval/deepseek_flash1017_challenge100_holdout_v2.json')
    assert r['final_verified'] and len(r['tasks'])==100
    assert r['totals']['evaluable']==100 and r['totals']['strict_evaluable']==96
    assert r['totals']['train_correct']==21 and r['totals']['strict_correct']==20
    assert r['totals']['holdout_correct']==9 and r['totals']['holdout_available']==21
    assert r['v1_totals']['holdout_correct']==8 and r['v1_totals']['train_correct_missing_holdout']==1
    assert len(r['new36_slices'])==10 and r['by_subset']['new36']['train_correct']==0
    history=[h for t in r['tasks'] for h in t['physical_history']]
    assert len(history)==126 and sum(h['remote_terminal'] for h in history)==101
    assert sum(h['administrative_exclusion'] for h in history)==11
    assert sum(t['selected_effort']=='high' for t in r['tasks'])==1
    for t in r['tasks']:
        first=next(h for h in sorted(t['physical_history'],key=lambda h:h['started_at']) if h['evaluable'])
        assert t['physical_attempt']==first['physical_attempt'] and t['selected_effort']==first['effort']
        assert t['holdout_v2']['holdout_available']==t['train_correct']
    u=r['usage']['physical'];known=u['known_totals']
    assert all(n==25 for n in u['unknown_counts'].values())
    assert known['output_tokens']==known['reasoning_tokens_within_output']+known['visible_tokens']==12526535
    assert known['total_tokens']==known['input_tokens']+known['output_tokens']==13122831

def test_current_caches_exact_old_prefix_and_new_stored_scores():
    old=rows(B/'eval/induction_challenge64_round1_eval_cache_v1.jsonl')
    train=rows(B/'eval/induction_challenge64_round1_eval_cache_v2.jsonl')
    assert train[:3072]==old and len(train)==3200
    flash_train=train[3072:3136]
    assert {x['model_id'] for x in flash_train}=={MID}
    holds=[x for x in rows(B/'eval/induction_challenge64_round1_holdout_eval_cache_v2.jsonl') if x['model_id']==MID]
    source={t['task_id']:t for t in load(B/'eval/deepseek_flash1017_challenge100_holdout_v2.json')['tasks']}
    validator=jsonschema.Draft202012Validator(load(B/'schemas/eval_cache_record.schema.json'))
    for row in flash_train+holds:validator.validate(row)
    for row in flash_train:
        t=source[row['instance_id']];s=t['stored_train_score']
        assert row['prediction']['formula']==t['formula'] and row['valid']==t['train_correct']
        for k in ('ast_size','quantifier_depth','equality_count'):assert row['prediction'][k]==s[k]
        for k in ('status','mismatch_count','false_positives','false_negatives'):assert row['evaluation'][k]==s[k]
    assert len(holds)==64 and sum(h['valid'] for h in holds)==9 and sum(h['completed'] for h in holds)==21
    for row in holds:
        t=source[row['instance_id']]
        assert row['valid']==(t['holdout_v2']['holdout_correct'] is True)
        assert row['metadata']['holdout_outcome']==t['holdout_v2']['holdout_correct']
        assert row['metadata']['holdout_world_count']==5

def test_affected_only_patch_predictions_and_privacy():
    r=load(B/'eval/deepseek_flash1017_challenge100_holdout_v2.json');patch=r['affected_holdout_v2_results']
    assert len(patch)==len({x['task_id'] for x in patch})==14
    assert sum(x['new_holdout_score_calls'] for x in patch)==3
    assert sum(not x['holdout_evaluated'] for x in patch)==11
    for x in patch:
        assert x['holdout_evaluated']==x['train_correct']
        for c in x['candidate_results']:
            assert len(c['world_results'])==5 and c['v2_correct']==all(w['mismatch_count']==0 for w in c['world_results'])
    preds=rows(B/'predictions/deepseek_flash1017_challenge100_predictions_v2.jsonl.gz')
    assert len(preds)==100 and all(x['rawResponse'] is None and x['thinking'] is None for x in preds)
    for item in [r,preds]:
        text=json.dumps(item)
        for forbidden in ('/Users/','/tmp/','pipeline.sqlite','prompt_id','response_id','api_call_id','reasoning_content','raw_text','API_KEY'):assert forbidden not in text
    old=load(B/'eval/challenge100_output_token_usage.json')
    new=load(B/'eval/challenge100_output_token_usage_v2.json')
    assert new['models'][:len(old['models'])]==old['models']
    assert next(m for m in new['models'] if m['model_id']==MID)['unreported_responses_lower_bound']==25
    for n,count in ((100,31),(64,50)):
        registry=yaml.safe_load((B/f'docs/challenge{n}_round1_model_registry_holdout_v2.yaml').read_text())
        assert len(registry['models'])==count and registry['models'][-2]['id']==MID

def test_current_rendered_flash_scores():
    text=(B/'docs/leaderboard.md').read_text()
    assert '| DeepSeek 4.1 Flash-1017 | 9.0% (9/100)' in text
    assert '| DeepSeek 4.1 Flash-1017 | 14.1% (9/64)' in text
    assert 'challenge100_output_token_usage_v2.json' in text
