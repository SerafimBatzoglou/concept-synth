"""Read stored artifacts only; no parsing/scoring or provider calls."""
from collections import Counter
import gzip
import json
from pathlib import Path
import yaml

BENCH=Path(__file__).resolve().parents[1]/'benchmarks/induction'
MODEL='gemini-4-argon'

def report():return json.loads((BENCH/'eval/gemini_4_argon_challenge100_round1_report.json').read_text())
def rows(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return [json.loads(x) for x in f]

def test_final_selection_counts_and_no_rescoring():
    r=report()
    with gzip.open(BENCH/'data/induction_fullobs_challenge100_v1.yaml.gz','rt') as f:ids=[d['instanceId'] for d in yaml.safe_load(f)]
    assert [t['task_id'] for t in r['tasks']]==ids
    assert r['publication_status']=='final_verified_high_medium_low' and not r['provisional'] and r['no_rescoring']
    assert r['frontiers']=={'post_round1':100,'symbolic_candidates':0}
    assert r['non_evaluable_task_ids']==[]
    assert Counter(t['selected_effort'] for t in r['tasks'])=={'high':84,'medium':15,'low':1}
    assert len({t['category_or_slice'] for t in r['tasks'][64:]})==len(r['new36_slices'])==10
    for subset,ts,ev,correct,hold,available,missing in [('all',r['tasks'],100,57,37,54,3),('challenge64',r['tasks'][:64],64,42,24,41,1),('new36',r['tasks'][64:],36,15,13,13,2)]:
        s=r['totals'] if subset=='all' else r['by_subset'][subset]
        assert sum(t['evaluable'] for t in ts)==s['evaluable']==ev
        assert sum(t['correct'] for t in ts)==s['train_correct']==correct
        assert sum(t['holdout']['correct'] is True for t in ts)==s['holdout_correct']==hold
        assert sum(t['holdout']['available'] for t in ts)==s['holdout_available']==available
        assert sum(t['correct'] and not t['holdout']['available'] for t in ts)==s['train_correct_missing_holdout']==missing
    for t in r['tasks']:
        assert t['strict_as_submitted_evaluable'] and t['parser_normalizations']==[]
        assert bool(t['stored_train_score']['train_valid'])==t['correct']==t['strict_as_submitted_correct']
        assert bool(t['stored_holdout_score'])==t['holdout']['available']
        if t['stored_holdout_score']:assert bool(t['stored_holdout_score']['train_valid'])==t['holdout']['correct']
        else:assert t['holdout']['correct'] is None
        h=t['physical_history'];assert [a['campaign_attempt'] for a in h]==list(range(1,len(h)+1))
        assert next(a for a in h if a['outcome']=='evaluable')['campaign_attempt']==t['campaign_attempt']
        assert h[-1]['outcome']=='evaluable' and all(a['outcome']!='evaluable' for a in h[:-1])
    assert sum(len(t['physical_history']) for t in r['tasks'])==145

def test_usage_and_uncertainty():
    r=report();u=r['known_physical_usage'];c=r['cost_accounting']
    assert r['generation_calls']['started']==r['generation_calls']['terminal_evaluated']==145
    assert u['input_tokens']==262449163 and u['cached_input_tokens']==35099096
    assert u['output_tokens']==57789684==u['visible_output_tokens']+u['reasoning_tokens_within_output']
    assert u['reasoning_tokens_within_output']==57777205
    assert u['total_tokens']==320444228==u['input_tokens']+u['output_tokens']+c['separate_provider_tool_use_tokens']
    assert c['separate_provider_tool_use_tokens']==205381 and c['actual_billed_usd'] is None
    assert abs(c['estimate_after_explicit_waiver_usd']-1035.4360436)<1e-8
    assert r['selected_response_usage']['output_tokens']==35203765
    assert sum(a['usage']['output_tokens'] for t in r['tasks'] for a in t['physical_history'])==u['output_tokens']

def test_projection_privacy_and_leaderboards():
    r=report()
    tr={x['instance_id']:x for x in rows(BENCH/'eval/induction_challenge64_round1_eval_cache_v1.jsonl') if x['model_id']==MODEL}
    ho={x['instance_id']:x for x in rows(BENCH/'eval/induction_challenge64_round1_holdout_eval_cache_v1.jsonl') if x['model_id']==MODEL}
    ps=rows(BENCH/'predictions/gemini_4_argon_challenge100_predictions.jsonl.gz')
    assert len(tr)==len(ho)==64 and len(ps)==100
    assert [p['extractedFormula'] for p in ps]==[t['formula'] for t in r['tasks']]
    for t in r['tasks'][:64]:
        e,h=tr[t['task_id']],ho[t['task_id']]
        assert (e['parse_ok'],e['valid'],e['prediction']['formula'])==(t['evaluable'],t['correct'],t['formula'])
        assert h['completed']==t['holdout']['available'] and h['valid']==(t['holdout']['correct'] is True)
        for target,s in [(e,t['stored_train_score']),(h,t['stored_holdout_score'])]:
            if s:
                for k in ('status','mismatch_count','false_positives','false_negatives'):assert target['evaluation'][k]==s[k]
                assert target['prediction']['ast_size']==s['ast_size']
    serial=json.dumps([r,list(tr.values()),list(ho.values()),ps])
    for forbidden in ('/Users/','/tmp/','pipeline.sqlite','candidate_id','prompt_id','response_id','reasoningTrace','API_KEY'):assert forbidden not in serial
    assert all(p['rawResponse'] is None and p['response'] is None and p['thinking'] is None for p in ps)
    text=(BENCH/'docs/leaderboard.md').read_text()
    assert '| Gemini 4 Argon | 37.0% (37/100) | 57/100 (57.0%) | 100/100 | 45.5 / 18.0 | 57.8 |' in text
    assert '| Gemini 4 Argon | 37.5% (24/64) | 42/64 (65.6%) | 64/64 | 55.9 / 18.0 |' in text
