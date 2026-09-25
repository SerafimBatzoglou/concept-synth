import gzip
import json
from pathlib import Path
import yaml

BENCH = Path(__file__).resolve().parents[1] / 'benchmarks/induction'

def test_luna_final_four_pass_projection_and_accounting():
    r = json.loads((BENCH/'eval/gpt6_luna_challenge100_round1_report.json').read_text())
    assert r['model'] == 'gpt-6-luna'
    assert r['publication_status'] == 'final_four_verified_physical_passes_empty_residual'
    with gzip.open(BENCH/'data/induction_fullobs_challenge100_v1.yaml.gz','rt') as f:
        ids = [t['instanceId'] for t in yaml.safe_load(f)]
    assert [t['task_id'] for t in r['tasks']] == ids and len(set(ids)) == 100
    assert r['frontiers'] == {'post_round1':100,'symbolic_candidates':0}
    assert (r['overall']['evaluable'],r['overall']['correct']) == (100,18)
    assert (r['overall']['strict_as_submitted_evaluable'],r['overall']['strict_as_submitted_correct']) == (83,10)
    assert r['overall']['parser_normalized_responses'] == 17
    assert (r['overall']['holdout']['correct'],r['overall']['holdout']['available']) == (12,17)
    assert r['non_evaluable_task_ids'] == []
    assert r['settings']['effort_sequence'] == ['max','xhigh','high','medium']
    assert r['settings']['pending_efforts'] == []
    assert r['settings']['skipped_efforts'] == []
    assert r['settings']['max_output_tokens'] == 128000
    assert [p['generation_calls']['submitted_requests'] for p in r['components']] == [100,82,14,5]
    assert [p['overall']['evaluable'] for p in r['components']] == [18,68,9,5]
    assert [p['overall']['correct'] for p in r['components']] == [18,0,0,0]
    calls = r['generation_calls']
    assert calls['submitted_requests'] == calls['terminal_records'] == 201
    assert (calls['attempted_confirmed'],calls['attempted_unknown']) == (171,30)
    assert (calls['accepted_confirmed'],calls['acceptance_unknown']) == (169,32)
    assert calls['batch_submissions'] == 4 and calls['retries'] == 0
    assert r['all_physical_response_token_usage']['total']['sum'] == 14312408
    assert r['all_physical_response_token_usage']['total']['count'] == 169
    assert r['overall']['token_usage']['total']['sum'] == 5187646
    assert 'unknown, not zero' in r['token_usage_note']
    assert r['selected_component_counts'] == {
        'pass1_max':{'selected':18,'selected_evaluable':18},
        'pass2_xhigh':{'selected':68,'selected_evaluable':68},
        'pass3_high':{'selected':9,'selected_evaluable':9},
        'pass4_medium':{'selected':5,'selected_evaluable':5}}
    assert r['monetary_cost'] is None
    assert [p['overall']['token_usage']['total']['sum'] for p in r['components']] == [10384358,3603370,247917,76763]
    assert [p['generation_calls']['acceptance_unknown'] for p in r['components']] == [15,12,5,0]
    assert [p['generation_calls']['attempted_unknown'] for p in r['components']] == [15,12,3,0]
    assert [p['generation_calls']['submitted_requests']-p['overall']['token_usage']['total']['count'] for p in r['components']] == [15,12,5,0]
    assert (r['by_subset']['new36']['evaluable'],r['by_subset']['new36']['correct']) == (36,0)
    assert r['final_combined_report_sha256'] == 'e7c23a85e2240ec43826c398a6ddb83af9dc01281d4b0b6a912d2c471dd17505'
    rows = [json.loads(x) for x in (BENCH/'eval/induction_challenge64_round1_eval_cache_v1.jsonl').read_text().splitlines()]
    projected = {x['instance_id']:x for x in rows if x['model_id']==r['model']}
    assert len(projected) == 64
    assert sum(x['parse_ok'] for x in projected.values()) == 64
    assert sum(x['valid'] for x in projected.values()) == 18
    for t in r['tasks'][:64]:
        assert projected[t['task_id']]['parse_ok'] == t['evaluable']
        assert projected[t['task_id']]['valid'] == t['correct']
    assert all(bool(t['formula']) == t['evaluable'] for t in r['tasks'])
    assert len({t['category_or_slice'] for t in r['tasks'] if t['subset']=='new36'}) == 10
    for name in ['leaderboard.md','challenge64_round1_results.md']:
        text = (BENCH/'docs'/name).read_text()
        assert 'GPT-6 Luna' in text and 'GPT-6 Luna (provisional)' not in text
    for name in ['challenge100_round1_model_registry.yaml','challenge64_round1_model_registry.yaml']:
        row = next(x for x in yaml.safe_load((BENCH/'docs'/name).read_text())['models'] if x['id']==r['model'])
        assert row['display_name'] == 'GPT-6 Luna'
    for forbidden in ['/Users/','pipeline.sqlite','reasoningTrace','provider_call_meta','candidate_id','OPENAI_API_KEY','batch_6','prompt_id']:
        assert forbidden not in json.dumps(r)
