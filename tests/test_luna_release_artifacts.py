import gzip
import json
from pathlib import Path
import yaml

BENCH = Path(__file__).resolve().parents[1] / 'benchmarks/induction'

def test_luna_provisional_snapshot_and_projection():
    r = json.loads((BENCH/'eval/gpt6_luna_challenge100_round1_report.json').read_text())
    assert r['model'] == 'gpt-6-luna'
    assert r['publication_status'] == 'provisional_two_verified_physical_passes'
    with gzip.open(BENCH/'data/induction_fullobs_challenge100_v1.yaml.gz','rt') as f:
        ids = [t['instanceId'] for t in yaml.safe_load(f)]
    assert [t['task_id'] for t in r['tasks']] == ids and len(set(ids)) == 100
    assert r['frontiers'] == {'post_round1':100,'symbolic_candidates':0}
    assert (r['overall']['evaluable'],r['overall']['correct']) == (86,18)
    assert (r['overall']['strict_as_submitted_evaluable'],r['overall']['strict_as_submitted_correct']) == (70,10)
    assert r['overall']['parser_normalized_responses'] == 16
    assert (r['overall']['holdout']['correct'],r['overall']['holdout']['available']) == (12,17)
    assert len(r['non_evaluable_task_ids']) == 14
    assert r['settings']['effort_sequence'] == ['max','xhigh']
    assert r['settings']['pending_efforts'] == ['high','medium']
    assert r['settings']['skipped_efforts'] == []
    assert r['settings']['max_output_tokens'] == 128000
    assert [p['generation_calls']['submitted_requests'] for p in r['components']] == [100,82]
    assert [p['overall']['evaluable'] for p in r['components']] == [18,68]
    assert [p['overall']['correct'] for p in r['components']] == [18,0]
    calls = r['generation_calls']
    assert calls['submitted_requests'] == calls['terminal_records'] == 182
    assert (calls['attempted_confirmed'],calls['attempted_unknown']) == (155,27)
    assert (calls['accepted_confirmed'],calls['acceptance_unknown']) == (155,27)
    assert calls['batch_submissions'] == 2 and calls['retries'] == 0
    assert r['all_physical_response_token_usage']['total']['sum'] == 13987728
    assert r['all_physical_response_token_usage']['total']['count'] == 155
    assert r['overall']['token_usage']['total']['sum'] == 6070544
    assert 'unknown, not zero' in r['token_usage_note']
    assert r['selected_component_counts'] == {'pass1_max':{'selected':32,'selected_evaluable':18},'pass2_xhigh':{'selected':68,'selected_evaluable':68}}
    rows = [json.loads(x) for x in (BENCH/'eval/induction_challenge64_round1_eval_cache_v1.jsonl').read_text().splitlines()]
    projected = {x['instance_id']:x for x in rows if x['model_id']==r['model']}
    assert len(projected) == 64
    assert sum(x['parse_ok'] for x in projected.values()) == 56
    assert sum(x['valid'] for x in projected.values()) == 18
    for t in r['tasks'][:64]:
        assert projected[t['task_id']]['parse_ok'] == t['evaluable']
        assert projected[t['task_id']]['valid'] == t['correct']
    assert all(bool(t['formula']) == t['evaluable'] for t in r['tasks'])
    assert len({t['category_or_slice'] for t in r['tasks'] if t['subset']=='new36'}) == 10
    for name in ['leaderboard.md','challenge64_round1_results.md']:
        assert 'GPT-6 Luna (provisional)' in (BENCH/'docs'/name).read_text()
    for name in ['challenge100_round1_model_registry.yaml','challenge64_round1_model_registry.yaml']:
        row = next(x for x in yaml.safe_load((BENCH/'docs'/name).read_text())['models'] if x['id']==r['model'])
        assert row['display_name'] == 'GPT-6 Luna (provisional)'
    for forbidden in ['/Users/','pipeline.sqlite','reasoningTrace','provider_call_meta','candidate_id','OPENAI_API_KEY','batch_6','prompt_id']:
        assert forbidden not in json.dumps(r)
