import gzip
import json
from pathlib import Path

import yaml

BENCH = Path(__file__).resolve().parents[1] / 'benchmarks/induction'


def test_sol_final_stored_scores_projection_and_accounting():
    report = json.loads((BENCH/'eval/gpt6_sol_challenge100_round1_report.json').read_text())
    assert report['model'] == 'gpt-6-sol'
    assert report['publication_status'] == 'final_two_verified_physical_passes_empty_residual'
    with gzip.open(BENCH/'data/induction_fullobs_challenge100_v1.yaml.gz', 'rt') as f:
        ids = [t['instanceId'] for t in yaml.safe_load(f)]
    assert [t['task_id'] for t in report['tasks']] == ids and len(set(ids)) == 100
    assert report['frontiers'] == {'post_round1':100, 'symbolic_candidates':0}
    assert (report['overall']['evaluable'], report['overall']['correct']) == (100,57)
    assert (report['overall']['strict_as_submitted_evaluable'], report['overall']['strict_as_submitted_correct']) == (99,56)
    assert report['overall']['parser_normalized_responses'] == 1
    assert (report['overall']['holdout']['correct'], report['overall']['holdout']['available']) == (35,53)
    assert report['non_evaluable_task_ids'] == []
    assert report['settings']['effort_sequence'] == ['max','xhigh']
    assert report['settings']['skipped_efforts'] == ['high','medium']
    assert report['settings']['max_output_tokens'] == 128000
    assert [p['generation_calls']['submitted_requests'] for p in report['components']] == [100,50]
    assert [p['overall']['evaluable'] for p in report['components']] == [50,50]
    assert [p['overall']['correct'] for p in report['components']] == [48,9]
    assert report['components'][1]['overall']['holdout']['correct'] == 0
    calls = report['generation_calls']
    assert calls['submitted_requests'] == calls['terminal_records'] == 150
    assert (calls['attempted_confirmed'],calls['attempted_unknown']) == (101,49)
    assert (calls['accepted_confirmed'],calls['acceptance_unknown']) == (100,50)
    assert calls['batch_submissions'] == 2 and calls['retries'] == 0
    assert report['all_physical_response_token_usage']['total']['sum'] == 5252531
    assert report['all_physical_response_token_usage']['total']['count'] == 100
    assert report['overall']['token_usage']['total']['sum'] == 5252531
    assert report['overall']['token_usage']['reasoning']['sum'] == 4656306
    assert 'unknown, not zero' in report['token_usage_note']
    assert [c['selected'] for c in report['selected_component_counts'].values()] == [50,50]
    assert len({t['category_or_slice'] for t in report['tasks'] if t['subset']=='new36'}) == 10
    rows = [json.loads(x) for x in (BENCH/'eval/induction_challenge64_round1_eval_cache_v1.jsonl').read_text().splitlines()]
    projected = {r['instance_id']:r for r in rows if r['model_id']==report['model']}
    assert len(projected) == 64
    assert sum(r['parse_ok'] for r in projected.values()) == 64
    assert sum(r['valid'] for r in projected.values()) == 42
    for t in report['tasks'][:64]:
        assert projected[t['task_id']]['parse_ok'] == t['evaluable']
        assert projected[t['task_id']]['valid'] == t['correct']
    assert all(t['formula'] and t['original_submitted_formula'] for t in report['tasks'])
    serialized = json.dumps(report)
    for forbidden in ['/Users/','pipeline.sqlite','reasoningTrace','provider_call_meta','candidate_id','OPENAI_API_KEY','batch_6','file-D','prompt_id']:
        assert forbidden not in serialized
