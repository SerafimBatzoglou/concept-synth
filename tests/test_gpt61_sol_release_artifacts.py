"""Stored-artifact checks only: never reparse formulas or recompute scores."""
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

import yaml

BENCH = Path(__file__).resolve().parents[1] / 'benchmarks/induction'
MODEL = 'gpt-6.1-sol'


def report():
    return json.loads((BENCH / 'eval/gpt6_1_sol_challenge100_round1_report.json').read_text())


def test_final_counts_strict_syntax_holdouts_and_complexity():
    r = report()
    with gzip.open(BENCH / 'data/induction_fullobs_challenge100_v1.yaml.gz', 'rt') as f:
        ids = [d['instanceId'] for d in yaml.safe_load(f)]
    assert [t['task_id'] for t in r['tasks']] == ids
    assert r['publication_status'] == 'final_verified_regular_api_retry_policy_v3'
    assert r['frontiers'] == {'post_round1': 100, 'symbolic_candidates': 0}
    assert r['provisional'] is False and r['no_rescoring'] and r['new_generations'] == 0
    expected = {'challenge100': (100, 98, 90, 90, 88, 8, 62, 84, 6),
                'challenge64': (64, 63, 61, 61, 60, 2, 42, 60, 1),
                'new36': (36, 35, 29, 29, 28, 6, 20, 24, 5)}
    for subset, (n, evaluable, correct, strict_eval, strict_correct, normalized, hold_correct, available, missing) in expected.items():
        tasks = [t for t in r['tasks'] if subset == 'challenge100' or t['subset'] == subset]
        s = r['by_subset'][subset]
        assert len(tasks) == n == s['denominator']
        assert sum(t['evaluable'] for t in tasks) == s['evaluable'] == evaluable
        assert sum(t['correct'] for t in tasks) == s['correct'] == correct
        assert sum(t['strict_as_submitted_evaluable'] for t in tasks) == s['strict_evaluable'] == strict_eval
        assert sum(t['strict_as_submitted_correct'] for t in tasks) == s['strict_correct'] == strict_correct
        assert sum(bool(t['parser_normalizations']) for t in tasks) == s['parser_normalized_responses'] == normalized
        assert sum(t['holdout']['correct'] is True for t in tasks) == s['holdout']['correct'] == hold_correct
        assert sum(t['holdout']['available'] for t in tasks) == s['holdout']['available'] == available
        assert sum(t['correct'] and not t['holdout']['available'] for t in tasks) == s['holdout']['train_correct_missing_worlds'] == missing
        asts = [t['correct_ast'] for t in tasks if t['correct']]
        assert (mean(asts), median(asts)) == (s['correct_formula_ast']['mean'], s['correct_formula_ast']['median'])
    assert len({t['category_or_slice'] for t in r['tasks'][64:]}) == 10
    for task in r['tasks']:
        assert bool(task['formula']) == task['evaluable']
        train, hold = task['stored_train_score'], task['stored_holdout_score']
        assert bool(train and train['parse_ok']) == task['evaluable']
        assert bool(train and train['train_valid']) == task['correct']
        assert bool(hold) == task['holdout']['available']
        if hold:
            assert task['correct'] and bool(hold['train_valid']) == task['holdout']['correct']
        else:
            assert task['holdout']['correct'] is None
        if task['parser_normalizations']:
            assert task['original_submitted_formula'] != task['formula']
    assert sum(t['correct'] and bool(t['parser_normalizations']) for t in r['tasks']) == 2


def test_first_evaluable_history_and_separate_budget_exhaustion():
    r = report()
    assert r['policy']['additional_error_budget'] == r['policy']['additional_token_cap_budget'] == 3
    assert r['policy']['third_error_downgrade_charges'] == 'error_budget_only'
    assert r['policy']['cap_triggered_calls'] == 0
    exhausted = {'extreme_context_010', 'fullobs_v2_nested_containment_qd3_006'}
    assert set(r['non_evaluable_task_ids']) == exhausted
    lengths = Counter()
    for t in r['tasks']:
        history = t['physical_history']; budget = t['retry_budget']
        lengths[len(history)] += 1
        assert [a['attempt_index'] for a in history] == list(range(len(history)))
        assert [a['effort'] for a in history] == ['max', 'max', 'max', 'xhigh'][:len(history)]
        assert [a['trigger'] for a in history] == ['initial'] + ['error_retry'] * (len(history) - 1)
        assert all(a['outcome'] == 'error' for a in history[:-1])
        assert budget['error_retries_used'] == len(history) - 1 and budget['token_cap_retries_used'] == 0
        if t['task_id'] in exhausted:
            assert len(history) == 4 and history[-1]['outcome'] == 'error'
            assert budget['action'] == 'stop_error_budget_exhausted' and t['selected_attempt_index'] == 0
            assert not t['evaluable'] and all(value is None for value in t['tokens'].values())
        else:
            assert history[-1]['outcome'] == 'evaluable' and budget['action'] == 'stop_evaluable'
            assert t['selected_attempt_index'] == len(history) - 1
            assert t['selected_component'] == history[-1]['component']
    assert lengths == {1: 55, 2: 10, 3: 9, 4: 26}
    assert sum(len(t['physical_history']) for t in r['tasks']) == 206
    assert sum(t['evaluable'] and not t['correct'] for t in r['tasks']) == 8


def test_physical_vs_selected_accounting_and_unknown_usage():
    r = report(); calls = r['generation_calls']
    assert calls['calls_started'] == calls['local_terminal_outcomes'] == calls['attempted_confirmed'] == 206
    assert calls['attempted_unknown'] == calls['rejected'] == 0
    assert calls['accepted_confirmed'] == 98 and calls['acceptance_unknown'] == calls['unknown_usage'] == 108
    assert calls['http_statuses'] == {'200': 98, '524': 96, '502': 6, '500': 4, '520': 2}
    assert [c['calls']['calls_started'] for c in r['components']] == [100, 45, 15, 16, 4, 3, 8, 5, 1, 4, 2, 3]
    assert sum(c['calls']['acceptance_unknown'] for c in r['components']) == 108
    for key, expected in [('input', 570320), ('output', 4134385), ('reasoning', 4111197), ('total', 4704705)]:
        assert r['known_physical_usage'][key]['sum'] == expected
        assert r['selected_response_usage'][key]['sum'] == expected
        assert r['known_physical_usage'][key]['count'] == r['selected_response_usage'][key]['count'] == 98
        assert sum(c['summary']['token_usage'][key]['sum'] for c in r['components']) == expected
        assert sum(t['tokens'][key] for t in r['tasks'] if t['tokens'][key] is not None) == expected
    assert r['cost_accounting']['actual_total_billed_usd'] is None
    assert all(v is None for v in r['cost_accounting']['actual_component_billed_usd'].values())
    assert r['initial_physical_calls'] == {'new': 98, 'reused_probes': 2, 'total': 100}
    assert r['failed_batch']['validation_failed'] and r['failed_batch']['billed_usd'] is None
    assert r['failed_batch']['regular_generation_calls'] == 0 and not r['failed_batch']['retrieved']
    assert r['settings']['sdk_automatic_retries'] == 0 and r['settings']['retry_timeout_seconds'] is None
    assert r['settings']['max_output_tokens'] == 128000 and r['settings']['total_workers'] == 20


def test_public_projection_privacy_and_source_hashes():
    r = report()
    evals = [json.loads(line) for line in (BENCH / 'eval/induction_challenge64_round1_eval_cache_v1.jsonl').read_text().splitlines()]
    evals = {x['instance_id']: x for x in evals if x['model_id'] == MODEL}
    holds = [json.loads(line) for line in (BENCH / 'eval/induction_challenge64_round1_holdout_eval_cache_v1.jsonl').read_text().splitlines()]
    holds = {x['instance_id']: x for x in holds if x['model_id'] == MODEL}
    with gzip.open(BENCH / 'predictions/induction_challenge64_round1_predictions_v1.jsonl.gz', 'rt') as f:
        preds = [json.loads(line) for line in f]
    preds = [x for x in preds if x['model'] == MODEL]
    assert len(evals) == len(holds) == len(preds) == 64
    for t in r['tasks'][:64]:
        e, h = evals[t['task_id']], holds[t['task_id']]
        assert (e['parse_ok'], e['valid']) == (t['evaluable'], t['correct'])
        assert (h['completed'], h['valid']) == (t['holdout']['available'], t['holdout']['correct'] is True)
        if t['stored_train_score']:
            assert e['prediction']['ast_size'] == t['stored_train_score']['ast_size']
        if t['stored_holdout_score']:
            for field in ['mismatch_count', 'false_positives', 'false_negatives', 'status']:
                assert h['evaluation'][field] == t['stored_holdout_score'][field]
    assert sum(x['completed'] for x in holds.values()) == 60 and sum(x['valid'] for x in holds.values()) == 42
    assert all(p['rawResponse'] is None and p['response'] is None and p['thinking'] is None for p in preds)
    serialized = json.dumps([r, list(evals.values()), list(holds.values()), preds])
    for pattern in [r'/Users/', r'/tmp/', r'pipeline.sqlite', r'candidate_id', r'prompt_id', r'reasoningTrace',
                    r'provider_call_meta', r'OPENAI_API_KEY', r'sk-[A-Za-z0-9_-]{12,}', r'resp_[A-Za-z0-9]{12,}']:
        assert not re.search(pattern, serialized), pattern
    assert r['final_combined_report_sha256'] == '1a9258a794240ff575b866a28b3b4ced5893d6cd86c115fe1cc37b436e99befc'
    assert r['campaign_completion_sha256'] == 'de7f5435edbd097191638b52defc8f3261372e0a367abdb52d3616d45691691a'
    assert r['verification']['source_files_preserved'] == 2958
    text = (BENCH / 'docs/leaderboard.md').read_text()
    assert '| GPT-6.1 Sol | 62.0% (62/100) | 90/100 (90.0%) | 98/100 | 61.9 / 18.0 |' in text
    assert '| GPT-6.1 Sol | 65.6% (42/64) | 61/64 (95.3%) | 63/64 | 45.7 / 18.0 |' in text
