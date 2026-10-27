"""Check released stored evidence only; never evaluate benchmark formulas."""
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
B = ROOT / 'benchmarks/induction'
MID = 'inception/mercury-3'


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return [json.loads(line) for line in f if line.strip()]


def report():
    return load(B / 'eval/mercury3_challenge100_holdout_v2.json')


def test_final_totals_holdout_gate_and_slices():
    r = report()
    assert r['final_verified'] and len(r['tasks']) == 100
    assert len({t['task_id'] for t in r['tasks']}) == 100
    totals = r['totals']
    for key, expected in [('evaluable', 96), ('strict_evaluable', 95),
                          ('train_correct', 9), ('strict_correct', 9),
                          ('holdout_available', 9), ('holdout_correct', 5),
                          ('normalizations', 1), ('train_correct_missing_holdout', 0)]:
        assert totals[key] == expected
    assert sum(t['evaluable'] for t in r['tasks']) == 96
    assert sum(t['train_correct'] for t in r['tasks']) == 9
    asts = []
    for t in r['tasks']:
        assert t['holdout_v2']['holdout_available'] == t['train_correct']
        assert (t['stored_holdout_v2_score'] is not None) == t['train_correct']
        assert (t['stored_train_score'] is not None) == t['evaluable']
        if t['train_correct']:
            assert t['strict_correct'] and t['holdout_v2']['holdout_world_count'] == 5
            asts.append(t['stored_train_score']['ast_size'])
        else:
            assert t['holdout_v2']['holdout_correct'] is None
        if t['normalizations']:
            assert not t['strict_evaluable'] and not t['train_correct']
    assert mean(asts) == totals['correct_ast_mean'] == pytest.approx(88.1111111111)
    assert median(asts) == totals['correct_ast_median'] == 18
    assert r['by_subset']['challenge64']['evaluable'] == 60
    assert r['by_subset']['new36']['evaluable'] == 36
    assert len(r['new36_slices']) == 10
    assert sum(s['count'] for s in r['new36_slices'].values()) == 36
    assert all(s['evaluable'] == s['count'] and s['train_correct'] == s['holdout_correct'] == 0
               for s in r['new36_slices'].values())


def test_bounded_attempts_and_first_evaluable_selection():
    r = report()
    history = [h for t in r['tasks'] for h in t['physical_history']]
    assert len(history) == 222
    assert Counter(h['physical_attempt'] for h in history) == {1: 100, 2: 55, 3: 29, 4: 25, 5: 8, 6: 5}
    assert Counter(h['physical_attempt'] for h in history if h['evaluable']) == {1: 45, 2: 26, 3: 4, 4: 17, 5: 3, 6: 1}
    assert r['physical_requests']['unique_accepted'] == 194
    assert sum(h['remote_terminal'] for h in history) == 142
    assert sum(not h['remote_terminal'] for h in history) == 80
    assert sum(h['http_status'] == 429 for h in history) == 28
    for t in r['tasks']:
        hs = t['physical_history']
        assert [h['physical_attempt'] for h in hs] == list(range(1, len(hs) + 1))
        assert hs == sorted(hs, key=lambda h: h['started_at'])
        assert all(not h['evaluable'] for h in hs[:-1])
        chosen = next((h for h in hs if h['evaluable']), hs[0])
        assert t['physical_attempt'] == chosen['physical_attempt']
        assert t['selected_effort'] == chosen['effort']
        assert t['usage'] == chosen['usage']
        assert t['selected_original_exclusion'] == (not t['evaluable'])
        for h in hs:
            n = h['physical_attempt']
            assert h['max_tokens'] == 131072
            assert h['effort'] == ('high' if n in (1, 2, 4) else 'medium')
            assert h['residual_pass'] == (1 if n in (4, 5) else 2 if n == 6 else 0)
            if n >= 4:
                assert h['residual_pass_attempt'] == (2 if n == 5 else 1)
    remaining = {t['task_id']: t['physical_history'][-1]['outcome'] for t in r['tasks'] if not t['evaluable']}
    assert remaining == {'hard_081': 'remote_unknown', 'hard_026': 'remote_unknown',
                         'hard_086': 'token_limit', 'hard_035': 'token_limit'}
    assert set(r['residual_non_evaluable']) == set(remaining)
    assert r['evaluation']['prior95_evaluable_selections_preserved']
    assert r['evaluation']['publication_scoring_calls'] == r['evaluation']['publication_model_calls'] == 0


def test_physical_and_selected_usage_include_reasoning_once():
    r = report()
    for kind, usages in [('physical', [h['usage'] for t in r['tasks'] for h in t['physical_history']]),
                         ('selected', [t['usage'] for t in r['tasks']])]:
        totals = r['usage'][kind]
        for key, expected in totals['known_totals'].items():
            assert sum(u[key] for u in usages if u[key] is not None) == pytest.approx(expected)
            assert sum(u[key] is None for u in usages) == totals['unknown_counts'][key]
        k = totals['known_totals']
        assert k['output_tokens'] == k['reasoning_tokens_within_output'] + k['visible_tokens']
        assert k['total_tokens'] == k['input_tokens'] + k['output_tokens']
    assert r['usage']['physical']['known_totals']['output_tokens'] == 11584662
    assert r['usage']['physical']['unknown_counts']['output_tokens'] == 108
    assert r['usage']['selected']['unknown_counts']['output_tokens'] == 3
    tokens = next(m for m in load(B / 'eval/challenge100_output_token_usage_v2.json')['models'] if m['model_id'] == MID)
    assert tokens['output_tokens'] == 11584662 and tokens['physical_attempts'] == 222
    assert tokens['unreported_responses_lower_bound'] == 108 and tokens['reasoning_included_once']


def test_exact_prior_cache_bytes_and_new_stored_scores():
    r = report()
    before = r['prior_release_preservation']['file_sha256_before_append']
    source = {t['task_id']: t for t in r['tasks']}
    validator = jsonschema.Draft202012Validator(load(B / 'schemas/eval_cache_record.schema.json'))
    for name, holdout in [('induction_challenge64_round1_eval_cache_v2.jsonl', False),
                          ('induction_challenge64_round1_holdout_eval_cache_v2.jsonl', True)]:
        path = B / 'eval' / name
        lines = path.read_bytes().splitlines(keepends=True)
        assert len(lines) == 3392
        assert hashlib.sha256(b''.join(lines[:3136])).hexdigest() == before[str(path.relative_to(ROOT))]
        added = [json.loads(line) for line in lines[3136:3200]]
        assert len({row['instance_id'] for row in added}) == 64
        for row in added:
            validator.validate(row)
            assert row['model_id'] == MID
            t = source[row['instance_id']]
            assert row['prediction']['formula'] == t['formula']
            s = t['stored_holdout_v2_score' if holdout else 'stored_train_score']
            assert row['parse_ok'] == bool(s and s['parse_ok'])
            assert row['completed'] == (t['train_correct'] if holdout else t['evaluable'])
            assert row['valid'] == (t['holdout_v2']['holdout_correct'] is True if holdout else t['train_correct'])
            if s:
                for k in ('status', 'mismatch_count', 'false_positives', 'false_negatives'):
                    assert row['evaluation'][k] == s[k]
            if holdout:
                assert row['metadata']['holdout_world_count'] == 5
            elif s:
                for k in ('ast_size', 'quantifier_depth', 'equality_count'):
                    assert row['prediction'][k] == s[k]


def test_predictions_privacy_registries_and_rendered_results():
    r = report()
    preds = rows(B / 'predictions/mercury3_challenge100_predictions_v2.jsonl.gz')
    assert len(preds) == 100 and sum(x['success'] for x in preds) == 96
    source = {t['task_id']: t for t in r['tasks']}
    validator = jsonschema.Draft202012Validator(load(B / 'schemas/prediction_record.schema.json'))
    for pred in preds:
        validator.validate(pred)
        assert pred['extractedFormula'] == source[pred['instanceId']]['formula']
        assert pred['rawResponse'] is pred['response'] is pred['thinking'] is None
        assert 'evaluation' not in pred and 'score' not in pred
        assert not {'strict_correct', 'train_correct', 'holdout_correct'}.intersection(pred['metadata'])
    text = json.dumps([r, preds])
    for forbidden in ('/Users/', '/tmp/', 'pipeline.sqlite', 'candidate_id', 'prompt_id',
                      'response_id', 'api_call_id', 'provider_usage', 'reasoning_content', 'raw_text', 'API_KEY'):
        assert forbidden not in text
    for n, count in [(100, 34), (64, 53)]:
        registry = yaml.safe_load((B / f'docs/challenge{n}_round1_model_registry_holdout_v2.yaml').read_text())
        assert len(registry['models']) == count and registry['models'][-4]['id'] == MID
    table = (B / 'docs/leaderboard.md').read_text()
    assert '| Inception Mercury 3 | 5.0% (5/100) | 9/100 (9.0%) | 96/100 | 88.1 / 18.0 | 11.6 |' in table
    assert '| Inception Mercury 3 | 7.8% (5/64) | 9/64 (14.1%) | 60/64 | 88.1 / 18.0 |' in table
    assert '[Protocol and accounting](mercury3_results.md)' in table
