import importlib.util
import json
from pathlib import Path

import yaml

BENCH = Path(__file__).resolve().parents[1] / 'benchmarks/induction'


def module(name):
    spec = importlib.util.spec_from_file_location(name, BENCH / 'analysis' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def table_rows(text):
    return [line for line in text.splitlines() if line.startswith('| ') and not line.startswith('| Model |')]


def test_published_tables_use_total_denominator_and_holdout_sort():
    render = module('make_challenge100_leaderboard')
    eval_path = BENCH / 'eval/induction_challenge64_round1_eval_cache_v1.jsonl'
    holdout_path = BENCH / 'eval/induction_challenge64_round1_holdout_eval_cache_v1.jsonl'
    registry64 = BENCH / 'docs/challenge64_round1_model_registry.yaml'
    c100 = yaml.safe_load((BENCH / 'docs/challenge100_round1_model_registry.yaml').read_text())
    c64 = yaml.safe_load(registry64.read_text())
    rows100 = render.render_challenge100(registry=c100, eval_path=eval_path, holdout_path=holdout_path)
    rows64 = render.render_challenge64(registry=c64, eval_path=eval_path, holdout_path=holdout_path)
    for rows, denominator in [(rows100, 100), (rows64, 64)]:
        order = []
        for row in rows:
            name, holdout, train, evaluable, complexity = [s.strip() for s in row.strip('|').split('|')]
            count = int(holdout.split('(')[1].split('/')[0])
            assert holdout == f'{100 * count / denominator:.1f}% ({count}/{denominator})'
            order.append((-count, -int(train.split('/')[0]), -int(evaluable.split('/')[0]), name))
        assert order == sorted(order)
    assert '| Claude Sonnet 5.5 | 37.0% (37/100) | 44/100 (44.0%) | 100/100 | 19.0 / 17.0 |' in rows100
    assert '| Claude Sonnet 5.5 | 40.6% (26/64) | 32/64 (50.0%) | 64/64 | 19.6 / 18.0 |' in rows64
    assert rows100.index(next(r for r in rows100 if 'Claude Sonnet 5.5' in r)) < rows100.index(next(r for r in rows100 if 'GPT-6 Sol' in r))
    standalone = module('make_challenge64_round1_table').render(
        dataset_path=BENCH / 'data/induction_fullobs_challenge64_v1.yaml.gz',
        eval_path=eval_path, holdout_path=holdout_path, registry_path=registry64)
    assert table_rows(standalone) == rows64
    assert standalone == (BENCH / 'docs/challenge64_round1_results.md').read_text()
    for path in ['docs/leaderboard.md', 'docs/challenge64_round1_results.md']:
        text = (BENCH / path).read_text()
        assert '| Model | Holdout Correct %<br>(all problems) | Train Correct | Evaluable |' in text
        assert 'missing' in text and 'unknown' in text


def test_rank_is_not_conditional_rate_and_missing_holdouts_get_no_credit(tmp_path):
    # A has 100% conditional holdout success, B only 50%, but B solves more
    # total problems. C's absent holdout outcome is not a verified success.
    configs = [('A', 2, 1, 1), ('B', 4, 2, 4), ('C', 1, 0, 0)]
    evals, holds, models = [], [], []
    for name, train, successes, available in configs:
        models.append({'id': name, 'display_name': name})
        for i in range(64):
            evals.append({'model_id': name, 'parse_ok': True, 'valid': i < train,
                          'prediction': {'ast_size': 10}})
            holds.append({'model_id': name, 'valid': i < successes,
                          'metadata': {'eligible_train_valid': i < train, 'holdout_available': i < available}})
    ep, hp = tmp_path / 'eval.jsonl', tmp_path / 'holdout.jsonl'
    ep.write_text('\n'.join(map(json.dumps, evals)))
    hp.write_text('\n'.join(map(json.dumps, holds)))
    original = hp.read_bytes()
    rows = module('make_challenge100_leaderboard').render_challenge64(
        registry={'models': models}, eval_path=ep, holdout_path=hp)
    assert [r.split('|')[1].strip() for r in rows] == ['B', 'A', 'C']
    assert '| B | 3.1% (2/64) |' in rows[0]
    assert '| C | 0.0% (0/64) |' in rows[2]
    assert hp.read_bytes() == original
