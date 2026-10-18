import copy
import json
import random
from pathlib import Path

import pytest

from concept_synth.induction.complete_holdouts import complete_task, draw, structure, validate, read_dataset, read_jsonl, sha
from concept_synth.sexpr_parser import parse_sexpr_formula
from concept_synth.target import compute_target_extension

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT/'benchmarks/induction/docs/holdout_v2_policy.json').read_text())


def fixture():
    return {"instanceId": "hard_070", "problem": {"worlds": [{
        "worldId": "train", "domain": [f"a{i}" for i in range(9)], "domainSize": 9,
        "predicates": {"P": ["a0"], "Q": ["a1"], "R": ["(a0, a1)"], "S": []},
        "targetExtension": {"T_true": ["a2"]}}]},
        "problemDescription": {"hiddenTarget": {"formula": "(forall y (or (not (R x y)) (exists z (R y z))))"}}}


def test_regular_sampler_structurally_cannot_balance_hard070():
    gold = parse_sexpr_formula(fixture()['problemDescription']['hiddenTarget']['formula'])
    rng = random.Random(13)
    for _ in range(30):
        model = draw(rng, 9, 'regular', POLICY)
        assert len(compute_target_extension(model, gold).T_true) == 9


def test_deterministic_complete_fallback_no_network(monkeypatch):
    import socket
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **k: pytest.fail('No network'))
    record = fixture()
    old = {'task_id': 'hard_070', 'worlds': []}
    row, trace = complete_task(record, old, POLICY)
    row2, trace2 = complete_task(record, old, POLICY)
    assert row == row2 and trace == trace2
    assert trace['sampling']['regular']['proposals'] == 0
    assert trace['sampling']['fallback']['accepted'] == 5
    assert validate([record], [old], [row])['worlds'] == 5


def test_prefix_preservation_and_no_aliasing():
    record = fixture()
    empty = {'task_id': 'hard_070', 'worlds': []}
    row, _ = complete_task(record, empty, POLICY)
    old = {'task_id': 'hard_070', 'worlds': copy.deepcopy(row['worlds'][:3])}
    before = copy.deepcopy(old)
    filled, _ = complete_task(record, old, POLICY)
    assert filled['worlds'][:3] == before['worlds'] and old == before
    assert validate([record], [old], [filled])['added'] == 2


def test_full_legacy_never_resampled():
    record = fixture()
    row, _ = complete_task(record, {'task_id': 'hard_070', 'worlds': []}, POLICY)
    preserved, trace = complete_task(record, row, POLICY)
    assert preserved['worlds'] == row['worlds']
    assert trace['added'] == 0
    assert all(x['proposals'] == 0 for x in trace['sampling'].values())


def test_budget_exhaustion_fails_closed():
    policy = {**POLICY, 'fallback_budget_per_incomplete_task': 0}
    with pytest.raises(RuntimeError, match='no release written'):
        complete_task(fixture(), {'task_id': 'hard_070', 'worlds': []}, policy)


def test_structure_ignores_ids_labels_and_predicate_representation():
    world = fixture()['problem']['worlds'][0]
    other = copy.deepcopy(world)
    other['worldId'] = 'different'
    other['targetExtension']['T_true'] = []
    other['predicates']['P'] = {'true': ['a0']}
    assert structure(world) == structure(other)


def test_bad_label_and_duplicates_rejected():
    record = fixture()
    old = {'task_id': 'hard_070', 'worlds': []}
    row, _ = complete_task(record, old, POLICY)
    broken = copy.deepcopy(row)
    broken['worlds'][0]['targetExtension']['T_true'] = []
    with pytest.raises(AssertionError):
        validate([record], [old], [broken])
    broken = copy.deepcopy(row)
    broken['worlds'][1] = copy.deepcopy(broken['worlds'][0])
    with pytest.raises(AssertionError):
        validate([record], [old], [broken])


def test_frozen_release_hashes_counts_projection_and_prompt_equality():
    from concept_synth.induction.prompting import build_prompt
    base = ROOT/'benchmarks/induction/data'
    out = base/'challenge_holdouts_v2'
    manifest = json.loads((out/'manifest.json').read_text())
    for name, artifact in manifest['artifacts'].items():
        assert sha(out/name) == artifact['sha256']
        assert (out/name).stat().st_size == artifact['bytes']
    assert sha(ROOT/'src/concept_synth/induction/complete_holdouts.py') == manifest['generator_sha256']
    old = read_dataset(base/'induction_fullobs_challenge100_v1.yaml.gz')
    new = read_dataset(out/'induction_fullobs_challenge100_v2.yaml.gz')
    assert old == new
    assert read_dataset(out/'induction_fullobs_challenge64_v2.yaml.gz') == new[:64]
    assert all(build_prompt(a) == build_prompt(b) for a, b in zip(old, new))
    rows = read_jsonl(out/'challenge100_holdouts_v2.jsonl')
    assert read_jsonl(out/'challenge64_holdouts_v2.jsonl') == rows[:64]
    result = validate(new, read_jsonl(base/'challenge100_legacy_holdouts_v1.jsonl'), rows)
    assert result == manifest['validation']
    assert result['preserved'] == 449 and result['added'] == 51
