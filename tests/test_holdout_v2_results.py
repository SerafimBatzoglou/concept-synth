"""Validate versioned stored scores without rerunning model-response evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import jsonschema
import yaml

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/'benchmarks/induction'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def test_affected_only_scores_and_eligibility():
    patch=rows(BENCH/'eval/challenge_holdouts_v2_affected_results.jsonl')
    audit=read(BENCH/'docs/holdout_v2_evaluation_verification.json')
    assert len(patch)==463 and len({(r['model_id'],r['task_id']) for r in patch})==463
    assert len({r['model_id'] for r in patch})==48
    assert {r['task_id'] for r in patch}==set(audit['affected_task_ids'])
    assert len(audit['affected_task_ids'])==14
    assert sum(r['new_holdout_score_calls'] for r in patch)==133
    assert sum(not r['holdout_evaluated'] for r in patch)==330
    for row in patch:
        assert row['holdout_evaluated']==row['train_correct']
        assert row['leaderboard_credit']==any(c['v2_correct'] for c in row['candidate_results'])
        if not row['train_correct']:
            assert row['v2_holdout_correct'] is None and not row['candidate_results']
        for candidate in row['candidate_results']:
            assert len(candidate['world_results'])==5
            assert candidate['v2_correct']==all(w['mismatch_count']==0 for w in candidate['world_results'])
    gains=[r for r in patch if r['leaderboard_credit'] and r['old_holdout_correct'] is not True]
    assert len(gains)==59 and all(r['previous_worlds']==0 for r in gains)
    assert not any(r['old_holdout_correct'] is True and not r['leaderboard_credit'] for r in patch)
    assert 'deepseek-v4.1-flash-1017' not in {r['model_id'] for r in patch}

def test_c64_copy_only_projection_and_schema():
    old=rows(BENCH/'eval/induction_challenge64_round1_holdout_eval_cache_v1.jsonl')
    new=rows(BENCH/'eval/induction_challenge64_round1_holdout_eval_cache_v2.jsonl')
    patch={(r['model_id'],r['task_id']):r for r in rows(BENCH/'eval/challenge_holdouts_v2_affected_results.jsonl') if r['subset']=='challenge64'}
    validator=jsonschema.Draft202012Validator(read(BENCH/'schemas/eval_cache_record.schema.json'))
    assert len(old)==3072 and len(new)==3264
    # The first release's 48 models are unchanged; supplements are additive.
    assert {r['model_id'] for r in new[3072:3136]}=={'deepseek-v4.1-flash-1017'}
    assert {r['model_id'] for r in new[3136:3200]}=={'inception/mercury-3'}
    assert {r['model_id'] for r in new[3200:]}=={'kimi-k3.1-pro'}
    unchanged=0
    for before,after in zip(old,new):
        key=(before['model_id'],before['instance_id'])
        assert key==(after['model_id'],after['instance_id'])
        if key not in patch:
            assert before==after
            unchanged+=1
        else:
            p=patch[key]
            validator.validate(after)
            assert after['valid']==p['leaderboard_credit']
            assert after['metadata']['holdout_outcome']==p['v2_holdout_correct']
            assert after['completed']==p['holdout_evaluated']
            assert after['metadata']['holdout_world_count']==5
            assert before['prediction']==after['prediction']
    assert unchanged==2928

def test_v2_totals_registry_and_preserved_training():
    old=yaml.safe_load((BENCH/'docs/challenge100_round1_model_registry.yaml').read_text())
    new=yaml.safe_load((BENCH/'docs/challenge100_round1_model_registry_holdout_v2.yaml').read_text())
    summary={m['model_id']:m for m in read(BENCH/'eval/challenge_holdouts_v2_model_summary.json')['models']}
    assert len(summary)==48 and len(new['models'])==32
    holds=rows(BENCH/'eval/induction_challenge64_round1_holdout_eval_cache_v2.jsonl')
    for before,after in zip(old['models'],new['models']):
        mid=before['id'];m=summary[mid]
        reverted=copy.deepcopy(after)
        for key in ('holdout_correct','holdout_evaluable'):
            reverted['benchmarked36'][key]=before['benchmarked36'][key]
        assert reverted==before
        assert after['benchmarked36']['holdout_evaluable']==after['benchmarked36']['correct']
        h64=sum(r['valid'] for r in holds if r['model_id']==after['challenge64']['registry_id'])
        assert h64==m['challenge64']['v2_holdout_correct']
        assert h64+after['benchmarked36']['holdout_correct']==m['challenge100']['v2_holdout_correct']
    assert summary['gpt-6-astra']['challenge100']['v2_holdout_correct']==85
    assert summary['gpt-6.1-sol']['challenge100']['v2_holdout_correct']==67
    assert summary['gemini-4-argon']['challenge100']['v2_holdout_correct']==40

def test_v2_tables_reproduce_without_scoring(tmp_path):
    subprocess.run([sys.executable,str(BENCH/'analysis/make_holdout_v2_leaderboards.py'),
                    '--output-dir',str(tmp_path)],check=True)
    for name in ('leaderboard.md','challenge100_leaderboard.svg','challenge64_round1_results.md'):
        assert (tmp_path/name).read_bytes()==(BENCH/'docs'/name).read_bytes()
    text=(tmp_path/'leaderboard.md').read_text()
    assert '**Holdouts v2:**' in text and 'only the 14 changed tasks' in text
    assert '| GPT-6 Astra | 85.0% (85/100)' in text
    assert '| Gemini 4 Argon | 40.0% (40/100)' in text

def test_release_hashes_and_no_private_payloads():
    manifest=read(BENCH/'holdout_v2_evaluation_release_manifest.json')
    assert manifest['version']=='challenge-holdouts-v2'
    for artifact in manifest['artifacts']:
        p=ROOT/artifact['path']
        assert sha(p)==artifact['sha256'] and p.stat().st_size==artifact['sizeBytes']
    for rel,h in manifest['preservedV1SourceHashes'].items():assert sha(ROOT/rel)==h
    for name in ('eval/challenge_holdouts_v2_affected_results.jsonl','eval/challenge_holdouts_v2_model_summary.json',
                 'eval/induction_challenge64_round1_holdout_eval_cache_v2.jsonl','docs/holdout_v2_evaluation_verification.json'):
        text=(BENCH/name).read_text()
        for forbidden in ('/Users/','/tmp/','pipeline.sqlite','raw_response','reasoning_content','api_key'):
            assert forbidden not in text
