#!/usr/bin/env python3
"""Refresh checksums after copy-only v2 publication. Never evaluates formulas."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BENCH=ROOT/'benchmarks/induction'
MANIFEST=BENCH/'holdout_v2_evaluation_release_manifest.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def artifact(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p),'sizeBytes':p.stat().st_size}
def save(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

def main():
    if MANIFEST.exists():
        preserved=json.loads(MANIFEST.read_text())['preservedV1SourceHashes']
        for rel,h in preserved.items():
            if sha(ROOT/rel)!=h:raise ValueError(f'Historical source changed: {rel}')
    else:
        protected=[p for folder in ('data','eval','predictions') for p in (BENCH/folder).iterdir()
                   if p.is_file() and '_v2' not in p.name]
        protected += [BENCH/f'docs/challenge{n}_round1_model_registry.yaml' for n in (100,64)]
        preserved={str(p.relative_to(ROOT)):sha(p) for p in sorted(protected)}
    # Historical score files/counts remain v1. These catalog manifests also list
    # maintained documentation/renderers, so refresh only their file checksums.
    for name in ('challenge100_round1_release_manifest.json','challenge64_round1_release_manifest.json',
                 'challenge_holdouts_v2_release_manifest.json'):
        p=BENCH/name;data=json.loads(p.read_text())
        data['artifacts']=[artifact(ROOT/a['path']) for a in data['artifacts']]
        if name!='challenge_holdouts_v2_release_manifest.json':
            data['currentLeaderboardVersion']='challenge-holdouts-v2'
            data['currentLeaderboardManifest']='holdout_v2_evaluation_release_manifest.json'
            data['historicalScoreFilesPreserved']=True
        save(p,data)
    paths=[
        BENCH/'eval/challenge_holdouts_v2_affected_results.jsonl',
        BENCH/'eval/challenge_holdouts_v2_model_summary.json',
        BENCH/'eval/induction_challenge64_round1_holdout_eval_cache_v2.jsonl',
        *[BENCH/f'docs/challenge{n}_round1_model_registry_holdout_v2.yaml' for n in (100,64)],
        *[BENCH/'docs'/n for n in ('leaderboard.md','challenge100_leaderboard.svg','challenge64_round1_results.md',
            'leaderboard_holdout_v1.md','challenge100_leaderboard_holdout_v1.svg','challenge64_round1_results_holdout_v1.md',
            'challenge_holdouts_v2.md','holdout_v2_evaluation_verification.json')],
        *[BENCH/'analysis'/n for n in ('make_holdout_v2_leaderboards.py','make_challenge100_leaderboard.py','make_challenge64_round1_table.py')],
        BENCH/'scripts/refresh_holdout_v2_release.py',ROOT/'tests/test_holdout_v2_results.py',
        BENCH/'data/challenge_holdouts_v2/manifest.json',ROOT/'README.md',BENCH/'README.md',
        BENCH/'eval/deepseek_flash1017_challenge100_holdout_v2.json',
        BENCH/'eval/induction_challenge64_round1_eval_cache_v2.jsonl',
        BENCH/'eval/challenge100_output_token_usage_v2.json',
        BENCH/'predictions/deepseek_flash1017_challenge100_predictions_v2.jsonl.gz',
        BENCH/'docs/deepseek_flash1017_results.md',ROOT/'tests/test_flash1017_results.py',
        ROOT/'tests/test_challenge100_token_table.py',
        BENCH/'eval/mercury3_challenge100_holdout_v2.json',
        BENCH/'predictions/mercury3_challenge100_predictions_v2.jsonl.gz',
        BENCH/'docs/mercury3_results.md',ROOT/'tests/test_mercury3_results.py',
    ]
    save(MANIFEST,{'schemaVersion':'induction_holdout_evaluation_release_manifest_v2',
        'version':'challenge-holdouts-v2','date':'2026-10-19','license':'MIT',
        'scope':'Verified completed configurations: 50 Challenge64, 31 Challenge100, including final Flash-1017 and Mercury 3.',
        'currentConfigurations':{'challenge64':50,'challenge100':31},
        'counts':{'affectedTasks':14,'affectedModelTaskRows':477,'eligibleHoldoutEvaluations':136,
                  'skippedIneligibleRows':341,'unchangedTasksNotRescored':86,'newlyMeasurablePasses':60},
        'initialRelease':{'challenge64Configurations':48,'challenge100Configurations':29,
            'affectedRows':463,'eligibleHoldoutEvaluations':133,'skippedIneligibleRows':330,
            'evidence':'benchmarks/induction/docs/holdout_v2_evaluation_verification.json'},
        'flash1017Supplement':{'affectedRows':14,'eligibleHoldoutEvaluations':3,'skippedIneligibleRows':11,
            'report':'benchmarks/induction/eval/deepseek_flash1017_challenge100_holdout_v2.json',
            'previous48ModelsPreserved':True,'initialV2TrainingRowsCopiedExactly':3072},
        'mercury3Supplement':{'selectedTasks':100,'evaluable':96,'strictEvaluable':95,
            'trainCorrect':9,'holdoutCorrect':5,'physicalRequests':222,
            'holdoutV2UsedFromStart':True,'previous49ModelsPreserved':True,
            'previousV2TrainingAndHoldoutRowsCopiedExactly':3136,
            'report':'benchmarks/induction/eval/mercury3_challenge100_holdout_v2.json'},
        'modelApiCalls':0,'publicationScorerCalls':0,'preservedV1SourceHashes':preserved,
        'artifacts':[artifact(p) for p in paths]})
    print(f'Refreshed {MANIFEST.name}; historical source hashes verified.')

if __name__=='__main__':main()
