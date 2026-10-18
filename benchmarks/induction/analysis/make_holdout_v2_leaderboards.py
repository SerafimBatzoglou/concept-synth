#!/usr/bin/env python3
"""Render current v2 leaderboards from frozen scores; no evaluation or API calls."""
import argparse
from pathlib import Path
import yaml

import make_challenge100_leaderboard as combined
import make_challenge64_round1_table as projection

BENCH=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BENCH/'docs')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    r100=BENCH/'docs/challenge100_round1_model_registry_holdout_v2.yaml'
    r64=BENCH/'docs/challenge64_round1_model_registry_holdout_v2.yaml'
    train=BENCH/'eval/induction_challenge64_round1_eval_cache_v2.jsonl'
    hold=BENCH/'eval/induction_challenge64_round1_holdout_eval_cache_v2.jsonl'
    tokens=BENCH/'eval/challenge100_output_token_usage_v2.json'
    text=combined.render(challenge100_registry_path=r100,challenge64_registry_path=r64,
                         challenge64_eval_path=train,challenge64_holdout_path=hold,token_usage_path=tokens)
    (args.output_dir/'leaderboard.md').write_text(text)
    registry=yaml.safe_load(r100.read_text())
    table=combined.render_challenge100(registry=registry,eval_path=train,holdout_path=hold)
    aligned=combined.token_aligned_rows(table,registry,combined.load_token_usage(tokens,registry))
    (args.output_dir/'challenge100_leaderboard.svg').write_text(combined.render_challenge100_svg(aligned))
    text=projection.render(dataset_path=BENCH/'data/challenge_holdouts_v2/induction_fullobs_challenge64_v2.yaml.gz',
                           eval_path=train,holdout_path=hold,registry_path=r64)
    (args.output_dir/'challenge64_round1_results.md').write_text(text)

if __name__=='__main__':main()
