from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "benchmarks" / "abduction" / "data" / "abd_instances_v1.yaml.gz"
HOLDOUTS_PATH = REPO_ROOT / "benchmarks" / "abduction" / "data" / "abd_holdouts_v1.jsonl.gz"
FROZEN_EVAL_PATH = REPO_ROOT / "benchmarks" / "abduction" / "eval" / "abd_eval_cache_v1.jsonl"


def test_build_prompt_smoke() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "concept_synth.abduction.cli",
            "build-prompt",
            "--dataset",
            str(DATASET_PATH),
            "--instance-id",
            "ABD_FULL_TH10_000",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Training Worlds" in proc.stdout
    assert "Ab" in proc.stdout


def test_evaluate_smoke_uses_canonical_predictions(tmp_path: Path) -> None:
    output_path = tmp_path / "eval.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "concept_synth.abduction.cli",
            "evaluate",
            "--dataset",
            str(DATASET_PATH),
            "--holdouts",
            str(HOLDOUTS_PATH),
            "--model",
            "gpt-5.4",
            "--limit",
            "1",
            "--timeout-ms",
            "10000",
            "--overwrite",
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["dataset_path"] == str(DATASET_PATH)


def test_table_regeneration_smoke(tmp_path: Path) -> None:
    outdir = tmp_path / "tables"
    proc = subprocess.run(
        [
            "./benchmarks/abduction/scripts/reproduce_tables.sh",
            str(FROZEN_EVAL_PATH),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "OUT_DIR": str(outdir)},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert any(outdir.glob("*.tex"))
    assert any(outdir.glob("*.csv"))
