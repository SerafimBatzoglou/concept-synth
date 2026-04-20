from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "benchmarks" / "abduction" / "data" / "abd_instances_v1.yaml.gz"
HOLDOUTS_PATH = REPO_ROOT / "benchmarks" / "abduction" / "data" / "abd_holdouts_v1.jsonl.gz"
PREDICTIONS_PATH = REPO_ROOT / "benchmarks" / "abduction" / "predictions" / "abd_predictions_v1.jsonl.gz"
FROZEN_EVAL_PATH = REPO_ROOT / "benchmarks" / "abduction" / "eval" / "abd_eval_cache_v1.jsonl"
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
TEST_ENV = {
    **os.environ,
    "PYTHONPATH": str(SRC_PATH)
    if not os.environ.get("PYTHONPATH")
    else f"{SRC_PATH}{os.pathsep}{os.environ['PYTHONPATH']}",
}

from concept_synth.abduction.benchmark_io import load_problem_index
from concept_synth.abduction.evaluate_abd_b1 import evaluate_on_holdouts, load_holdouts_from_jsonl


def _iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize_eval_row(row: dict) -> dict:
    prediction = row["prediction"]
    return {
        "instance_id": row["instance_id"],
        "scenario": row["scenario"],
        "theory": row["theory"],
        "difficulty": row["difficulty"],
        "num_train_worlds": row["num_train_worlds"],
        "num_holdout_worlds": row["num_holdout_worlds"],
        "model_id": row["model_id"],
        "prediction": {
            "formula": prediction["formula"],
            "original_formula": prediction.get("original_formula"),
            "parse_ok": prediction["parse_ok"],
            "parse_error": prediction.get("parse_error"),
            "ast_size": prediction.get("ast_size"),
            "auto_closed_parens": prediction.get("auto_closed_parens"),
        },
        "train_eval": row["train_eval"],
        "holdout_eval": row["holdout_eval"],
        "notes": row.get("notes"),
    }


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
        env=TEST_ENV,
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
        env=TEST_ENV,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["dataset_path"] == str(DATASET_PATH)


def test_evaluate_partial_and_skeptical_match_frozen_cache(tmp_path: Path) -> None:
    cases = [
        ("ABD_PARTIAL_TH10_000", "claude-opus-4-5-20251101"),
        ("abd_skeptical_v2_TH11_00", "claude-opus-4-5-20251101"),
    ]
    wanted = set(cases)

    selected_rows = [
        row for row in _iter_jsonl(PREDICTIONS_PATH) if (row["instanceId"], row["model"]) in wanted
    ]
    assert len(selected_rows) == len(cases)

    predictions_path = tmp_path / "selected_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected_rows),
        encoding="utf-8",
    )

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
            "--predictions",
            str(predictions_path),
            "--overwrite",
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=TEST_ENV,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    actual_rows = {
        (row["instance_id"], row["model_id"]): _normalize_eval_row(row) for row in _iter_jsonl(output_path)
    }
    frozen_rows = {
        (row["instance_id"], row["model_id"]): _normalize_eval_row(row)
        for row in _iter_jsonl(FROZEN_EVAL_PATH)
        if (row["instance_id"], row["model_id"]) in wanted
    }

    assert actual_rows == frozen_rows


def test_holdout_eval_passes_timeout_to_checker() -> None:
    problem = load_problem_index(str(DATASET_PATH))["ABD_PARTIAL_TH10_000"]
    holdout_world = load_holdouts_from_jsonl(str(HOLDOUTS_PATH))["ABD_PARTIAL_TH10_000"][0]

    with patch(
        "concept_synth.abduction.evaluate_abd_b1.check_abd_partial_validity",
        return_value=SimpleNamespace(valid=False, reason="stub"),
    ) as mock_check:
        evaluate_on_holdouts(
            problem,
            "(exists y (and (S x y) (P y)))",
            [holdout_world],
            timeout_ms=12345,
        )

    assert mock_check.call_count == 1
    args, kwargs = mock_check.call_args
    assert len(args) == 3
    assert kwargs["timeout_ms"] == 12345


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
        env={**TEST_ENV, "OUT_DIR": str(outdir)},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert any(outdir.glob("*.tex"))
    assert any(outdir.glob("*.csv"))
