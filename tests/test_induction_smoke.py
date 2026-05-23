from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

TEST_ENV = {
    **os.environ,
    "PYTHONPATH": str(SRC_PATH)
    if not os.environ.get("PYTHONPATH")
    else f"{SRC_PATH}{os.pathsep}{os.environ['PYTHONPATH']}",
}

BENCH_ROOT = REPO_ROOT / "benchmarks" / "induction"
DATASET_PATHS = {
    "FullObs": BENCH_ROOT / "data" / "induction_fullobs_v1.yaml.gz",
    "CI": BENCH_ROOT / "data" / "induction_ci_v1.yaml.gz",
    "EC": BENCH_ROOT / "data" / "induction_ec_v1.yaml.gz",
}
PREDICTIONS_PATH = BENCH_ROOT / "predictions" / "induction_predictions_v1.jsonl.gz"
FROZEN_EVAL_PATH = BENCH_ROOT / "eval" / "induction_eval_cache_v1.jsonl"

from concept_synth.induction.evaluator import evaluate_prediction
from concept_synth.induction.generator import generate_task_bundle, validate_generated_records


def _load_first_record(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return yaml.safe_load(handle)[0]


def _iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def test_build_prompt_smoke_for_each_task() -> None:
    cases = [
        (DATASET_PATHS["FullObs"], "simple_001", "Training Worlds"),
        (DATASET_PATHS["CI"], "C_core_001", "YES Worlds"),
        (DATASET_PATHS["EC"], "E_core_0001", "Partial Observations"),
    ]
    for dataset, instance_id, expected in cases:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "concept_synth.induction.cli",
                "build-prompt",
                "--dataset",
                str(dataset),
                "--instance-id",
                instance_id,
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            env=TEST_ENV,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert expected in proc.stdout
        assert "Output Format" in proc.stdout


def test_gold_formulas_validate_on_first_record_of_each_task() -> None:
    for task, dataset_path in DATASET_PATHS.items():
        record = _load_first_record(dataset_path)
        formula = record["problemDescription"]["hiddenTarget"]["formula"]
        row = evaluate_prediction(
            record,
            {"instanceId": record["instanceId"], "model": "gold", "extractedFormula": formula},
            dataset_path=str(dataset_path),
            run_id="test_gold",
            timeout_ms=30000,
            compute_diagnostics=False,
        )
        assert row["parse_ok"], task
        assert row["valid"], task
        assert row["gold_plus_0"], task


def test_evaluate_smoke_uses_canonical_predictions(tmp_path: Path) -> None:
    output_path = tmp_path / "eval.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "concept_synth.induction.cli",
            "evaluate",
            "--dataset",
            str(DATASET_PATHS["FullObs"]),
            "--predictions",
            str(PREDICTIONS_PATH),
            "--limit",
            "1",
            "--no-diagnostics",
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
    assert rows[0]["dataset_path"] == str(DATASET_PATHS["FullObs"])
    assert rows[0]["task"] == "FullObs"


def test_table_regeneration_smoke(tmp_path: Path) -> None:
    outdir = tmp_path / "tables"
    proc = subprocess.run(
        [
            "./benchmarks/induction/scripts/reproduce_tables.sh",
            str(FROZEN_EVAL_PATH),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={**TEST_ENV, "OUT_DIR": str(outdir)},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (outdir / "induction_summary.tex").exists()
    assert (outdir / "induction_summary.csv").exists()


def test_public_generator_produces_valid_records_for_all_tasks() -> None:
    records = generate_task_bundle(
        task="all",
        n=2,
        seed=123,
        worlds=3,
        yes_worlds=3,
        no_worlds=1,
        domain_min=4,
        domain_max=5,
        unknown_rate=0.3,
    )

    assert len(records) == 6
    assert {record["task"] for record in records} == {"FullObs", "CI", "EC"}
    assert not validate_generated_records(records, timeout_ms=30000)
    for record in records:
        assert record["schemaVersion"] == "induction_benchmark_record_v1"
        assert record["problemDescription"]["benchmark_name"] == "induction_public_generator"
        assert record["problemDescription"]["hiddenTarget"]["formula"]
        for world in record["problem"]["worlds"]:
            assert "fullPredicates" not in world
            assert "targetExtension" in world


def test_generate_cli_round_trip_prompt_and_eval(tmp_path: Path) -> None:
    generated_path = tmp_path / "generated.yaml"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "concept_synth.induction.cli",
            "generate",
            "--task",
            "all",
            "--n",
            "1",
            "--seed",
            "456",
            "--validate",
            "--worlds",
            "2",
            "--yes-worlds",
            "2",
            "--no-worlds",
            "1",
            "--domain-min",
            "4",
            "--domain-max",
            "5",
            "--output",
            str(generated_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=TEST_ENV,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    prompt_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "concept_synth.induction.cli",
            "build-prompt",
            "--dataset",
            str(generated_path),
            "--index",
            "0",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=TEST_ENV,
        check=False,
    )
    assert prompt_proc.returncode == 0, prompt_proc.stderr
    assert "Problem Instance" in prompt_proc.stdout

    with generated_path.open("r", encoding="utf-8") as handle:
        generated_records = yaml.safe_load(handle)
    assert len(generated_records) == 3
    for record in generated_records:
        formula = record["problemDescription"]["hiddenTarget"]["formula"]
        row = evaluate_prediction(
            record,
            {"instanceId": record["instanceId"], "model": "gold", "extractedFormula": formula},
            dataset_path=str(generated_path),
            run_id="test_generated_cli",
            timeout_ms=30000,
            compute_diagnostics=False,
        )
        assert row["valid"], record["instanceId"]
