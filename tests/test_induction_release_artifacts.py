from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "benchmarks" / "induction"
DATASET_PATHS = {
    "FullObs": BENCH_ROOT / "data" / "induction_fullobs_v1.yaml.gz",
    "CI": BENCH_ROOT / "data" / "induction_ci_v1.yaml.gz",
    "EC": BENCH_ROOT / "data" / "induction_ec_v1.yaml.gz",
}
PREDICTIONS_PATH = BENCH_ROOT / "predictions" / "induction_predictions_v1.jsonl.gz"
EVAL_PATH = BENCH_ROOT / "eval" / "induction_eval_cache_v1.jsonl"
META_PATH = BENCH_ROOT / "eval" / "induction_eval_cache_v1.meta.json"
MANIFEST_PATH = BENCH_ROOT / "release_manifest.json"


def _load_json_schema(path: Path) -> Draft202012Validator:
    return Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))


def _iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_canonical_files_exist() -> None:
    paths = [*DATASET_PATHS.values(), PREDICTIONS_PATH, EVAL_PATH, META_PATH, MANIFEST_PATH]
    paths.extend((BENCH_ROOT / "schemas").glob("*.schema.json"))
    for path in paths:
        assert path.exists(), path


def test_benchmark_records_match_schema_and_are_sanitized() -> None:
    validator = _load_json_schema(BENCH_ROOT / "schemas" / "benchmark_record.schema.json")
    expected_counts = {"FullObs": 375, "CI": 200, "EC": 200}

    for task, path in DATASET_PATHS.items():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            records = yaml.safe_load(handle)

        assert isinstance(records, list)
        assert len(records) == expected_counts[task]
        for record in records:
            validator.validate(record)
            assert record["task"] == task
            assert record["instanceId"] == record["problem"]["instanceId"]
            assert "llmResults" not in record
            assert "modelResponses" not in record
            assert "rawResponse" not in json.dumps(record)
            assert "evaluation" not in json.dumps(record)
            for world in record["problem"]["worlds"]:
                assert "fullPredicates" not in world


def test_predictions_match_schema_and_keep_model_record_details() -> None:
    validator = _load_json_schema(BENCH_ROOT / "schemas" / "prediction_record.schema.json")
    count = 0
    tasks = Counter()
    has_raw_response = False
    has_timestamp = False
    has_thinking = False

    for row in _iter_jsonl(PREDICTIONS_PATH):
        validator.validate(row)
        assert "evaluation" not in row
        count += 1
        tasks[row["task"]] += 1
        has_raw_response = has_raw_response or bool(row.get("rawResponse"))
        has_timestamp = has_timestamp or bool(row.get("timestamp"))
        has_thinking = has_thinking or bool(row.get("thinking"))

    assert count == 11413
    assert tasks == {"FullObs": 5561, "CI": 2939, "EC": 2913}
    assert has_raw_response
    assert has_timestamp
    assert has_thinking


def test_eval_cache_is_sanitized() -> None:
    validator = _load_json_schema(BENCH_ROOT / "schemas" / "eval_cache_record.schema.json")
    count = 0
    tasks = Counter()

    for row in _iter_jsonl(EVAL_PATH):
        validator.validate(row)
        assert row["dataset_path"].startswith("benchmarks/induction/data/")
        assert "/Users/" not in row["dataset_path"]
        assert row["run_id"] == "released_induction_eval_cache_v1"
        count += 1
        tasks[row["task"]] += 1

    assert count == 11413
    assert tasks == {"FullObs": 5561, "CI": 2939, "EC": 2913}


def test_manifest_hashes_and_counts() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["schemaVersion"] == "induction_release_manifest_v1"
    assert manifest["releaseTag"] == "induction-v1.0"
    assert manifest["license"] == "MIT"
    assert manifest["counts"]["benchmarkInstances"] == 775
    assert manifest["counts"]["benchmarkInstancesByTask"] == {"CI": 200, "EC": 200, "FullObs": 375}
    assert manifest["counts"]["predictionRows"] == 11413
    assert manifest["counts"]["evalRows"] == 11413

    for artifact in manifest["artifacts"]:
        path = REPO_ROOT / artifact["path"]
        assert path.exists(), artifact["path"]
        assert path.stat().st_size == artifact["sizeBytes"]
        assert _sha256(path) == artifact["sha256"]
