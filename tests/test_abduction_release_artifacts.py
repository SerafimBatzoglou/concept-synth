from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "benchmarks" / "abduction"
DATASET_PATH = BENCH_ROOT / "data" / "abd_instances_v1.yaml.gz"
HOLDOUTS_PATH = BENCH_ROOT / "data" / "abd_holdouts_v1.jsonl.gz"
PREDICTIONS_PATH = BENCH_ROOT / "predictions" / "abd_predictions_v1.jsonl.gz"
EVAL_PATH = BENCH_ROOT / "eval" / "abd_eval_cache_v1.jsonl"
META_PATH = BENCH_ROOT / "eval" / "abd_eval_cache_v1.meta.json"
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
    for path in [DATASET_PATH, HOLDOUTS_PATH, PREDICTIONS_PATH, EVAL_PATH, META_PATH, MANIFEST_PATH]:
        assert path.exists(), path


def test_benchmark_records_match_schema() -> None:
    validator = _load_json_schema(BENCH_ROOT / "schemas" / "benchmark_record.schema.json")
    with gzip.open(DATASET_PATH, "rt", encoding="utf-8") as handle:
        records = yaml.safe_load(handle)

    assert isinstance(records, list)
    assert len(records) == 600

    scenario_counts = Counter()
    for record in records:
        validator.validate(record)
        assert "llmResults" not in record
        assert "modelResponses" not in record
        assert record["instanceId"] == record["problem"]["instanceId"]
        assert record["problemDescription"]["instanceId"] == record["instanceId"]
        assert record["problemDescription"]["numTrainWorlds"] == len(record["problem"]["trainWorlds"])
        scenario_counts[record["problem"]["scenario"]] += 1

    assert scenario_counts == {
        "ABD_FULL": 195,
        "ABD_PARTIAL": 243,
        "ABD_SKEPTICAL": 162,
    }


def test_holdout_records_match_schema() -> None:
    validator = _load_json_schema(BENCH_ROOT / "schemas" / "holdout_record.schema.json")
    rows = list(_iter_jsonl(HOLDOUTS_PATH))

    assert len(rows) == 2983
    for row in rows:
        validator.validate(row)

    instance_counts = Counter(row["instanceId"] for row in rows)
    assert len(instance_counts) == 598
    assert min(instance_counts.values()) >= 1
    assert max(instance_counts.values()) <= 5


def test_predictions_match_schema() -> None:
    validator = _load_json_schema(BENCH_ROOT / "schemas" / "prediction_record.schema.json")
    rows = list(_iter_jsonl(PREDICTIONS_PATH))

    assert rows
    assert len(rows) == 7171
    for row in rows:
        validator.validate(row)
        assert "evaluation" not in row


def test_eval_cache_is_sanitized() -> None:
    validator = _load_json_schema(BENCH_ROOT / "schemas" / "eval_cache_record.schema.json")
    rows = list(_iter_jsonl(EVAL_PATH))

    assert len(rows) == 7171
    for row in rows:
        validator.validate(row)
        assert row["dataset_name"] == "abd_instances_v1"
        assert row["dataset_path"] == "benchmarks/abduction/data/abd_instances_v1.yaml.gz"
        assert "results/abduction" not in row["dataset_path"]
        assert row["run_id"] == "released_abd_eval_cache_v1"


def test_manifest_hashes_and_counts() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["releaseTag"] == "abduction-v1.2"
    assert manifest["license"] == "MIT"
    assert manifest["counts"]["benchmarkInstances"] == 600
    assert manifest["counts"]["predictionRows"] == 7171
    assert manifest["counts"]["evalRows"] == 7171
    assert manifest["counts"]["holdoutRows"] == 2983
    assert manifest["counts"]["instancesWithoutHoldouts"] == [
        "ABD_PARTIAL_TH11_062b",
        "ABD_PARTIAL_TH7_087",
    ]

    for artifact in manifest["artifacts"]:
        path = REPO_ROOT / artifact["path"]
        assert path.exists(), artifact["path"]
        assert path.stat().st_size == artifact["sizeBytes"]
        assert _sha256(path) == artifact["sha256"]
