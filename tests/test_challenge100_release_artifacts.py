from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "benchmarks" / "induction"
C64_DATA = BENCH_ROOT / "data" / "induction_fullobs_challenge64_v1.yaml.gz"
NEW36_DATA = BENCH_ROOT / "data" / "induction_fullobs_benchmarked36_v1.yaml.gz"
NEW36_HOLDOUT = BENCH_ROOT / "data" / "induction_fullobs_benchmarked36_generated_iid_holdout_v1.jsonl"
C100_DATA = BENCH_ROOT / "data" / "induction_fullobs_challenge100_v1.yaml.gz"
C100_REGISTRY = BENCH_ROOT / "docs" / "challenge100_round1_model_registry.yaml"
C64_REGISTRY = BENCH_ROOT / "docs" / "challenge64_round1_model_registry.yaml"
C64_EVAL = BENCH_ROOT / "eval" / "induction_challenge64_round1_eval_cache_v1.jsonl"
C64_HOLDOUT = BENCH_ROOT / "eval" / "induction_challenge64_round1_holdout_eval_cache_v1.jsonl"
LEADERBOARD = BENCH_ROOT / "docs" / "leaderboard.md"
TABLE_SCRIPT = BENCH_ROOT / "analysis" / "make_challenge100_leaderboard.py"
MANIFEST = BENCH_ROOT / "challenge100_round1_release_manifest.json"


def load_yaml_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_challenge100_is_the_ordered_disjoint_union() -> None:
    challenge64 = load_yaml_gz(C64_DATA)
    new36 = load_yaml_gz(NEW36_DATA)
    challenge100 = load_yaml_gz(C100_DATA)
    assert len(challenge64) == 64
    assert len(new36) == 36
    assert len(challenge100) == 100
    assert challenge100 == challenge64 + new36
    c64_ids = [row["instanceId"] for row in challenge64]
    new36_ids = [row["instanceId"] for row in new36]
    assert len(set(c64_ids)) == 64
    assert len(set(new36_ids)) == 36
    assert not set(c64_ids) & set(new36_ids)


def test_new36_holdout_sidecar_matches_the_benchmark() -> None:
    new36 = load_yaml_gz(NEW36_DATA)
    records = [
        json.loads(line)
        for line in NEW36_HOLDOUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["task_id"] for record in records] == [row["instanceId"] for row in new36]
    assert sum(bool(record["worlds"]) for record in records) == 29
    assert sum(len(record["worlds"]) for record in records) == 136


def test_challenge100_registry_is_arithmetically_consistent() -> None:
    registry = yaml.safe_load(C100_REGISTRY.read_text(encoding="utf-8"))
    models = registry["models"]
    assert len(models) == 21
    assert len({model["id"] for model in models}) == len(models)
    for model in models:
        c64 = model["challenge64"]
        new36 = model["benchmarked36"]
        c100 = model["challenge100"]
        assert 0 <= c64["correct"] <= c64["evaluable"] <= 64
        assert 0 <= new36["correct"] <= new36["evaluable"] <= 36
        assert len(new36["correct_formula_ast_sizes"]) == new36["correct"]
        assert 0 <= new36["holdout_correct"] <= new36["holdout_evaluable"] <= new36["correct"]
        assert c100["evaluable"] == c64["evaluable"] + new36["evaluable"]
        assert c100["correct"] == c64["correct"] + new36["correct"]


def test_combined_leaderboard_is_reproducible(tmp_path: Path) -> None:
    regenerated = tmp_path / "leaderboard.md"
    subprocess.run(
        [
            sys.executable,
            str(TABLE_SCRIPT),
            "--challenge100-registry", str(C100_REGISTRY),
            "--challenge64-registry", str(C64_REGISTRY),
            "--challenge64-eval", str(C64_EVAL),
            "--challenge64-holdout", str(C64_HOLDOUT),
            "--out", str(regenerated),
        ],
        check=True,
    )
    assert regenerated.read_text(encoding="utf-8") == LEADERBOARD.read_text(encoding="utf-8")


def test_challenge100_manifest_hashes_and_counts() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schemaVersion"] == "induction_challenge100_round1_release_manifest_v1"
    assert manifest["counts"] == {
        "benchmarked36HoldoutTasksWithWorlds": 29,
        "benchmarked36Tasks": 36,
        "challenge100Models": 21,
        "challenge100Tasks": 100,
        "challenge64Tasks": 64,
    }
    for artifact in manifest["artifacts"]:
        path = REPO_ROOT / artifact["path"]
        assert path.exists(), artifact["path"]
        assert path.stat().st_size == artifact["sizeBytes"]
        assert sha256(path) == artifact["sha256"]


def test_deepseek_v41_flash_verified_results_and_projection() -> None:
    report = json.loads((BENCH_ROOT / "eval/deepseek_v4_1_flash_challenge100_round1_report.json").read_text())
    assert report["model"] == "deepseek-v4.1-flash"
    assert len(report["tasks"]) == 100
    assert len({row["task_id"] for row in report["tasks"]}) == 100
    assert report["overall"]["evaluable"] == 97
    assert report["overall"]["correct"] == 11
    assert report["overall"]["holdout"]["available"] == 10
    assert report["overall"]["holdout"]["correct"] == 3
    assert report["frontiers"] == {"post_round1": 100, "symbolic_candidates": 0}
    assert report["settings"]["thinking_effort"] == "max"
    assert report["settings"]["max_output_tokens"] == 384000
    assert report["settings"]["workers"] == 50
    assert report["generation_calls"]["started_once"] == 100
    assert report["generation_calls"]["retries"] == 0
    assert report["provider_terminal"]["output_cap"] == 2
    assert report["provider_terminal"]["provider_errors"] == 1
    rows = [json.loads(line) for line in C64_EVAL.read_text().splitlines()]
    projected = [row for row in rows if row["model_id"] == "deepseek-v4.1-flash"]
    assert len(projected) == 64
    assert sum(row["parse_ok"] for row in projected) == 63
    assert sum(row["valid"] for row in projected) == 11
    serialized = json.dumps(report)
    for private_value in ["/Users/", "pipeline.sqlite", "reasoningTrace", "provider_call_meta", "responseId", "candidate_id"]:
        assert private_value not in serialized


def test_challenge64_manifest_includes_new_projection() -> None:
    manifest = json.loads((BENCH_ROOT / "challenge64_round1_release_manifest.json").read_text())
    registry = yaml.safe_load(C64_REGISTRY.read_text())
    assert set(manifest["includedModels"]) == {model["id"] for model in registry["models"]}
    assert "deepseek-v4.1-flash" in manifest["includedModels"]
    assert all(count == 64 * len(registry["models"]) for count in manifest["counts"].values())
    for artifact in manifest["artifacts"]:
        path = REPO_ROOT / artifact["path"]
        assert path.stat().st_size == artifact["sizeBytes"]
        assert sha256(path) == artifact["sha256"]


@pytest.mark.parametrize("error", ["missing model", "wrong counts", "wrong name"])
def test_leaderboard_rejects_inconsistent_challenge64_projection(tmp_path: Path, error: str) -> None:
    spec = importlib.util.spec_from_file_location("challenge100_table", TABLE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c100 = yaml.safe_load(C100_REGISTRY.read_text(encoding="utf-8"))
    c64 = yaml.safe_load(C64_REGISTRY.read_text(encoding="utf-8"))
    target = c100["models"][0]["challenge64"]["registry_id"]
    if error == "missing model":
        c64["models"] = [model for model in c64["models"] if model["id"] != target]
        message = "missing from Challenge64 registry"
    elif error == "wrong counts":
        c100["models"][0]["challenge64"]["evaluable"] -= 1
        message = "projection counts disagree"
    else:
        c100["models"][0]["display_name"] += " inconsistent"
        message = "inconsistent display name"
    c100_path, c64_path = tmp_path / "c100.yaml", tmp_path / "c64.yaml"
    c100_path.write_text(yaml.safe_dump(c100), encoding="utf-8")
    c64_path.write_text(yaml.safe_dump(c64), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        module.render(
            challenge100_registry_path=c100_path,
            challenge64_registry_path=c64_path,
            challenge64_eval_path=C64_EVAL,
            challenge64_holdout_path=C64_HOLDOUT,
        )
