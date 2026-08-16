from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "benchmarks" / "induction"
C64_DATA = BENCH_ROOT / "data" / "induction_fullobs_challenge64_v1.yaml.gz"
NEW36_DATA = BENCH_ROOT / "data" / "induction_fullobs_benchmarked36_v1.yaml.gz"
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


def test_challenge100_registry_is_arithmetically_consistent() -> None:
    registry = yaml.safe_load(C100_REGISTRY.read_text(encoding="utf-8"))
    models = registry["models"]
    assert len(models) == 15
    assert len({model["id"] for model in models}) == len(models)
    for model in models:
        c64 = model["challenge64"]
        new36 = model["benchmarked36"]
        c100 = model["challenge100"]
        assert 0 <= c64["correct"] <= c64["evaluable"] <= 64
        assert 0 <= new36["correct"] <= new36["evaluable"] <= 36
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
        "benchmarked36Tasks": 36,
        "challenge100Models": 15,
        "challenge100Tasks": 100,
        "challenge64Tasks": 64,
    }
    for artifact in manifest["artifacts"]:
        path = REPO_ROOT / artifact["path"]
        assert path.exists(), artifact["path"]
        assert path.stat().st_size == artifact["sizeBytes"]
        assert sha256(path) == artifact["sha256"]
