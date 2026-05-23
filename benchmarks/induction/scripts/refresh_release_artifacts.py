#!/usr/bin/env python3
"""Refresh INDUCTION release metadata after curated artifact updates."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_ROOT = REPO_ROOT / "benchmarks" / "induction"
ARTIFACT_PATHS = [
    "benchmarks/induction/data/induction_fullobs_v1.yaml.gz",
    "benchmarks/induction/data/induction_ci_v1.yaml.gz",
    "benchmarks/induction/data/induction_ec_v1.yaml.gz",
    "benchmarks/induction/predictions/induction_predictions_v1.jsonl.gz",
    "benchmarks/induction/eval/induction_eval_cache_v1.jsonl",
    "benchmarks/induction/eval/induction_eval_cache_v1.meta.json",
    "benchmarks/induction/schemas/benchmark_record.schema.json",
    "benchmarks/induction/schemas/prediction_record.schema.json",
    "benchmarks/induction/schemas/eval_cache_record.schema.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_yaml_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> int:
    datasets = {
        "FullObs": BENCH_ROOT / "data" / "induction_fullobs_v1.yaml.gz",
        "CI": BENCH_ROOT / "data" / "induction_ci_v1.yaml.gz",
        "EC": BENCH_ROOT / "data" / "induction_ec_v1.yaml.gz",
    }

    benchmark_counts = {task: len(load_yaml_gz(path)) for task, path in datasets.items()}
    prediction_rows = list(iter_jsonl(BENCH_ROOT / "predictions" / "induction_predictions_v1.jsonl.gz"))
    eval_rows = list(iter_jsonl(BENCH_ROOT / "eval" / "induction_eval_cache_v1.jsonl"))
    prediction_by_task = Counter(row.get("task") for row in prediction_rows)
    eval_by_task = Counter(row.get("task") for row in eval_rows)
    prediction_by_model = Counter(row.get("model") for row in prediction_rows)
    eval_by_model = Counter(row.get("model_id") for row in eval_rows)

    manifest = {
        "schemaVersion": "induction_release_manifest_v1",
        "releaseTag": "induction-v1.0",
        "releaseDate": "2026-05-23",
        "benchmark": "INDUCTION",
        "license": "MIT",
        "paper": {
            "title": "INDUCTION: Finite-Structure Concept Synthesis in First-Order Logic",
            "venue": "ICML 2026",
        },
        "sourceRepository": "https://github.com/SerafimBatzoglou/concept-synth",
        "counts": {
            "benchmarkInstances": sum(benchmark_counts.values()),
            "benchmarkInstancesByTask": benchmark_counts,
            "predictionRows": len(prediction_rows),
            "predictionRowsByTask": dict(sorted(prediction_by_task.items())),
            "predictionModelCounts": dict(sorted(prediction_by_model.items())),
            "evalRows": len(eval_rows),
            "evalRowsByTask": dict(sorted(eval_by_task.items())),
            "evalModelCounts": dict(sorted(eval_by_model.items())),
        },
        "artifacts": [],
    }

    for rel_path in ARTIFACT_PATHS:
        path = REPO_ROOT / rel_path
        manifest["artifacts"].append(
            {
                "path": rel_path,
                "sizeBytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    (BENCH_ROOT / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {BENCH_ROOT / 'release_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
