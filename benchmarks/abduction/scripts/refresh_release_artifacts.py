#!/usr/bin/env python3
"""Sanitize a regenerated ABD eval cache and refresh release metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


def _open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_dataset_records(path: Path) -> List[Dict[str, Any]]:
    with _open_text(path) as handle:
        records = yaml.safe_load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a top-level YAML list in {path}")
    return records


def _count_models(rows: Iterable[Dict[str, Any]], key: str) -> Tuple[List[str], Dict[str, int], int]:
    model_order: List[str] = []
    model_counts: Dict[str, int] = {}
    total = 0
    for row in rows:
        model = str(row[key])
        if model not in model_counts:
            model_order.append(model)
            model_counts[model] = 0
        model_counts[model] += 1
        total += 1
    return model_order, model_counts, total


def _sanitize_eval_cache(
    input_path: Path,
    output_path: Path,
    dataset_relpath: str,
    run_id: str | None,
) -> Tuple[str, str, int, int, List[str], Dict[str, int]]:
    schema_version = ""
    dataset_name = ""
    row_count = 0
    instance_ids = set()
    model_order: List[str] = []
    model_counts: Dict[str, int] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            schema_version = schema_version or str(row.get("schema_version", "abd_eval_v1"))
            dataset_name = dataset_name or str(row.get("dataset_name", ""))
            row["dataset_path"] = dataset_relpath
            row["gold_formula"] = None
            if run_id:
                row["run_id"] = run_id

            model_id = str(row.get("model_id", ""))
            if model_id not in model_counts:
                model_order.append(model_id)
                model_counts[model_id] = 0
            model_counts[model_id] += 1
            row_count += 1
            instance_ids.add(str(row.get("instance_id", "")))

            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    return schema_version, dataset_name, row_count, len(instance_ids), model_order, model_counts


def _build_meta(
    existing_meta: Dict[str, Any],
    eval_schema_version: str,
    release_tag: str,
    dataset_name: str,
    dataset_relpath: str,
    dataset_path: Path,
    holdout_relpath: str,
    holdout_path: Path,
    predictions_relpath: str,
    predictions_path: Path,
    eval_relpath: str,
    eval_path: Path,
    run_id: str,
    row_count: int,
    instance_count: int,
    model_order: List[str],
    model_counts: Dict[str, int],
) -> Dict[str, Any]:
    notes = list(existing_meta.get("notes", []))
    regen_note = "Regenerated from the current public ABD evaluation pipeline."
    if regen_note not in notes:
        notes.append(regen_note)

    return {
        "schema_version": eval_schema_version,
        "release_tag": release_tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "dataset_path": dataset_relpath,
        "dataset_sha256": _sha256(dataset_path),
        "holdout_path": holdout_relpath,
        "holdout_sha256": _sha256(holdout_path),
        "predictions_path": predictions_relpath,
        "predictions_sha256": _sha256(predictions_path),
        "eval_path": eval_relpath,
        "eval_sha256": _sha256(eval_path),
        "run_id": run_id,
        "row_count": row_count,
        "instance_count": instance_count,
        "models": model_order,
        "model_counts": model_counts,
        "notes": notes,
    }


def _build_counts(
    dataset_records: List[Dict[str, Any]],
    holdout_path: Path,
    predictions_path: Path,
    eval_rows: int,
    eval_model_counts: Dict[str, int],
) -> Dict[str, Any]:
    scenario_counts: Counter[str] = Counter()
    theory_counts: Counter[str] = Counter()
    instance_ids: List[str] = []
    for record in dataset_records:
        instance_id = str(record.get("instanceId") or record.get("problem", {}).get("instanceId", ""))
        if instance_id:
            instance_ids.append(instance_id)
        problem = record.get("problem", record)
        scenario_counts[str(problem.get("scenario", ""))] += 1
        theory_id = problem.get("theoryId") or problem.get("theory", {}).get("theoryId", "")
        theory_counts[str(theory_id)] += 1

    holdout_rows = list(_iter_jsonl(holdout_path))
    holdout_counts: Dict[str, int] = {}
    for row in holdout_rows:
        instance_id = str(row.get("instanceId", ""))
        holdout_counts[instance_id] = holdout_counts.get(instance_id, 0) + 1

    prediction_order, prediction_model_counts, prediction_rows = _count_models(
        _iter_jsonl(predictions_path), "model"
    )
    _ = prediction_order

    instances_without_holdouts = sorted(iid for iid in instance_ids if iid not in holdout_counts)

    return {
        "benchmarkInstances": len(dataset_records),
        "scenarioCounts": dict(scenario_counts),
        "theoryCounts": dict(theory_counts),
        "holdoutRows": len(holdout_rows),
        "instancesWithHoldouts": len(holdout_counts),
        "instancesWithoutHoldouts": instances_without_holdouts,
        "predictionRows": prediction_rows,
        "predictionModelCounts": prediction_model_counts,
        "evalRows": eval_rows,
        "evalModelCounts": eval_model_counts,
    }


def _refresh_manifest(
    manifest_path: Path,
    dataset_records: List[Dict[str, Any]],
    holdout_path: Path,
    predictions_path: Path,
    eval_rows: int,
    eval_model_counts: Dict[str, int],
    release_tag: str,
    release_date: str,
) -> None:
    manifest = _load_json(manifest_path)
    repo_root = manifest_path.parents[2]

    for artifact in manifest.get("artifacts", []):
        path = repo_root / artifact["path"]
        artifact["sizeBytes"] = path.stat().st_size
        artifact["sha256"] = _sha256(path)

    manifest["releaseTag"] = release_tag
    manifest["releaseDate"] = release_date
    manifest["counts"] = _build_counts(
        dataset_records=dataset_records,
        holdout_path=holdout_path,
        predictions_path=predictions_path,
        eval_rows=eval_rows,
        eval_model_counts=eval_model_counts,
    )
    _write_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanitize a regenerated ABD eval cache and refresh release metadata."
    )
    parser.add_argument("--eval-input", required=True, help="Raw eval-cache JSONL produced by the public CLI.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Path to the concept-synth repository root.",
    )
    parser.add_argument("--run-id", help="Override the run_id stored in the sanitized cache.")
    parser.add_argument("--release-tag", help="Override the release tag stored in metadata and manifest.")
    parser.add_argument("--release-date", help="Override the release date stored in the manifest (YYYY-MM-DD).")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    bench_root = repo_root / "benchmarks" / "abduction"
    dataset_path = bench_root / "data" / "abd_instances_v1.yaml.gz"
    holdout_path = bench_root / "data" / "abd_holdouts_v1.jsonl.gz"
    predictions_path = bench_root / "predictions" / "abd_predictions_v1.jsonl.gz"
    eval_output_path = bench_root / "eval" / "abd_eval_cache_v1.jsonl"
    meta_path = bench_root / "eval" / "abd_eval_cache_v1.meta.json"
    manifest_path = bench_root / "release_manifest.json"

    dataset_relpath = "benchmarks/abduction/data/abd_instances_v1.yaml.gz"
    holdout_relpath = "benchmarks/abduction/data/abd_holdouts_v1.jsonl.gz"
    predictions_relpath = "benchmarks/abduction/predictions/abd_predictions_v1.jsonl.gz"
    eval_relpath = "benchmarks/abduction/eval/abd_eval_cache_v1.jsonl"

    existing_meta = _load_json(meta_path)
    run_id = args.run_id or str(existing_meta.get("run_id", "released_abd_eval_cache_v1"))
    release_tag = args.release_tag or str(existing_meta.get("release_tag", "abduction-v1.1"))
    manifest = _load_json(manifest_path)
    release_date = args.release_date or str(manifest.get("releaseDate", ""))

    eval_schema_version, dataset_name, row_count, instance_count, model_order, model_counts = _sanitize_eval_cache(
        input_path=Path(args.eval_input).resolve(),
        output_path=eval_output_path,
        dataset_relpath=dataset_relpath,
        run_id=run_id,
    )

    meta = _build_meta(
        existing_meta=existing_meta,
        eval_schema_version=eval_schema_version,
        release_tag=release_tag,
        dataset_name=dataset_name,
        dataset_relpath=dataset_relpath,
        dataset_path=dataset_path,
        holdout_relpath=holdout_relpath,
        holdout_path=holdout_path,
        predictions_relpath=predictions_relpath,
        predictions_path=predictions_path,
        eval_relpath=eval_relpath,
        eval_path=eval_output_path,
        run_id=run_id,
        row_count=row_count,
        instance_count=instance_count,
        model_order=model_order,
        model_counts=model_counts,
    )
    _write_json(meta_path, meta)

    dataset_records = _load_dataset_records(dataset_path)
    _refresh_manifest(
        manifest_path=manifest_path,
        dataset_records=dataset_records,
        holdout_path=holdout_path,
        predictions_path=predictions_path,
        eval_rows=row_count,
        eval_model_counts=model_counts,
        release_tag=release_tag,
        release_date=release_date,
    )


if __name__ == "__main__":
    main()
