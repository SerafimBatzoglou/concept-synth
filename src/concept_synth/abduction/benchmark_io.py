"""I/O helpers for public benchmark bundles and prediction files."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def open_text(path: str):
    """Open a plain-text or gzip-compressed text file."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_yaml(path: str) -> Any:
    """Load a YAML file, including .yaml.gz bundles."""
    with open_text(path) as handle:
        return yaml.safe_load(handle)


def load_problem_records(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the top-level list of benchmark records from a released bundle."""
    data = load_yaml(dataset_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level YAML list in {dataset_path}")
    return data


def get_instance_id(record: Dict[str, Any]) -> str:
    """Extract the canonical instance identifier from a benchmark record."""
    return (
        record.get("problemId", "")
        or record.get("problem", record).get("instanceId", "")
        or record.get("instanceId", "")
    )


def index_problem_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index benchmark records by instance id."""
    problems: Dict[str, Dict[str, Any]] = {}
    for record in records:
        instance_id = get_instance_id(record)
        if not instance_id:
            continue
        problems[instance_id] = record
    return problems


def load_problem_index(dataset_path: str) -> Dict[str, Dict[str, Any]]:
    """Load and index a benchmark bundle by instance id."""
    return index_problem_records(load_problem_records(dataset_path))


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into memory."""
    rows: List[Dict[str, Any]] = []
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_embedded_results(
    problems: Dict[str, Dict[str, Any]],
    models: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract embedded llmResults entries from a benchmark bundle."""
    rows: List[Dict[str, Any]] = []
    for instance_id, problem in problems.items():
        for result in problem.get("llmResults", []):
            model_id = result.get("model")
            if models and model_id not in models:
                continue
            row = dict(result)
            row.setdefault("instanceId", instance_id)
            rows.append(row)
    return rows


def guess_holdout_path(dataset_path: str) -> Optional[str]:
    """Find the shipped holdout sidecar next to a dataset when it is unambiguous."""
    dataset_file = Path(dataset_path)
    dataset_name = dataset_name_from_path(dataset_path)

    canonical_name = re.sub(r"_instances_", "_holdouts_", dataset_name)
    canonical_candidates = []
    if canonical_name != dataset_name:
        canonical_candidates = [
            dataset_file.parent / f"{canonical_name}.jsonl.gz",
            dataset_file.parent / f"{canonical_name}.jsonl",
        ]
        for candidate in canonical_candidates:
            if candidate.exists():
                return str(candidate)

    base_name = dataset_file.name[:-3] if dataset_file.name.endswith(".gz") else dataset_file.name
    candidates = sorted(dataset_file.parent.glob(f"{base_name}.holdout*.jsonl*"))
    if len(candidates) == 1:
        return str(candidates[0])
    return None


def guess_predictions_path(dataset_path: str) -> Optional[str]:
    """Find the canonical released predictions file for a benchmark dataset."""
    dataset_file = Path(dataset_path)
    dataset_name = dataset_name_from_path(dataset_path)
    predictions_dir = dataset_file.parent.parent / "predictions"

    canonical_name = re.sub(r"_instances_", "_predictions_", dataset_name)
    candidates = []
    if canonical_name != dataset_name:
        candidates.extend(
            [
                predictions_dir / f"{canonical_name}.jsonl.gz",
                predictions_dir / f"{canonical_name}.jsonl",
            ]
        )

    candidates.extend(
        sorted(predictions_dir.glob(f"{dataset_name}*predictions*.jsonl*"))
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def dataset_name_from_path(dataset_path: str) -> str:
    """Return a stable dataset name without .yaml or .yaml.gz suffixes."""
    name = Path(dataset_path).name
    for suffix in (".yaml.gz", ".yml.gz", ".yaml", ".yml"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(dataset_path).stem
