"""I/O helpers for public benchmark bundles and prediction files."""

from __future__ import annotations

import gzip
import json
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
    with open(path, "r", encoding="utf-8") as handle:
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
    base_name = dataset_file.name[:-3] if dataset_file.name.endswith(".gz") else dataset_file.name
    candidates = sorted(dataset_file.parent.glob(f"{base_name}.holdout*.jsonl"))
    if len(candidates) == 1:
        return str(candidates[0])
    return None


def dataset_name_from_path(dataset_path: str) -> str:
    """Return a stable dataset name without .yaml or .yaml.gz suffixes."""
    name = Path(dataset_path).name
    for suffix in (".yaml.gz", ".yml.gz", ".yaml", ".yml"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(dataset_path).stem

