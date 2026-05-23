"""I/O helpers for released INDUCTION benchmark artifacts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import yaml


TASK_DATASETS = {
    "FullObs": "induction_fullobs_v1.yaml.gz",
    "CI": "induction_ci_v1.yaml.gz",
    "EC": "induction_ec_v1.yaml.gz",
}


def repo_root() -> Path:
    """Return the repository root for an editable/source checkout."""
    return Path(__file__).resolve().parents[3]


def induction_root() -> Path:
    return repo_root() / "benchmarks" / "induction"


def canonical_dataset_paths() -> Dict[str, Path]:
    data_dir = induction_root() / "data"
    return {task: data_dir / filename for task, filename in TASK_DATASETS.items()}


def canonical_predictions_path() -> Path:
    return induction_root() / "predictions" / "induction_predictions_v1.jsonl.gz"


def canonical_eval_cache_path() -> Path:
    return induction_root() / "eval" / "induction_eval_cache_v1.jsonl"


def generated_eval_path() -> Path:
    return induction_root() / "generated_eval" / "induction_eval_cache_v1.jsonl"


def open_text(path: str | Path, mode: str = "rt"):
    """Open a plain-text or gzip-compressed text file."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def load_yaml(path: str | Path) -> Any:
    with open_text(path) as handle:
        return yaml.safe_load(handle)


def load_problem_records(dataset_path: str | Path) -> List[Dict[str, Any]]:
    """Load released INDUCTION records from a YAML or YAML.GZ file."""
    data = load_yaml(dataset_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level YAML list in {dataset_path}")
    return data


def get_instance_id(record: Dict[str, Any]) -> str:
    """Extract the canonical instance id from either released or legacy records."""
    return (
        str(record.get("instanceId") or "")
        or str(record.get("problem", {}).get("instanceId") or "")
        or str(record.get("problemId") or "")
    )


def index_problem_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    problems: Dict[str, Dict[str, Any]] = {}
    for record in records:
        instance_id = get_instance_id(record)
        if instance_id:
            problems[instance_id] = record
    return problems


def load_problem_index(dataset_path: str | Path) -> Dict[str, Dict[str, Any]]:
    return index_problem_records(load_problem_records(dataset_path))


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path, *, append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    mode = "at" if append else "wt"
    with open_text(path, mode) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def dataset_name_from_path(dataset_path: str | Path) -> str:
    name = Path(dataset_path).name
    for suffix in (".yaml.gz", ".yml.gz", ".yaml", ".yml"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(dataset_path).stem


def infer_task_from_dataset(dataset_path: str | Path) -> Optional[str]:
    name = dataset_name_from_path(dataset_path)
    if "fullobs" in name:
        return "FullObs"
    if name.endswith("_ci_v1") or "_ci_" in name:
        return "CI"
    if name.endswith("_ec_v1") or "_ec_" in name:
        return "EC"
    return None
