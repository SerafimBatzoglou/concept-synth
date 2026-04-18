"""
eval_cache_reader.py — Minimal eval-cache reader for the review artifact.

Auto-generated from the full eval_cache.py; contains only the
dataclass definitions and EvalCacheReader needed to regenerate tables.
Requires only Python 3.8+ stdlib (no PyYAML, no Z3).
"""


import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # Not needed for reading cached JSONL


# Schema version - increment when schema changes
SCHEMA_VERSION = "abd_eval_v1"


@dataclass
class Prediction:
    """Prediction object within an evaluation record."""
    raw_text: Optional[str] = None
    formula: Optional[str] = None
    original_formula: Optional[str] = None
    parse_ok: bool = False
    parse_error: Optional[str] = None
    auto_closed_parens: int = 0
    ast_size: Optional[int] = None


@dataclass
class TrainEval:
    """Training evaluation results."""
    train_all_valid: bool = False
    train_invalid_worlds: Optional[List[str]] = None
    pred_cost_train_sum: Optional[int] = None
    gap_vs_opt_train_sum: Optional[int] = None
    gap_vs_gold_train_sum: Optional[int] = None
    gap_vs_opt_train_norm: Optional[float] = None
    gap_vs_gold_train_norm: Optional[float] = None


@dataclass
class HoldoutEval:
    """Holdout evaluation results."""
    holdout_all_valid: bool = False
    holdout_invalid_worlds: Optional[List[str]] = None
    pred_cost_holdout_sum: Optional[int] = None
    gap_vs_opt_holdout_sum: Optional[int] = None
    gap_vs_gold_holdout_sum: Optional[int] = None
    gap_vs_opt_holdout_norm: Optional[float] = None
    gap_vs_gold_holdout_norm: Optional[float] = None


@dataclass
class Timing:
    """Timing information for the evaluation."""
    parse_ms: Optional[float] = None
    train_eval_ms: Optional[float] = None
    holdout_eval_ms: Optional[float] = None
    total_ms: Optional[float] = None


@dataclass
class EvalCacheRecord:
    """
    Single evaluation cache record (one JSONL line).

    Represents the evaluation of one (instance_id, model_id, run_id) tuple.
    """
    # Identifiers
    schema_version: str = SCHEMA_VERSION
    dataset_name: str = ""
    dataset_path: str = ""
    dataset_sha256: str = ""
    holdout_sha256: str = ""  # Empty string if embedded
    instance_id: str = ""

    # Problem metadata
    scenario: str = ""  # ABD_FULL / ABD_PARTIAL
    theory: str = ""  # TH7, TH10, etc.
    difficulty: str = ""  # easy / hard
    num_train_worlds: int = 0
    num_holdout_worlds: int = 0

    # Gold/optimal baselines
    gold_formula: Optional[str] = None  # The planted gold formula text
    gold_ast: Optional[int] = None
    gold_cost_train_sum: Optional[int] = None
    gold_cost_holdout_sum: Optional[int] = None
    opt_cost_train_sum: int = 0
    opt_cost_holdout_sum: Optional[int] = None

    # Model info
    model_id: str = ""
    run_id: str = ""

    # Evaluation components
    prediction: Prediction = field(default_factory=Prediction)
    train_eval: TrainEval = field(default_factory=TrainEval)
    holdout_eval: HoldoutEval = field(default_factory=HoldoutEval)
    timing: Timing = field(default_factory=Timing)

    # Optional notes
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "holdout_sha256": self.holdout_sha256,
            "instance_id": self.instance_id,
            "scenario": self.scenario,
            "theory": self.theory,
            "difficulty": self.difficulty,
            "num_train_worlds": self.num_train_worlds,
            "num_holdout_worlds": self.num_holdout_worlds,
            "gold_formula": self.gold_formula,
            "gold_ast": self.gold_ast,
            "gold_cost_train_sum": self.gold_cost_train_sum,
            "gold_cost_holdout_sum": self.gold_cost_holdout_sum,
            "opt_cost_train_sum": self.opt_cost_train_sum,
            "opt_cost_holdout_sum": self.opt_cost_holdout_sum,
            "model_id": self.model_id,
            "run_id": self.run_id,
            "prediction": asdict(self.prediction),
            "train_eval": asdict(self.train_eval),
            "holdout_eval": asdict(self.holdout_eval),
            "timing": asdict(self.timing),
            "notes": self.notes,
        }

    def to_json_line(self) -> str:
        """Convert to JSON string for JSONL output."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalCacheRecord":
        """Create from dictionary (for reading JSONL)."""
        record = cls()
        record.schema_version = d.get("schema_version", SCHEMA_VERSION)
        record.dataset_name = d.get("dataset_name", "")
        record.dataset_path = d.get("dataset_path", "")
        record.dataset_sha256 = d.get("dataset_sha256", "")
        record.holdout_sha256 = d.get("holdout_sha256", "")
        record.instance_id = d.get("instance_id", "")
        record.scenario = d.get("scenario", "")
        record.theory = d.get("theory", "")
        record.difficulty = d.get("difficulty", "")
        record.num_train_worlds = d.get("num_train_worlds", 0)
        record.num_holdout_worlds = d.get("num_holdout_worlds", 0)
        record.gold_formula = d.get("gold_formula")
        record.gold_ast = d.get("gold_ast")
        record.gold_cost_train_sum = d.get("gold_cost_train_sum")
        record.gold_cost_holdout_sum = d.get("gold_cost_holdout_sum")
        record.opt_cost_train_sum = d.get("opt_cost_train_sum", 0)
        record.opt_cost_holdout_sum = d.get("opt_cost_holdout_sum")
        record.model_id = d.get("model_id", "")
        record.run_id = d.get("run_id", "")

        # Nested objects
        pred_d = d.get("prediction", {})
        record.prediction = Prediction(
            raw_text=pred_d.get("raw_text"),
            formula=pred_d.get("formula"),
            original_formula=pred_d.get("original_formula"),
            parse_ok=pred_d.get("parse_ok", False),
            parse_error=pred_d.get("parse_error"),
            auto_closed_parens=pred_d.get("auto_closed_parens", 0),
            ast_size=pred_d.get("ast_size"),
        )

        train_d = d.get("train_eval", {})
        record.train_eval = TrainEval(
            train_all_valid=train_d.get("train_all_valid", False),
            train_invalid_worlds=train_d.get("train_invalid_worlds"),
            pred_cost_train_sum=train_d.get("pred_cost_train_sum"),
            gap_vs_opt_train_sum=train_d.get("gap_vs_opt_train_sum"),
            gap_vs_gold_train_sum=train_d.get("gap_vs_gold_train_sum"),
            gap_vs_opt_train_norm=train_d.get("gap_vs_opt_train_norm"),
            gap_vs_gold_train_norm=train_d.get("gap_vs_gold_train_norm"),
        )

        hold_d = d.get("holdout_eval", {})
        record.holdout_eval = HoldoutEval(
            holdout_all_valid=hold_d.get("holdout_all_valid", False),
            holdout_invalid_worlds=hold_d.get("holdout_invalid_worlds"),
            pred_cost_holdout_sum=hold_d.get("pred_cost_holdout_sum"),
            gap_vs_opt_holdout_sum=hold_d.get("gap_vs_opt_holdout_sum"),
            gap_vs_gold_holdout_sum=hold_d.get("gap_vs_gold_holdout_sum"),
            gap_vs_opt_holdout_norm=hold_d.get("gap_vs_opt_holdout_norm"),
            gap_vs_gold_holdout_norm=hold_d.get("gap_vs_gold_holdout_norm"),
        )

        timing_d = d.get("timing", {})
        record.timing = Timing(
            parse_ms=timing_d.get("parse_ms"),
            train_eval_ms=timing_d.get("train_eval_ms"),
            holdout_eval_ms=timing_d.get("holdout_eval_ms"),
            total_ms=timing_d.get("total_ms"),
        )

        record.notes = d.get("notes")
        return record


def compute_file_sha256(file_path: str) -> str:
    """Compute SHA256 hash of a file's contents."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_yaml_canonical_sha256(yaml_path: str) -> str:
    """
    Compute SHA256 of YAML after loading and re-dumping canonically.

    This ensures the hash is stable across formatting differences.
    For large files or files with non-serializable objects, falls back to file hash.
    """
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Custom JSON encoder to handle sets and other non-serializable types
        def default_encoder(obj):
            if isinstance(obj, set):
                return sorted(list(obj))
            if isinstance(obj, frozenset):
                return sorted(list(obj))
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # Canonical JSON dump (sorted keys, no indentation)
        canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, default=default_encoder)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception as e:
        # Fallback to file hash if canonical serialization fails
        print(f"Warning: Using file hash for {yaml_path} (canonical serialization failed: {e})")
        return compute_file_sha256(yaml_path)


def get_git_sha() -> str:
    """Get current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def generate_run_id(prefix: str = "eval") -> str:
    """Generate a run_id with date, git sha, and optional prefix."""
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = get_git_sha()
    return f"{prefix}_{date_str}_{git_sha}"




class EvalCacheReader:
    """
    Reader for evaluation cache JSONL files.

    Supports:
    - Loading all records or filtering by criteria
    - Getting latest run_id
    - Aggregating statistics
    """

    def __init__(self, *jsonl_paths: str):
        self.jsonl_paths = [Path(p) for p in jsonl_paths]
        self._records: Optional[List[EvalCacheRecord]] = None

    def load_records(
        self,
        run_id: Optional[str] = None,
        model_id: Optional[str] = None,
        dataset_sha256: Optional[str] = None,
        scenario: Optional[str] = None,
        theory: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> List[EvalCacheRecord]:
        """
        Load records with optional filtering.

        Args:
            run_id: Filter to specific run_id, or "latest" for most recent
            model_id: Filter to specific model
            dataset_sha256: Filter to specific dataset version
            scenario: Filter to ABD_FULL or ABD_PARTIAL
            theory: Filter to specific theory (TH7, TH10, etc.)
            difficulty: Filter to easy or hard

        Returns:
            List of EvalCacheRecord objects
        """
        records = []

        for path in self.jsonl_paths:
            if not path.exists():
                print(f"Warning: {path} does not exist, skipping")
                continue

            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        record = EvalCacheRecord.from_dict(d)
                        records.append(record)
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"Warning: Could not parse line: {e}")
                        continue

        # Handle "latest" run_id
        if run_id == "latest":
            run_ids = sorted(set(r.run_id for r in records))
            if run_ids:
                run_id = run_ids[-1]
                print(f"Using latest run_id: {run_id}")
            else:
                run_id = None

        # Apply filters
        filtered = []
        for r in records:
            if run_id and r.run_id != run_id:
                continue
            if model_id and r.model_id != model_id:
                continue
            if dataset_sha256 and r.dataset_sha256 != dataset_sha256:
                continue
            if scenario and r.scenario != scenario:
                continue
            if theory and r.theory != theory:
                continue
            if difficulty and r.difficulty != difficulty:
                continue
            filtered.append(r)

        self._records = filtered
        return filtered

    def get_run_ids(self) -> List[str]:
        """Get all unique run_ids in the cache."""
        if self._records is None:
            self.load_records()
        return sorted(set(r.run_id for r in self._records))

    def get_models(self) -> List[str]:
        """Get all unique model_ids in the cache."""
        if self._records is None:
            self.load_records()
        return sorted(set(r.model_id for r in self._records))

    def get_latest_run_id(self) -> Optional[str]:
        """Get the most recent run_id (lexicographically)."""
        run_ids = self.get_run_ids()
        return run_ids[-1] if run_ids else None
