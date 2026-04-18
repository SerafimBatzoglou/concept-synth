#!/usr/bin/env python3
"""
Evaluation Cache for ABD-B1 Benchmark

Defines the JSONL cache schema (abd_eval_v1) and provides utilities for:
- Writing evaluation results incrementally to JSONL
- Reading cached results with optional filtering
- Computing dataset/holdout SHA256 hashes for reproducibility
- Resume/skip logic for efficient re-evaluation

Schema version: abd_eval_v1

Each JSONL line contains one evaluation record:
{
    "schema_version": "abd_eval_v1",
    "dataset_name": str,
    "dataset_path": str,
    "dataset_sha256": str,
    "holdout_sha256": str (or "" if embedded),
    "instance_id": str,
    "scenario": str (ABD_FULL / ABD_PARTIAL),
    "theory": str (TH7, TH10, TH11, TH12, TH2),
    "difficulty": str (easy / hard),
    "num_train_worlds": int,
    "num_holdout_worlds": int,
    "gold_ast": int or null,
    "gold_cost_train_sum": int or null,
    "gold_cost_holdout_sum": int or null,
    "opt_cost_train_sum": int,
    "opt_cost_holdout_sum": int or null,
    "model_id": str,
    "run_id": str,
    "prediction": {
        "raw_text": str or null,
        "formula": str or null,
        "original_formula": str or null,
        "parse_ok": bool,
        "parse_error": str or null,
        "auto_closed_parens": int,
        "ast_size": int or null
    },
    "train_eval": {
        "train_all_valid": bool,
        "train_invalid_worlds": list[str] or null,
        "pred_cost_train_sum": int or null,
        "gap_vs_opt_train_sum": int or null,
        "gap_vs_gold_train_sum": int or null,
        "gap_vs_opt_train_norm": float or null,
        "gap_vs_gold_train_norm": float or null
    },
    "holdout_eval": {
        "holdout_all_valid": bool,
        "holdout_invalid_worlds": list[str] or null,
        "pred_cost_holdout_sum": int or null,
        "gap_vs_opt_holdout_sum": int or null,
        "gap_vs_gold_holdout_sum": int or null,
        "gap_vs_opt_holdout_norm": float or null,
        "gap_vs_gold_holdout_norm": float or null
    },
    "timing_ms": {
        "parse_ms": float or null,
        "train_eval_ms": float or null,
        "holdout_eval_ms": float or null,
        "total_ms": float or null
    },
    "notes": str or null
}
"""

import hashlib
import gzip
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from concept_synth.benchmark_io import dataset_name_from_path


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
        open_fn = gzip.open if yaml_path.endswith(".gz") else open
        with open_fn(yaml_path, "rt", encoding="utf-8") as f:
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


class EvalCacheWriter:
    """
    Incremental writer for evaluation cache JSONL files.

    Supports:
    - Append-mode writing (crash-safe)
    - Resume detection (skip already-evaluated instances)
    - Overwrite mode (clears existing cache when resume=False)
    - Metadata file generation
    """

    def __init__(
        self,
        output_path: str,
        dataset_path: str,
        holdout_path: Optional[str] = None,
        run_id: Optional[str] = None,
        resume: bool = True,
    ):
        self.output_path = Path(output_path)
        self.dataset_path = dataset_path
        self.holdout_path = holdout_path
        self.run_id = run_id or generate_run_id()
        self.resume = resume

        # Compute hashes
        self.dataset_sha256 = compute_yaml_canonical_sha256(dataset_path)
        self.holdout_sha256 = compute_file_sha256(holdout_path) if holdout_path else ""
        self.dataset_name = dataset_name_from_path(dataset_path)

        # Track already-evaluated instances for resume
        self.evaluated_keys: Set[Tuple[str, str, str]] = set()  # (instance_id, model_id, run_id)
        self._evaluated_instance_model_keys: Set[Tuple[str, str]] = set()  # (instance_id, model_id)

        # Load existing records if resuming, or clear cache if overwriting
        if resume and self.output_path.exists():
            self._load_existing_keys()
        elif not resume and self.output_path.exists():
            # Clear the cache file when not resuming (--overwrite mode)
            self.output_path.unlink()
            print(f"Cleared existing cache: {self.output_path}")

        # File handle (opened on first write)
        self._file_handle = None
        self._records_written = 0

    def _load_existing_keys(self):
        """Load existing (instance_id, model_id, run_id) tuples from JSONL."""
        try:
            with open(self.output_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        key = (
                            d.get("instance_id", ""),
                            d.get("model_id", ""),
                            d.get("run_id", ""),
                        )
                        self.evaluated_keys.add(key)
                        self._evaluated_instance_model_keys.add((key[0], key[1]))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Could not load existing cache: {e}")

    def should_skip(self, instance_id: str, model_id: str, run_id: Optional[str] = None) -> bool:
        """Check if this (instance, model) should be skipped (already evaluated).

        Note: run_id is ignored for resume matching — any existing evaluation
        for the same (instance, model) pair is sufficient to skip.
        """
        if not self.resume:
            return False
        return (instance_id, model_id) in self._evaluated_instance_model_keys

    def write_record(self, record: EvalCacheRecord):
        """Write a single evaluation record to the JSONL file."""
        if self._file_handle is None:
            # Open in append mode
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.output_path, "a", encoding="utf-8")

        # Ensure record has correct metadata
        record.schema_version = SCHEMA_VERSION
        record.dataset_name = self.dataset_name
        record.dataset_path = str(self.dataset_path)
        record.dataset_sha256 = self.dataset_sha256
        record.holdout_sha256 = self.holdout_sha256
        if not record.run_id:
            record.run_id = self.run_id

        # Write and flush immediately
        self._file_handle.write(record.to_json_line() + "\n")
        self._file_handle.flush()
        self._records_written += 1

        # Track for resume
        key = (record.instance_id, record.model_id, record.run_id)
        self.evaluated_keys.add(key)

    def write_metadata(self, models: List[str], command_line: str = ""):
        """Write a metadata JSON file alongside the JSONL."""
        meta_path = Path(str(self.output_path) + ".meta.json")
        meta = {
            "schema_version": SCHEMA_VERSION,
            "dataset_name": self.dataset_name,
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": self.dataset_sha256,
            "holdout_path": str(self.holdout_path) if self.holdout_path else None,
            "holdout_sha256": self.holdout_sha256,
            "run_id": self.run_id,
            "git_sha": get_git_sha(),
            "timestamp": datetime.now().isoformat(),
            "models": models,
            "command_line": command_line,
            "records_written": self._records_written,
            "total_keys": len(self.evaluated_keys),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def close(self):
        """Close the file handle."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


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


def create_eval_cache_record(
    problem: Dict[str, Any],
    result: Dict[str, Any],
    train_eval_result: Any,  # AbdEvalResult
    holdout_eval_result: Any,  # HoldoutEvalResult or None
    model_id: str,
    run_id: str,
    num_train_worlds: int,
    num_holdout_worlds: int,
    holdout_worlds: Optional[List[Dict]] = None,
    timing: Optional[Dict[str, float]] = None,
) -> EvalCacheRecord:
    """
    Create an EvalCacheRecord from evaluation results.

    This is the main conversion function used by the evaluator.
    """
    # Extract problem metadata
    prob_data = problem.get("problem", problem)
    desc = problem.get("problemDescription", {})
    gold = prob_data.get("gold", {})

    instance_id = (
        problem.get("problemDescription", {}).get("instanceId") or
        problem.get("instanceId") or
        result.get("instanceId", "")
    )
    scenario = prob_data.get("scenario", "ABD_FULL")
    theory = prob_data.get("theoryId") or prob_data.get("theory", {}).get("theoryId", "Unknown")
    difficulty = desc.get("difficulty", "unknown")

    # Gold info
    gold_alpha = gold.get("alpha") or gold.get("goldAlpha")
    gold_ast_size = None
    if gold_alpha:
        try:
            from concept_synth.sexpr_parser import parse_sexpr_formula
            from concept_synth.ast_utils import ast_size
            gold_ast = parse_sexpr_formula(gold_alpha)
            gold_ast_size = ast_size(gold_ast)
        except Exception:
            pass

    gold_cost_train = gold.get("totalGoldAlphaCost")
    gold_cost_holdout = None

    # Optimal costs (from world data)
    opt_cost_train = 0
    train_worlds = prob_data.get("trainWorlds") or prob_data.get("worlds", [])
    for w in train_worlds:
        if not w.get("isHeldout", False):
            opt_cost_train += w.get("optCost", 0)

    opt_cost_holdout = None
    if holdout_worlds:
        opt_cost_holdout = sum(w.get("optCost", 0) for w in holdout_worlds)
        # Compute gold cost for holdout worlds (each world has goldAlphaCost)
        gold_costs = [w.get("goldAlphaCost") for w in holdout_worlds]
        if all(c is not None for c in gold_costs):
            gold_cost_holdout = sum(gold_costs)

    # Prediction info
    raw_text = result.get("response") or result.get("rawResponse")
    formula = train_eval_result.alpha_sexpr or result.get("extractedFormula")
    original_formula = getattr(train_eval_result, "alpha_original_sexpr", None)
    parse_ok = train_eval_result.parse_error is None
    parse_error = train_eval_result.parse_error
    auto_closed_parens = getattr(train_eval_result, "trailing_parens_added", 0)

    ast_size_val = None
    if formula and parse_ok:
        try:
            from concept_synth.sexpr_parser import parse_sexpr_formula
            from concept_synth.ast_utils import ast_size
            pred_ast = parse_sexpr_formula(formula)
            ast_size_val = ast_size(pred_ast)
        except Exception:
            pass

    notes = None
    if auto_closed_parens > 0:
        plural = "s" if auto_closed_parens != 1 else ""
        notes = f"Auto-closed {auto_closed_parens} trailing parenthesis{plural} before evaluation"

    # Train evaluation
    train_all_valid = train_eval_result.valid
    train_invalid_worlds = None
    pred_cost_train = None
    gap_vs_opt_train = None
    gap_vs_gold_train = None
    gap_vs_opt_train_norm = None
    gap_vs_gold_train_norm = None

    if train_all_valid and train_eval_result.total_cost is not None:
        pred_cost_train = train_eval_result.total_cost
        if train_eval_result.total_opt_cost is not None:
            gap_vs_opt_train = pred_cost_train - train_eval_result.total_opt_cost
            if num_train_worlds > 0:
                gap_vs_opt_train_norm = gap_vs_opt_train / num_train_worlds
        if gold_cost_train is not None:
            gap_vs_gold_train = pred_cost_train - gold_cost_train
            if num_train_worlds > 0:
                gap_vs_gold_train_norm = gap_vs_gold_train / num_train_worlds
    elif train_eval_result.per_world:
        # Extract invalid world IDs
        train_invalid_worlds = [
            pw.get("worldId", str(i))
            for i, pw in enumerate(train_eval_result.per_world)
            if not pw.get("valid", False)
        ]

    # Holdout evaluation
    holdout_all_valid = False
    holdout_invalid_worlds = None
    pred_cost_holdout = None
    gap_vs_opt_holdout = None
    gap_vs_gold_holdout = None
    gap_vs_opt_holdout_norm = None
    gap_vs_gold_holdout_norm = None

    if holdout_eval_result is not None:
        holdout_all_valid = holdout_eval_result.holdout_valid
        if holdout_all_valid and holdout_eval_result.holdout_total_cost is not None:
            pred_cost_holdout = holdout_eval_result.holdout_total_cost
            if holdout_eval_result.holdout_total_opt_cost is not None:
                gap_vs_opt_holdout = pred_cost_holdout - holdout_eval_result.holdout_total_opt_cost
                if num_holdout_worlds > 0:
                    gap_vs_opt_holdout_norm = gap_vs_opt_holdout / num_holdout_worlds
            if gold_cost_holdout is not None:
                gap_vs_gold_holdout = pred_cost_holdout - gold_cost_holdout
                if num_holdout_worlds > 0:
                    gap_vs_gold_holdout_norm = gap_vs_gold_holdout / num_holdout_worlds
        elif holdout_eval_result.per_holdout:
            holdout_invalid_worlds = [
                ph.get("worldId", str(i))
                for i, ph in enumerate(holdout_eval_result.per_holdout)
                if not ph.get("valid", False)
            ]

    # Timing
    timing_obj = Timing()
    if timing:
        timing_obj.parse_ms = timing.get("parse_ms")
        timing_obj.train_eval_ms = timing.get("train_eval_ms")
        timing_obj.holdout_eval_ms = timing.get("holdout_eval_ms")
        timing_obj.total_ms = timing.get("total_ms")

    # Build record
    record = EvalCacheRecord(
        instance_id=instance_id,
        scenario=scenario,
        theory=theory,
        difficulty=difficulty,
        num_train_worlds=num_train_worlds,
        num_holdout_worlds=num_holdout_worlds,
        gold_formula=gold_alpha,
        gold_ast=gold_ast_size,
        gold_cost_train_sum=gold_cost_train,
        gold_cost_holdout_sum=gold_cost_holdout,
        opt_cost_train_sum=opt_cost_train,
        opt_cost_holdout_sum=opt_cost_holdout,
        model_id=model_id,
        run_id=run_id,
        prediction=Prediction(
            raw_text=raw_text[:1000] if raw_text else None,  # Truncate for space
            formula=formula,
            original_formula=original_formula,
            parse_ok=parse_ok,
            parse_error=parse_error,
            auto_closed_parens=auto_closed_parens,
            ast_size=ast_size_val,
        ),
        train_eval=TrainEval(
            train_all_valid=train_all_valid,
            train_invalid_worlds=train_invalid_worlds,
            pred_cost_train_sum=pred_cost_train,
            gap_vs_opt_train_sum=gap_vs_opt_train,
            gap_vs_gold_train_sum=gap_vs_gold_train,
            gap_vs_opt_train_norm=gap_vs_opt_train_norm,
            gap_vs_gold_train_norm=gap_vs_gold_train_norm,
        ),
        holdout_eval=HoldoutEval(
            holdout_all_valid=holdout_all_valid,
            holdout_invalid_worlds=holdout_invalid_worlds,
            pred_cost_holdout_sum=pred_cost_holdout,
            gap_vs_opt_holdout_sum=gap_vs_opt_holdout,
            gap_vs_gold_holdout_sum=gap_vs_gold_holdout,
            gap_vs_opt_holdout_norm=gap_vs_opt_holdout_norm,
            gap_vs_gold_holdout_norm=gap_vs_gold_holdout_norm,
        ),
        timing=timing_obj,
        notes=notes,
    )

    return record
