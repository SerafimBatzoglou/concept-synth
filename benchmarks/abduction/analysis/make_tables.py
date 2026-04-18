#!/usr/bin/env python3
"""
make_tables.py - Generate LaTeX and CSV tables from ABD-B1 evaluation cache

Reads JSONL evaluation cache files and produces paper-ready tables:
- Dataset summary
- Main model comparison (TRAIN)
- Scenario breakdown (ABD_FULL vs ABD_PARTIAL)
- Theory breakdown
- Holdout generalization summary
- Complexity vs generalization (AST bins)

Usage:
    python scripts/make_tables.py --input eval_cache.jsonl --outdir paper/tables
    python scripts/make_tables.py --input eval_cache.jsonl --run-id latest --outdir paper/tables

Output:
    paper/tables/*.tex (LaTeX booktabs tables)
    paper/tables/*.csv (CSV versions for debugging)
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from concept_synth.abduction.eval_cache import EvalCacheReader, EvalCacheRecord

# =============================================================================
# Model display configuration
# =============================================================================

MODEL_DISPLAY_NAMES = {
    "claude-opus-4-6": "Opus-4.6",
    "grok4.1fast": "Grok4.1f",
    "deepseek-reasoner": "DSR",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.4": "GPT-5.4",
    "grok4": "Grok4",
    "gemini-3-pro-preview": "Gemini-3",
    "gemini-3.1-pro-preview": "Gemini-3.1",
    "kimi-k2-thinking": "Kimi-K2t",
    "gpt-4o": "GPT-4o",
    "hermes4": "Hermes4",
}

# Models to exclude from paper tables (kept in eval cache)
EXCLUDE_MODELS = {"claude-opus-4-5-20251101"}

# Fallback model order for stable tie-breaking.
MODEL_ORDER = [
    "claude-opus-4-6",      # 0.51 GapG
    "grok4",                # 0.68 GapG
    "gpt-5.2",              # 0.70 GapG (92% V)
    "gpt-5.4",
    "gemini-3.1-pro-preview", # 0.76 GapG (98% V)
    "gemini-3-pro-preview", # 0.70 GapG (77% V)
    "grok4.1fast",           # 1.00 GapG
    "deepseek-reasoner",     # 1.02 GapG
    "kimi-k2-thinking",     # above Hermes4/GPT-4o
    "hermes4",              # 2.29 GapG
    "gpt-4o",               # 2.52 GapG
]
MODEL_ORDER_OVERRIDE: Optional[List[str]] = None

VALIDITY_GROUP_SPECS = [
    (0, "High Validity ($>90\\%$)"),
    (1, "Intermediate Validity"),
    (2, "Low Validity ($<50\\%$)"),
]

SCENARIO_ORDER = ["ABD_FULL", "ABD_PARTIAL", "ABD_SKEPTICAL"]
SCENARIO_DISPLAY_NAMES = {
    "ABD_FULL": "ABD-Full",
    "ABD_PARTIAL": "ABD-Partial",
    "ABD_SKEPTICAL": "ABD-Skeptical",
}
SCENARIO_SHORT_NAMES = {
    "ABD_FULL": "FULL",
    "ABD_PARTIAL": "PARTIAL",
    "ABD_SKEPTICAL": "SKEPTICAL",
}

# AST size bins for complexity analysis
AST_BIN_EDGES = [0, 5, 10, 15, 20, 30, 50, float("inf")]

# =============================================================================
# Theory display configuration (consecutive numbering for paper)
# =============================================================================

# Map internal theory IDs to paper-friendly consecutive IDs
THEORY_PAPER_IDS = {
    "TH2": "T1",
    "TH7": "T2",
    "TH10": "T3",
    "TH11": "T4",
    "TH12": "T5",
    # SKEPTICAL-specific theories
    "TH3": "T6",
    "TH5": "T7",
}

# Theory formulas for the paper (LaTeX format)
THEORY_FORMULAS = {
    "TH2": r"$\exists y(R(x,y) \land P(y)) \land \neg\mathit{Ab}(x) \to Q(x)$",
    "TH7": r"$\exists y(R(x,y) \land P(y)) \land \neg\mathit{Ab}(x) \to \exists z(S(x,z) \land Q(z))$",
    "TH10": r"$\exists y(S(x,y) \land P(y)) \land \neg\mathit{Ab}(x) \to \exists z(R(x,z) \land Q(z))$",
    "TH11": r"$\exists y(R(x,y) \land P(y)) \land \neg\mathit{Ab}(x) \to \exists z(S(x,z) \land \forall w(R(z,w) \to P(w)))$",
    "TH12": r"$\exists y(R(x,y) \land P(y)) \land \neg\mathit{Ab}(x) \to \forall z(S(x,z) \to Q(z))$",
    # SKEPTICAL-specific theories
    "TH3": r"$P(x) \land \neg\mathit{Ab}(x) \to \exists y(R(x,y))$",
    "TH5": r"$P(x) \land \neg\mathit{Ab}(x) \to \forall y(R(x,y) \to Q(y))$",
}

# Theory descriptions for the paper
THEORY_DESCRIPTIONS = {
    "TH2": "Relational antecedent, unary consequent",
    "TH7": "Relational antecedent and consequent",
    "TH10": "Swapped relations (S-antecedent, R-consequent)",
    "TH11": "Nested universal in consequent",
    "TH12": "Universal consequent",
    # SKEPTICAL-specific theories
    "TH3": "Unary antecedent, existential consequent",
    "TH5": "Unary antecedent, universal consequent",
}


def get_theory_paper_id(theory_id: str) -> str:
    """Get paper-friendly theory ID."""
    return THEORY_PAPER_IDS.get(theory_id, theory_id)


def get_theory_formula(theory_id: str) -> str:
    """Get LaTeX formula for a theory."""
    return THEORY_FORMULAS.get(theory_id, "---")


def get_model_display_name(model_id: str) -> str:
    """Get display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def get_scenario_display_name(scenario: str) -> str:
    """Get paper-friendly scenario display name."""
    return SCENARIO_DISPLAY_NAMES.get(scenario, scenario.replace("_", "-"))


def get_scenario_short_name(scenario: str) -> str:
    """Get short scenario display name."""
    return SCENARIO_SHORT_NAMES.get(scenario, scenario.replace("ABD_", ""))


def get_validity_group_index(valid_pct: float) -> int:
    """Bucket models by overall repaired train validity."""
    if valid_pct > 90.0:
        return 0
    if valid_pct < 50.0:
        return 2
    return 1


def compute_model_order(records: List[EvalCacheRecord]) -> List[str]:
    """Compute paper model order: validity band, then overall train Gap."""
    by_model = aggregate_by_model(records)
    fallback_order = {m: i for i, m in enumerate(MODEL_ORDER)}

    def model_key(model_id: str) -> Tuple[int, float, int, str]:
        metrics = by_model[model_id]
        gap = metrics.avg_gap_vs_opt_train
        gap_key = gap if gap is not None else float("inf")
        return (
            get_validity_group_index(metrics.valid_pct),
            gap_key,
            fallback_order.get(model_id, 999),
            model_id,
        )

    return sorted(by_model.keys(), key=model_key)


def set_model_order(records: List[EvalCacheRecord]) -> None:
    """Set the dynamic paper model order for the current generation run."""
    global MODEL_ORDER_OVERRIDE
    MODEL_ORDER_OVERRIDE = compute_model_order(records)


def sort_models(model_ids: List[str]) -> List[str]:
    """Sort model IDs in presentation order."""
    order_source = MODEL_ORDER_OVERRIDE or MODEL_ORDER
    order_map = {m: i for i, m in enumerate(order_source)}
    fallback_map = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(model_ids, key=lambda m: (order_map.get(m, 999), fallback_map.get(m, 999), m))


def sort_scenarios(scenarios: List[str]) -> List[str]:
    """Sort scenarios in paper presentation order."""
    order_map = {s: i for i, s in enumerate(SCENARIO_ORDER)}
    return sorted(scenarios, key=lambda s: order_map.get(s, 999))


def sort_theories(theories: List[str]) -> List[str]:
    """Sort theories by paper ID order (T1..T7)."""
    def theory_key(theory_id: str) -> Tuple[int, str]:
        paper_id = get_theory_paper_id(theory_id)
        if paper_id.startswith("T") and paper_id[1:].isdigit():
            return (int(paper_id[1:]), theory_id)
        return (999, theory_id)

    return sorted(theories, key=theory_key)


# =============================================================================
# Utility functions for manifest and provenance
# =============================================================================


def get_git_sha() -> str:
    """Return static SHA for artifact builds."""
    return "artifact"


def compute_file_sha256(file_path: str) -> str:
    """Compute SHA256 hash of a file's contents."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()[:16]  # First 16 chars for brevity
    except Exception:
        return "unknown"


# =============================================================================
# Gold formula filtering
# =============================================================================


def is_exact_gold_match(r: EvalCacheRecord) -> bool:
    """Check if a record's prediction exactly matches the gold formula."""
    if r.gold_formula is not None and r.prediction.formula is not None:
        return r.prediction.formula == r.gold_formula
    return False


def filter_out_gold_matches(records: List[EvalCacheRecord]) -> List[EvalCacheRecord]:
    """Filter out records where the prediction exactly matches the gold formula."""
    return [r for r in records if not is_exact_gold_match(r)]


def dedupe_records_by_instance_model(
    records: List[EvalCacheRecord],
) -> Tuple[List[EvalCacheRecord], int, int]:
    """Keep only the last record for each (instance_id, model_id) pair.

    Returns:
        deduped_records: Unique records keyed by (instance_id, model_id)
        duplicate_rows_removed: Number of rows dropped
        duplicate_pairs: Number of repeated keys observed
    """
    latest_by_key: Dict[Tuple[str, str], EvalCacheRecord] = {}
    duplicate_rows_removed = 0
    duplicate_keys_seen: set[Tuple[str, str]] = set()

    for record in records:
        key = (record.instance_id, record.model_id)
        if key in latest_by_key:
            duplicate_rows_removed += 1
            duplicate_keys_seen.add(key)
        latest_by_key[key] = record

    return list(latest_by_key.values()), duplicate_rows_removed, len(duplicate_keys_seen)


# =============================================================================
# Aggregation functions
# =============================================================================

@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a group of records."""
    total: int = 0
    train_valid: int = 0
    strict_train_valid: int = 0
    holdout_valid: int = 0
    parse_ok: int = 0
    parse_error: int = 0
    auto_repaired: int = 0
    auto_closed_parens: int = 0

    # Failure mode tracking
    missing: int = 0  # No response at all
    invalid: int = 0  # Parsed but not valid on all train worlds

    # Sums for averaging (only valid records)
    gap_vs_opt_train_sum: float = 0.0
    gap_vs_opt_train_count: int = 0
    gap_vs_gold_train_sum: float = 0.0
    gap_vs_gold_train_count: int = 0
    gap_vs_opt_holdout_sum: float = 0.0
    gap_vs_opt_holdout_count: int = 0

    ast_size_sum: float = 0.0
    ast_size_count: int = 0
    valid_ast_size_sum: float = 0.0
    valid_ast_size_count: int = 0

    # Beats gold tracking
    beats_gold: int = 0
    beats_gold_improvement_sum: float = 0.0  # Sum of (gold_cost - pred_cost) for beaters
    beats_gold_ast_sum: float = 0.0  # Sum of AST sizes for beaters

    # Train-valid with holdout tracking
    train_valid_with_holdout: int = 0
    holdout_valid_given_train_valid: int = 0

    # Survivor-only tracking (train_valid AND holdout_valid)
    # For proper conditional ΔGap computation with same denominator
    survivor_count: int = 0
    survivor_gap_train_sum: float = 0.0
    survivor_gap_holdout_sum: float = 0.0

    def add(self, record: EvalCacheRecord):
        """Add a record to the aggregation."""
        self.total += 1

        # Check for missing response (no raw_text or empty)
        has_response = record.prediction.raw_text and len(record.prediction.raw_text.strip()) > 10

        if record.prediction.parse_ok:
            self.parse_ok += 1
        else:
            self.parse_error += 1
            if not has_response:
                self.missing += 1

        if record.prediction.auto_closed_parens > 0:
            self.auto_repaired += 1
            self.auto_closed_parens += record.prediction.auto_closed_parens

        if record.train_eval.train_all_valid:
            self.train_valid += 1
            if record.prediction.auto_closed_parens == 0:
                self.strict_train_valid += 1
            if record.prediction.ast_size is not None:
                self.valid_ast_size_sum += record.prediction.ast_size
                self.valid_ast_size_count += 1

            # Gap metrics (normalized per world)
            if record.train_eval.gap_vs_opt_train_norm is not None:
                self.gap_vs_opt_train_sum += record.train_eval.gap_vs_opt_train_norm
                self.gap_vs_opt_train_count += 1

            if record.train_eval.gap_vs_gold_train_norm is not None:
                self.gap_vs_gold_train_sum += record.train_eval.gap_vs_gold_train_norm
                self.gap_vs_gold_train_count += 1

                # Reference-beating: gap_vs_gold < 0 means model beats the planted reference
                if record.train_eval.gap_vs_gold_train_sum is not None and record.train_eval.gap_vs_gold_train_sum < 0:
                    self.beats_gold += 1
                    # Track improvement magnitude (per world)
                    n_worlds = record.num_train_worlds or 1
                    self.beats_gold_improvement_sum += abs(record.train_eval.gap_vs_gold_train_sum) / n_worlds
                    if record.prediction.ast_size is not None:
                        self.beats_gold_ast_sum += record.prediction.ast_size
        else:
            # Track invalid (parsed but not valid on all train worlds)
            if record.prediction.parse_ok:
                self.invalid += 1

        if record.holdout_eval.holdout_all_valid:
            self.holdout_valid += 1

            if record.holdout_eval.gap_vs_opt_holdout_norm is not None:
                self.gap_vs_opt_holdout_sum += record.holdout_eval.gap_vs_opt_holdout_norm
                self.gap_vs_opt_holdout_count += 1

        # AST size (for parse_ok records)
        if record.prediction.ast_size is not None:
            self.ast_size_sum += record.prediction.ast_size
            self.ast_size_count += 1

        # Conditional holdout metrics
        if record.train_eval.train_all_valid and record.num_holdout_worlds > 0:
            self.train_valid_with_holdout += 1
            if record.holdout_eval.holdout_all_valid:
                self.holdout_valid_given_train_valid += 1

        # Survivor tracking: train_valid AND holdout_valid (same denominator for ΔGap)
        if record.train_eval.train_all_valid and record.holdout_eval.holdout_all_valid:
            if (record.train_eval.gap_vs_opt_train_norm is not None and
                record.holdout_eval.gap_vs_opt_holdout_norm is not None):
                self.survivor_count += 1
                self.survivor_gap_train_sum += record.train_eval.gap_vs_opt_train_norm
                self.survivor_gap_holdout_sum += record.holdout_eval.gap_vs_opt_holdout_norm

    @property
    def valid_pct(self) -> float:
        return 100 * self.train_valid / max(1, self.total)

    @property
    def strict_valid_pct(self) -> float:
        return 100 * self.strict_train_valid / max(1, self.total)

    @property
    def holdout_valid_pct(self) -> float:
        return 100 * self.holdout_valid / max(1, self.total)

    @property
    def holdout_valid_given_train_pct(self) -> float:
        return 100 * self.holdout_valid_given_train_valid / max(1, self.train_valid_with_holdout)

    @property
    def parse_error_pct(self) -> float:
        # Exclude missing from parse errors to avoid double-counting
        return 100 * (self.parse_error - self.missing) / max(1, self.total)

    @property
    def repair_pct(self) -> float:
        return 100 * self.auto_repaired / max(1, self.total)

    @property
    def avg_gap_vs_opt_train(self) -> Optional[float]:
        if self.gap_vs_opt_train_count == 0:
            return None
        return self.gap_vs_opt_train_sum / self.gap_vs_opt_train_count

    @property
    def avg_gap_vs_gold_train(self) -> Optional[float]:
        if self.gap_vs_gold_train_count == 0:
            return None
        return self.gap_vs_gold_train_sum / self.gap_vs_gold_train_count

    @property
    def avg_gap_vs_opt_holdout(self) -> Optional[float]:
        if self.gap_vs_opt_holdout_count == 0:
            return None
        return self.gap_vs_opt_holdout_sum / self.gap_vs_opt_holdout_count

    @property
    def avg_ast_size(self) -> Optional[float]:
        if self.ast_size_count == 0:
            return None
        return self.ast_size_sum / self.ast_size_count

    @property
    def avg_valid_ast_size(self) -> Optional[float]:
        if self.valid_ast_size_count == 0:
            return None
        return self.valid_ast_size_sum / self.valid_ast_size_count

    @property
    def beats_gold_pct(self) -> float:
        return 100 * self.beats_gold / max(1, self.train_valid)

    @property
    def delta_gap(self) -> Optional[float]:
        """Holdout gap minus train gap, computed over survivors only.

        Survivors = instances where train_valid AND holdout_valid.
        This ensures we use the same denominator for both train and holdout gaps,
        avoiding comparison of different instance populations.
        """
        if self.survivor_count == 0:
            return None
        avg_train = self.survivor_gap_train_sum / self.survivor_count
        avg_holdout = self.survivor_gap_holdout_sum / self.survivor_count
        return avg_holdout - avg_train

    @property
    def survivor_avg_gap_train(self) -> Optional[float]:
        """Average train gap among survivors."""
        if self.survivor_count == 0:
            return None
        return self.survivor_gap_train_sum / self.survivor_count

    @property
    def survivor_avg_gap_holdout(self) -> Optional[float]:
        """Average holdout gap among survivors."""
        if self.survivor_count == 0:
            return None
        return self.survivor_gap_holdout_sum / self.survivor_count

    @property
    def missing_pct(self) -> float:
        """Percentage of records with no response."""
        return 100 * self.missing / max(1, self.total)

    @property
    def invalid_pct(self) -> float:
        """Percentage of records that parsed but weren't valid on all train worlds."""
        return 100 * self.invalid / max(1, self.total)

    @property
    def avg_beats_gold_improvement(self) -> Optional[float]:
        """Average per-world improvement (in exceptions) among gold-beaters."""
        if self.beats_gold == 0:
            return None
        return self.beats_gold_improvement_sum / self.beats_gold

    @property
    def avg_beats_gold_ast(self) -> Optional[float]:
        """Average AST size among gold-beaters."""
        if self.beats_gold == 0:
            return None
        return self.beats_gold_ast_sum / self.beats_gold


def aggregate_by_model(records: List[EvalCacheRecord]) -> Dict[str, AggregatedMetrics]:
    """Aggregate records by model."""
    by_model: Dict[str, AggregatedMetrics] = defaultdict(AggregatedMetrics)
    for r in records:
        by_model[r.model_id].add(r)
    return dict(by_model)


def aggregate_by_model_and_scenario(records: List[EvalCacheRecord]) -> Dict[str, Dict[str, AggregatedMetrics]]:
    """Aggregate records by model and scenario."""
    by_model_scenario: Dict[str, Dict[str, AggregatedMetrics]] = defaultdict(lambda: defaultdict(AggregatedMetrics))
    for r in records:
        by_model_scenario[r.model_id][r.scenario].add(r)
    return {m: dict(s) for m, s in by_model_scenario.items()}


def aggregate_by_model_and_theory(records: List[EvalCacheRecord]) -> Dict[str, Dict[str, AggregatedMetrics]]:
    """Aggregate records by model and theory."""
    by_model_theory: Dict[str, Dict[str, AggregatedMetrics]] = defaultdict(lambda: defaultdict(AggregatedMetrics))
    for r in records:
        by_model_theory[r.model_id][r.theory].add(r)
    return {m: dict(t) for m, t in by_model_theory.items()}


def aggregate_by_model_and_ast_bin(records: List[EvalCacheRecord]) -> Dict[str, Dict[str, AggregatedMetrics]]:
    """Aggregate records by model and AST size bin."""
    by_model_bin: Dict[str, Dict[str, AggregatedMetrics]] = defaultdict(lambda: defaultdict(AggregatedMetrics))

    for r in records:
        ast_size = r.prediction.ast_size
        if ast_size is None:
            continue

        # Find bin
        for i, (low, high) in enumerate(zip(AST_BIN_EDGES[:-1], AST_BIN_EDGES[1:])):
            if low <= ast_size < high:
                if high == float("inf"):
                    bin_label = f"[{int(low)},∞)"
                else:
                    bin_label = f"[{int(low)},{int(high)})"
                by_model_bin[r.model_id][bin_label].add(r)
                break

    return {m: dict(b) for m, b in by_model_bin.items()}


def get_dataset_summary(records: List[EvalCacheRecord]) -> Dict[str, Any]:
    """Get dataset summary statistics."""
    by_scenario = defaultdict(lambda: {"count": 0, "train_worlds": [], "holdout_worlds": [], "theories": set()})
    by_difficulty = defaultdict(int)

    seen_instances = set()
    for r in records:
        # Dedupe by instance (multiple models evaluate same instance)
        if r.instance_id in seen_instances:
            continue
        seen_instances.add(r.instance_id)

        by_scenario[r.scenario]["count"] += 1
        by_scenario[r.scenario]["train_worlds"].append(r.num_train_worlds)
        by_scenario[r.scenario]["holdout_worlds"].append(r.num_holdout_worlds)
        by_scenario[r.scenario]["theories"].add(r.theory)

        key = f"{r.scenario}_{r.difficulty}"
        by_difficulty[key] += 1

    # Compute averages
    summary = {
        "total_instances": len(seen_instances),
        "by_scenario": {},
        "by_difficulty": dict(by_difficulty),
    }
    for scenario, data in by_scenario.items():
        n = data["count"]
        summary["by_scenario"][scenario] = {
            "count": n,
            "avg_train_worlds": sum(data["train_worlds"]) / n if n else 0,
            "avg_holdout_worlds": sum(data["holdout_worlds"]) / n if n else 0,
            "theories": sorted(data["theories"]),
        }

    return summary


# =============================================================================
# LaTeX table generation
# =============================================================================

def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format percentage value."""
    return f"{value:.{decimals}f}\\%"


def fmt_float(value: Optional[float], digits: int = 2) -> str:
    """Format float value."""
    if value is None:
        return "---"
    return f"{value:.{digits}f}"


def fmt_int(value: int) -> str:
    """Format integer value."""
    return str(value)


def percentile_linear(values: List[float], q: float) -> Optional[float]:
    """Compute a percentile with linear interpolation."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def bold_if_best(value: float, values: List[float], higher_better: bool = True, fmt_func=fmt_pct) -> str:
    """Format value, bolding if it's the best."""
    if not values:
        return fmt_func(value)
    if higher_better:
        best = max(values)
    else:
        best = min(values)
    formatted = fmt_func(value)
    if abs(value - best) < 0.001:
        return f"\\textbf{{{formatted}}}"
    return formatted


def generate_dataset_summary_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate dataset summary table with theory definitions."""
    summary = get_dataset_summary(records)

    # Domain size ranges per scenario (from benchmark YAML source files)
    DOMAIN_SIZE_RANGES = {
        "ABD_FULL": "9--11",
        "ABD_PARTIAL": "9--11",
        "ABD_SKEPTICAL": "10--12",
    }

    # LaTeX table - Part 1: Dataset overview
    lines = [
        "% Dataset Summary Table with Theory Definitions",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Abduction Benchmark Summary.} (a) Dataset statistics by scenario. (b) Default theories used in the benchmark; each theory defines when $\\mathit{Ab}(x)$ blocks a default conclusion.}",
        "\\label{tab:dataset_summary}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "",
        "% Part (a): Dataset overview",
        "\\begin{tabular}{@{}lrrrcl@{}}",
        "\\toprule",
        "\\multicolumn{6}{@{}l}{\\textbf{(a) Dataset Statistics}} \\\\",
        "\\midrule",
        "Scenario & N & Prompt & Holdout & $|D|$ & Theories \\\\",
        "\\midrule",
    ]

    for scenario in sorted(summary["by_scenario"].keys()):
        data = summary["by_scenario"][scenario]
        # Use paper theory IDs (T1-T5)
        theories_paper = sorted([get_theory_paper_id(t) for t in data["theories"]])
        # Wrap theories across two lines if more than 5 (for SKEPTICAL)
        if len(theories_paper) > 5:
            # Split: first 5 on line 1, rest on line 2
            line1 = ", ".join(theories_paper[:5])
            line2 = ", ".join(theories_paper[5:])
            theories_str = f"\\makecell[l]{{{line1},\\\\{line2}}}"
        else:
            theories_str = ", ".join(theories_paper)
        scenario_short = scenario.replace("ABD_", "")
        domain_range = DOMAIN_SIZE_RANGES.get(scenario, "---")
        lines.append(
            f"{scenario_short} & {data['count']} & {data['avg_train_worlds']:.1f} & "
            f"{data['avg_holdout_worlds']:.1f} & {domain_range} & {theories_str} \\\\"
        )

    lines.append("\\midrule")
    lines.append(f"\\textbf{{Total}} & \\textbf{{{summary['total_instances']}}} & & & & \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    lines.append("\\vspace{1em}")
    lines.append("")

    # Part (b): Theory definitions
    lines.append("% Part (b): Theory definitions")
    lines.append("\\footnotesize")
    lines.append("\\begin{tabular}{@{}cp{5.8cm}@{}}")
    lines.append("\\toprule")
    lines.append("\\multicolumn{2}{@{}l}{\\textbf{(b) Default Theories}: $\\phi(x) \\land \\neg\\mathit{Ab}(x) \\to \\psi(x)$} \\\\")
    lines.append("\\midrule")
    lines.append("ID & Antecedent $\\to$ Consequent \\\\")
    lines.append("\\midrule")

    # Sort theories by paper ID - use abbreviated formulas
    theory_short = {
        "TH2": (r"$\exists y(R(x,y) \land P(y))$", r"$Q(x)$"),
        "TH7": (r"$\exists y(R(x,y) \land P(y))$", r"$\exists z(S(x,z) \land Q(z))$"),
        "TH10": (r"$\exists y(S(x,y) \land P(y))$", r"$\exists z(R(x,z) \land Q(z))$"),
        "TH11": (r"$\exists y(R(x,y) \land P(y))$", r"$\exists z(S(x,z) \land \forall w(R(z,w) \to P(w)))$"),
        "TH12": (r"$\exists y(R(x,y) \land P(y))$", r"$\forall z(S(x,z) \to Q(z))$"),
        # SKEPTICAL-specific theories
        "TH3": (r"$P(x)$", r"$\exists y(R(x,y))$"),
        "TH5": (r"$P(x)$", r"$\forall y(R(x,y) \to Q(y))$"),
    }
    sorted_theories = sorted(THEORY_PAPER_IDS.keys(), key=lambda t: THEORY_PAPER_IDS[t])
    for theory_id in sorted_theories:
        paper_id = get_theory_paper_id(theory_id)
        ante, cons = theory_short.get(theory_id, ("---", "---"))
        lines.append(f"{paper_id} & {ante} $\\to$ {cons} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Write LaTeX
    tex_path = outdir / "abd_dataset_summary.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")

    # Write CSV
    csv_path = outdir / "abd_dataset_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Scenario", "Count", "AvgTrainWorlds", "AvgHoldoutWorlds", "DomainSizes", "Theories"])
        for scenario in sorted(summary["by_scenario"].keys()):
            data = summary["by_scenario"][scenario]
            writer.writerow([
                scenario,
                data["count"],
                f"{data['avg_train_worlds']:.1f}",
                f"{data['avg_holdout_worlds']:.1f}",
                DOMAIN_SIZE_RANGES.get(scenario, ""),
                ", ".join(data["theories"]),
            ])
    print(f"Wrote {csv_path}")


def generate_main_train_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate main model comparison table (TRAIN metrics) with scenario breakdown.

    Double-column table showing ABD_FULL, ABD_PARTIAL, and Overall metrics.
    """
    set_model_order(records)
    by_model = aggregate_by_model(records)
    by_model_scenario = aggregate_by_model_and_scenario(records)
    models = sort_models(list(by_model.keys()))

    if not models:
        print("No models found, skipping main train table")
        return

    # Dynamically get scenarios from data (supports ABD_FULL, ABD_PARTIAL, ABD_SKEPTICAL)
    scenarios = sort_scenarios(list(set(r.scenario for r in records)))

    # Collect best values for highlighting (per scenario)
    best_values = {}
    for scenario in scenarios:
        valid_pcts = []
        strict_valid_pcts = []
        avg_valid_asts = []
        gap_opts = []
        gap_golds = []
        for model in models:
            m = by_model_scenario.get(model, {}).get(scenario, AggregatedMetrics())
            if m.total > 0:
                valid_pcts.append(m.valid_pct)
                strict_valid_pcts.append(m.strict_valid_pct)
                if m.avg_valid_ast_size is not None:
                    avg_valid_asts.append(m.avg_valid_ast_size)
                if m.avg_gap_vs_opt_train is not None:
                    gap_opts.append(m.avg_gap_vs_opt_train)
                if m.avg_gap_vs_gold_train is not None:
                    gap_golds.append(m.avg_gap_vs_gold_train)
        best_values[scenario] = {
            "valid": max(valid_pcts) if valid_pcts else None,
            "strict_valid": max(strict_valid_pcts) if strict_valid_pcts else None,
            "ast": min(avg_valid_asts) if avg_valid_asts else None,
            "gap": min(gap_opts) if gap_opts else None,
            "gap_gold": min(gap_golds) if gap_golds else None,
        }

    # Overall best values
    overall_valid = [by_model[m].valid_pct for m in models]
    overall_strict_valid = [by_model[m].strict_valid_pct for m in models]
    overall_valid_ast = [by_model[m].avg_valid_ast_size for m in models if by_model[m].avg_valid_ast_size is not None]
    overall_gap_opt = [by_model[m].avg_gap_vs_opt_train for m in models if by_model[m].avg_gap_vs_opt_train is not None]
    overall_gap_gold = [by_model[m].avg_gap_vs_gold_train for m in models if by_model[m].avg_gap_vs_gold_train is not None]
    best_values["Overall"] = {
        "valid": max(overall_valid) if overall_valid else None,
        "strict_valid": max(overall_strict_valid) if overall_strict_valid else None,
        "ast": min(overall_valid_ast) if overall_valid_ast else None,
        "gap": min(overall_gap_opt) if overall_gap_opt else None,
        "gap_gold": min(overall_gap_gold) if overall_gap_gold else None,
    }

    # Compact table with dynamic scenario breakdown (PV%, PSV%, AST, Gap, GRef).
    # Always use table* for 3 scenarios with 5 cols each + overall = 20 data cols.
    num_scenarios = len(scenarios)
    table_env = "table*"  # Always wide for this many columns

    # Build tabular spec: Model + 5 cols per scenario (PV%, PSV%, AST, Gap, GRef) + 5 cols for Overall
    tabular_cols = "l " + "rrrrr " * num_scenarios + "rrrrr"

    # Build multicolumn header
    header_parts = [""]
    for scenario in scenarios:
        scenario_short = get_scenario_short_name(scenario)
        header_parts.append(f"\\multicolumn{{5}}{{c}}{{{scenario_short}}}")
    header_parts.append("\\multicolumn{5}{c}{Overall}")
    header_line = " & ".join(header_parts) + " \\\\"

    # Build cmidrule separators
    cmidrule_parts = []
    col = 2
    for _ in scenarios:
        cmidrule_parts.append(f"\\cmidrule(lr){{{col}-{col+4}}}")
        col += 5
    cmidrule_parts.append(f"\\cmidrule(lr){{{col}-{col+4}}}")  # Overall (5 cols)
    cmidrule_line = " ".join(cmidrule_parts)

    # Build subheader
    subheader_parts = ["Model"]
    for _ in scenarios:
        subheader_parts.extend(["PV\\%", "PSV\\%", "AST", "Gap", "GRef"])
    subheader_parts.extend(["PV\\%", "PSV\\%", "AST", "Gap", "GRef"])  # Overall
    subheader_line = " & ".join(subheader_parts) + " \\\\"

    lines = [
        "% Main Model Comparison Table (PROMPT) - Dynamic scenario breakdown with GRef",
        f"\\begin{{{table_env}}}[t]",
        "\\centering",
        "\\caption{\\textbf{Abduction Prompt-Set Performance.} Models are grouped by overall repaired prompt validity into high ($>90\\%$), intermediate, and low ($<50\\%$) bands; within each band, rows are ordered by overall Gap. Percentages are benchmark instances, not worlds. PV\\% = prompt-valid after conservative suffix repair; PSV\\% = strict prompt-validity without repair; AST = mean formula size among prompt-valid formulas; Gap = normalized extra exceptions above the solver lower bound; GRef = normalized extra exceptions vs. the planted generator reference. Gap and GRef average over prompt-valid formulas only. Lower is better for AST, Gap, and GRef. Best values bold.}",
        "\\label{tab:abd_main_train}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{1.2pt}",
        f"\\begin{{tabular}}{{@{{}}{tabular_cols}@{{}}}}",
        "\\toprule",
        header_line,
        cmidrule_line,
        subheader_line,
        "\\midrule",
    ]

    total_cols = 1 + 5 * num_scenarios + 5
    grouped_models: Dict[int, List[str]] = {group_idx: [] for group_idx, _ in VALIDITY_GROUP_SPECS}
    for model in models:
        grouped_models[get_validity_group_index(by_model[model].valid_pct)].append(model)

    emitted_group = False
    for group_idx, group_label in VALIDITY_GROUP_SPECS:
        group_models = grouped_models.get(group_idx, [])
        if not group_models:
            continue
        if emitted_group:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{total_cols}}}{{@{{}}l}}{{\\textit{{{group_label}}}}} \\\\")
        emitted_group = True

        for model in group_models:
            display = get_model_display_name(model)
            row_parts = [display]

            # Per-scenario metrics (PV%, PSV%, AST, Gap, GRef)
            for scenario in scenarios:
                m = by_model_scenario.get(model, {}).get(scenario, AggregatedMetrics())
                if m.total > 0:
                    # Valid% as integer
                    valid_int = int(round(m.valid_pct))
                    valid_str = str(valid_int)
                    if best_values[scenario]["valid"] is not None and abs(m.valid_pct - best_values[scenario]["valid"]) < 0.5:
                        valid_str = f"\\textbf{{{valid_str}}}"

                    strict_valid_int = int(round(m.strict_valid_pct))
                    strict_valid_str = str(strict_valid_int)
                    if (best_values[scenario]["strict_valid"] is not None and
                            abs(m.strict_valid_pct - best_values[scenario]["strict_valid"]) < 0.5):
                        strict_valid_str = f"\\textbf{{{strict_valid_str}}}"

                    avg_valid_ast = m.avg_valid_ast_size
                    if avg_valid_ast is not None:
                        ast_str = f"{avg_valid_ast:.1f}"
                        if (best_values[scenario]["ast"] is not None and
                                abs(avg_valid_ast - best_values[scenario]["ast"]) < 0.05):
                            ast_str = f"\\textbf{{{ast_str}}}"
                    else:
                        ast_str = "---"

                    gap = m.avg_gap_vs_opt_train
                    if gap is not None:
                        gap_str = f"{gap:.2f}"
                        if best_values[scenario]["gap"] is not None and abs(gap - best_values[scenario]["gap"]) < 0.005:
                            gap_str = f"\\textbf{{{gap_str}}}"
                    else:
                        gap_str = "---"

                    gap_gold = m.avg_gap_vs_gold_train
                    if gap_gold is not None:
                        gap_gold_str = f"{gap_gold:.2f}"
                        if best_values[scenario]["gap_gold"] is not None and abs(gap_gold - best_values[scenario]["gap_gold"]) < 0.005:
                            gap_gold_str = f"\\textbf{{{gap_gold_str}}}"
                    else:
                        gap_gold_str = "---"

                    row_parts.extend([valid_str, strict_valid_str, ast_str, gap_str, gap_gold_str])
                else:
                    row_parts.extend(["---", "---", "---", "---", "---"])

            m_overall = by_model[model]
            overall_valid_int = int(round(m_overall.valid_pct))
            overall_valid_str = str(overall_valid_int)
            if best_values["Overall"]["valid"] is not None and abs(m_overall.valid_pct - best_values["Overall"]["valid"]) < 0.5:
                overall_valid_str = f"\\textbf{{{overall_valid_str}}}"

            overall_strict_valid_int = int(round(m_overall.strict_valid_pct))
            overall_strict_valid_str = str(overall_strict_valid_int)
            if (best_values["Overall"]["strict_valid"] is not None and
                    abs(m_overall.strict_valid_pct - best_values["Overall"]["strict_valid"]) < 0.5):
                overall_strict_valid_str = f"\\textbf{{{overall_strict_valid_str}}}"

            overall_avg_valid_ast = m_overall.avg_valid_ast_size
            if overall_avg_valid_ast is not None:
                overall_ast_str = f"{overall_avg_valid_ast:.1f}"
                if (best_values["Overall"]["ast"] is not None and
                        abs(overall_avg_valid_ast - best_values["Overall"]["ast"]) < 0.05):
                    overall_ast_str = f"\\textbf{{{overall_ast_str}}}"
            else:
                overall_ast_str = "---"

            overall_gap_opt = m_overall.avg_gap_vs_opt_train
            if overall_gap_opt is not None:
                overall_gap_opt_str = f"{overall_gap_opt:.2f}"
                if best_values["Overall"]["gap"] is not None and abs(overall_gap_opt - best_values["Overall"]["gap"]) < 0.005:
                    overall_gap_opt_str = f"\\textbf{{{overall_gap_opt_str}}}"
            else:
                overall_gap_opt_str = "---"

            overall_gap_gold = m_overall.avg_gap_vs_gold_train
            if overall_gap_gold is not None:
                overall_gap_gold_str = f"{overall_gap_gold:.2f}"
                if best_values["Overall"]["gap_gold"] is not None and abs(overall_gap_gold - best_values["Overall"]["gap_gold"]) < 0.005:
                    overall_gap_gold_str = f"\\textbf{{{overall_gap_gold_str}}}"
            else:
                overall_gap_gold_str = "---"

            row_parts.extend([
                overall_valid_str,
                overall_strict_valid_str,
                overall_ast_str,
                overall_gap_opt_str,
                overall_gap_gold_str,
            ])

            lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\end{{{table_env}}}")

    # Write LaTeX
    tex_path = outdir / "abd_main_train.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")

    # Write CSV
    csv_path = outdir / "abd_main_train.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Build dynamic header
        csv_header = ["Model"]
        for scenario in scenarios:
            scenario_short = get_scenario_short_name(scenario)
            csv_header.extend([
                f"{scenario_short}_ValidPct",
                f"{scenario_short}_StrictValidPct",
                f"{scenario_short}_ValidAST",
                f"{scenario_short}_Gap",
                f"{scenario_short}_GapRef",
            ])
        csv_header.extend(["Overall_ValidPct", "Overall_StrictValidPct", "Overall_ValidAST", "Overall_Gap", "Overall_GapRef"])
        writer.writerow(csv_header)
        for model in models:
            m_overall = by_model[model]
            row = [get_model_display_name(model)]
            for scenario in scenarios:
                m = by_model_scenario.get(model, {}).get(scenario, AggregatedMetrics())
                if m.total > 0:
                    row.extend([
                        f"{m.valid_pct:.1f}",
                        f"{m.strict_valid_pct:.1f}",
                        fmt_float(m.avg_valid_ast_size, 1),
                        fmt_float(m.avg_gap_vs_opt_train),
                        fmt_float(m.avg_gap_vs_gold_train),
                    ])
                else:
                    row.extend(["", "", "", "", ""])
            row.extend([
                f"{m_overall.valid_pct:.1f}",
                f"{m_overall.strict_valid_pct:.1f}",
                fmt_float(m_overall.avg_valid_ast_size, 1),
                fmt_float(m_overall.avg_gap_vs_opt_train),
                fmt_float(m_overall.avg_gap_vs_gold_train),
            ])
            writer.writerow(row)
    print(f"Wrote {csv_path}")


def generate_scenario_breakdown_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate scenario breakdown table with Gap and Gap Gold for all scenarios (ABD_FULL, ABD_PARTIAL, ABD_SKEPTICAL)."""
    by_model_scenario = aggregate_by_model_and_scenario(records)
    models = sort_models(list(by_model_scenario.keys()))
    scenarios = sorted(set(r.scenario for r in records))

    if not models or not scenarios:
        print("No data for scenario breakdown, skipping")
        return

    # 3 columns per scenario: PV%, Gap (vs opt), GapRef
    lines = [
        "% Scenario Breakdown Table (PROMPT) with Gap and GapRef",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Abduction Prompt-Set Performance by Scenario.} PV\\% = prompt-validity after conservative suffix repair; Gap = normalized extra exceptions above the solver lower bound, averaged over prompt-valid formulas only; GapRef = normalized extra exceptions vs. the planted generator reference, averaged over prompt-valid formulas only. Lower is better for Gap and GapRef.}",
        "\\label{tab:abd_scenario_breakdown}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}l" + "rrr" * len(scenarios) + "@{}}",
        "\\toprule",
    ]

    # Header with multicolumn for scenarios
    header_parts = ["Model"]
    for scenario in scenarios:
        scenario_escaped = scenario.replace("_", "\\_")
        header_parts.append(f"\\multicolumn{{3}}{{c}}{{{scenario_escaped}}}")
    lines.append(" & ".join(header_parts) + " \\\\")

    # Cmidrule separators
    cmidrule_parts = []
    col = 2
    for _ in scenarios:
        cmidrule_parts.append(f"\\cmidrule(lr){{{col}-{col+2}}}")
        col += 3
    lines.append(" ".join(cmidrule_parts))

    # Sub-header
    subheader_parts = [""]
    for _ in scenarios:
        subheader_parts.extend(["PV\\%", "Gap", "GapRef"])
    lines.append(" & ".join(subheader_parts) + " \\\\")
    lines.append("\\midrule")

    for model in models:
        display = get_model_display_name(model)
        row_parts = [display]
        for scenario in scenarios:
            m = by_model_scenario[model].get(scenario, AggregatedMetrics())
            row_parts.append(fmt_pct(m.valid_pct))
            row_parts.append(fmt_float(m.avg_gap_vs_opt_train))
            row_parts.append(fmt_float(m.avg_gap_vs_gold_train))
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Write LaTeX
    tex_path = outdir / "abd_scenario_breakdown.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")

    # Write CSV
    csv_path = outdir / "abd_scenario_breakdown.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Model"]
        for s in scenarios:
            header.extend([f"{s}_ValidPct", f"{s}_Gap", f"{s}_GapRef"])
        writer.writerow(header)
        for model in models:
            row = [get_model_display_name(model)]
            for scenario in scenarios:
                m = by_model_scenario[model].get(scenario, AggregatedMetrics())
                row.append(f"{m.valid_pct:.1f}")
                row.append(fmt_float(m.avg_gap_vs_opt_train))
                row.append(fmt_float(m.avg_gap_vs_gold_train))
            writer.writerow(row)
    print(f"Wrote {csv_path}")


def generate_theory_breakdown_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate theory breakdown table with PV% and GRef."""
    by_model_theory = aggregate_by_model_and_theory(records)
    models = sort_models(list(by_model_theory.keys()))
    theories = sorted(set(r.theory for r in records))

    if not models or not theories:
        print("No data for theory breakdown, skipping")
        return

    # Always use table* for this wide table (2 cols per model)
    table_env = "table*"

    lines = [
        "% Theory Breakdown Table (PROMPT) - PV% and GRef",
        f"\\begin{{{table_env}}}[t]",
        "\\centering",
        "\\caption{\\textbf{Abduction Prompt-Set Performance by Theory.} PV\\% = prompt-validity after conservative suffix repair; GRef = normalized extra exceptions vs. the planted generator reference, averaged over prompt-valid formulas only. T1--T5 aggregate across all scenarios; T6--T7 are ABD-Skeptical only. Lower is better for GRef. Best values bold.}",
        "\\label{tab:abd_theory_breakdown}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{1.0pt}",
        "\\begin{tabular}{@{}l" + "rr" * len(models) + "@{}}",
        "\\toprule",
    ]

    # Header row 1: Model names spanning 2 columns each
    header1_parts = [""]
    for m in models:
        header1_parts.append(f"\\multicolumn{{2}}{{c}}{{{get_model_display_name(m)}}}")
    lines.append(" & ".join(header1_parts) + " \\\\")

    # Cmidrules under each model
    cmidrule_parts = []
    col = 2
    for _ in models:
        cmidrule_parts.append(f"\\cmidrule(lr){{{col}-{col+1}}}")
        col += 2
    lines.append(" ".join(cmidrule_parts))

    # Header row 2: PV% and GRef for each model
    header2_parts = ["Theory"]
    for _ in models:
        header2_parts.extend(["PV\\%", "GRef"])
    lines.append(" & ".join(header2_parts) + " \\\\")
    lines.append("\\midrule")

    # Sort theories by paper ID
    theories_sorted = sorted(theories, key=lambda t: THEORY_PAPER_IDS.get(t, t))

    # Compute best values per theory for bolding
    best_by_theory: Dict[str, Dict[str, Optional[float]]] = {}
    for theory in theories_sorted:
        v_pcts = []
        gap_golds = []
        for model in models:
            m = by_model_theory[model].get(theory, AggregatedMetrics())
            if m.total > 0:
                v_pcts.append(m.valid_pct)
                if m.avg_gap_vs_gold_train is not None:
                    gap_golds.append(m.avg_gap_vs_gold_train)
        best_by_theory[theory] = {
            "best_v": max(v_pcts) if v_pcts else None,
            "best_gap_gold": min(gap_golds) if gap_golds else None,  # Lower is better
        }

    for theory in theories_sorted:
        paper_id = get_theory_paper_id(theory)
        row = [paper_id]
        best = best_by_theory[theory]
        for model in models:
            m = by_model_theory[model].get(theory, AggregatedMetrics())
            # Valid% as integer
            valid_int = int(round(m.valid_pct))
            v_str = str(valid_int)
            if best["best_v"] is not None and abs(m.valid_pct - best["best_v"]) < 0.5:
                v_str = f"\\textbf{{{v_str}}}"
            row.append(v_str)
            # GapRef
            gap_gold = m.avg_gap_vs_gold_train
            if gap_gold is not None:
                g_str = f"{gap_gold:.2f}"
                if best["best_gap_gold"] is not None and abs(gap_gold - best["best_gap_gold"]) < 0.005:
                    g_str = f"\\textbf{{{g_str}}}"
                row.append(g_str)
            else:
                row.append("---")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\end{{{table_env}}}")

    # Write LaTeX
    tex_path = outdir / "abd_theory_breakdown.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")

    # Write CSV
    csv_path = outdir / "abd_theory_breakdown.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header with PV% and GapRef for each model
        csv_header = ["Theory"]
        for m in models:
            display = get_model_display_name(m)
            csv_header.extend([f"{display}_PV%", f"{display}_GapRef"])
        writer.writerow(csv_header)
        for theory in theories_sorted:
            paper_id = get_theory_paper_id(theory)
            row = [paper_id]
            for model in models:
                m = by_model_theory[model].get(theory, AggregatedMetrics())
                row.append(f"{m.valid_pct:.1f}")
                row.append(fmt_float(m.avg_gap_vs_gold_train))
            writer.writerow(row)
    print(f"Wrote {csv_path}")


def generate_holdout_by_scenario_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate holdout generalization by scenario table.

    Shows PV%, HV%, PGap, HGap, ΔGap for each model × scenario.
    Includes all valid predictions (including exact gold matches).
    """
    by_model_scenario = aggregate_by_model_and_scenario(records)
    models = sort_models(list(by_model_scenario.keys()))
    scenarios = sorted(set(r.scenario for r in records))

    if not models or not scenarios:
        print("No data for holdout by scenario, skipping")
        return

    # Compute best values per scenario for bolding
    best_by_scenario: Dict[str, Dict[str, Optional[float]]] = {}
    for scenario in scenarios:
        h_valids = []
        deltas = []
        for model in models:
            m = by_model_scenario[model].get(scenario, AggregatedMetrics())
            if m.total > 0:
                if m.holdout_valid_pct is not None:
                    h_valids.append(m.holdout_valid_pct)
                if m.delta_gap is not None:
                    deltas.append(m.delta_gap)
        best_by_scenario[scenario] = {
            "best_h_valid": max(h_valids) if h_valids else None,
            "best_delta": min(deltas) if deltas else None,  # Lower is better
        }

    # 5 columns per scenario: PV%, HV%, PGap, HGap, ΔGap
    lines = [
        "% Holdout Generalization by Scenario Table",
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{\\textbf{Holdout Generalization by Scenario.} PV\\%/HV\\% are prompt/holdout validity; PGap/HGap are normalized per-world gaps on their respective valid subsets; $\\Delta$Gap is computed on survivors valid on both prompt and holdout. Best HV\\% and $\\Delta$Gap per scenario bold.}",
        "\\label{tab:abd_holdout_by_scenario}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}l" + "rrrrr" * len(scenarios) + "@{}}",
        "\\toprule",
    ]

    # Header with multicolumn for scenarios
    header_parts = ["Model"]
    for scenario in scenarios:
        scenario_short = scenario.replace("ABD_", "")
        header_parts.append(f"\\multicolumn{{5}}{{c}}{{{scenario_short}}}")
    lines.append(" & ".join(header_parts) + " \\\\")

    # Cmidrule separators
    cmidrule_parts = []
    col = 2
    for _ in scenarios:
        cmidrule_parts.append(f"\\cmidrule(lr){{{col}-{col+4}}}")
        col += 5
    lines.append(" ".join(cmidrule_parts))

    # Sub-header
    subheader_parts = [""]
    for _ in scenarios:
        subheader_parts.extend(["PV\\%", "HV\\%", "PGap", "HGap", "$\\Delta$Gap"])
    lines.append(" & ".join(subheader_parts) + " \\\\")
    lines.append("\\midrule")

    for model in models:
        display = get_model_display_name(model)
        row_parts = [display]
        for scenario in scenarios:
            m = by_model_scenario[model].get(scenario, AggregatedMetrics())
            best = best_by_scenario[scenario]

            row_parts.append(fmt_pct(m.valid_pct, decimals=0))

            # HV% with bold for best
            h_valid_str = fmt_pct(m.holdout_valid_pct, decimals=0)
            if best["best_h_valid"] is not None and m.holdout_valid_pct is not None and abs(m.holdout_valid_pct - best["best_h_valid"]) < 0.5:
                h_valid_str = f"\\textbf{{{h_valid_str}}}"
            row_parts.append(h_valid_str)

            row_parts.append(fmt_float(m.avg_gap_vs_opt_train))
            row_parts.append(fmt_float(m.avg_gap_vs_opt_holdout))

            # ΔGap with bold for best (lowest)
            delta = m.delta_gap
            if delta is not None:
                delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
                if best["best_delta"] is not None and abs(delta - best["best_delta"]) < 0.005:
                    delta_str = f"\\textbf{{{delta_str}}}"
                row_parts.append(delta_str)
            else:
                row_parts.append("---")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    # Write LaTeX
    tex_path = outdir / "abd_holdout_by_scenario.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")


def generate_holdout_by_theory_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate holdout generalization by theory table.

    Shows PV%, HV%, PGap, HGap, ΔGap for each model × theory.
    Includes all valid predictions (including exact gold matches).
    """
    by_model_theory = aggregate_by_model_and_theory(records)
    models = sort_models(list(by_model_theory.keys()))
    theories = sorted(set(r.theory for r in records))
    min_survivors_display = 5

    if not models or not theories:
        print("No data for holdout by theory, skipping")
        return

    # Sort theories by paper ID
    theories_sorted = sorted(theories, key=lambda t: THEORY_PAPER_IDS.get(t, t))

    # Compute best ΔGap per theory for bolding (lowest is better)
    best_by_theory: Dict[str, Optional[float]] = {}
    for theory in theories_sorted:
        deltas = []
        for model in models:
            m = by_model_theory[model].get(theory, AggregatedMetrics())
            if m.delta_gap is not None and m.survivor_count >= min_survivors_display:
                deltas.append(m.delta_gap)
        best_by_theory[theory] = min(deltas) if deltas else None

    # Compact table: models as columns, theories as rows
    # Each cell shows "ΔGap (N)" where N is count
    # Use table* for two-column span when many models
    use_wide = len(models) > 4
    table_env = "table*" if use_wide else "table"
    # Use footnotesize for wide tables
    font_size = "\\footnotesize" if use_wide else "\\small"

    lines = [
        "% Holdout Generalization by Theory Table",
        f"\\begin{{{table_env}}}[t]",
        "\\centering",
        "\\caption{\\textbf{Holdout $\\Delta$Gap by Theory.} T1--T5 aggregate across all three scenarios; T6--T7 are ABD-Skeptical only. Each cell shows $\\Delta$Gap = HGap $-$ PGap over \\emph{survivors} valid on both prompt and holdout, with N = survivor count. Cells with fewer than 5 survivors are suppressed. Lower is better. Best per theory bold.}",
        "\\label{tab:abd_holdout_by_theory}",
        font_size,
    ]
    if use_wide:
        lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.extend([
        "\\begin{tabular}{@{}l" + "r" * len(models) + "@{}}",
        "\\toprule",
    ])

    # Header with models
    header = ["Theory"] + [get_model_display_name(m) for m in models]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for theory in theories_sorted:
        paper_id = get_theory_paper_id(theory)
        row = [paper_id]
        best_delta = best_by_theory[theory]
        for model in models:
            m = by_model_theory[model].get(theory, AggregatedMetrics())
            delta = m.delta_gap
            # Use survivor_count since ΔGap is computed over survivors
            if delta is not None and m.survivor_count >= min_survivors_display:
                delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
                cell = f"{delta_str} ({m.survivor_count})"
                if best_delta is not None and abs(delta - best_delta) < 0.05:
                    cell = f"\\textbf{{{delta_str}}} ({m.survivor_count})"
                row.append(cell)
            else:
                row.append("---")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\end{{{table_env}}}")

    # Write LaTeX
    tex_path = outdir / "abd_holdout_by_theory.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")


def generate_complexity_by_model_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate complexity vs generalization table for all models.

    AST bins as rows, models as columns, showing ΔGap for each.
    """
    by_model_bin = aggregate_by_model_and_ast_bin(records)
    models = sort_models(list(by_model_bin.keys()))

    if not models:
        print("No data for complexity by model, skipping")
        return

    # Get all bins across all models
    all_bins = set()
    for model_bins in by_model_bin.values():
        all_bins.update(model_bins.keys())

    # Sort bins
    def bin_sort_key(b: str) -> int:
        try:
            return int(b.split(",")[0].strip("["))
        except Exception:
            return 999
    bins = sorted(all_bins, key=bin_sort_key)

    if not bins:
        print("No AST bins found, skipping complexity by model table")
        return

    # Use table* for two-column span when many models
    use_wide = len(models) > 4
    table_env = "table*" if use_wide else "table"
    # Use scriptsize for 8+ models to fit margins
    font_size = "\\scriptsize" if len(models) >= 8 else ("\\footnotesize" if use_wide else "\\small")

    lines = [
        "% Complexity vs Generalization by Model Table",
        f"\\begin{{{table_env}}}[t]",
        "\\centering",
        "\\caption{\\textbf{$\\Delta$Gap by AST Bin and Model.} Aggregated across FULL and PARTIAL tasks. Each cell shows $\\Delta$Gap (N valid). Lower is better.}",
        "\\label{tab:abd_complexity_by_model}",
        font_size,
    ]
    if use_wide:
        lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.extend([
        "\\begin{tabular}{@{}l" + "r" * len(models) + "@{}}",
        "\\toprule",
    ])

    # Header with models
    header = ["AST Bin"] + [get_model_display_name(m) for m in models]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for bin_label in bins:
        # Escape brackets for LaTeX
        escaped_label = bin_label.replace("[", "{[}").replace(")", "{)}")
        row = [escaped_label]
        for model in models:
            m = by_model_bin[model].get(bin_label, AggregatedMetrics())
            delta = m.delta_gap
            if delta is not None and m.train_valid > 0:
                delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
                row.append(f"{delta_str} ({m.train_valid})")
            else:
                row.append("---")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\end{{{table_env}}}")

    # Write LaTeX
    tex_path = outdir / "abd_complexity_by_model.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")


def generate_holdout_summary_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate holdout generalization summary table (B1 enhanced with AvgAST)."""
    by_model = aggregate_by_model(records)
    models = sort_models(list(by_model.keys()))

    if not models:
        print("No models found, skipping holdout summary table")
        return

    # Compute best values for bolding
    t_vals = [by_model[m].valid_pct for m in models]
    h_vals = [by_model[m].holdout_valid_pct for m in models]
    t_gaps = [by_model[m].avg_gap_vs_opt_train for m in models if by_model[m].avg_gap_vs_opt_train is not None]
    h_gaps = [by_model[m].avg_gap_vs_opt_holdout for m in models if by_model[m].avg_gap_vs_opt_holdout is not None]
    deltas = [by_model[m].delta_gap for m in models if by_model[m].delta_gap is not None]

    best_t_val = max(t_vals) if t_vals else None
    best_h_val = max(h_vals) if h_vals else None
    best_t_gap = min(t_gaps) if t_gaps else None
    best_h_gap = min(h_gaps) if h_gaps else None
    best_delta = min(deltas) if deltas else None  # Lower is better

    lines = [
        "% Holdout Generalization Summary Table (with AvgAST)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Generalization Drop Profile.} Aggregated across all scenarios. PV\\%/HV\\% are instance-level prompt/holdout validity; PGap/HGap are normalized per-world gaps on their respective valid subsets; $\\Delta$Gap is computed on survivors valid on both prompt and holdout; AST = mean formula size. Best values bold.}",
        "\\label{tab:abd_holdout_summary}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Model & PV\\% & HV\\% & PGap & HGap & $\\Delta$Gap & AST \\\\",
        "\\midrule",
    ]

    for model in models:
        m = by_model[model]
        display = get_model_display_name(model)

        train_gap = m.avg_gap_vs_opt_train
        holdout_gap = m.avg_gap_vs_opt_holdout
        delta = m.delta_gap
        avg_ast = m.avg_ast_size

        # Format with bold for best values
        t_val_str = f"{m.valid_pct:.1f}\\%"
        if best_t_val and abs(m.valid_pct - best_t_val) < 0.5:
            t_val_str = f"\\textbf{{{t_val_str}}}"

        h_val_str = f"{m.holdout_valid_pct:.1f}\\%"
        if best_h_val and abs(m.holdout_valid_pct - best_h_val) < 0.5:
            h_val_str = f"\\textbf{{{h_val_str}}}"

        t_gap_str = fmt_float(train_gap)
        if train_gap is not None and best_t_gap is not None and abs(train_gap - best_t_gap) < 0.005:
            t_gap_str = f"\\textbf{{{t_gap_str}}}"

        h_gap_str = fmt_float(holdout_gap)
        if holdout_gap is not None and best_h_gap is not None and abs(holdout_gap - best_h_gap) < 0.005:
            h_gap_str = f"\\textbf{{{h_gap_str}}}"

        delta_str = f"{'+' if delta and delta > 0 else ''}{fmt_float(delta)}"
        if delta is not None and best_delta is not None and abs(delta - best_delta) < 0.005:
            delta_str = f"\\textbf{{{delta_str}}}"

        lines.append(
            f"{display} & {t_val_str} & {h_val_str} & "
            f"{t_gap_str} & {h_gap_str} & "
            f"{delta_str} & {fmt_float(avg_ast, 1)} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Write LaTeX
    tex_path = outdir / "abd_holdout_summary.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")

    # Write CSV
    csv_path = outdir / "abd_holdout_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "TrainValidPct", "HoldoutValidPct", "TrainGap", "HoldoutGap", "DeltaGap", "AvgAST"])
        for model in models:
            m = by_model[model]
            writer.writerow([
                get_model_display_name(model),
                f"{m.valid_pct:.1f}",
                f"{m.holdout_valid_pct:.1f}",
                fmt_float(m.avg_gap_vs_opt_train),
                fmt_float(m.avg_gap_vs_opt_holdout),
                fmt_float(m.delta_gap),
                fmt_float(m.avg_ast_size, 1),
            ])
    print(f"Wrote {csv_path}")


def generate_conditional_holdout_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate conditional holdout metrics table (HoldoutValid|TrainValid)."""
    by_model = aggregate_by_model(records)
    models = sort_models(list(by_model.keys()))

    if not models:
        return

    # Compute best values for bolding (all higher is better)
    t_vals = [by_model[m].train_valid for m in models]
    t_h_vals = [by_model[m].train_valid_with_holdout for m in models]
    h_t_vals = [by_model[m].holdout_valid_given_train_valid for m in models]
    h_pcts = [by_model[m].holdout_valid_given_train_pct for m in models if by_model[m].holdout_valid_given_train_pct is not None]

    best_t_val = max(t_vals) if t_vals else None
    best_t_h = max(t_h_vals) if t_h_vals else None
    best_h_t = max(h_t_vals) if h_t_vals else None
    best_h_pct = max(h_pcts) if h_pcts else None

    lines = [
        "% Conditional Holdout Metrics Table",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Instance-Level Holdout Survival Counts.} All counts are benchmark instances. H$|$P\\% = holdout validity conditional on prompt validity. Best values bold.}",
        "\\label{tab:abd_holdout_conditional}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Model & PV & P+H & H$|$P & H$|$P\\% \\\\",
        "\\midrule",
    ]

    for model in models:
        m = by_model[model]
        display = get_model_display_name(model)

        # Bold T-Val if best
        t_val_str = str(m.train_valid)
        if best_t_val is not None and m.train_valid == best_t_val:
            t_val_str = f"\\textbf{{{t_val_str}}}"

        # Bold T+H if best
        t_h_str = str(m.train_valid_with_holdout)
        if best_t_h is not None and m.train_valid_with_holdout == best_t_h:
            t_h_str = f"\\textbf{{{t_h_str}}}"

        # Bold H|T if best
        h_t_str = str(m.holdout_valid_given_train_valid)
        if best_h_t is not None and m.holdout_valid_given_train_valid == best_h_t:
            h_t_str = f"\\textbf{{{h_t_str}}}"

        # Bold H%|T if best
        h_pct_str = fmt_pct(m.holdout_valid_given_train_pct)
        if best_h_pct is not None and m.holdout_valid_given_train_pct is not None and abs(m.holdout_valid_given_train_pct - best_h_pct) < 0.5:
            h_pct_str = f"\\textbf{{{h_pct_str}}}"

        lines.append(
            f"{display} & {t_val_str} & {t_h_str} & "
            f"{h_t_str} & {h_pct_str} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Write LaTeX
    tex_path = outdir / "abd_holdout_conditional.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")


def generate_complexity_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate complexity vs generalization table (AST bins)."""
    by_model_bin = aggregate_by_model_and_ast_bin(records)
    models = sort_models(list(by_model_bin.keys()))

    if not models:
        print("No models found, skipping complexity table")
        return

    # Get all bins across models
    all_bins = set()
    for model_bins in by_model_bin.values():
        all_bins.update(model_bins.keys())

    # Sort bins by lower bound
    def bin_sort_key(b: str) -> int:
        try:
            return int(b.split(",")[0].strip("["))
        except:
            return 999

    bins = sorted(all_bins, key=bin_sort_key)

    if not bins:
        print("No AST bins found, skipping complexity table")
        return

    # Generate one CSV with all models and bins
    csv_path = outdir / "abd_complexity_vs_gen.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "ASTBin", "N", "HoldoutValidPct", "AvgHoldoutGap", "DeltaGap"])
        for model in models:
            for bin_label in bins:
                m = by_model_bin[model].get(bin_label, AggregatedMetrics())
                if m.total > 0:
                    writer.writerow([
                        get_model_display_name(model),
                        bin_label,
                        m.total,
                        f"{m.holdout_valid_pct:.1f}",
                        fmt_float(m.avg_gap_vs_opt_holdout),
                        fmt_float(m.delta_gap),
                    ])
    print(f"Wrote {csv_path}")

    # Generate a compact LaTeX table for key models (top 3)
    key_models = models[:3] if len(models) >= 3 else models

    lines = [
        "% Complexity vs Generalization Table",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Holdout Gap by AST Size (FULL and PARTIAL tasks)}",
        "\\label{tab:abd_complexity}",
        "\\small",
        "\\begin{tabular}{@{}l" + "rr" * len(key_models) + "@{}}",
        "\\toprule",
    ]

    # Header
    header = ["AST Bin"]
    for model in key_models:
        header.append(f"\\multicolumn{{2}}{{c}}{{{get_model_display_name(model)}}}")
    lines.append(" & ".join(header) + " \\\\")

    subheader = [""]
    for _ in key_models:
        subheader.extend(["N", "$\\Delta$Gap"])
    lines.append(" & ".join(subheader) + " \\\\")
    lines.append("\\midrule")

    for bin_label in bins:
        # Wrap in braces to prevent [x,y) being parsed as optional arg to \\
        escaped_label = "{" + bin_label.replace("∞", "$\\infty$") + "}"
        row = [escaped_label]
        for model in key_models:
            m = by_model_bin[model].get(bin_label, AggregatedMetrics())
            row.append(str(m.total))
            delta = m.delta_gap
            row.append(f"{'+' if delta and delta > 0 else ''}{fmt_float(delta)}")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_path = outdir / "abd_complexity.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")


# =============================================================================
# NEW TABLES (Part B additions)
# =============================================================================


def generate_failure_modes_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate failure mode breakdown table (B3).

    Shows: Valid%, Invalid%, Parse%, Repair%, Missing% for each model.
    """
    by_model = aggregate_by_model(records)
    models = sort_models(list(by_model.keys()))

    if not models:
        print("No models found, skipping failure modes table")
        return

    # Compute best values for bolding (Valid=higher better, others=lower better)
    valid_vals = [by_model[m].valid_pct for m in models]
    invalid_vals = [by_model[m].invalid_pct for m in models]
    parse_vals = [by_model[m].parse_error_pct for m in models]
    missing_vals = [by_model[m].missing_pct for m in models]

    best_valid = max(valid_vals) if valid_vals else None
    best_invalid = min(invalid_vals) if invalid_vals else None
    best_parse = min(parse_vals) if parse_vals else None
    best_missing = min(missing_vals) if missing_vals else None

    lines = [
        "% Failure Mode Breakdown Table (auto-generated)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Failure Mode Breakdown.} Aggregated across all abduction tasks. Parse = unrepaired parse failure; Repair = trailing right-parenthesis repair applied before evaluation; Invalid = parsed but not valid on all prompt worlds; Valid = prompt-valid predictions. Best values bold.}",
        "\\label{tab:abd_failure_modes}",
        "\\small",
        "\\begin{tabular}{@{}lrrrrr@{}}",
        "\\toprule",
        "Model & Valid\\% & Invalid\\% & Parse\\% & Repair\\% & Missing\\% \\\\",
        "\\midrule",
    ]

    for model in models:
        m = by_model[model]
        display = get_model_display_name(model)

        # Format with bold for best values
        valid_str = fmt_pct(m.valid_pct)
        if best_valid is not None and abs(m.valid_pct - best_valid) < 0.5:
            valid_str = f"\\textbf{{{valid_str}}}"

        invalid_str = fmt_pct(m.invalid_pct)
        if best_invalid is not None and abs(m.invalid_pct - best_invalid) < 0.5:
            invalid_str = f"\\textbf{{{invalid_str}}}"

        parse_str = fmt_pct(m.parse_error_pct)
        if best_parse is not None and abs(m.parse_error_pct - best_parse) < 0.5:
            parse_str = f"\\textbf{{{parse_str}}}"

        repair_str = fmt_pct(m.repair_pct)
        missing_str = fmt_pct(m.missing_pct)
        if best_missing is not None and abs(m.missing_pct - best_missing) < 0.5:
            missing_str = f"\\textbf{{{missing_str}}}"

        lines.append(
            f"{display} & {valid_str} & {invalid_str} & "
            f"{parse_str} & {repair_str} & {missing_str} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_path = outdir / "abd_failure_modes.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")

    # CSV version
    csv_path = outdir / "abd_failure_modes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "ValidPct", "InvalidPct", "ParsePct", "RepairPct", "MissingPct"])
        for model in models:
            m = by_model[model]
            writer.writerow([
                get_model_display_name(model),
                f"{m.valid_pct:.1f}",
                f"{m.invalid_pct:.1f}",
                f"{m.parse_error_pct:.1f}",
                f"{m.repair_pct:.1f}",
                f"{m.missing_pct:.1f}",
            ])
    print(f"Wrote {csv_path}")


def generate_failure_mode_breakdown_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate failure-mode counts by scenario (for appendix).

    Classifies each prediction into the first matching category:
    1. Auto-repaired: trailing ')' were appended and evaluation succeeded
    2. Parse error: formula still does not parse
    3. All invalid (train): valid on zero training worlds
    4. Partial invalid (train): valid on some but not all training worlds
    5. Brittle (holdout): valid on all train, invalid on >=1 holdout
       - Sub-category catastrophic: <50% holdout worlds valid
    6. Parsimony inflation: valid on both, but delta-gap > 2
    7. Success: valid on both, delta-gap <= 2
    """
    scenarios = ["ABD_FULL", "ABD_PARTIAL", "ABD_SKEPTICAL"]
    scenario_labels = {"ABD_FULL": "ABD-Full", "ABD_PARTIAL": "ABD-Partial", "ABD_SKEPTICAL": "ABD-Skeptical"}
    categories = [
        "auto_repaired",
        "parse_error",
        "all_invalid",
        "partial_invalid",
        "brittle",
        "catastrophic",
        "parsimony",
    ]

    from collections import defaultdict
    counts: Dict[str, Dict[str, int]] = {sc: defaultdict(int) for sc in scenarios}
    totals: Dict[str, int] = defaultdict(int)

    for r in records:
        sc = r.scenario
        if sc not in scenarios:
            continue
        totals[sc] += 1

        if r.prediction.auto_closed_parens > 0:
            counts[sc]["auto_repaired"] += 1

        # 2. Parse error
        if not r.prediction.parse_ok:
            counts[sc]["parse_error"] += 1
            continue

        # 3. All invalid (train)
        if not r.train_eval.train_all_valid:
            inv = r.train_eval.train_invalid_worlds
            if inv is not None and r.num_train_worlds > 0:
                valid_count = r.num_train_worlds - len(inv)
            else:
                valid_count = 0
            if valid_count == 0:
                counts[sc]["all_invalid"] += 1
            else:
                # 4. Partial invalid (train)
                counts[sc]["partial_invalid"] += 1
            continue

        # train_all_valid is True
        # 5. Brittle
        if not r.holdout_eval.holdout_all_valid:
            counts[sc]["brittle"] += 1
            inv_h = r.holdout_eval.holdout_invalid_worlds
            if inv_h is not None and r.num_holdout_worlds > 0:
                h_valid = r.num_holdout_worlds - len(inv_h)
                if h_valid / r.num_holdout_worlds < 0.5:
                    counts[sc]["catastrophic"] += 1
            continue

        # 6. Parsimony inflation
        t_gap = r.train_eval.gap_vs_opt_train_norm
        h_gap = r.holdout_eval.gap_vs_opt_holdout_norm
        if t_gap is not None and h_gap is not None:
            delta = h_gap - t_gap
            if delta > 2:
                counts[sc]["parsimony"] += 1
                continue

        # 7. Success (not emitted in table)

    total_predictions = sum(totals.values())

    # Build LaTeX table
    lines = [
        "% Failure-mode counts by scenario (auto-generated)",
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{Failure-mode counts across all {total_predictions:,} model predictions.}}",
        "\\label{tab:failure-modes}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Failure Mode & " + " & ".join(scenario_labels[sc] for sc in scenarios) + " \\\\",
        "\\midrule",
    ]

    row_labels = {
        "auto_repaired": "Auto-repaired parse",
        "parse_error": "Parse error",
        "all_invalid": "All invalid (prompt)",
        "partial_invalid": "Partial invalid (prompt)",
        "brittle": "Brittle (holdout)",
        "catastrophic": "\\quad of which catastrophic",
        "parsimony": "Parsimony inflation",
    }

    for cat in categories:
        vals = [str(counts[sc][cat]) for sc in scenarios]
        pad = max(len(v) for v in vals)
        vals_padded = [v.rjust(pad) for v in vals]
        lines.append(f"{row_labels[cat]:30s} & {'& '.join(f'{v} ' for v in vals_padded)}\\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_path = outdir / "abd_failure_mode_breakdown.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")

    # CSV version
    csv_path = outdir / "abd_failure_mode_breakdown.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FailureMode"] + [scenario_labels[sc] for sc in scenarios])
        for cat in categories:
            writer.writerow([row_labels[cat].replace("\\quad ", "")] + [counts[sc][cat] for sc in scenarios])
    print(f"Wrote {csv_path}")

    # Return counts for prose generation
    return counts, totals


def generate_beats_gold_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate gold-beating diagnostics table (B4).

    Shows: BeatsRef%, AvgImprovement, AvgAST (beaters), N_beaters for each model.
    """
    by_model = aggregate_by_model(records)
    models = sort_models(list(by_model.keys()))

    if not models:
        print("No models found, skipping reference-beating table")
        return

    # Compute best values for bolding
    # BeatsRef%: higher is better
    # AvgImpr: higher is better (more improvement)
    # AvgAST: lower is better (shorter formulas)
    # N: higher is better (more gold-beaters)
    beats_pct_vals = [by_model[m].beats_gold_pct for m in models if by_model[m].beats_gold_pct is not None]
    avg_impr_vals = [by_model[m].avg_beats_gold_improvement for m in models if by_model[m].avg_beats_gold_improvement is not None]
    avg_ast_vals = [by_model[m].avg_beats_gold_ast for m in models if by_model[m].avg_beats_gold_ast is not None]
    n_beaters_vals = [by_model[m].beats_gold for m in models]

    best_beats_pct = max(beats_pct_vals) if beats_pct_vals else None
    best_avg_impr = max(avg_impr_vals) if avg_impr_vals else None
    best_avg_ast = min(avg_ast_vals) if avg_ast_vals else None
    best_n = max(n_beaters_vals) if n_beaters_vals else None

    lines = [
        "% Gold-Beating Diagnostics Table (auto-generated)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Reference-Beating Diagnostics} (FULL and PARTIAL tasks). BeatsRef\\% = fraction of prompt-valid instances where model cost $<$ reference cost; AvgImpr = mean per-world cost improvement; AvgAST = mean AST size among reference-beaters. Best values bold.}",
        "\\label{tab:abd_beats_gold}",
        "\\small",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Model & BeatsRef\\% & AvgImpr & AvgAST & N \\\\",
        "\\midrule",
    ]

    for model in models:
        m = by_model[model]
        display = get_model_display_name(model)

        # Format with bold for best values
        beats_pct = fmt_pct(m.beats_gold_pct)
        if best_beats_pct is not None and m.beats_gold_pct is not None and abs(m.beats_gold_pct - best_beats_pct) < 0.5:
            beats_pct = f"\\textbf{{{beats_pct}}}"

        avg_impr = fmt_float(m.avg_beats_gold_improvement, 1)
        if best_avg_impr is not None and m.avg_beats_gold_improvement is not None and abs(m.avg_beats_gold_improvement - best_avg_impr) < 0.05:
            avg_impr = f"\\textbf{{{avg_impr}}}"

        avg_ast = fmt_float(m.avg_beats_gold_ast, 1)
        if best_avg_ast is not None and m.avg_beats_gold_ast is not None and abs(m.avg_beats_gold_ast - best_avg_ast) < 0.5:
            avg_ast = f"\\textbf{{{avg_ast}}}"

        n_beaters = str(m.beats_gold)
        if best_n is not None and m.beats_gold == best_n:
            n_beaters = f"\\textbf{{{n_beaters}}}"

        lines.append(f"{display} & {beats_pct} & {avg_impr} & {avg_ast} & {n_beaters} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_path = outdir / "abd_beats_gold.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")

    # CSV version
    csv_path = outdir / "abd_beats_gold.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "BeatsRefPct", "AvgImprovement", "AvgAST", "N"])
        for model in models:
            m = by_model[model]
            writer.writerow([
                get_model_display_name(model),
                f"{m.beats_gold_pct:.1f}",
                fmt_float(m.avg_beats_gold_improvement, 1),
                fmt_float(m.avg_beats_gold_ast, 1),
                m.beats_gold,
            ])
    print(f"Wrote {csv_path}")


def generate_complexity_by_scenario_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate consolidated complexity vs generalization table (replaces Tables 8 and 9).

    Shows Validity% and ΔGap by AST bin for each task (FULL, PARTIAL, SKEPTICAL).
    Structure: Task as row header, then models with V% and ΔGap per bin.
    Bins: [0,15], [15,30], [30,∞]
    Includes all valid predictions (including exact gold matches).
    """

    # Define simplified bins as requested
    # Format: (latex_label, csv_label, lo, hi)
    ast_bins = [
        ("{[}0,15{)}", "[0,15)", 0, 15),
        ("{[}15,30{)}", "[15,30)", 15, 30),
        ("{[}30,$\\infty${)}", "[30,inf)", 30, float("inf")),
    ]

    # Aggregation structure for validity and delta gap
    @dataclass
    class BinMetrics:
        total: int = 0
        train_valid: int = 0
        holdout_valid: int = 0  # Count of instances valid on both train and holdout
        delta_gap_sum: float = 0.0
        delta_gap_count: int = 0
        delta_gap: Optional[float] = None

        @property
        def holdout_valid_pct(self) -> Optional[float]:
            """Holdout validity % given train validity."""
            if self.train_valid == 0:
                return None
            return 100 * self.holdout_valid / self.train_valid

    # Aggregate by scenario, model, and bin
    # Structure: scenario -> model -> bin_label -> BinMetrics
    by_scenario_model: Dict[str, Dict[str, Dict[str, BinMetrics]]] = {}

    for r in records:
        scenario = r.scenario
        model = r.model_id

        if scenario not in by_scenario_model:
            by_scenario_model[scenario] = {}
        if model not in by_scenario_model[scenario]:
            by_scenario_model[scenario][model] = {csv_label: BinMetrics() for _, csv_label, _, _ in ast_bins}

        # Get AST size from prediction
        ast_size = r.prediction.ast_size
        if ast_size is None:
            continue

        # Determine which bin
        for _, csv_label, lo, hi in ast_bins:
            if lo <= ast_size < hi:
                m = by_scenario_model[scenario][model][csv_label]
                m.total += 1

                # Track train validity
                if r.train_eval.train_all_valid:
                    m.train_valid += 1

                    # Track holdout validity (given train valid)
                    if r.holdout_eval.holdout_all_valid:
                        m.holdout_valid += 1

                    # Compute delta gap (only when both gaps available)
                    holdout_gap = r.holdout_eval.gap_vs_opt_holdout_norm
                    train_gap = r.train_eval.gap_vs_opt_train_norm
                    if holdout_gap is not None and train_gap is not None:
                        delta = holdout_gap - train_gap
                        m.delta_gap_sum += delta
                        m.delta_gap_count += 1
                break

    # Compute averages
    for scenario_data in by_scenario_model.values():
        for model_data in scenario_data.values():
            for m in model_data.values():
                if m.delta_gap_count > 0:
                    m.delta_gap = m.delta_gap_sum / m.delta_gap_count

    scenarios = sorted(by_scenario_model.keys())
    all_models = set()
    for scenario_data in by_scenario_model.values():
        all_models.update(scenario_data.keys())
    models = sort_models(list(all_models))

    if not models or not scenarios:
        print("No data found, skipping complexity by scenario table")
        return

    # Skip models with insufficient valid responses for this table
    excluded_models = {"hermes4", "gpt-4o"}

    # Compute best values per scenario and bin for bolding
    # Structure: scenario -> bin_label -> {"best_v": float, "best_delta": float}
    # Only consider non-excluded models for best computation
    best_by_scenario_bin: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for scenario in scenarios:
        best_by_scenario_bin[scenario] = {}
        for _, csv_label, _, _ in ast_bins:
            v_pcts = []
            deltas = []
            for model in models:
                if model in excluded_models:
                    continue
                model_bins = by_scenario_model[scenario].get(model, {})
                m = model_bins.get(csv_label)
                if m and m.holdout_valid_pct is not None and m.train_valid > 0:
                    v_pcts.append(m.holdout_valid_pct)
                if m and m.delta_gap is not None and m.delta_gap_count > 0:
                    deltas.append(m.delta_gap)
            best_by_scenario_bin[scenario][csv_label] = {
                "best_v": max(v_pcts) if v_pcts else None,
                "best_delta": min(deltas) if deltas else None,  # Lower is better
            }

    # Build wide table, matching the style of Tables 2 and 3.
    latex_labels = [latex_label for latex_label, _, _, _ in ast_bins]
    csv_labels = [csv_label for _, csv_label, _, _ in ast_bins]
    num_bins = len(ast_bins)
    metrics_per_scenario = 2 * num_bins

    lines = [
        "% Complexity vs Generalization Table (auto-generated)",
        "% Shows H|P and DeltaGap per AST bin and scenario",
        "\\begin{table*}[!b]",
        "\\centering",
        "\\caption{\\textbf{Holdout Generalization by Formula Complexity.} H$|$P columns report holdout validity conditional on prompt validity (percent); $\\Delta$ columns report survivor-conditioned $\\Delta$Gap. Hermes4 and GPT-4o are omitted due to too few valid responses.}",
        "\\label{tab:abd_complexity_by_scenario}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{1.8pt}",
        "\\begin{tabular}{@{}l" + "rr" * num_bins * len(scenarios) + "@{}}",
        "\\toprule",
    ]

    header1_parts = [""]
    for scenario in scenarios:
        scenario_short = scenario.replace("ABD_", "")
        header1_parts.append(
            f"\\multicolumn{{{metrics_per_scenario}}}{{c}}{{{scenario_short}}}"
        )
    lines.append(" & ".join(header1_parts) + " \\\\")

    cmidrule_parts = []
    col = 2
    for _ in scenarios:
        cmidrule_parts.append(f"\\cmidrule(lr){{{col}-{col + metrics_per_scenario - 1}}}")
        col += metrics_per_scenario
    lines.append(" ".join(cmidrule_parts))

    header2_parts = [""]
    for _ in scenarios:
        for latex_label in latex_labels:
            header2_parts.append(f"\\multicolumn{{2}}{{c}}{{{latex_label}}}")
    lines.append(" & ".join(header2_parts) + " \\\\")

    cmidrule_parts = []
    col = 2
    for _ in scenarios:
        for _ in ast_bins:
            cmidrule_parts.append(f"\\cmidrule(lr){{{col}-{col+1}}}")
            col += 2
    lines.append(" ".join(cmidrule_parts))

    header3_parts = ["Model"]
    for _ in scenarios:
        for _ in range(num_bins):
            header3_parts.extend(["H$|$P", "$\\Delta$"])
    lines.append(" & ".join(header3_parts) + " \\\\")
    lines.append("\\midrule")

    for model in models:
        if model in excluded_models:
            continue
        display = get_model_display_name(model)
        row_parts = [display]

        for scenario in scenarios:
            model_bins = by_scenario_model[scenario].get(model, {})
            for _, csv_label, _, _ in ast_bins:
                m = model_bins.get(csv_label, BinMetrics())
                best = best_by_scenario_bin[scenario][csv_label]

                v_pct = m.holdout_valid_pct
                if v_pct is not None and m.train_valid > 0:
                    v_str = f"{v_pct:.0f}"
                    if best["best_v"] is not None and abs(v_pct - best["best_v"]) < 0.5:
                        v_str = f"\\textbf{{{v_str}}}"
                    row_parts.append(v_str)
                else:
                    row_parts.append("---")

                if m.delta_gap_count > 0 and m.delta_gap is not None:
                    delta = m.delta_gap
                    delta_str = f"{'+' if delta > 0 else ''}{delta:.1f}"
                    if best["best_delta"] is not None and abs(delta - best["best_delta"]) < 0.05:
                        delta_str = f"\\textbf{{{delta_str}}}"
                    row_parts.append(delta_str)
                else:
                    row_parts.append("---")

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    tex_path = outdir / "abd_complexity_by_scenario.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")

    # CSV version
    csv_path = outdir / "abd_complexity_by_scenario.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header with H|P% and ΔGap for each bin
        csv_header = ["Task", "Model"]
        for csv_label in csv_labels:
            csv_header.extend([f"{csv_label}_H_given_P_pct", f"{csv_label}_DeltaGap"])
        writer.writerow(csv_header)

        for scenario in scenarios:
            scenario_short = scenario.replace("ABD_", "")
            for model in models:
                if model in excluded_models:
                    continue
                row = [scenario_short, get_model_display_name(model)]
                model_bins = by_scenario_model[scenario].get(model, {})
                for _, csv_label, _, _ in ast_bins:
                    m = model_bins.get(csv_label, BinMetrics())
                    # V%
                    v_pct = m.holdout_valid_pct
                    if v_pct is not None:
                        row.append(f"{v_pct:.1f}")
                    else:
                        row.append("")
                    # ΔGap
                    if m.delta_gap_count > 0 and m.delta_gap is not None:
                        row.append(f"{m.delta_gap:.2f}")
                    else:
                        row.append("")
                writer.writerow(row)
    print(f"Wrote {csv_path}")


# =============================================================================
# Paired Formula Comparison (Shorter vs Longer than Gold)
# =============================================================================


def generate_shorter_vs_longer_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate table comparing shorter vs longer formulas on the same problems.

    Uses problem-level averaging (macro-average) to avoid bias where easier problems
    contribute more formulas. For each qualifying problem, we compute per-problem
    metrics (V% and ΔGap for shorter and longer categories), then average those
    across problems.

    Criteria:
    - Find problems where at least 2 models have valid predictions
    - At least one formula has AST <= gold AST (shorter, excluding exact gold matches)
    - At least one formula has AST > gold AST (longer)

    For qualifying problems, compare:
    - Holdout validity % for shorter vs longer formulas (macro-averaged across problems)
    - ΔGap for shorter vs longer formulas (macro-averaged across problems)
    """
    # Group records by instance_id
    by_instance: Dict[str, List[EvalCacheRecord]] = defaultdict(list)
    for r in records:
        by_instance[r.instance_id].append(r)

    # Per-problem metrics for macro-averaging
    @dataclass
    class ProblemMetrics:
        """Metrics for one category (shorter/longer) within a single problem."""
        count: int = 0  # Number of formulas in this category for this problem
        holdout_valid: int = 0
        delta_gap_sum: float = 0.0
        delta_gap_count: int = 0

        @property
        def holdout_valid_pct(self) -> Optional[float]:
            if self.count == 0:
                return None
            return 100 * self.holdout_valid / self.count

        @property
        def avg_delta_gap(self) -> Optional[float]:
            if self.delta_gap_count == 0:
                return None
            return self.delta_gap_sum / self.delta_gap_count

    # Collect per-problem metrics by scenario
    # Structure: scenario -> list of (shorter_metrics, longer_metrics) tuples
    problem_metrics_by_scenario: Dict[str, List[Tuple[ProblemMetrics, ProblemMetrics]]] = defaultdict(list)

    # Track totals for reporting
    qualifying_instances = 0
    total_shorter = 0
    total_longer = 0

    for instance_id, instance_records in by_instance.items():
        # Get gold AST size from any record (should be same for all)
        gold_ast = None
        scenario = None
        for r in instance_records:
            if r.gold_ast is not None:
                gold_ast = r.gold_ast
                scenario = r.scenario
                break

        if gold_ast is None or scenario is None:
            continue

        # Find valid predictions with AST sizes
        valid_records = []
        for r in instance_records:
            if r.train_eval.train_all_valid and r.prediction.ast_size is not None:
                valid_records.append(r)

        if len(valid_records) < 2:
            continue

        # Categorize as shorter (AST <= gold) or longer (AST > gold)
        # Exclude formulas that exactly match the gold formula text
        def is_exact_gold_match(r):
            """Returns True if formula is exactly the gold formula."""
            # Compare actual formula text if available
            if r.gold_formula is not None and r.prediction.formula is not None:
                return r.prediction.formula == r.gold_formula
            # Fallback: can't determine, assume not a match
            return False

        shorter_records = [r for r in valid_records
                          if r.prediction.ast_size <= gold_ast and not is_exact_gold_match(r)]
        longer_records = [r for r in valid_records if r.prediction.ast_size > gold_ast]

        # Must have at least one of each to qualify
        if not shorter_records or not longer_records:
            continue

        qualifying_instances += 1
        total_shorter += len(shorter_records)
        total_longer += len(longer_records)

        # Compute per-problem metrics for shorter formulas
        shorter_metrics = ProblemMetrics()
        for r in shorter_records:
            shorter_metrics.count += 1
            if r.holdout_eval.holdout_all_valid:
                shorter_metrics.holdout_valid += 1
            holdout_gap = r.holdout_eval.gap_vs_opt_holdout_norm
            train_gap = r.train_eval.gap_vs_opt_train_norm
            if holdout_gap is not None and train_gap is not None:
                shorter_metrics.delta_gap_sum += holdout_gap - train_gap
                shorter_metrics.delta_gap_count += 1

        # Compute per-problem metrics for longer formulas
        longer_metrics = ProblemMetrics()
        for r in longer_records:
            longer_metrics.count += 1
            if r.holdout_eval.holdout_all_valid:
                longer_metrics.holdout_valid += 1
            holdout_gap = r.holdout_eval.gap_vs_opt_holdout_norm
            train_gap = r.train_eval.gap_vs_opt_train_norm
            if holdout_gap is not None and train_gap is not None:
                longer_metrics.delta_gap_sum += holdout_gap - train_gap
                longer_metrics.delta_gap_count += 1

        problem_metrics_by_scenario[scenario].append((shorter_metrics, longer_metrics))

    if qualifying_instances == 0:
        print("No qualifying instances for shorter vs longer comparison, skipping")
        return

    # Macro-average: average per-problem metrics across problems
    @dataclass
    class MacroAveragedMetrics:
        """Macro-averaged metrics across problems."""
        num_problems: int = 0
        total_formulas: int = 0  # For reporting
        valid_pct_sum: float = 0.0
        valid_pct_count: int = 0
        delta_gap_sum: float = 0.0
        delta_gap_count: int = 0

        @property
        def avg_valid_pct(self) -> Optional[float]:
            if self.valid_pct_count == 0:
                return None
            return self.valid_pct_sum / self.valid_pct_count

        @property
        def avg_delta_gap(self) -> Optional[float]:
            if self.delta_gap_count == 0:
                return None
            return self.delta_gap_sum / self.delta_gap_count

    # Compute macro-averaged metrics per scenario
    by_scenario_macro: Dict[str, Dict[str, MacroAveragedMetrics]] = {}

    for scenario, problem_pairs in problem_metrics_by_scenario.items():
        shorter_macro = MacroAveragedMetrics()
        longer_macro = MacroAveragedMetrics()

        for shorter_m, longer_m in problem_pairs:
            # Shorter
            shorter_macro.num_problems += 1
            shorter_macro.total_formulas += shorter_m.count
            if shorter_m.holdout_valid_pct is not None:
                shorter_macro.valid_pct_sum += shorter_m.holdout_valid_pct
                shorter_macro.valid_pct_count += 1
            if shorter_m.avg_delta_gap is not None:
                shorter_macro.delta_gap_sum += shorter_m.avg_delta_gap
                shorter_macro.delta_gap_count += 1

            # Longer
            longer_macro.num_problems += 1
            longer_macro.total_formulas += longer_m.count
            if longer_m.holdout_valid_pct is not None:
                longer_macro.valid_pct_sum += longer_m.holdout_valid_pct
                longer_macro.valid_pct_count += 1
            if longer_m.avg_delta_gap is not None:
                longer_macro.delta_gap_sum += longer_m.avg_delta_gap
                longer_macro.delta_gap_count += 1

        by_scenario_macro[scenario] = {"shorter": shorter_macro, "longer": longer_macro}

    scenarios = sorted(by_scenario_macro.keys())

    # Build table
    lines = [
        "% Shorter vs Longer Formula Comparison (auto-generated)",
        "% Compares formulas with AST <= gold vs AST > gold on same problems",
        "% Uses problem-level averaging to avoid bias from easier problems",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Shorter vs. Longer Formulas on Paired Problems.} "
        "On problems where multiple models produced valid formulas, we compare "
        "those with AST $\\leq$ the planted generator reference (shorter) vs AST $>$ the reference (longer). "
        "Formulas exactly matching the planted generator reference are excluded. "
        "Metrics are macro-averaged across problems to avoid bias from easier problems. "
        "H$|$P\\% = holdout validity given prompt-validity; $\\Delta$Gap = HGap $-$ PGap.}",
        "\\label{tab:abd_shorter_vs_longer}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        " & \\multicolumn{3}{c}{Shorter ($\\leq$ ref)} & \\multicolumn{3}{c}{Longer ($>$ ref)} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
        "Task & N & H$|$P\\% & $\\Delta$Gap & N & H$|$P\\% & $\\Delta$Gap \\\\",
        "\\midrule",
    ]

    # Overall macro-averaged metrics
    overall_shorter = MacroAveragedMetrics()
    overall_longer = MacroAveragedMetrics()

    for scenario in scenarios:
        scenario_short = scenario.replace("ABD_", "")
        shorter = by_scenario_macro[scenario]["shorter"]
        longer = by_scenario_macro[scenario]["longer"]

        # Accumulate overall (across all problems from all scenarios)
        for shorter_m, longer_m in problem_metrics_by_scenario[scenario]:
            overall_shorter.num_problems += 1
            overall_shorter.total_formulas += shorter_m.count
            if shorter_m.holdout_valid_pct is not None:
                overall_shorter.valid_pct_sum += shorter_m.holdout_valid_pct
                overall_shorter.valid_pct_count += 1
            if shorter_m.avg_delta_gap is not None:
                overall_shorter.delta_gap_sum += shorter_m.avg_delta_gap
                overall_shorter.delta_gap_count += 1

            overall_longer.num_problems += 1
            overall_longer.total_formulas += longer_m.count
            if longer_m.holdout_valid_pct is not None:
                overall_longer.valid_pct_sum += longer_m.holdout_valid_pct
                overall_longer.valid_pct_count += 1
            if longer_m.avg_delta_gap is not None:
                overall_longer.delta_gap_sum += longer_m.avg_delta_gap
                overall_longer.delta_gap_count += 1

        # Format row - N is now number of problems, not formulas
        s_vpct = f"{shorter.avg_valid_pct:.0f}" if shorter.avg_valid_pct is not None else "---"
        l_vpct = f"{longer.avg_valid_pct:.0f}" if longer.avg_valid_pct is not None else "---"

        s_delta = f"{'+' if shorter.avg_delta_gap and shorter.avg_delta_gap > 0 else ''}{shorter.avg_delta_gap:.2f}" if shorter.avg_delta_gap is not None else "---"
        l_delta = f"{'+' if longer.avg_delta_gap and longer.avg_delta_gap > 0 else ''}{longer.avg_delta_gap:.2f}" if longer.avg_delta_gap is not None else "---"

        lines.append(f"{scenario_short} & {shorter.num_problems} & {s_vpct} & {s_delta} & {longer.num_problems} & {l_vpct} & {l_delta} \\\\")

    # Overall row
    lines.append("\\midrule")
    s_vpct = f"{overall_shorter.avg_valid_pct:.0f}" if overall_shorter.avg_valid_pct is not None else "---"
    l_vpct = f"{overall_longer.avg_valid_pct:.0f}" if overall_longer.avg_valid_pct is not None else "---"

    s_delta = f"{'+' if overall_shorter.avg_delta_gap and overall_shorter.avg_delta_gap > 0 else ''}{overall_shorter.avg_delta_gap:.2f}" if overall_shorter.avg_delta_gap is not None else "---"
    l_delta = f"{'+' if overall_longer.avg_delta_gap and overall_longer.avg_delta_gap > 0 else ''}{overall_longer.avg_delta_gap:.2f}" if overall_longer.avg_delta_gap is not None else "---"

    lines.append(f"\\textbf{{Overall}} & {overall_shorter.num_problems} & {s_vpct} & {s_delta} & {overall_longer.num_problems} & {l_vpct} & {l_delta} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_path = outdir / "abd_shorter_vs_longer.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path} ({qualifying_instances} problems, {total_shorter} shorter formulas, {total_longer} longer formulas)")

    # CSV version
    csv_path = outdir / "abd_shorter_vs_longer.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Shorter_N_problems", "Shorter_V%", "Shorter_DeltaGap",
                         "Longer_N_problems", "Longer_V%", "Longer_DeltaGap"])
        for scenario in scenarios:
            scenario_short = scenario.replace("ABD_", "")
            shorter = by_scenario_macro[scenario]["shorter"]
            longer = by_scenario_macro[scenario]["longer"]
            writer.writerow([
                scenario_short,
                shorter.num_problems,
                f"{shorter.avg_valid_pct:.1f}" if shorter.avg_valid_pct is not None else "",
                f"{shorter.avg_delta_gap:.2f}" if shorter.avg_delta_gap is not None else "",
                longer.num_problems,
                f"{longer.avg_valid_pct:.1f}" if longer.avg_valid_pct is not None else "",
                f"{longer.avg_delta_gap:.2f}" if longer.avg_delta_gap is not None else "",
            ])
        # Overall
        writer.writerow([
            "Overall",
            overall_shorter.num_problems,
            f"{overall_shorter.avg_valid_pct:.1f}" if overall_shorter.avg_valid_pct is not None else "",
            f"{overall_shorter.avg_delta_gap:.2f}" if overall_shorter.avg_delta_gap is not None else "",
            overall_longer.num_problems,
            f"{overall_longer.avg_valid_pct:.1f}" if overall_longer.avg_valid_pct is not None else "",
            f"{overall_longer.avg_delta_gap:.2f}" if overall_longer.avg_delta_gap is not None else "",
        ])
    print(f"Wrote {csv_path}")


def generate_holdout_match_sanity_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate holdout distribution match sanity check table.

    Shows that train and holdout worlds are drawn from similar distributions
    by comparing world counts and per-world gold costs.
    """
    # Group by scenario, deduplicated by instance_id (multiple models per instance)
    by_scenario: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_instances: set = set()

    for r in records:
        # Skip records without holdout data
        if r.num_holdout_worlds == 0:
            continue
        # Deduplicate by instance_id (world counts/gold costs are instance properties)
        if r.instance_id in seen_instances:
            continue
        seen_instances.add(r.instance_id)

        # Compute per-world gold costs
        train_gold_per_world = None
        holdout_gold_per_world = None

        if r.gold_cost_train_sum is not None and r.num_train_worlds > 0:
            train_gold_per_world = r.gold_cost_train_sum / r.num_train_worlds
        if r.gold_cost_holdout_sum is not None and r.num_holdout_worlds > 0:
            holdout_gold_per_world = r.gold_cost_holdout_sum / r.num_holdout_worlds

        by_scenario[r.scenario].append({
            "num_train": r.num_train_worlds,
            "num_holdout": r.num_holdout_worlds,
            "train_gold_per_world": train_gold_per_world,
            "holdout_gold_per_world": holdout_gold_per_world,
        })

    if not by_scenario:
        print("No holdout data for sanity check table")
        return

    # Compute means per scenario (deduplicated by instance_id)
    scenario_stats = {}
    for scenario, instances in by_scenario.items():
        n = len(instances)
        if n == 0:
            continue

        mean_train_worlds = sum(i["num_train"] for i in instances) / n
        mean_holdout_worlds = sum(i["num_holdout"] for i in instances) / n

        train_golds = [i["train_gold_per_world"] for i in instances if i["train_gold_per_world"] is not None]
        holdout_golds = [i["holdout_gold_per_world"] for i in instances if i["holdout_gold_per_world"] is not None]

        mean_train_gold = sum(train_golds) / len(train_golds) if train_golds else None
        mean_holdout_gold = sum(holdout_golds) / len(holdout_golds) if holdout_golds else None

        scenario_stats[scenario] = {
            "n_instances": n,
            "mean_train_worlds": mean_train_worlds,
            "mean_holdout_worlds": mean_holdout_worlds,
            "mean_train_gold": mean_train_gold,
            "mean_holdout_gold": mean_holdout_gold,
        }

    # Generate LaTeX table with compact two-level header
    lines = [
        "% Holdout Distribution Match Sanity Check",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Prompt vs Holdout Distribution Match.} Mean world counts and per-world reference costs. N counts instances with available holdout worlds (may differ from Table~\\ref{tab:dataset_summary} if some instances lack holdouts).}",
        "\\label{tab:abd_holdout_match}",
        "\\small",
        "\\begin{tabular}{@{}lrcccc@{}}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Worlds} & \\multicolumn{2}{c}{Reference cost/world} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "Scenario & N & Prompt & Holdout & Prompt & Holdout \\\\",
        "\\midrule",
    ]

    scenarios = ["ABD_FULL", "ABD_PARTIAL", "ABD_SKEPTICAL"]
    for scenario in scenarios:
        if scenario not in scenario_stats:
            continue
        s = scenario_stats[scenario]
        scenario_short = scenario.replace("ABD_", "")

        t_gold = f"{s['mean_train_gold']:.2f}" if s['mean_train_gold'] is not None else "---"
        h_gold = f"{s['mean_holdout_gold']:.2f}" if s['mean_holdout_gold'] is not None else "---"

        lines.append(
            f"{scenario_short} & {s['n_instances']} & {s['mean_train_worlds']:.1f} & "
            f"{s['mean_holdout_worlds']:.1f} & {t_gold} & {h_gold} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    tex_path = outdir / "abd_holdout_match_sanity.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")


def generate_appendix_gap_by_theory_tables(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate appendix per-theory train/holdout gap tables for each scenario."""
    table_specs = {
        "ABD_FULL": {
            "filename": "abd_appendix_gap_full.tex",
            "label": "tab:g1-full",
            "caption": r"Per-theory results: \textbf{ABD-Full}. Each cell shows PGap\,/\,HGap\,(survivors).",
        },
        "ABD_PARTIAL": {
            "filename": "abd_appendix_gap_partial.tex",
            "label": "tab:g1-partial",
            "caption": r"Per-theory results: \textbf{ABD-Partial}. Format as Table~\ref{tab:g1-full}.",
        },
        "ABD_SKEPTICAL": {
            "filename": "abd_appendix_gap_skeptical.tex",
            "label": "tab:g1-skeptical",
            "caption": r"Per-theory results: \textbf{ABD-Skeptical}. Format as Table~\ref{tab:g1-full}. Note negative $\Delta$Gap for T5: holdout gap is often \emph{lower} than prompt gap because worst-case costs can be smaller in holdout worlds for this theory.",
        },
    }

    scenarios = sort_scenarios(list({r.scenario for r in records}))
    all_models = sort_models(list({r.model_id for r in records}))

    for scenario in scenarios:
        spec = table_specs.get(scenario)
        if spec is None:
            continue

        scenario_records = [r for r in records if r.scenario == scenario]
        if not scenario_records:
            continue

        by_model_theory = aggregate_by_model_and_theory(scenario_records)
        theories = sort_theories(list({r.theory for r in scenario_records}))

        lines = [
            "% Appendix per-theory train/holdout gap table (auto-generated)",
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{{spec['caption']}}}",
            f"\\label{{{spec['label']}}}",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{@{}l" + "c" * len(all_models) + "@{}}",
            "\\toprule",
            " & ".join([""] + [get_model_display_name(m) for m in all_models]) + " \\\\",
            "\\midrule",
        ]

        csv_rows: List[List[str]] = [["Theory"] + [get_model_display_name(m) for m in all_models]]

        for theory in theories:
            paper_id = get_theory_paper_id(theory)
            row = [paper_id]
            csv_row = [paper_id]
            for model in all_models:
                m = by_model_theory.get(model, {}).get(theory, AggregatedMetrics())
                if m.train_valid == 0:
                    cell = "---"
                else:
                    train_gap = fmt_float(m.avg_gap_vs_opt_train, 1)
                    holdout_gap = fmt_float(m.avg_gap_vs_opt_holdout, 1)
                    cell = f"{train_gap}/{holdout_gap}\\,({m.survivor_count})"
                row.append(cell)
                csv_row.append(cell)
            lines.append(" & ".join(row) + " \\\\")
            csv_rows.append(csv_row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ])

        tex_path = outdir / spec["filename"]
        with open(tex_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Wrote {tex_path}")

        csv_path = outdir / spec["filename"].replace(".tex", ".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"Wrote {csv_path}")


def generate_gap_distribution_summary_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate appendix train-gap distribution summary by scenario and model."""
    by_scenario_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    scenarios = sort_scenarios(list({r.scenario for r in records}))
    models = sort_models(list({r.model_id for r in records}))

    for r in records:
        gap = r.train_eval.gap_vs_opt_train_norm
        if r.train_eval.train_all_valid and gap is not None:
            by_scenario_model[r.scenario][r.model_id].append(gap)

    lines = [
        "% Gap distribution summary table (auto-generated)",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Normalised prompt-gap distribution by scenario. N = predictions valid on all prompt worlds. $>$3 and $>$5 are tail percentages.}",
        "\\label{tab:gap-dist}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrrr@{}}",
        "\\toprule",
        "Model & N & Mean & Med & P90 & Max & $>$3 & $>$5 \\\\",
        "\\midrule",
    ]

    csv_rows: List[List[str]] = [[
        "Scenario", "Model", "N", "Mean", "Median", "P90", "Max", "PctGT3", "PctGT5"
    ]]

    for idx, scenario in enumerate(scenarios):
        if idx > 0:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textbf{{{get_scenario_display_name(scenario)}}}}} \\\\")
        lines.append("\\cmidrule(l){1-8}")

        for model in models:
            values = sorted(by_scenario_model.get(scenario, {}).get(model, []))
            n = len(values)
            display = get_model_display_name(model)
            if n == 0:
                lines.append(f"{display} & 0 & --- & --- & --- & --- & --- & --- \\\\")
                csv_rows.append([get_scenario_short_name(scenario), display, 0, "", "", "", "", "", ""])
                continue

            mean_val = sum(values) / n
            med_val = percentile_linear(values, 0.5)
            p90_val = percentile_linear(values, 0.9)
            max_val = max(values)
            pct_gt3 = 100 * sum(v > 3 for v in values) / n
            pct_gt5 = 100 * sum(v > 5 for v in values) / n

            lines.append(
                f"{display} & {n} & {mean_val:.2f} & {med_val:.2f} & {p90_val:.2f} & "
                f"{max_val:.1f} & {pct_gt3:.1f}\\% & {pct_gt5:.1f}\\% \\\\"
            )
            csv_rows.append([
                get_scenario_short_name(scenario),
                display,
                n,
                f"{mean_val:.2f}",
                f"{med_val:.2f}",
                f"{p90_val:.2f}",
                f"{max_val:.1f}",
                f"{pct_gt3:.1f}",
                f"{pct_gt5:.1f}",
            ])

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    tex_path = outdir / "abd_gap_distribution.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")

    csv_path = outdir / "abd_gap_distribution.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Wrote {csv_path}")


def generate_beats_gold_by_theory_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate appendix reference-beating table pooled across models, grouped by theory."""
    scenarios = sort_scenarios(list({r.scenario for r in records}))
    theories = sort_theories(list({r.theory for r in records}))
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"beats": 0, "total": 0})
    )

    for r in records:
        if not r.train_eval.train_all_valid:
            continue
        gap_gold_sum = r.train_eval.gap_vs_gold_train_sum
        if gap_gold_sum is None:
            continue
        bucket = counts[r.theory][r.scenario]
        bucket["total"] += 1
        if gap_gold_sum < 0:
            bucket["beats"] += 1

    lines = [
        "% Reference-beating by theory table (auto-generated)",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Reference-beating rate by theory (all models pooled). $k/N$ = predictions beating the planted generator reference out of prompt-valid predictions.}",
        "\\label{tab:beats-theory}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}l" + "r" * len(scenarios) + "@{}}",
        "\\toprule",
        "Theory & " + " & ".join(get_scenario_display_name(s) for s in scenarios) + " \\\\",
        "\\midrule",
    ]

    csv_rows: List[List[str]] = [[
        "Theory",
        *[f"{get_scenario_short_name(s)}_Beats" for s in scenarios],
        *[f"{get_scenario_short_name(s)}_Total" for s in scenarios],
        *[f"{get_scenario_short_name(s)}_Pct" for s in scenarios],
    ]]

    for theory in theories:
        row = [get_theory_paper_id(theory)]
        csv_row = [get_theory_paper_id(theory)]
        for scenario in scenarios:
            bucket = counts[theory][scenario]
            total = bucket["total"]
            beats = bucket["beats"]
            if total == 0:
                row.append("\\multicolumn{1}{c}{---}")
                csv_row.extend([0, 0, ""])
            else:
                pct = 100 * beats / total
                row.append(f"{beats}/{total}\\;({pct:.1f}\\%)")
                csv_row.extend([beats, total, f"{pct:.1f}"])
        lines.append(" & ".join(row) + " \\\\")
        csv_rows.append(csv_row)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    tex_path = outdir / "abd_beats_gold_by_theory.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")

    csv_path = outdir / "abd_beats_gold_by_theory.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Wrote {csv_path}")


def generate_beats_gold_by_ast_table(records: List[EvalCacheRecord], outdir: Path) -> None:
    """Generate appendix reference-beating table grouped by reference AST size."""
    ast_bins = [
        (r"$[0,5)$", "[0,5)", 0, 5),
        (r"$[5,10)$", "[5,10)", 5, 10),
        (r"$[10,15)$", "[10,15)", 10, 15),
        (r"$[15,20)$", "[15,20)", 15, 20),
        (r"$[20,30)$", "[20,30)", 20, 30),
        (r"$[30,+)$", "[30,+)", 30, float("inf")),
    ]

    scenarios = sort_scenarios(list({r.scenario for r in records}))
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"beats": 0, "total": 0})
    )

    for r in records:
        if not r.train_eval.train_all_valid or r.gold_ast is None:
            continue
        gap_gold_sum = r.train_eval.gap_vs_gold_train_sum
        if gap_gold_sum is None:
            continue
        for _, csv_label, lo, hi in ast_bins:
            if lo <= r.gold_ast < hi:
                bucket = counts[csv_label][r.scenario]
                bucket["total"] += 1
                if gap_gold_sum < 0:
                    bucket["beats"] += 1
                break

    lines = [
        "% Reference-beating by reference AST table (auto-generated)",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Reference-beating rate by planted generator reference AST size (all models pooled).}",
        "\\label{tab:beats-ast}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}l" + "r" * len(scenarios) + "@{}}",
        "\\toprule",
        "Ref AST & " + " & ".join(get_scenario_display_name(s) for s in scenarios) + " \\\\",
        "\\midrule",
    ]

    csv_rows: List[List[str]] = [[
        "GoldAST",
        *[f"{get_scenario_short_name(s)}_Beats" for s in scenarios],
        *[f"{get_scenario_short_name(s)}_Total" for s in scenarios],
        *[f"{get_scenario_short_name(s)}_Pct" for s in scenarios],
    ]]

    for latex_label, csv_label, _, _ in ast_bins:
        row = [latex_label]
        csv_row = [csv_label]
        for scenario in scenarios:
            bucket = counts[csv_label][scenario]
            total = bucket["total"]
            beats = bucket["beats"]
            if total == 0:
                row.append("\\multicolumn{1}{c}{---}")
                csv_row.extend([0, 0, ""])
            else:
                pct = 100 * beats / total
                row.append(f"{beats}/{total}\\;({pct:.1f}\\%)")
                csv_row.extend([beats, total, f"{pct:.1f}"])
        lines.append(" & ".join(row) + " \\\\")
        csv_rows.append(csv_row)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    tex_path = outdir / "abd_beats_gold_by_ast.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {tex_path}")

    csv_path = outdir / "abd_beats_gold_by_ast.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Wrote {csv_path}")


# =============================================================================
# Manifest and Paper Metrics (Part A)
# =============================================================================


def generate_manifest(
    records: List[EvalCacheRecord],
    input_paths: List[str],
    outdir: Path,
    generated_tables: List[str],
) -> None:
    """Generate manifest.json with provenance information (A1)."""

    # Collect metadata
    models = sorted(set(r.model_id for r in records))
    scenarios = sorted(set(r.scenario for r in records))
    theories = sorted(set(r.theory for r in records))
    run_ids = sorted(set(r.run_id for r in records))

    manifest = {
        "created_at": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "schema_version": "abd_eval_v1",
        "cache_files": [str(p) for p in input_paths],
        "cache_sha256": {str(p): compute_file_sha256(p) for p in input_paths},
        "record_count": len(records),
        "models": models,
        "scenarios": scenarios,
        "theories": theories,
        "run_ids": run_ids,
        "output_tables": generated_tables,
    }

    manifest_path = outdir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")


def generate_paper_metrics(
    records: List[EvalCacheRecord],
    outdir: Path,
) -> None:
    """Generate paper_metrics.json with headline scalar numbers (A2)."""

    by_model = aggregate_by_model(records)
    models = sort_models(list(by_model.keys()))

    if not models:
        return

    # Find best values
    best_valid_pct = max(by_model[m].valid_pct for m in models)
    best_gap_opt = min(
        by_model[m].avg_gap_vs_opt_train for m in models
        if by_model[m].avg_gap_vs_opt_train is not None
    )

    # Compute averages
    delta_gaps = [
        by_model[m].delta_gap for m in models
        if by_model[m].delta_gap is not None
    ]
    avg_delta_gap = sum(delta_gaps) / len(delta_gaps) if delta_gaps else None

    # Total instances (deduplicated)
    instance_ids = set(r.instance_id for r in records)

    paper_metrics = {
        "total_instances": len(instance_ids),
        "total_records": len(records),
        "models_evaluated": models,
        "best_train_valid_pct": round(best_valid_pct, 1),
        "best_train_gap_opt": round(best_gap_opt, 2) if best_gap_opt else None,
        "avg_holdout_delta_gap": round(avg_delta_gap, 2) if avg_delta_gap else None,
        "per_model": {
            get_model_display_name(m): {
                "train_valid_pct": round(by_model[m].valid_pct, 1),
                "holdout_valid_pct": round(by_model[m].holdout_valid_pct, 1),
                "train_gap": round(by_model[m].avg_gap_vs_opt_train, 2) if by_model[m].avg_gap_vs_opt_train else None,
                "holdout_gap": round(by_model[m].avg_gap_vs_opt_holdout, 2) if by_model[m].avg_gap_vs_opt_holdout else None,
                "delta_gap": round(by_model[m].delta_gap, 2) if by_model[m].delta_gap else None,
                "beats_gold_pct": round(by_model[m].beats_gold_pct, 1),
            }
            for m in models
        },
    }

    metrics_path = outdir / "paper_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(paper_metrics, f, indent=2)
    print(f"Wrote {metrics_path}")


# =============================================================================
# Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX and CSV tables from ABD-B1 evaluation cache",
    )
    parser.add_argument(
        "--input", "-i", required=True, nargs="+",
        help="Path(s) to JSONL evaluation cache file(s)"
    )
    parser.add_argument(
        "--outdir", "-o", default="paper/tables",
        help="Output directory for tables (default: paper/tables)"
    )
    parser.add_argument(
        "--run-id",
        help="Filter to specific run_id (use 'latest' for most recent)"
    )
    parser.add_argument(
        "--model",
        help="Filter to specific model"
    )
    parser.add_argument(
        "--manifest", action="store_true",
        help="Generate manifest.json and paper_metrics.json with provenance info"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Check cache files exist
    for path in args.input:
        if not Path(path).exists():
            print(f"ERROR: Cache file not found: {path}")
            print("Run 'make eval' first to generate the cache, or check the path.")
            sys.exit(1)

    # Load records
    reader = EvalCacheReader(*args.input)
    records = reader.load_records(
        run_id=args.run_id,
        model_id=args.model,
    )

    if not records:
        print(f"No records found in {args.input}")
        return

    # Filter out excluded models
    if EXCLUDE_MODELS:
        before = len(records)
        records = [r for r in records if r.model_id not in EXCLUDE_MODELS]
        excluded = before - len(records)
        if excluded:
            print(f"Excluded {excluded} records from models: {EXCLUDE_MODELS}")

    raw_record_count = len(records)
    records, duplicate_rows_removed, duplicate_pairs = dedupe_records_by_instance_model(records)
    if duplicate_rows_removed:
        print(
            "Deduped evaluation records by (instance_id, model_id): "
            f"removed {duplicate_rows_removed} rows across {duplicate_pairs} repeated pairs "
            f"(keeping the last record for each pair)"
        )

    print(f"Loaded {len(records)} evaluation records ({raw_record_count} raw rows)")
    models = sorted(set(r.model_id for r in records))
    print(f"Models: {models}")
    set_model_order(records)

    if args.run_id:
        run_ids = sorted(set(r.run_id for r in records))
        print(f"Run IDs: {run_ids}")

    # Track generated tables for manifest
    generated_tables = []

    # Generate all tables
    print(f"\nGenerating tables to {outdir}...")

    # Core tables
    generate_dataset_summary_table(records, outdir)
    generated_tables.append("abd_dataset_summary.tex")

    generate_main_train_table(records, outdir)
    generated_tables.append("abd_main_train.tex")

    # NOTE: Scenario breakdown table removed - redundant with main_train_table
    # which already shows per-scenario breakdown

    generate_theory_breakdown_table(records, outdir)
    generated_tables.append("abd_theory_breakdown.tex")

    generate_holdout_summary_table(records, outdir)
    generated_tables.append("abd_holdout_summary.tex")

    generate_holdout_match_sanity_table(records, outdir)
    generated_tables.append("abd_holdout_match_sanity.tex")

    generate_conditional_holdout_table(records, outdir)
    generated_tables.append("abd_holdout_conditional.tex")

    # Consolidated complexity vs generalization table (replaces Tables 8 and 9)
    generate_complexity_by_scenario_table(records, outdir)
    generated_tables.append("abd_complexity_by_scenario.tex")

    # New tables (Part B)
    generate_failure_modes_table(records, outdir)
    generated_tables.append("abd_failure_modes.tex")

    generate_failure_mode_breakdown_table(records, outdir)
    generated_tables.append("abd_failure_mode_breakdown.tex")

    generate_beats_gold_table(records, outdir)
    generated_tables.append("abd_beats_gold.tex")

    # New holdout breakdown tables
    generate_holdout_by_scenario_table(records, outdir)
    generated_tables.append("abd_holdout_by_scenario.tex")

    generate_holdout_by_theory_table(records, outdir)
    generated_tables.append("abd_holdout_by_theory.tex")

    generate_complexity_by_model_table(records, outdir)
    generated_tables.append("abd_complexity_by_model.tex")

    # Appendix tables that were previously hand-maintained
    generate_appendix_gap_by_theory_tables(records, outdir)
    generated_tables.extend([
        "abd_appendix_gap_full.tex",
        "abd_appendix_gap_partial.tex",
        "abd_appendix_gap_skeptical.tex",
    ])

    generate_gap_distribution_summary_table(records, outdir)
    generated_tables.append("abd_gap_distribution.tex")

    generate_beats_gold_by_theory_table(records, outdir)
    generated_tables.append("abd_beats_gold_by_theory.tex")

    generate_beats_gold_by_ast_table(records, outdir)
    generated_tables.append("abd_beats_gold_by_ast.tex")

    # Paired comparison: shorter vs longer formulas on same problems
    generate_shorter_vs_longer_table(records, outdir)
    generated_tables.append("abd_shorter_vs_longer.tex")

    # Manifest and paper metrics (Part A)
    if args.manifest:
        generate_manifest(records, args.input, outdir, generated_tables)
        generate_paper_metrics(records, outdir)

    print(f"\nDone! Tables written to {outdir}")
    print(f"Generated {len(generated_tables)} tables: {', '.join(generated_tables)}")


if __name__ == "__main__":
    main()
