"""
target.py - Target Extension Computation

Computes the extension of the target predicate T(x) by evaluating
a formula φ(x) on each domain element.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from concept_synth.bootstrap import add_repo_root
except ModuleNotFoundError:
    import os as _os
    import sys as _sys

    _path = _os.path.abspath(__file__)
    while True:
        parent = _os.path.dirname(_path)
        if _os.path.basename(_path) == "concept_synth":
            if parent not in _sys.path:
                _sys.path.insert(0, parent)
            break
        if parent == _path:
            break
        _path = parent
    from concept_synth.bootstrap import add_repo_root
add_repo_root(__file__)
from concept_synth.fol.formulas import FOFormula
from concept_synth.fol.model import FiniteModel, evaluate


@dataclass
class TargetExtension:
    """Extension of the target predicate T(x) in a finite world."""

    T_true: List[str]
    T_false: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {"T_true": self.T_true}
        if self.T_false is not None:
            result["T_false"] = self.T_false
        return result


def compute_target_extension(model: FiniteModel, phi: FOFormula) -> TargetExtension:
    """
    Compute the target extension T(x) = {c : φ(c) is true in model}.

    Args:
        model: The finite model
        phi: The formula with free variable x

    Returns:
        TargetExtension with T_true and T_false lists
    """
    T_true = []
    T_false = []

    for i in range(model.n):
        const = model.index_to_const(i)
        # Evaluate φ with x bound to element i
        result = evaluate(phi, model, {"x": i})

        if result:
            T_true.append(const)
        else:
            T_false.append(const)

    return TargetExtension(T_true=T_true, T_false=T_false)


def validate_target(
    target: TargetExtension,
    domain_size: int,
    min_true_frac: float = 0.15,
    max_true_frac: float = 0.85,
) -> Tuple[bool, str]:
    """
    Validate that a target extension is non-trivial.

    Args:
        target: The target extension to validate
        domain_size: Size of the domain
        min_true_frac: Minimum fraction of true elements
        max_true_frac: Maximum fraction of true elements

    Returns:
        Tuple of (is_valid, reason)
    """
    num_true = len(target.T_true)

    if num_true == 0:
        return False, "No true elements"

    if num_true == domain_size:
        return False, "All elements are true"

    true_frac = num_true / domain_size

    if true_frac < min_true_frac:
        return False, f"Too few true elements: {true_frac:.2f} < {min_true_frac}"

    if true_frac > max_true_frac:
        return False, f"Too many true elements: {true_frac:.2f} > {max_true_frac}"

    return True, "Valid"


class TargetConfig:
    """Configuration for target computation."""

    def __init__(self, min_true_frac: float = 0.15, max_true_frac: float = 0.85):
        self.min_true_frac = min_true_frac
        self.max_true_frac = max_true_frac

    def validate(self, target: TargetExtension, domain_size: int) -> Tuple[bool, str]:
        return validate_target(target, domain_size, self.min_true_frac, self.max_true_frac)
