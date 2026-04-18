"""
Predicate format compatibility layer.

This module provides functions to read predicates in various YAML formats:
1. Legacy format: {'true': ['a0', 'a1'], 'false': ['a2']}
2. Compact list format: ['a0', 'a1'] (for unary) or ['(a0, a1)', '(a2, a3)'] (for binary)
3. True/False dict list format: [{'True': [...], 'False': [...]}] (for binary with both)

All functions return standard Python lists for consistency.
"""

from typing import Any, Dict, List, Optional, Tuple


def get_unary_extension(pred_data: Any) -> List[str]:
    """
    Extract unary predicate extension (true elements) from various formats.

    Handles:
    - None -> []
    - List ['a0', 'a1'] -> ['a0', 'a1']
    - Dict {'true': ['a0', 'a1']} -> ['a0', 'a1']
    - Old dict format {'a0': True, 'a1': False} -> ['a0']

    Args:
        pred_data: Predicate data in any supported format

    Returns:
        List of elements where predicate is true
    """
    if pred_data is None:
        return []

    # New compact format: just a list of elements
    if isinstance(pred_data, list):
        return list(pred_data)

    # Dict format
    if isinstance(pred_data, dict):
        # Legacy format: {'true': [...], 'false': [...]}
        if "true" in pred_data or "false" in pred_data:
            return list(pred_data.get("true", []))
        # Very old format: {'a0': True, 'a1': False, ...}
        return [k for k, v in pred_data.items() if v is True]

    return []


def get_unary_false_extension(pred_data: Any) -> List[str]:
    """
    Extract unary predicate false extension from various formats.

    Args:
        pred_data: Predicate data in any supported format

    Returns:
        List of elements where predicate is explicitly false
    """
    if pred_data is None:
        return []

    # New compact format: no false info stored for unary
    if isinstance(pred_data, list):
        return []

    # Dict format
    if isinstance(pred_data, dict):
        # Legacy format: {'true': [...], 'false': [...]}
        if "true" in pred_data or "false" in pred_data:
            return list(pred_data.get("false", []))
        # Very old format: {'a0': True, 'a1': False, ...}
        return [k for k, v in pred_data.items() if v is False]

    return []


def get_binary_extension(pred_data: Any) -> List[str]:
    """
    Extract binary predicate true extension from various formats.

    Handles:
    - None -> []
    - List ['(a0, a1)', '(a2, a3)'] -> ['(a0, a1)', '(a2, a3)']
    - List [{'True': [...], 'False': [...]}] -> true pairs
    - Dict {'true': ['(a0, a1)'], 'false': [...]} -> ['(a0, a1)']

    Args:
        pred_data: Predicate data in any supported format

    Returns:
        List of pair strings like '(a0, a1)' where predicate is true
    """
    if pred_data is None:
        return []

    # New compact format: list
    if isinstance(pred_data, list):
        # Check if it's the True/False dict list format
        if pred_data and isinstance(pred_data[0], dict):
            for item in pred_data:
                if "True" in item:
                    return list(item["True"])
            return []
        # Otherwise it's a list of pair strings
        return list(pred_data)

    # Legacy dict format: {'true': [...], 'false': [...]}
    if isinstance(pred_data, dict):
        return list(pred_data.get("true", []))

    return []


def get_binary_false_extension(pred_data: Any) -> List[str]:
    """
    Extract binary predicate false extension from various formats.

    Args:
        pred_data: Predicate data in any supported format

    Returns:
        List of pair strings like '(a0, a1)' where predicate is explicitly false
    """
    if pred_data is None:
        return []

    # New compact format: list
    if isinstance(pred_data, list):
        # Check if it's the True/False dict list format
        if pred_data and isinstance(pred_data[0], dict):
            for item in pred_data:
                if "False" in item:
                    return list(item["False"])
            return []
        # Plain list format has no false info
        return []

    # Legacy dict format: {'true': [...], 'false': [...]}
    if isinstance(pred_data, dict):
        return list(pred_data.get("false", []))

    return []


def parse_binary_pair(pair_str: str) -> Tuple[str, str]:
    """
    Parse a binary pair string like '(a0, a1)' into a tuple ('a0', 'a1').

    Args:
        pair_str: String like '(a0, a1)' or 'a0, a1'

    Returns:
        Tuple of two element names
    """
    pair_str = str(pair_str).strip("()")
    parts = [p.strip() for p in pair_str.split(",")]
    if len(parts) == 2:
        return (parts[0], parts[1])
    raise ValueError(f"Invalid binary pair format: {pair_str}")


def get_predicate_true_count(
    predicates: Dict[str, Any], pred_name: str, is_binary: bool = False
) -> int:
    """
    Get count of true atoms for a predicate.

    Args:
        predicates: Dict of predicate name -> predicate data
        pred_name: Name of predicate
        is_binary: Whether this is a binary predicate

    Returns:
        Count of true atoms
    """
    pred_data = predicates.get(pred_name)
    if is_binary:
        return len(get_binary_extension(pred_data))
    return len(get_unary_extension(pred_data))


def get_predicate_false_count(
    predicates: Dict[str, Any], pred_name: str, is_binary: bool = False
) -> int:
    """
    Get count of false atoms for a predicate.

    Args:
        predicates: Dict of predicate name -> predicate data
        pred_name: Name of predicate
        is_binary: Whether this is a binary predicate

    Returns:
        Count of explicitly false atoms
    """
    pred_data = predicates.get(pred_name)
    if is_binary:
        return len(get_binary_false_extension(pred_data))
    return len(get_unary_false_extension(pred_data))


# Convenience functions for common predicates
def get_P_extension(predicates: Dict[str, Any]) -> List[str]:
    """Get true extension for unary predicate P."""
    return get_unary_extension(predicates.get("P"))


def get_Q_extension(predicates: Dict[str, Any]) -> List[str]:
    """Get true extension for unary predicate Q."""
    return get_unary_extension(predicates.get("Q"))


def get_R_extension(predicates: Dict[str, Any]) -> List[str]:
    """Get true extension for binary predicate R."""
    return get_binary_extension(predicates.get("R"))


def get_S_extension(predicates: Dict[str, Any]) -> List[str]:
    """Get true extension for binary predicate S."""
    return get_binary_extension(predicates.get("S"))
