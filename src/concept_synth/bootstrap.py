"""Compatibility shim for copied modules from the internal repo."""

from __future__ import annotations


def add_repo_root(_path: str) -> None:
    """Public package installs do not need repo-root path injection."""
    return None

