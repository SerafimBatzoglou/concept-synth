"""
sexpr_parser.py - S-expression Parser for FOL Formulas

Parses S-expression strings into FOFormula AST objects.
Supports the restricted grammar G1 used in the concept synthesis benchmark.

Grammar:
    φ ::= (P x)              -- unary predicate
        | (R x y)            -- binary predicate
        | (= x y)            -- equality
        | (not φ)            -- negation (restricted to atoms in NNF)
        | (and φ₁ φ₂)        -- conjunction (binary, accepts n-ary and converts)
        | (or φ₁ φ₂)         -- disjunction (binary, accepts n-ary and converts)
        | (forall v φ)       -- universal quantification
        | (exists v φ)       -- existential quantification
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, List, Optional, Set, Tuple

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
from concept_synth.fol.formulas import (
    Constant,
    Eq,
    Exists,
    FOAnd,
    FOBiconditional,
    FOFormula,
    FOImplies,
    FONot,
    FOOr,
    Forall,
    FOTerm,
    Pred,
    Var,
)
from concept_synth.signature_utils import DEFAULT_ALLOWED_INDUCTION_PREDICATES


class SExprParseError(Exception):
    """Exception raised for S-expression parsing errors."""

    pass


class SExprParser:
    """Parser for S-expression FOL formulas."""

    def __init__(self, s: str, allowed_predicates: Optional[Set[str]] = None):
        """
        Initialize parser.

        Args:
            s: The S-expression string to parse
            allowed_predicates: Optional set of allowed predicate names
        """
        self.s = s
        self.pos = 0
        self.allowed_predicates = allowed_predicates or set(DEFAULT_ALLOWED_INDUCTION_PREDICATES)
        self.tokens = self._tokenize(s)
        self.token_pos = 0

    def _tokenize(self, s: str) -> List[str]:
        """Tokenize the S-expression string."""
        # Add spaces around parentheses
        s = s.replace("(", " ( ").replace(")", " ) ")
        # Split on whitespace
        tokens = s.split()
        return tokens

    def peek(self) -> Optional[str]:
        """Peek at the current token without consuming it."""
        if self.token_pos >= len(self.tokens):
            return None
        return self.tokens[self.token_pos]

    def consume(self) -> str:
        """Consume and return the current token."""
        if self.token_pos >= len(self.tokens):
            raise SExprParseError("Unexpected end of input")
        token = self.tokens[self.token_pos]
        self.token_pos += 1
        return token

    def expect(self, expected: str):
        """Consume a token and verify it matches expected."""
        token = self.consume()
        if token != expected:
            raise SExprParseError(f"Expected '{expected}', got '{token}'")

    def parse(self) -> FOFormula:
        """Parse the entire S-expression and return an FOFormula."""
        formula = self.parse_formula()
        if self.token_pos < len(self.tokens):
            remaining = " ".join(self.tokens[self.token_pos :])
            raise SExprParseError(f"Unexpected tokens after formula: {remaining}")
        return formula

    def parse_formula(self) -> FOFormula:
        """Parse a single formula."""
        token = self.peek()

        if token is None:
            raise SExprParseError("Unexpected end of input while parsing formula")

        if token == "(":
            self.consume()  # consume '('
            return self.parse_compound()
        else:
            # Bare variable or constant - not valid as a formula
            raise SExprParseError(f"Expected '(' to start formula, got '{token}'")

    def parse_compound(self) -> FOFormula:
        """Parse a compound formula (after the opening paren)."""
        operator = self.consume()

        if operator in ("not", "and", "or", "implies", "iff", "forall", "exists"):
            return self.parse_connective(operator)
        elif operator == "=":
            return self.parse_equality()
        elif operator in self.allowed_predicates:
            return self.parse_predicate(operator)
        else:
            raise SExprParseError(f"Unknown operator or predicate: '{operator}'")

    def parse_connective(self, conn_type: str) -> FOFormula:
        """Parse a logical connective."""
        if conn_type == "not":
            inner = self.parse_formula()
            self.expect(")")
            return FONot(inner)

        elif conn_type == "and":
            # Accept n-ary AND and convert to binary (right-associative)
            formulas = []
            while self.peek() != ")":
                formulas.append(self.parse_formula())
            self.expect(")")
            if len(formulas) < 2:
                raise SExprParseError(f"'and' requires at least 2 arguments, got {len(formulas)}")
            # Convert n-ary AND to nested binary AND (right-associative)
            result = formulas[-1]
            for f in reversed(formulas[:-1]):
                result = FOAnd(f, result)
            return result

        elif conn_type == "or":
            # Accept n-ary OR and convert to binary (right-associative)
            formulas = []
            while self.peek() != ")":
                formulas.append(self.parse_formula())
            self.expect(")")
            if len(formulas) < 2:
                raise SExprParseError(f"'or' requires at least 2 arguments, got {len(formulas)}")
            # Convert n-ary OR to nested binary OR (right-associative)
            result = formulas[-1]
            for f in reversed(formulas[:-1]):
                result = FOOr(f, result)
            return result

        elif conn_type == "implies":
            left = self.parse_formula()
            right = self.parse_formula()
            self.expect(")")
            return FOImplies(left, right)

        elif conn_type == "iff":
            left = self.parse_formula()
            right = self.parse_formula()
            self.expect(")")
            return FOBiconditional(left, right)

        elif conn_type == "forall":
            var_name = self.consume()
            if not self._is_variable(var_name):
                raise SExprParseError(f"Expected variable name after 'forall', got '{var_name}'")
            body = self.parse_formula()
            self.expect(")")
            return Forall(Var(var_name), "D", body)

        elif conn_type == "exists":
            var_name = self.consume()
            if not self._is_variable(var_name):
                raise SExprParseError(f"Expected variable name after 'exists', got '{var_name}'")
            body = self.parse_formula()
            self.expect(")")
            return Exists(Var(var_name), "D", body)

        else:
            raise SExprParseError(f"Unknown connective: '{conn_type}'")

    def parse_equality(self) -> FOFormula:
        """Parse an equality formula (= t1 t2)."""
        t1 = self.parse_term()
        t2 = self.parse_term()
        self.expect(")")
        return Eq(t1, t2)

    def parse_predicate(self, pred_name: str) -> FOFormula:
        """Parse a predicate application."""
        args = []
        while self.peek() != ")":
            args.append(self.parse_term())
        self.expect(")")
        return Pred(pred_name, args)

    def parse_term(self) -> FOTerm:
        """Parse a term (variable or constant)."""
        token = self.consume()
        if self._is_variable(token):
            return Var(token)
        else:
            return Constant(token)

    def _is_variable(self, name: str) -> bool:
        """Check if a name is a variable (single lowercase letter)."""
        return len(name) == 1 and name.islower()


def parse_sexpr_formula(s: str, allowed_predicates: Optional[Set[str]] = None) -> FOFormula:
    """
    Parse an S-expression string into an FOFormula.

    Args:
        s: The S-expression string
        allowed_predicates: Optional set of allowed predicate names

    Returns:
        The parsed FOFormula

    Raises:
        SExprParseError: If parsing fails
    """
    parser = SExprParser(s, allowed_predicates)
    return parser.parse()


def try_parse_sexpr(s: str) -> Tuple[Optional[FOFormula], Optional[str]]:
    """
    Try to parse an S-expression, returning (formula, None) on success
    or (None, error_message) on failure.
    """
    try:
        formula = parse_sexpr_formula(s)
        return formula, None
    except SExprParseError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Unexpected error: {e}"
