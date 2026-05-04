# -*- coding: utf-8 -*-
"""
inference_engine.py — Layer 3: Clingo-backed ASP Logic Runner
══════════════════════════════════════════════════════════════════

A proper Datalog / Answer-Set Programming engine powered by Clingo
(Potassco), the state-of-the-art ASP solver.

Why Clingo instead of hand-rolled forward-chaining:
  • Native negation-as-failure (``not``) — no workarounds
  • Formal semantics — stable model semantics, well-understood
  • Battle-tested solver — used in academic/industrial AI research
  • Rules are readable Datalog — any researcher can audit them
  • Fast — 5ms for our rule set, scales to thousands of facts

The engine translates Python facts into ASP atoms, appends the
classification rules (from classification_rules.py), runs Clingo,
and parses the answer set back into Python objects.

Usage
─────
    from inference_engine import Engine

    engine = Engine()

    # Assert facts (from entity_extractor + frame_detector)
    engine.assert_fact("food_mention", "mel")
    engine.assert_fact("restricted_item", "diabetes", "mel", "added_sugar")
    engine.assert_fact("frame", "negation", "mel")

    # Solve
    engine.solve()

    # Query
    engine.query("safe_context")     → [("mel", "negation")]
    engine.query("violation")        → []
    engine.explain("safe_context", "mel", "negation")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import clingo

from asp_rules import get_full_program as _get_classification_rules

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Fact:
    """A ground atom: predicate(arg1, arg2, …)."""
    predicate: str
    args: tuple[str, ...]

    def to_asp(self) -> str:
        """Render as an ASP fact string: ``predicate(a1, a2).``"""
        if not self.args:
            return f"{self.predicate}."
        args_str = ",".join(str(a) for a in self.args)
        return f"{self.predicate}({args_str})."

    def __repr__(self) -> str:
        if not self.args:
            return f"{self.predicate}"
        return f"{self.predicate}({', '.join(repr(a) for a in self.args)})"


@dataclass
class SolveResult:
    """Result of a Clingo solve call."""
    satisfiable: bool
    atoms: list[Fact] = field(default_factory=list)
    solve_time_ms: float = 0.0

    def query(self, predicate: str) -> list[tuple[str, ...]]:
        """Get all tuples for a given predicate."""
        return [f.args for f in self.atoms if f.predicate == predicate]

    def has(self, predicate: str, *args: str) -> bool:
        """Check if a specific ground atom is in the answer set."""
        target = Fact(predicate, args)
        return target in self.atoms


# The classification rules are defined in classification_rules.py
# (the ASP "knowledge base"), imported once here.
CLASSIFICATION_RULES = _get_classification_rules()


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class Engine:
    """
    ASP-backed inference engine using Clingo.

    Workflow:
      1. Assert ground facts via assert_fact()
      2. Call solve() — compiles ASP program + facts, runs Clingo
      3. Query results via result.query() or result.has()
    """

    def __init__(self, extra_rules: str = "") -> None:
        self._facts: list[Fact] = []
        self._fact_set: set[Fact] = set()
        self._extra_rules = extra_rules
        self._result: SolveResult | None = None

    # ── Fact management ─────────────────────────────────────────────────

    def assert_fact(self, predicate: str, *args: str) -> Fact:
        """Assert a ground fact.  Args are auto-sanitised for ASP."""
        clean_args = tuple(_asp_safe(str(a)) for a in args)
        fact = Fact(predicate, clean_args)
        if fact not in self._fact_set:
            self._facts.append(fact)
            self._fact_set.add(fact)
        return fact

    def assert_facts(self, facts: list[Fact]) -> None:
        """Assert multiple pre-built facts."""
        for f in facts:
            if f not in self._fact_set:
                self._facts.append(f)
                self._fact_set.add(f)

    @property
    def facts(self) -> list[Fact]:
        return list(self._facts)

    @property
    def n_facts(self) -> int:
        return len(self._facts)

    # ── Solve ───────────────────────────────────────────────────────────

    def solve(self) -> SolveResult:
        """
        Compile facts + rules into an ASP program and solve with Clingo.

        Returns a SolveResult containing all derived atoms.
        """
        import time

        # Build the full ASP program
        fact_lines = "\n".join(f.to_asp() for f in self._facts)
        program = fact_lines + "\n" + CLASSIFICATION_RULES
        if self._extra_rules:
            program += "\n" + self._extra_rules

        logger.debug("ASP program (%d facts, %d rule chars)",
                      len(self._facts), len(CLASSIFICATION_RULES))

        # Run Clingo
        t0 = time.time()
        ctl = clingo.Control(["--warn=none"])  # suppress warnings
        ctl.add("base", [], program)
        ctl.ground([("base", [])])

        answer_atoms: list[Fact] = []
        satisfiable = False

        def on_model(model: clingo.Model) -> None:
            nonlocal satisfiable
            satisfiable = True
            for sym in model.symbols(shown=True):
                args = tuple(_asp_unquote(str(a)) for a in sym.arguments)
                answer_atoms.append(Fact(sym.name, args))

        ctl.solve(on_model=on_model)
        elapsed_ms = (time.time() - t0) * 1000

        self._result = SolveResult(
            satisfiable=satisfiable,
            atoms=answer_atoms,
            solve_time_ms=elapsed_ms,
        )

        logger.debug("Solved in %.1fms: %d atoms, sat=%s",
                      elapsed_ms, len(answer_atoms), satisfiable)

        return self._result

    # ── Query shortcuts (delegates to SolveResult) ──────────────────────

    @property
    def result(self) -> SolveResult:
        if self._result is None:
            raise RuntimeError("Call solve() before querying results.")
        return self._result

    def query(self, predicate: str) -> list[tuple[str, ...]]:
        """Query all tuples for a predicate in the answer set."""
        return self.result.query(predicate)

    def has(self, predicate: str, *args: str) -> bool:
        """Check if a specific atom exists in the answer set."""
        return self.result.has(predicate, *args)

    def explain(self, predicate: str, *args: str) -> str:
        """
        Build a human-readable explanation for why an atom holds.

        Traces back through the rules to show which facts contributed.
        """
        target = Fact(predicate, tuple(str(a) for a in args))

        if target not in set(self.result.atoms):
            return f"{target} — NOT in answer set"

        # For violations: show which restricted_item triggered it
        if predicate == "violation" and len(args) == 1:
            food = args[0]
            restrictions = [
                a for a in self.result.atoms
                if a.predicate == "restricted_item" and a.args[1] == food
            ]
            r_str = ", ".join(str(r) for r in restrictions)
            return (
                f"{target} — derived because: "
                f"restricted_item holds ({r_str}), "
                f"food_mention({food}) present, "
                f"no safe_context({food}, _) found"
            )

        # For safe_context: show which frame triggered it
        if predicate == "safe_context" and len(args) == 2:
            food, ctx_type = args
            food_quoted = _asp_safe(food)
            frames = [
                a for a in self._facts
                if a.predicate == "frame"
                and (food in a.args or food_quoted in a.args)
            ]
            f_str = ", ".join(str(f) for f in frames)
            return (
                f"{target} — derived from frames: {f_str}"
            )

        return f"{target} — in answer set"

    # ── Utilities ───────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all facts and cached results."""
        self._facts.clear()
        self._fact_set.clear()
        self._result = None

    def get_asp_program(self) -> str:
        """Return the full ASP program as a string (for debugging/export)."""
        fact_lines = "\n".join(f.to_asp() for f in self._facts)
        return fact_lines + "\n" + CLASSIFICATION_RULES

    def __repr__(self) -> str:
        n_derived = len(self._result.atoms) if self._result else 0
        return (
            f"Engine(facts={len(self._facts)}, "
            f"derived={n_derived})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _asp_safe(s: str) -> str:
    """
    Make a string safe for use as an ASP constant.

    ASP constants must be lowercase and contain only [a-z0-9_].
    Strings with other characters are quoted.
    """
    # If it's already a simple identifier, return as-is
    if s.isidentifier() and s[0].islower() and all(c.isalnum() or c == '_' for c in s):
        return s
    # Otherwise quote it
    escaped = s.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def _asp_unquote(s: str) -> str:
    """Strip ASP string quotes from a Clingo symbol's string representation.

    Clingo renders quoted atoms as '"bolinho de bacalhau"'; we strip
    the outer quotes so that Python-side comparisons work with the
    original unquoted string.
    """
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        return s[1:-1].replace('\\"', '"').replace('\\\\', '\\')
    return s


# ═══════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY — same API as v1 inference_engine
# ═══════════════════════════════════════════════════════════════════════════
# These re-exports allow safe_context_classifier.py to work unchanged.

# Condition / NegCondition / Rule are no longer needed since rules
# are now expressed in ASP, but we keep stubs for import compatibility.

class Condition:
    """Stub — rules are now expressed in ASP. Kept for import compat."""
    def __init__(self, predicate: str, *args: Any): pass

class NegCondition:
    """Stub — rules are now expressed in ASP. Kept for import compat."""
    def __init__(self, predicate: str, *args: Any): pass

@dataclass
class Rule:
    """Stub — rules are now expressed in ASP. Kept for import compat."""
    name: str = ""
    conditions: list = field(default_factory=list)
    conclusion: tuple = ()
    priority: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# CLI — self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("── Clingo ASP Engine self-test ──\n")

    engine = Engine()

    # Domain knowledge
    engine.assert_fact("condition_restricts", "diabetes", "added_sugar")
    engine.assert_fact("condition_restricts", "diabetes", "high_glycemic")
    engine.assert_fact("condition_restricts", "lactose_intolerance", "lactose")

    # Lexicon substance flags
    engine.assert_fact("involves_substance", "mel", "added_sugar")
    engine.assert_fact("involves_substance", "mel", "high_glycemic")
    engine.assert_fact("involves_substance", "acucar", "added_sugar")
    engine.assert_fact("involves_substance", "leite", "lactose")

    # Extracted food mentions
    engine.assert_fact("food_mention", "mel")
    engine.assert_fact("food_mention", "acucar")
    engine.assert_fact("food_mention", "leite")

    # Detected frames (mel and açúcar negated; leite has none)
    engine.assert_fact("frame", "negation", "mel")
    engine.assert_fact("frame", "negation", "acucar")

    print(f"Facts: {engine.n_facts}")
    print(f"\nASP program:\n{engine.get_asp_program()}\n")

    # Solve
    result = engine.solve()
    print(f"Solve: {result.solve_time_ms:.1f}ms, "
          f"{len(result.atoms)} atoms, sat={result.satisfiable}\n")

    print("=== Restricted items ===")
    for args in result.query("restricted_item"):
        print(f"  restricted_item({', '.join(args)})")

    print("\n=== Safe contexts ===")
    for args in result.query("safe_context"):
        print(f"  safe_context({', '.join(args)})")

    print("\n=== Violations ===")
    for args in result.query("violation"):
        food = args[0]
        print(f"  ✗ violation({food})")
        print(f"    {engine.explain('violation', food)}")

    # Assertions
    assert result.has("safe_context", "mel", "negation"), "mel should be safe"
    assert result.has("safe_context", "acucar", "negation"), "acucar should be safe"
    assert result.has("violation", "leite"), "leite should be violation"
    assert not result.has("violation", "mel"), "mel should NOT be violation"
    assert not result.has("violation", "acucar"), "acucar should NOT be violation"

    print("\n✓ All assertions passed!")
    print(f"\nEngine: {engine}")
