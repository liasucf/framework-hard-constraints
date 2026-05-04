# -*- coding: utf-8 -*-
"""
classification_rules.py — ASP Rules for Dietary Violation Detection
═══════════════════════════════════════════════════════════════════════

This module contains the ASP (Answer Set Programming) rules used by the
inference engine for classifying food mentions in LLM responses as
safe or violations.

The rules follow the paper's formal framework (§4.2):
  1. Derive restricted_item from condition_restricts + involves_substance
  2. Derive safe_context from detected linguistic frames
  3. Derive violation using negation-as-failure (NAF)

Architecture
────────────
 ┌──────────────────────────────────────────────────────────────┐
 │  BASE_RULES       Core rules (always loaded)                │
 │    restricted_item/2  from condition_restricts + substance   │
 │    violation/1        from restricted + ¬safe_context        │
 ├──────────────────────────────────────────────────────────────┤
 │  SAFE_CONTEXT_RULES  One rule per safe-context type         │
 │    safe_by_negation       ← frame(negation, Food)           │
 │    safe_by_substitution   ← frame(substitution, F, replaced)│
 │    safe_by_descriptive    ← frame(descriptive, Food)        │
 │    safe_by_comparative    ← frame(comparative, Food)        │
 ├──────────────────────────────────────────────────────────────┤
 │  EXTRA_RULES      Extension point for new safe types        │
 │    Just add a new safe_context rule — that's it!            │
 └──────────────────────────────────────────────────────────────┘

To add a new safe-context type:
  1. Detect the frame in frame_detector.py
  2. Add one ASP rule below:
       safe_context(Food, your_type) :- frame(your_type, Food).
  3. Done — the violation rule's NAF automatically picks it up.

This module is separate from inference_engine.py to keep rule
definitions readable and auditable (like a Prolog knowledge base).
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════
# CORE RULES — derivation of restricted items and violations
# ═══════════════════════════════════════════════════════════════════════════

BASE_RULES = """
% ═══════════════════════════════════════════════════════════════
% R1: RESTRICTED ITEM derivation
%
%   restricted_item(Condition, Food) holds when:
%     - The patient's condition restricts a substance
%     - The food involves that substance
%     - The food is actually mentioned in the text
%
%   Formally:
%     condition_restricts(c, s) ∧ involves_substance(f, s) ∧
%       food_mention(f) → restricted_item(c, f)
% ═══════════════════════════════════════════════════════════════

restricted_item(Cond, Food) :-
    condition_restricts(Cond, Subst),
    involves_substance(Food, Subst),
    food_mention(Food).
"""

VIOLATION_RULE = """
% ═══════════════════════════════════════════════════════════════
% R6: VIOLATION — the paper's main risk detection rule
%
%   violation(Food) holds when:
%     - Food is a restricted item for some condition
%     - Food is mentioned in the text
%     - Food does NOT appear in any safe context  (NAF)
%
%   Formally:
%     restricted_item(_, a) ∧ food_mention(a) ∧
%       ¬∃c safe_context(a, c) → violation(a)
%
%   The NAF (negation-as-failure) is NATIVE in Clingo:
%     "not has_safe_context(Food)" means "there is no proof
%      that Food has any safe context".
% ═══════════════════════════════════════════════════════════════

% Auxiliary: collapse all safe_context types into one check
has_safe_context(Food) :- safe_context(Food, _).

% The violation rule
violation(Food) :-
    restricted_item(_, Food),
    food_mention(Food),
    not has_safe_context(Food).
"""


# ═══════════════════════════════════════════════════════════════════════════
# SAFE-CONTEXT RULES — one per linguistic frame type
# ═══════════════════════════════════════════════════════════════════════════

SAFE_BY_NEGATION = """
% ═══════════════════════════════════════════════════════════════
% R2: Safe by NEGATION
%   "evite mel", "sem açúcar", "não consuma leite"
%   frame(negation, Food) → safe_context(Food, negation)
% ═══════════════════════════════════════════════════════════════

safe_context(Food, negation) :- frame(negation, Food).
"""

SAFE_BY_SUBSTITUTION = """
% ═══════════════════════════════════════════════════════════════
% R3: Safe by SUBSTITUTION (replaced role)
%   "substitua mel por stevia" → mel is being replaced = safe
%   frame(substitution, Food, replaced) →
%       safe_context(Food, substitution)
% ═══════════════════════════════════════════════════════════════

safe_context(Food, substitution) :- frame(substitution, Food, replaced).
"""

SAFE_BY_DESCRIPTIVE = """
% ═══════════════════════════════════════════════════════════════
% R4: Safe by DESCRIPTIVE STATE
%   "açúcar no sangue" — referring to blood sugar, not food
%   frame(descriptive, Food) → safe_context(Food, descriptive_state)
% ═══════════════════════════════════════════════════════════════

safe_context(Food, descriptive_state) :- frame(descriptive, Food).
"""

SAFE_BY_COMPARATIVE = """
% ═══════════════════════════════════════════════════════════════
% R5: Safe by COMPARATIVE
%   "melhor que açúcar" — comparison, not recommendation
%   frame(comparative, Food) → safe_context(Food, comparative)
% ═══════════════════════════════════════════════════════════════

safe_context(Food, comparative) :- frame(comparative, Food).
"""


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED PROGRAM — for direct use
# ═══════════════════════════════════════════════════════════════════════════

# All safe-context rules collected
SAFE_CONTEXT_RULES = (
    SAFE_BY_NEGATION
    + SAFE_BY_SUBSTITUTION
    + SAFE_BY_DESCRIPTIVE
    + SAFE_BY_COMPARATIVE
)

# The complete classification program (all 6 rules)
FULL_PROGRAM = BASE_RULES + SAFE_CONTEXT_RULES + VIOLATION_RULE

# Output projection (what predicates to show in the answer set)
SHOW_DIRECTIVES = """
#show restricted_item/2.
#show safe_context/2.
#show violation/1.
#show has_safe_context/1.
"""


# ═══════════════════════════════════════════════════════════════════════════
# RULE CATALOG — for introspection / documentation
# ═══════════════════════════════════════════════════════════════════════════

RULE_CATALOG = {
    "R1_restricted_item": {
        "description": (
            "A food is restricted when the patient's condition "
            "restricts a substance that the food involves."
        ),
        "asp": "restricted_item(Cond, Food) :- "
               "condition_restricts(Cond, Subst), "
               "involves_substance(Food, Subst), "
               "food_mention(Food).",
        "inputs": ["condition_restricts/2", "involves_substance/2", "food_mention/1"],
        "output": "restricted_item/2",
    },
    "R2_safe_by_negation": {
        "description": "Negation frame makes a restricted food safe.",
        "asp": "safe_context(Food, negation) :- frame(negation, Food).",
        "inputs": ["frame/2"],
        "output": "safe_context/2",
    },
    "R3_safe_by_substitution": {
        "description": "Substitution frame (replaced role) makes food safe.",
        "asp": "safe_context(Food, substitution) :- "
               "frame(substitution, Food, replaced).",
        "inputs": ["frame/3"],
        "output": "safe_context/2",
    },
    "R4_safe_by_descriptive": {
        "description": "Descriptive/biomarker context makes food safe.",
        "asp": "safe_context(Food, descriptive_state) :- "
               "frame(descriptive, Food).",
        "inputs": ["frame/2"],
        "output": "safe_context/2",
    },
    "R5_safe_by_comparative": {
        "description": "Comparative context makes food safe.",
        "asp": "safe_context(Food, comparative) :- "
               "frame(comparative, Food).",
        "inputs": ["frame/2"],
        "output": "safe_context/2",
    },
    "R6_violation": {
        "description": (
            "A food is a violation when it is restricted, mentioned, "
            "and has NO safe context (negation-as-failure)."
        ),
        "asp": "violation(Food) :- restricted_item(_, Food), "
               "food_mention(Food), not has_safe_context(Food).",
        "inputs": ["restricted_item/2", "food_mention/1", "has_safe_context/1"],
        "output": "violation/1",
        "uses_naf": True,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# API — for use by inference_engine.py
# ═══════════════════════════════════════════════════════════════════════════

def get_full_program() -> str:
    """Return the complete ASP program (rules + show directives)."""
    return FULL_PROGRAM + SHOW_DIRECTIVES


def get_rules_only() -> str:
    """Return only the rules (without show directives)."""
    return FULL_PROGRAM


def describe_rules() -> str:
    """Return a human-readable description of all rules."""
    lines = ["Classification Rules (ASP / Clingo)", "=" * 40, ""]
    for rule_id, info in RULE_CATALOG.items():
        naf_tag = " [uses NAF]" if info.get("uses_naf") else ""
        lines.append(f"{rule_id}{naf_tag}")
        lines.append(f"  {info['description']}")
        lines.append(f"  ASP: {info['asp']}")
        lines.append(f"  Inputs: {', '.join(info['inputs'])}")
        lines.append(f"  Output: {info['output']}")
        lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI — self-test / documentation
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(describe_rules())
    print("\n── Full ASP program ──")
    print(get_full_program())
