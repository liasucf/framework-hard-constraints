# -*- coding: utf-8 -*-
"""
entity_extractor.py — Module 2: Extract Food Entities & Derive Restricted Items
════════════════════════════════════════════════════════════════════════════════

Given:
  • An LLM response text
  • A patient health condition
  • A pre-built Aho-Corasick automaton (from lexicon_builder.py)

Produces:
  • A list of food entities found in the text
  • For each entity, whether it is a restricted_item for the patient

Logic (from the paper §4.1–4.2):

  condition_restricts(condition, substance)    [domain KB]
    ∧ involves_substance(food, substance)      [from lexicon]
    → restricted_item(condition, food)         [derived]

Usage:
    from lexicon_builder import load_lexicon
    from entity_extractor import extract_restricted_items

    _, _, automaton = load_lexicon()
    items = extract_restricted_items(
        text="Adicione mel ao chá de gengibre.",
        condition="diabetes",
        automaton=automaton,
    )
    for item in items:
        print(item)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from lexicon_builder import norm, _NON_STANDALONE_WORDS

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN KNOWLEDGE BASE — condition_restricts(condition, substance)
# ═══════════════════════════════════════════════════════════════════════════
# Covers all 19 clinical conditions from the paper / restricoes.csv.

CONDITION_RESTRICTS: dict[str, list[str]] = {
    "diabetes":                  ["added_sugar", "high_glycemic"],
    "hypertension":              ["sodium"],
    "chronic_kidney_disease":    ["phosphorus", "potassium", "sodium"],
    "heart_failure":             ["sodium", "saturated_fat"],
    "irritable_bowel_syndrome":  ["fodmap"],
    "short_bowel_syndrome":      ["saturated_fat", "added_sugar", "insoluble_fiber"],
    "gout":                      ["purine", "sodium"],
    "hyperuricemia":             ["purine"],        # purine metabolism → uric acid
    "dyslipidemia":              ["saturated_fat", "trans_fat", "cholesterol"],
    "nafld":                     ["saturated_fat", "added_sugar", "alcohol"],
    "crohns_disease":            ["saturated_fat", "caffeine", "alcohol", "insoluble_fiber"],
    "celiac_disease":            ["gluten"],
    "lactose_intolerance":       ["lactose"],
    "high_triglycerides":        ["saturated_fat", "added_sugar", "trans_fat"],
    "gastroesophageal_reflux":   ["caffeine", "saturated_fat"],
    "alcoholic_fatty_liver":     ["alcohol", "saturated_fat", "added_sugar"],
    "peanut_allergy":            ["peanut"],
    "nut_allergy":               ["nut", "peanut"],
    "egg_allergy":               ["egg"],
}

# Mapping from Portuguese condition names (as they appear in restricoes.csv)
# to the English keys used in CONDITION_RESTRICTS.
CONDITION_PT_TO_KEY: dict[str, str] = {
    "diabetes tipo 2":                 "diabetes",
    "diabetes":                        "diabetes",
    "hipertensao":                     "hypertension",
    "hipertensão":                     "hypertension",
    "doenca renal cronica":            "chronic_kidney_disease",
    "doença renal crônica":            "chronic_kidney_disease",
    "insuficiencia cardiaca":          "heart_failure",
    "insuficiência cardíaca":          "heart_failure",
    "sindrome do intestino irritavel": "irritable_bowel_syndrome",
    "síndrome do intestino irritável": "irritable_bowel_syndrome",
    "sindrome do intestino curto":     "short_bowel_syndrome",
    "síndrome do intestino curto":     "short_bowel_syndrome",
    "gota":                            "gout",
    "hiperuricemia":                   "hyperuricemia",
    "dislipidemia":                    "dyslipidemia",
    "esteatose hepatica":              "nafld",
    "esteatose hepática":              "nafld",
    "doenca de crohn":                 "crohns_disease",
    "doença de crohn":                 "crohns_disease",
    "doenca celiaca":                  "celiac_disease",
    "doença celíaca":                  "celiac_disease",
    "intolerancia a lactose":          "lactose_intolerance",
    "intolerância à lactose":          "lactose_intolerance",
    "triglicerides altos":             "high_triglycerides",
    "triglicérides altos":             "high_triglycerides",
    "refluxo gastroesofagico":         "gastroesophageal_reflux",
    "refluxo gastroesofágico":         "gastroesophageal_reflux",
    "esteatose hepatica alcoolica":    "alcoholic_fatty_liver",
    "esteatose hepática alcoólica":    "alcoholic_fatty_liver",
    "alergia a amendoim":              "peanut_allergy",
    "alergia ao amendoim":             "peanut_allergy",
    "alergia a nozes":                 "nut_allergy",
    "alergia a castanhas":             "nut_allergy",
    "alergia a oleaginosas":           "nut_allergy",
    "alergia a ovo":                   "egg_allergy",
    "alergia ao ovo":                  "egg_allergy",
    "alergia a ovos":                  "egg_allergy",
    "triglicerideos elevados":         "high_triglycerides",
    "triglicerídeos elevados":         "high_triglycerides",
    "doenca hepatica gordurosa nao alcoolica": "nafld",
    "doença hepática gordurosa não alcoólica": "nafld",
    "dhgna":                           "nafld",
}


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FoodMatch:
    """A food entity detected in the text by the automaton."""
    matched_norm: str               # normalised variant that matched
    canonical: str                  # base food name (for display)
    tbca_code: str | None           # TBCA code (None for synthetic variants)
    substances: frozenset[str]      # substance flags from the lexicon
    start: int                      # character offset in normalised text
    end: int                        # character offset end


@dataclass
class RestrictedItem:
    """A food entity that is restricted for the patient's condition.

    Corresponds to: restricted_item(condition, food) in the paper.
    """
    food: FoodMatch                 # the matched food entity
    condition: str                  # condition key
    matching_substances: list[str]  # substances causing the restriction
    is_restricted: bool = True      # always True for this class


# ═══════════════════════════════════════════════════════════════════════════
# ENTITY EXTRACTION (Aho-Corasick)
# ═══════════════════════════════════════════════════════════════════════════

def _singularize(word: str) -> str:
    """Basic Portuguese plural → singular (mirrors lexicon_builder)."""
    if len(word) <= 3:
        return word
    if word == "paes":
        return "pao"
    if word.endswith("oes") and len(word) >= 5:
        return word[:-3] + "ao"
    if word.endswith("aes") and len(word) >= 5:
        return word[:-3] + "ao"
    if word.endswith("eis") and len(word) >= 5:
        return word[:-3] + "el"
    if word.endswith("ais") and len(word) >= 5:
        return word[:-3] + "al"
    if word.endswith("ns") and len(word) >= 4:
        return word[:-2] + "m"
    if word.endswith("zes") and len(word) >= 5:
        return word[:-2]
    if word.endswith("res") and len(word) >= 5:
        return word[:-2]
    if word.endswith("ses") and len(word) >= 5:
        return word[:-2]
    if word.endswith("es") and len(word) >= 5:
        return word[:-1]
    if word.endswith("is") and len(word) >= 5:
        return word[:-2] + "il"
    if word.endswith("s") and len(word) >= 4 and not word.endswith("us"):
        return word[:-1]
    return word


def _canonicalize_match(matched_norm: str) -> str:
    """Normalise a matched term to its canonical (singular) form.

    Multi-word matches are returned as-is because their plural form
    is already specific enough ("frutos do mar" stays as-is).
    Single-word matches are singularised so that "ovos" → "ovo",
    "frutas" → "fruta", "carnes" → "carne", etc.
    """
    if " " in matched_norm:
        return matched_norm
    sg = _singularize(matched_norm)
    return sg if len(sg) >= 3 else matched_norm


# ── Post-match structural filters ──────────────────────────────────────
# These are applied to automaton results to handle multi-word expressions
# that cannot be resolved by the automaton alone.

#: Spice/condiment TBCA base_food names — these are dried spices
#: consumed in 2-5 g portions, making *all* per-100 g nutrient flags
#: (high_glycemic, potassium, phosphorus, saturated_fat, insoluble_fiber,
#: cholesterol, etc.) clinically negligible.  We strip every substance
#: flag from these entries.
_SPICE_CONDIMENT_BASES: frozenset[str] = frozenset([
    "acafrao", "alecrim", "anis", "baunilha", "canela", "cardamomo",
    "cebolinha", "coentro", "cominho", "cravo", "curcuma", "curry",
    "dill", "endro", "erva doce", "funcho", "gengibre", "hortela", "louro",
    "manjericao", "manjerona", "mostarda", "noz moscada", "oregano",
    "paprica", "pimenta", "salsa", "tomilho", "urucum",
])

#: Generic category terms that are too vague to constitute a specific food.
#: These match TBCA entries whose substance flags come from one particular
#: product (e.g. "caldo" → bouillon cube) but the word itself can refer to
#: harmless variants (e.g. homemade broth).  Excluded from restriction.
_GENERIC_CATEGORY_TERMS: frozenset[str] = frozenset([
    "tempero", "suplemento", "caldo", "bebida", "sopa", "pate",
    "molho",  # generic "sauce" — too vague; specific variants kept
])

# ── Fix 8 constants: compound/adjective/lean-food neutralisation ────────

#: Compound modifiers: "FOOD de MODIFIER" → neutralise substances.
_SAFE_COMPOUND_MODIFIERS: dict[str, frozenset[str]] = {
    "amendoim": frozenset({"lactose", "gluten"}),
    "coco": frozenset({"lactose"}),
    "arroz": frozenset({"lactose", "gluten"}),
    "soja": frozenset({"lactose", "gluten"}),
    "aveia": frozenset({"lactose"}),
    "abacate": frozenset({"lactose"}),
    "amendoas": frozenset({"lactose"}),
    "castanha": frozenset({"lactose"}),
    "caju": frozenset({"lactose"}),
    "macadamia": frozenset({"lactose"}),
    "girassol": frozenset({"lactose", "nut"}),
    "cacau": frozenset({"lactose"}),
    "mandioca": frozenset({"gluten", "lactose"}),
    "frango": frozenset({"saturated_fat", "cholesterol", "trans_fat"}),
    "peru": frozenset({"saturated_fat", "cholesterol", "trans_fat"}),
    "peixe": frozenset({"saturated_fat", "cholesterol", "trans_fat"}),
    "atum": frozenset({"saturated_fat", "cholesterol", "trans_fat"}),
    "salmao": frozenset({"saturated_fat", "trans_fat"}),
    "tilapia": frozenset({"saturated_fat", "cholesterol", "trans_fat"}),
}

#: Adjective modifiers: "FOOD ADJECTIVE" → neutralise substances.
_SAFE_ADJECTIVE_MODIFIERS: dict[str, frozenset[str]] = {
    "vegetal": frozenset({"lactose"}),
    "vegetais": frozenset({"lactose"}),
    "vegano": frozenset({"lactose", "egg"}),
    "vegana": frozenset({"lactose", "egg"}),
    "veganos": frozenset({"lactose", "egg"}),
    "veganas": frozenset({"lactose", "egg"}),
    "plant-based": frozenset({"lactose", "egg"}),
    "plant": frozenset({"lactose", "egg"}),  # "FOOD plant based"
    "magra": frozenset({"saturated_fat", "cholesterol", "trans_fat"}),
    "magro": frozenset({"saturated_fat", "cholesterol", "trans_fat"}),
    "magras": frozenset({"saturated_fat", "cholesterol", "trans_fat"}),
    "magros": frozenset({"saturated_fat", "cholesterol", "trans_fat"}),
    "grelhada": frozenset({"saturated_fat", "trans_fat"}),
    "grelhado": frozenset({"saturated_fat", "trans_fat"}),
    "cozida": frozenset({"saturated_fat", "trans_fat"}),
    "cozido": frozenset({"saturated_fat", "trans_fat"}),
    "integral": frozenset({"high_glycemic"}),
    "integrais": frozenset({"high_glycemic"}),
}

#: Generic foods too broad for lipid flags unless qualified.
_GENERIC_LEAN_FOODS: dict[str, frozenset[str]] = {
    "carne":  frozenset({"cholesterol", "saturated_fat", "trans_fat"}),
    "carnes": frozenset({"cholesterol", "saturated_fat", "trans_fat"}),
}

#: Adjectives that make generic meat UNSAFE (re-enable substances).
_UNSAFE_FOOD_ADJECTIVES: frozenset[str] = frozenset({
    "vermelha", "vermelhas", "vermelho",
    "gorda", "gordas", "gordo", "gordos", "gordurosa", "gordurosas",
    "processada", "processadas", "processado", "processados",
    "curada", "curadas", "curado", "curados",
    "defumada", "defumadas", "defumado", "defumados",
    "embutida", "embutidas", "embutido", "embutidos",
    "frita", "fritas", "frito", "fritos",
    "seca", "secas", "seco", "secos",
    "suina", "suinas", "suino", "suinos",
    "bovina", "bovinas", "bovino", "bovinos",
})

#: Meal-time continuations — if text after a match starts with one of these,
#: the match is part of a meal-name compound ("café da manhã"), not a food.
_MEAL_TIME_CONTINUATIONS = re.compile(
    r"^\s+da\s+(?:manha|tarde|noite)\b"
)

#: Substance names as they appear in Portuguese text, mapped to internal
#: substance keys.  Used to detect "food + sem + substance" patterns and
#: remove that substance from the food's flag set.
_SUBSTANCE_PT_TO_KEY: dict[str, str] = {
    "gluten": "gluten", "glutem": "gluten",
    "lactose": "lactose",
    "cafeina": "caffeine",
    "acucar": "added_sugar",
    "sodio": "sodium", "sal": "sodium",
    "gordura saturada": "saturated_fat",
    "gorduras saturadas": "saturated_fat",
    "gordura trans": "trans_fat",
    "colesterol": "cholesterol",
    "alcool": "alcohol",
    "purina": "purine",
    "fosforo": "phosphorus",
    "potassio": "potassium",
    "fodmap": "fodmap",
    "amendoim": "peanut",
    "ovo": "egg", "ovos": "egg",
}

#: Regex to detect "sem {substance}" immediately after a food match.
#: Built dynamically from the substance vocabulary above.
_SEM_SUBSTANCE_RE = re.compile(
    r"^\s+sem\s+("
    + "|".join(
        re.escape(pt) for pt in sorted(_SUBSTANCE_PT_TO_KEY, key=len, reverse=True)
    )
    + r")\b"
)

#: Global regex — detects "sem {substance}" appearing ANYWHERE in text.
#: Used for Lever 2b: when a celiac recipe says "sem glúten" or "tudo
#: sem glúten", strip that substance from ALL foods in the response.
#: This complements Fix 2 (which only looks immediately after the food).
_GLOBAL_SEM_SUBSTANCE_RE = re.compile(
    r"\bsem\s+("
    + "|".join(
        re.escape(pt) for pt in sorted(_SUBSTANCE_PT_TO_KEY, key=len, reverse=True)
    )
    + r")\b"
)

#: Fix 9: English/alternative "free-from" patterns.
#: Maps a regex pattern to the substance key it clears.
#: Matches "gluten-free", "gluten free", "lactose-free", "dairy-free",
#: "livre de glúten", etc.  Applied globally (same as Lever 2b).
_FREE_FROM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bgluten[- ]?free\b"), "gluten"),
    (re.compile(r"\blivre\s+de\s+gluten\b"), "gluten"),
    (re.compile(r"\blactose[- ]?free\b"), "lactose"),
    (re.compile(r"\bdairy[- ]?free\b"), "lactose"),
    (re.compile(r"\blivre\s+de\s+lactose\b"), "lactose"),
    (re.compile(r"\begg[- ]?free\b"), "egg"),
    (re.compile(r"\blivre\s+de\s+ovo\b"), "egg"),
]


#: Punctuation characters that, when present between words in the
#: original text, indicate a structural boundary (heading/list separator)
#: rather than a true multi-word food name.  E.g., "Peixe: Salmão"
#: should NOT match as "peixe salmao".
_STRUCT_PUNCT = frozenset(":;,|(){}[]")

#: Regex to detect structural punctuation (colon, semicolon, comma,
#: parens, etc.) between words of a multi-word match.  Commas are safe
#: to include because no automaton key contains commas (they are always
#: stripped during normalization of TBCA names).
_STRUCT_PUNCT_RE = re.compile(r"[:;,|(){}\[\]]")


def _build_norm_to_orig_map(text: str) -> tuple[str, list[int]]:
    """Normalise *text* while building a position map from normalised → original.

    Returns (text_norm, orig_positions) where orig_positions[i] is the
    index in the original *text* that produced normalised character i.
    """
    import unicodedata as _ud

    s = str(text).strip().lower()
    # Expand c/ and s/ abbreviations (mirror lexicon_builder.norm)
    s = re.sub(r"\bc/\s*", "com ", s)
    s = re.sub(r"\bs/\s*", "sem ", s)
    # Strip accents — keep track of character ↔ original index
    nfd = _ud.normalize("NFD", s)
    # Build (char, orig_idx) pairs, skipping combining marks
    pairs: list[tuple[str, int]] = []
    orig_idx = 0
    for ch in nfd:
        if _ud.category(ch) == "Mn":
            continue  # combining accent mark — skip
        pairs.append((ch, orig_idx))
        orig_idx += 1
    # Replace non-alphanumerics with space, collapse whitespace
    buf: list[str] = []
    idx_buf: list[int] = []
    prev_space = False
    for ch, oi in pairs:
        if ch.isalnum():
            buf.append(ch)
            idx_buf.append(oi)
            prev_space = False
        else:
            if not prev_space and buf:
                buf.append(" ")
                idx_buf.append(oi)
                prev_space = True
    # Strip trailing space
    if buf and buf[-1] == " ":
        buf.pop()
        idx_buf.pop()
    return "".join(buf), idx_buf


def extract_food_entities(text: str, automaton: Any) -> list[FoodMatch]:
    """Extract all food entity mentions from *text* using the automaton.

    Uses longest-match, non-overlapping spans. Matches are validated
    with word-boundary checks to avoid partial matches inside longer words.
    Single-word matches are normalised to singular form for canonical output.

    Post-processing:
      • Suppresses matches that are part of meal-time compounds
        ("café da manhã") — structural, no hardcoded food names.
      • When "sem {substance}" follows a food, removes that substance
        from the food's flag set ("aveia sem glúten" → gluten removed).

    Returns a list of FoodMatch sorted by position.
    """
    text_norm, norm_to_orig = _build_norm_to_orig_map(text)
    # Pre-compute lowercased original for punctuation checks
    text_lower = str(text).strip().lower()
    # Expand c/ s/ to match norm() expansion
    text_lower = re.sub(r"\bc/\s*", "com ", text_lower)
    text_lower = re.sub(r"\bs/\s*", "sem ", text_lower)
    raw_matches: list[dict] = []

    for end_idx, payload in automaton.iter(text_norm):
        term = payload["matched_norm"]
        start_idx = end_idx - len(term) + 1

        # Word-boundary check: character before start must not be alphanumeric
        if start_idx > 0 and text_norm[start_idx - 1].isalnum():
            continue
        # Character after end must not be alphanumeric
        if end_idx + 1 < len(text_norm) and text_norm[end_idx + 1].isalnum():
            continue

        # ── Fix 1: Meal-name suppression ────────────────────────────
        # If the text after the match starts with " da manhã/tarde/noite",
        # this is a meal compound, not a food mention.  Skip it.
        after = text_norm[end_idx + 1:]
        if _MEAL_TIME_CONTINUATIONS.match(after):
            continue

        # ── Fix 3: Variant-collision guard ──────────────────────────
        # Reject short variants (≤2 words) whose text doesn't contain
        # the base_food name.  These are spurious lexicon fragments
        # (e.g. "micro ondas" → milho, "preta" → azeitona).
        # Allow Portuguese plural forms where stems overlap
        # (e.g. amendoim→amendoins, pão→pães, mel→meis).
        base_food = payload.get("base_food", "")
        n_words = len(term.split())
        if base_food and n_words <= 2 and base_food not in term:
            # Allow if term shares a common stem prefix (≥2 chars)
            # Handles PT plurals: amendoim→amendoins, pão→pães, mel→meis
            stem = base_food[:-1]
            if len(stem) < 2 or not term.startswith(stem):
                continue

        # ── Fix 7: Structural punctuation guard ─────────────────────
        # Multi-word matches that span a structural separator in the
        # original text (colon, semicolon, parentheses, etc.) are
        # rejected.  This prevents "Peixe: Salmão" → "peixe salmao"
        # from matching, while keeping real multi-word entries like
        # "peixe salgado" or "carne de porco" intact.
        if n_words >= 2 and norm_to_orig:
            # Check original text between start and end for struct punct
            orig_start = norm_to_orig[start_idx] if start_idx < len(norm_to_orig) else 0
            orig_end = norm_to_orig[min(end_idx, len(norm_to_orig) - 1)] if norm_to_orig else 0
            orig_span = text_lower[orig_start:orig_end + 1]
            if _STRUCT_PUNCT_RE.search(orig_span):
                continue

        raw_matches.append({
            **payload,
            "start": start_idx,
            "end": end_idx + 1,
        })

    # Sort by match length descending (longest first) for greedy selection
    raw_matches.sort(key=lambda x: x["end"] - x["start"], reverse=True)

    # Non-overlapping greedy selection (longest match wins)
    final: list[FoodMatch] = []
    used_spans: list[tuple[int, int]] = []

    for m in raw_matches:
        span = (m["start"], m["end"])
        overlap = any(
            not (span[1] <= a or span[0] >= b) for a, b in used_spans
        )
        if not overlap:
            subs = m.get("substances", frozenset())
            if isinstance(subs, (list, set)):
                subs = frozenset(subs)

            # ── Fix 2: Substance-qualifier removal ──────────────────
            # "aveia sem glúten" → remove 'gluten' from aveia's substances.
            # Structural: uses the substance vocabulary, not food names.
            after_text = text_norm[span[1]:]
            sem_match = _SEM_SUBSTANCE_RE.match(after_text)
            if sem_match:
                pt_substance = sem_match.group(1)
                substance_key = _SUBSTANCE_PT_TO_KEY.get(pt_substance)
                if substance_key and substance_key in subs:
                    subs = subs - {substance_key}

            # ── Fix 8: Compound modifier neutralisation ─────────────
            # "manteiga de amendoim" → lactose neutralised
            # "iogurte vegetal" → lactose neutralised
            # "pão integral" → high_glycemic neutralised
            # "carne magra" → saturated_fat/cholesterol neutralised
            after_words = after_text.strip().split()[:3]
            matched_term = m["matched_norm"]

            # Pattern A: "FOOD de MODIFIER" (compound, e.g. "leite de coco")
            if len(after_words) >= 2 and after_words[0] == "de":
                mod = after_words[1].strip("()[]{}.,;:!?\"'/\\-")
                neut = _SAFE_COMPOUND_MODIFIERS.get(mod, frozenset())
                if neut:
                    subs = subs - neut

            # Pattern B: "FOOD ADJECTIVE" (e.g. "iogurte vegetal")
            if after_words:
                adj = after_words[0].strip("()[]{}.,;:!?")
                neut = _SAFE_ADJECTIVE_MODIFIERS.get(adj, frozenset())
                if neut:
                    subs = subs - neut

            # Pattern C: Multi-word match ending with adjective
            # (e.g. "pao integral" matched as single automaton entry)
            if " " in matched_term:
                last_tok = matched_term.rsplit(None, 1)[-1]
                neut = _SAFE_ADJECTIVE_MODIFIERS.get(last_tok, frozenset())
                if neut:
                    subs = subs - neut

            # Pattern D: Generic lean food neutralisation
            # Plain "carne" is too broad for lipid flags unless
            # followed by an unsafe adjective ("vermelha", "gorda").
            base_token = matched_term.split()[0] if " " in matched_term else matched_term
            generic_neut = _GENERIC_LEAN_FOODS.get(base_token, frozenset())
            if generic_neut:
                has_unsafe = False
                if after_words:
                    adj = after_words[0].strip("()[]{}.,;:!?")
                    if adj in _UNSAFE_FOOD_ADJECTIVES:
                        has_unsafe = True
                if not has_unsafe:
                    subs = subs - generic_neut

            # ── Fix 4: Spice/condiment — strip ALL flags ─────────
            # Dried spices are consumed at 2-5 g per serving, making
            # every per-100 g nutrient flag (potassium, phosphorus,
            # saturated_fat, high_glycemic, insoluble_fiber …)
            # clinically negligible.  Clear the full set.
            entry_base = m.get("base_food", "")
            if entry_base in _SPICE_CONDIMENT_BASES:
                subs = frozenset()

            # ── Fix 6: Generic category term exclusion ────────────
            # Terms like "tempero", "caldo", "suplemento" are too vague
            # to constitute a specific food. Skip them entirely.
            # But keep specific multi-word variants (e.g. "caldo de galinha").
            canon_norm = _canonicalize_match(m["matched_norm"])
            if canon_norm in _GENERIC_CATEGORY_TERMS:
                continue

            # ── Plural→singular substance alignment ──────────────
            # When a plural form (e.g. "azeitonas") is matched and
            # canonicalized to its singular ("azeitona"), use the
            # singular's authoritative substance flags from the
            # automaton.  Plural TBCA entries often have wrong flags
            # inherited from compound recipes.
            if canon_norm != m["matched_norm"] and canon_norm in automaton:
                singular_entry = automaton.get(canon_norm)
                singular_subs = singular_entry.get("substances", frozenset())
                subs = singular_subs

            # ── Non-standalone word guard ─────────────────────────
            # Single-word matches that are adjectives, generic categories,
            # or food components (e.g. "oleo", "algodao", "salgado") are
            # only meaningful in compounds. Block bare standalone matches.
            if " " not in canon_norm and canon_norm in _NON_STANDALONE_WORDS:
                continue

            # Convert normalized positions to original-text positions
            orig_s = norm_to_orig[span[0]] if span[0] < len(norm_to_orig) else span[0]
            orig_e = (norm_to_orig[span[1] - 1] + 1) if span[1] - 1 < len(norm_to_orig) else span[1]

            final.append(FoodMatch(
                matched_norm=canon_norm,
                canonical=m.get("canonical", canon_norm),
                tbca_code=m.get("tbca_code"),
                substances=subs,
                start=orig_s,
                end=orig_e,
            ))
            used_spans.append(span)

    # ── Lever 2b: Global "sem {substance}" scanning ────────────────
    # When the text contains "sem glúten" / "sem lactose" etc. anywhere
    # (not just after a specific food), strip that substance from ALL
    # foods. Handles patterns like "tudo sem glúten", "certificado como
    # sem glúten", recipe titles, etc.
    global_stripped: set[str] = set()
    for gm in _GLOBAL_SEM_SUBSTANCE_RE.finditer(text_norm):
        pt_sub = gm.group(1)
        sub_key = _SUBSTANCE_PT_TO_KEY.get(pt_sub)
        if sub_key:
            global_stripped.add(sub_key)

    # ── Fix 9: English / alternative "free-from" patterns ──────────
    for pat, sub_key in _FREE_FROM_PATTERNS:
        if pat.search(text_norm):
            global_stripped.add(sub_key)

    if global_stripped:
        updated: list[FoodMatch] = []
        for fm in final:
            new_subs = fm.substances - global_stripped
            if new_subs != fm.substances:
                fm = FoodMatch(
                    matched_norm=fm.matched_norm,
                    canonical=fm.canonical,
                    tbca_code=fm.tbca_code,
                    substances=new_subs,
                    start=fm.start,
                    end=fm.end,
                )
            updated.append(fm)
        final = updated

    return sorted(final, key=lambda f: f.start)


# ═══════════════════════════════════════════════════════════════════════════
# CONDITION RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════

def resolve_condition(condition: str) -> str:
    """Resolve a condition name (Portuguese or English) to its canonical key.

    Raises ValueError if the condition is not recognised.
    """
    # Strip parentheticals like "(DHGNA)" or "(Alergia ao Glúten)"
    clean = re.sub(r'\([^)]*\)', '', condition).strip()

    # Try direct match first
    key = clean.lower().replace(" ", "_").replace("-", "_")
    if key in CONDITION_RESTRICTS:
        return key

    # Try Portuguese name mapping (normalised)
    condition_norm = norm(clean)
    if condition_norm in CONDITION_PT_TO_KEY:
        return CONDITION_PT_TO_KEY[condition_norm]

    # Try partial match on Portuguese names
    for pt_name, en_key in CONDITION_PT_TO_KEY.items():
        if norm(pt_name) in condition_norm or condition_norm in norm(pt_name):
            return en_key

    raise ValueError(
        f"Unknown condition: {condition!r}. "
        f"Known keys: {sorted(CONDITION_RESTRICTS.keys())}"
    )


def get_restricted_substances(condition: str) -> list[str]:
    """Return the list of restricted substances for a condition."""
    key = resolve_condition(condition)
    return CONDITION_RESTRICTS[key]


# ═══════════════════════════════════════════════════════════════════════════
# RESTRICTED ITEM DERIVATION
# ═══════════════════════════════════════════════════════════════════════════

def filter_restricted_items(
    food_entities: list[FoodMatch],
    condition: str,
) -> list[RestrictedItem]:
    """Filter extracted food entities to those restricted for the condition.

    Implements the forward-chaining rule:
        condition_restricts(condition, substance)
          ∧ involves_substance(food, substance)
          → restricted_item(condition, food)

    A food is restricted if its substance set (from the lexicon) overlaps
    with the condition's restricted substance set.
    """
    key = resolve_condition(condition)
    restricted_subs = set(CONDITION_RESTRICTS[key])

    results: list[RestrictedItem] = []
    seen_foods: set[str] = set()

    for food in food_entities:
        # Skip duplicates (same canonical food mentioned multiple times)
        if food.canonical in seen_foods:
            continue

        # ── Fix 5: Zero-substance guard ─────────────────────────────
        # Skip foods with no substance flags (shouldn't be restricted).
        if not food.substances:
            continue

        overlap = food.substances & restricted_subs
        if overlap:
            seen_foods.add(food.canonical)
            results.append(RestrictedItem(
                food=food,
                condition=key,
                matching_substances=sorted(overlap),
            ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: Full extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════

def extract_restricted_items(
    text: str,
    condition: str,
    automaton: Any,
) -> tuple[list[FoodMatch], list[RestrictedItem]]:
    """Full Module 2 pipeline: extract food entities → derive restricted items.

    Parameters
    ──────────
    text       : LLM response text to analyse.
    condition  : Patient health condition (Portuguese or English).
    automaton  : Aho-Corasick automaton from lexicon_builder.

    Returns
    ───────
    (all_foods, restricted_items)
        all_foods        : every food mention found in the text.
        restricted_items : only those foods restricted for the condition.
    """
    all_foods = extract_food_entities(text, automaton)
    restricted = filter_restricted_items(all_foods, condition)

    logger.debug(
        "Condition=%s | foods=%d | restricted=%d",
        condition, len(all_foods), len(restricted),
    )
    return all_foods, restricted


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract food entities and restricted items from text.",
    )
    ap.add_argument("condition", help="Patient condition (e.g. diabetes)")
    ap.add_argument("--text", required=True, help="LLM response text to analyse")
    ap.add_argument(
        "--data-dir", default=None,
        help="Directory with lexicon artifacts (default: data/)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    from pathlib import Path
    from lexicon_builder import load_lexicon, DATA_DIR

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    _, _, automaton = load_lexicon(data_dir)

    all_foods, restricted = extract_restricted_items(
        args.text, args.condition, automaton,
    )

    print(f"\nCondition: {args.condition}")
    print(f"All food mentions ({len(all_foods)}):")
    for f in all_foods:
        print(f"  • {f.canonical:30s}  substances={set(f.substances)}")

    print(f"\nRestricted items ({len(restricted)}):")
    for r in restricted:
        print(
            f"  ✗ {r.food.canonical:30s}  "
            f"because: {r.matching_substances}"
        )

    if not restricted:
        print("  (none)")
