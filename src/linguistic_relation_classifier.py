# -*- coding: utf-8 -*-
"""
frame_detector.py — Layer 2: Syntactic Frame Detection
═══════════════════════════════════════════════════════════

Detects linguistic frames (negation, substitution, comparative,
descriptive) around food mentions using spaCy dependency parsing
combined with the verb lexicon from verb_classifier.py.

Instead of 300+ regex patterns matching exact word sequences,
this module matches **syntactic shapes**:

  • NEGATION frames:
      - verb[avoidance] → obj[food]         "Evite mel"
      - ADV[neg] → verb[action] → obj[food] "Não consuma leite"
      - ADP[sem] → case → noun[food]        "sem mel"
      - particle[sem] → bridge → de → food  "sem adição de açúcar"

  • SUBSTITUTION frames:
      - verb[subst] → obj[old] → obl[por → new]  "Substitua mel por stevia"
      - ADP[ao invés de] → food                   "Ao invés de mel, use X"

  • DESCRIPTIVE frames:
      - food → nmod → {sangue, glicemia}    "açúcar no sangue"
      - {nível, teor} → de → food           "nível de açúcar"

  • COMPARATIVE frames:
      - {melhor, pior} → que → food          "melhor que açúcar"

Key feature: the verb_lexicon handles the imperative-as-PROPN problem.
When spaCy tags "Evite" as PROPN, we check the verb lexicon and
correct the classification before frame matching.

Usage
─────
    from frame_detector import FrameDetector

    fd = FrameDetector.create()
    frames = fd.detect(text, food_positions)
    # → [Frame(type="negation", food="mel", ...),
    #    Frame(type="substitution", food="mel", role="replaced", ...)]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import spacy
from spacy.tokens import Doc, Token

from lexicon_builder import norm as _lex_norm
from lexicon_builder import plural_re as _plural_re
from verb_lexicon_builder import VerbClassifier

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FoodPosition:
    """A food entity mention with its character offsets in the text."""
    text: str           # matched text surface form
    norm: str           # normalised form (from entity_extractor)
    start: int          # char offset start
    end: int            # char offset end


@dataclass
class Frame:
    """A detected linguistic frame around a food mention."""
    type: str               # negation | substitution | comparative | descriptive
    food_norm: str          # normalised food name
    food_text: str          # surface form in text
    role: str = ""          # for substitution: "replaced" or "replacement"
    trigger: str = ""       # the word/phrase that triggered the frame
    trigger_lemma: str = "" # lemma of the trigger
    verb_category: str = "" # verb_classifier category if trigger is a verb
    detail: str = ""        # human-readable explanation

    def __repr__(self) -> str:
        parts = [f"Frame({self.type!r}, food={self.food_norm!r}"]
        if self.role:
            parts.append(f", role={self.role!r}")
        parts.append(f", trigger={self.trigger!r}")
        if self.detail:
            parts.append(f", {self.detail}")
        parts.append(")")
        return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# NEGATION PARTICLES AND MULTI-WORD MARKERS
# ═══════════════════════════════════════════════════════════════════════════

#: Particles that negate a verb when attached as advmod/dep child.
NEGATION_PARTICLES = frozenset({"não", "nunca", "jamais", "nem"})

#: "sem" as ADP — directly negates its noun complement
SEM_TOKEN = "sem"

#: Words that block "sem" from negating the food (property negation).
#: "sem casca" = property of the food, not absence of food.
SEM_BLOCKED_COMPLEMENTS = frozenset({
    "casca", "pele", "semente", "sementes", "caroço", "carocos",
    "osso", "ossos", "recheio", "miolo", "bagaço",
    # Substance labels — "queijo sem lactose" = queijo IS present
    "lactose", "glúten", "gluten", "fodmap", "caseína", "caseina",
})

#: Bridge nouns for "sem adição de X" pattern
SEM_BRIDGE_NOUNS = frozenset({
    "adição", "adicao", "presença", "presenca", "ausência", "ausencia",
    "uso", "aditivo", "aditivos", "necessidade",
})

#: Multi-word substitution markers (normalised)
SUBSTITUTION_MARKERS_RE = re.compile(
    r"\b(ao\s+inv[ée]s\s+de|em\s+vez\s+de|no\s+lugar\s+de|"
    r"em\s+substitui[çc][ãa]o\s+a[os]?|em\s+lugar\s+de)\b",
    re.IGNORECASE,
)

#: Multi-word negation markers that work like "sem"
#  Allow plural forms: livres de, isentos de, ausências de
MULTI_WORD_NEG_RE = re.compile(
    r"\b(livres?\s+de|isentos?\s+de|aus[êe]ncias?\s+de|"
    r"sem\s+adi[çc][ãa]o\s+de|nada\s+de|zero)\b",
    re.IGNORECASE,
)

#: Biomarker / descriptive context nouns
BIOMARKER_NOUNS = frozenset({
    "sangue", "glicemia", "glicêmico", "glicemico", "insulina",
    "nível", "nivel", "níveis", "niveis", "teor",
    "colesterol", "triglicerídeos", "triglicerideos",
    "pressão", "pressao",
})

#: Consumption / control / quantity nouns — when food is governed by these,
#: it's being discussed as a substance category or intake concern, not
#: recommended as an ingredient.  WordNet-derived:
#:   consumption.n.01 → consumo, ingestão;  control.n.01 → controle;
#:   quantity.n.01 → quantidade;  excess.n.01 → excesso;
#:   moderation.n.01 → moderação;  restriction.n.01 → restrição;
#:   reduction.n.01 → redução;  intake.n.01 → ingestão, uso
CONSUMPTION_CONTROL_NOUNS = frozenset({
    "consumo", "ingestao", "ingestão", "controle",
    "quantidade", "excesso",
    "restricao", "restrição", "restricoes", "restrições",
    "reducao", "redução", "uso", "limite", "limites",
    "necessidade", "taxa", "dose", "dosagem",
})

# ── Fix 5: Cross-contamination vocabulary (WordNet-derived) ────────────
# Source synsets and their Portuguese lemmas:
#   trace.n.01 → traço, vestígio;  trace.n.06 → rastro
#   remainder.n.01 → resíduo;  contamination.n.01 → contaminação
#   contact.n.01 → contato;  exposure/vulnerability → exposição
#   risk.n.02 / hazard.n.01 → risco
#   factory.n.01 → fábrica, usina;  facility.n.01 → instalação
#   share.v.02 → compartilhar (linhas compartilhadas)
#   handle.v.04 → manusear;  manipulate.v.02 → manipular, manejar
#   manufacture.v.01 → fabricar;  process.v.02 → processar
#   produce.v.02 → produzir;  manage.v.02 → lidar
#   derivative.n.02 / by-product.n.02 → derivado
#   label.n.04 → rótulo, etiqueta
# These mark food as a contamination RISK, not an ingredient.
# Cooking verbs (cozinhar, preparar, misturar) are EXCLUDED —
# "cozinha com amendoim" = food IS used = violation.

#: Trace / residue / vestige noun stems (WordNet: trace.n.01/06, remainder.n.01)
_TRACE_NOUN_STEMS = r"tracos?|rastros?|vestigios?|residuos?"

#: Contamination / exposure / risk noun stems
#  (WordNet: contamination.n.01, contact.n.01, vulnerability.n.01, hazard.n.01)
_CONTAM_NOUN_STEMS = r"contaminac\w+|contato|exposicao|risco"

#: Facility noun stems (WordNet: factory.n.01, facility.n.01)
#  + "linha compartilhada" (share.v.02 → compartilhar)
_FACILITY_NOUN_STEMS = (
    r"fabricas?|usinas?|instalac\w+|ambientes?"
    r"|linhas?\s+(?:de\s+producao|compartilhad\w*)"
)

#: Handling verb stems for facility/contamination context ONLY (WordNet:
#  handle.v.04 → manusear; manipulate.v.02 → manipular, manejar;
#  manufacture.v.01 → fabricar; process.v.02 → processar;
#  produce.v.02 → produzir; manage.v.02 → lidar).
#  NOT cooking verbs.
_HANDLING_VERB_STEMS = (
    r"manipul|manuse|manej"   # handle/manipulate
    r"|process|fabric"        # process/manufacture
    r"|produz|lid"            # produce / lidar (handle)
)

#: Derivative / byproduct nouns (WordNet: derivative.n.02, by-product.n.02)
_DERIVATIVE_STEMS = r"derivad\w*|subprodut\w*"

#: Label-reading context nouns (WordNet: label.n.04)
_LABEL_NOUN_STEMS = r"rotulos?|etiquetas?"
#: Comparative markers
COMPARATIVE_RE = re.compile(
    r"\b(melhor|pior|menos\s+saud[áa]vel|mais\s+perigoso|"
    r"mais\s+segur[oa]|menos\s+segur[oa]|"
    r"mais\s+adequad[oa]|menos\s+adequad[oa])\s+"
    r"(?:do\s+)?que\b",
    re.IGNORECASE,
)

#: Avoidance-header + colon pattern (Fix 10)
#  Matches "Gorduras a evitar:", "O que evitar:", "Alimentos proibidos:",
#  etc. — an avoidance verb/phrase before a colon.
#  Foods appearing AFTER the colon in the same sentence are in avoidance
#  context even though the dep tree doesn't connect them.
AVOIDANCE_HEADER_RE = re.compile(
    r"(?:"
    r"(?:alimentos?|gorduras?|bebidas?|prote[ií]nas?|pe[iy]xes?"
    r"|carnes?|fontes?|itens?|ingredientes?|o\s+que|comidas?"
    r"|opcoes|op[çc][oõ]es|categorias?|tipos?|grupo)"
    r"\s+(?:a\s+(?:serem?\s+)?|que\s+(?:devem?\s+(?:ser\s+)?)?)?"
    r")?"
    r"(?:evit\w+|proibi\w+|eliminad?\w*|restri\w+|exclu[ií]\w+"
    r"|nao\s+(?:recomendad|permitid|indicad|consumir)\w*"
    r"|banid\w+|vetad\w+)"
    r"\s*(?:[^:\n]{0,40}?)"   # up to 40 chars before colon
    r":\s*",
    re.IGNORECASE,
)

#: Passive negation warnings (Portuguese)
PASSIVE_NEGATION_RE = re.compile(
    r"\b(n[ãa]o\s+[ée]\s+recomendad[oa]|"
    r"n[ãa]o\s+s[ãa]o\s+recomendad[oa]s|"
    r"deve[m]?\s+ser\s+evitad[oa]s?|"
    r"[ée]\s+inadequad[oa]|"
    r"[ée]\s+prejudicial|s[ãa]o\s+prejudiciais|"
    r"[ée]\s+perigoso|s[ãa]o\s+perigosos|"
    r"[ée]\s+restrit[oa]|est[áa]\s+restrit[oa]|"
    r"est[ãa]o\s+restrit[oa]s|"
    r"contraindicad[oa]s?|desaconselhad[oa]s?|"
    r"proibid[oa]s?)\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════
# NORMALISER
# ═══════════════════════════════════════════════════════════════════════════

_ACCENT_MAP = str.maketrans(
    "áàãâéêíóôõúüçÁÀÃÂÉÊÍÓÔÕÚÜÇ",
    "aaaaeeiooouucAAAAEEIOOOUUC",
)

def _norm(text: str) -> str:
    """Strip accents and lowercase."""
    return text.lower().translate(_ACCENT_MAP)


# Portuguese deverbal noun suffixes — these words are derived from verbs
# but express states, results, or abstract qualities, NOT actions.
# "moderação" (from moderar), "monitoramento" (from monitorar),
# "recomendação" (from recomendar) — these should NOT be treated as verbs
# by the imperative-as-PROPN fallback.
_DEVERBAL_NOUN_SUFFIXES = (
    "cao", "coes",       # -ção, -ções  (moderação, recomendação)
    "mento", "mentos",   # -mento       (monitoramento, tratamento)
    "ancia", "ancias",   # -ância       (tolerância)
    "encia", "encias",   # -ência       (frequência)
    "agem", "agens",     # -agem        (dosagem)
    "ura", "uras",       # -ura         (mistura) — but short words excluded by min len
    "dade", "dades",     # -dade        (qualidade)
)

# Nouns that are deverbatives (verb stem as noun) expressing recommendation
# or selection — NOT avoidance or substitution.  When these words are tagged
# as NOUN by spaCy and the verb-lexicon fallback fires, they should be
# blocked from emitting safe-context frames.
# "escolha" = choice, "opcao" = option, "compra" = purchase
_RECOMMENDATION_NOUNS = frozenset({
    "escolha", "escolhas", "opcao", "opcoes", "compra", "compras",
    "preferencia", "preferencias",
})


def _is_deverbal_noun(norm_word: str) -> bool:
    """Return True if the word looks like a deverbal noun (not an action)."""
    if norm_word in _RECOMMENDATION_NOUNS:
        return True
    if len(norm_word) < 6:
        return False
    return norm_word.endswith(_DEVERBAL_NOUN_SUFFIXES)



# ═══════════════════════════════════════════════════════════════════════════
# FRAME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FrameDetector:
    """
    Detects syntactic frames around food mentions in text.

    Combines spaCy dependency parsing with the verb lexicon for
    robust frame detection that doesn't depend on hardcoded verb lists.
    """
    verb_classifier: VerbClassifier = field(default_factory=lambda: None)  # type: ignore
    nlp: Optional[object] = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        verb_lexicon_path: str = "data/verb_lexicon.pkl",
        spacy_model: str = "pt_core_news_md",
    ) -> FrameDetector:
        """Create a FrameDetector with loaded resources."""
        import spacy as _spacy
        vc = VerbClassifier.load(verb_lexicon_path)
        nlp = _spacy.load(spacy_model, disable=["ner"])
        return cls(verb_classifier=vc, nlp=nlp)

    # ── Main entry point ────────────────────────────────────────────────

    def detect(
        self,
        text: str,
        food_positions: list[FoodPosition],
        pre_parsed_doc=None,
    ) -> list[Frame]:
        """
        Detect all linguistic frames around food mentions.

        Parameters
        ──────────
        text            : The full text (LLM response or sentence).
        food_positions  : Food mentions with char offsets from entity_extractor.

        Returns
        ───────
        List of Frame objects, one per detected frame.
        A food may appear in multiple frames (e.g., negation + descriptive).
        """
        if not food_positions or not text.strip():
            return []

        doc = pre_parsed_doc if pre_parsed_doc is not None else self.nlp(text)

        # Map food positions to spaCy tokens
        food_token_map = self._map_foods_to_tokens(doc, food_positions)

        frames: list[Frame] = []

        # ── Dependency-based detection ──
        for fpos, ftokens in food_token_map.items():
            if not ftokens:
                continue
            frames.extend(self._detect_dep_frames(doc, fpos, ftokens))

        # ── Regex-based detection for multi-word patterns ──
        # (These patterns span multiple tokens and are harder to express
        #  as dependency patterns — e.g., "ao invés de X")
        frames.extend(self._detect_regex_frames(text, food_positions))

        # ── Fix 10: Avoidance-header + colon context ──
        # "Gorduras a evitar: manteiga, banha" — food in same sentence
        # as avoidance verb but after colon; dep tree doesn't connect them.
        frames.extend(
            self._check_avoidance_colon_context(text, food_positions, frames)
        )

        # ── List-propagation: avoidance header → list items ──
        # When a sentence has an avoidance verb and ends with ":",
        # the following sentences that are list items (no verb of their
        # own) inherit the negation.  Uses verb_classifier + spaCy
        # sentence segmentation — no raw regex on text.
        frames.extend(
            self._propagate_list_negation(doc, food_token_map)
        )

        # Deduplicate: same food + same frame type → keep first
        frames = self._deduplicate(frames)

        return frames

    # ── Token mapping ───────────────────────────────────────────────────

    def _map_foods_to_tokens(
        self, doc: Doc, food_positions: list[FoodPosition]
    ) -> dict[FoodPosition, list[Token]]:
        """Map each FoodPosition to spaCy tokens by char offset alignment."""
        result: dict[FoodPosition, list[Token]] = {}
        for fp in food_positions:
            tokens = []
            for tok in doc:
                # Token overlaps with food mention
                if tok.idx < fp.end and tok.idx + len(tok.text) > fp.start:
                    tokens.append(tok)
            result[fp] = tokens
        return result

    # ── Dependency-based frame detection ────────────────────────────────

    def _detect_dep_frames(
        self, doc: Doc, fpos: FoodPosition, ftokens: list[Token]
    ) -> list[Frame]:
        """Detect frames by walking the dependency tree around food tokens."""
        frames: list[Frame] = []

        for ftok in ftokens:
            # ── Check for "exceto" exception negation ──
            frames.extend(self._check_exceto_negation(ftok, fpos))

            # ── Check ancestors for verb frames ──
            frames.extend(self._check_verb_ancestors(ftok, fpos))

            # ── Check for "sem" (ADP) as case marker ──
            frames.extend(self._check_sem_negation(ftok, fpos, doc))

            # ── Check for passive negation warnings ──
            frames.extend(self._check_passive_negation(ftok, fpos, doc))

            # ── Check for biomarker/descriptive context ──
            frames.extend(self._check_descriptive_context(ftok, fpos))

        return frames

    # ── "exceto" exception negation ─────────────────────────────────────

    def _check_exceto_negation(
        self, ftok: Token, fpos: FoodPosition
    ) -> list[Frame]:
        """Detect the 'exceto FOOD' pattern — an exception list that
        explicitly singles out a food as forbidden.

        Dep structure: 'exceto' appears as ADP/SCONJ with dep=case or
        dep=mark on the food token.  Coordinated items ('exceto X e Y')
        are reached via dep=conj on the first food.
        """
        frames: list[Frame] = []

        # Direct: ftok has 'exceto' as a case/mark child
        has_exceto = any(
            _norm(ch.text) == "exceto" and ch.dep_ in ("case", "mark")
            for ch in ftok.children
        )

        # Conjunct: ftok is conjoined to another token that has 'exceto'
        if not has_exceto and ftok.dep_ == "conj":
            head = ftok.head
            has_exceto = any(
                _norm(ch.text) == "exceto" and ch.dep_ in ("case", "mark")
                for ch in head.children
            )

        if has_exceto:
            frames.append(Frame(
                type="negation",
                food_norm=fpos.norm,
                food_text=fpos.text,
                trigger="exceto",
                trigger_lemma="exceto",
                detail="'exceto' exception list",
            ))

        return frames

    def _get_verb_category(self, token: Token) -> Optional[str]:
        """
        Get the verb category for a token, handling the imperative-as-PROPN
        problem by checking the verb lexicon as fallback.
        """
        # First check spaCy's analysis
        lemma = token.lemma_.lower()
        text_lower = token.text.lower()

        # If spaCy says it's a VERB, use its lemma
        if token.pos_ == "VERB":
            cat = self.verb_classifier.classify(lemma)
            if cat:
                return cat
            # Try the surface form too (spaCy may not lemmatise correctly)
            return self.verb_classifier.classify(text_lower)

        # If spaCy says PROPN/NOUN but our lexicon knows this word as a verb
        # → imperative fix (e.g. "Evite" tagged PROPN)
        # GUARD: reject deverbal nouns — suffixes like -ção, -mento, -ância,
        # -ência, -agem express states/results, not actions.
        # "moderação", "monitoramento" are NOT avoidance verbs.
        if token.pos_ in ("PROPN", "NOUN"):
            norm_text = _norm(text_lower)
            if _is_deverbal_noun(norm_text) or _is_deverbal_noun(_norm(lemma)):
                return None
            cat = self.verb_classifier.classify(text_lower)
            if cat:
                return cat
            cat = self.verb_classifier.classify(lemma)
            if cat:
                return cat

        return None

    def _check_verb_ancestors(
        self, ftok: Token, fpos: FoodPosition
    ) -> list[Frame]:
        """Walk up the dependency tree to find governing verbs."""
        frames: list[Frame] = []
        max_depth = 4

        # Check the food token's head chain
        current = ftok
        for depth in range(max_depth):
            head = current.head
            if head == current:
                break  # root

            cat = self._get_verb_category(head)
            if cat:
                frame = self._make_verb_frame(head, cat, ftok, fpos, depth)
                if frame:
                    frames.append(frame)
                    # Also check if this verb is negated
                    if self._is_negated_verb(head):
                        neg_frame = self._make_negated_verb_frame(
                            head, cat, ftok, fpos
                        )
                        if neg_frame:
                            frames.append(neg_frame)
                    break  # found governing verb that produced a frame
                # action/recommendation verbs produce no frame on their own;
                # keep climbing — an avoidance verb may govern them
                # (e.g. "Evite adicionar açúcar, mel")
                if self._is_negated_verb(head):
                    neg_frame = self._make_negated_verb_frame(
                        head, cat, ftok, fpos
                    )
                    if neg_frame:
                        frames.append(neg_frame)
                        break

            current = head

        # Also check: is the ROOT of the sentence a verb that governs this food?
        # (handles flat structures where food is deep in the tree)
        sent = ftok.sent if ftok.sent else ftok.doc[:]
        root = None
        for t in sent:
            if t.dep_ == "ROOT":
                root = t
                break
        if root and root != ftok:
            cat = self._get_verb_category(root)
            if cat and not any(f.trigger_lemma == root.lemma_ for f in frames):
                # Check if food is in the root's subtree
                if ftok in root.subtree:
                    frame = self._make_verb_frame(root, cat, ftok, fpos, -1)
                    if frame:
                        frames.append(frame)
                    if self._is_negated_verb(root):
                        neg_frame = self._make_negated_verb_frame(
                            root, cat, ftok, fpos
                        )
                        if neg_frame:
                            frames.append(neg_frame)

        return frames

    def _is_negated_verb(self, verb_tok: Token) -> bool:
        """Check if a verb has a negation particle as a child."""
        for child in verb_tok.children:
            if (child.dep_ in ("advmod", "neg", "dep")
                    and _norm(child.text) in {"nao", "nunca", "jamais", "nem"}):
                return True
        return False

    def _make_verb_frame(
        self, verb: Token, category: str, ftok: Token, fpos: FoodPosition,
        depth: int,
    ) -> Optional[Frame]:
        """Create a frame from a governing verb."""
        if category == "avoidance":
            return Frame(
                type="negation",
                food_norm=fpos.norm,
                food_text=fpos.text,
                trigger=verb.text,
                trigger_lemma=verb.lemma_.lower(),
                verb_category=category,
                detail=f"avoidance verb '{verb.lemma_}' governs food (depth={depth})",
            )
        elif category == "substitution":
            # Determine role: is this food the obj (replaced) or obl (replacement)?
            role = self._get_substitution_role(verb, ftok)
            return Frame(
                type="substitution",
                food_norm=fpos.norm,
                food_text=fpos.text,
                role=role,
                trigger=verb.text,
                trigger_lemma=verb.lemma_.lower(),
                verb_category=category,
                detail=f"substitution verb '{verb.lemma_}', food role={role}",
            )
        elif category == "action":
            # Action verb governing food = recommendation context (potential violation)
            # We don't emit a "safe" frame here — this is the default blocked case.
            # The inference engine will derive violation from absence of safe frames.
            return None
        elif category == "recommendation":
            # Same as action — recommendation of food = potential violation
            return None
        elif category == "moderation":
            # Moderation verbs (limitar, reduzir, diminuir, moderar, controlar)
            # imply reduced consumption, NOT elimination.  The food IS still
            # being consumed → no safe frame → violation stands.
            return None

        return None

    def _make_negated_verb_frame(
        self, verb: Token, original_category: str, ftok: Token, fpos: FoodPosition,
    ) -> Optional[Frame]:
        """Create a negation frame when an action/recommendation verb is negated."""
        if original_category in ("action", "recommendation"):
            # "não consuma leite" → negation
            return Frame(
                type="negation",
                food_norm=fpos.norm,
                food_text=fpos.text,
                trigger=verb.text,
                trigger_lemma=verb.lemma_.lower(),
                verb_category=original_category,
                detail=f"negated {original_category} verb '{verb.lemma_}' → food safe",
            )
        return None

    def _get_substitution_role(self, verb: Token, ftok: Token) -> str:
        """Determine if the food is being replaced or is the replacement."""
        # In "substitua X por Y": X = obj (replaced), Y = obl (replacement)
        # Check if food is direct object
        if ftok.dep_ in ("obj", "dobj"):
            return "replaced"
        # Check if food is in a "por" prepositional phrase
        if ftok.dep_ in ("obl", "nmod"):
            for child in ftok.children:
                if child.dep_ == "case" and _norm(child.text) == "por":
                    return "replacement"
            return "replacement"  # default for oblique
        # Walk up to check
        current = ftok
        while current.head != current:
            if current.dep_ in ("obj", "dobj"):
                return "replaced"
            if current.dep_ in ("obl", "nmod"):
                return "replacement"
            current = current.head
        return "replaced"  # default

    def _check_sem_negation(
        self, ftok: Token, fpos: FoodPosition, doc: Doc
    ) -> list[Frame]:
        """Detect 'sem X' pattern — ADP negating a food noun."""
        frames: list[Frame] = []

        # ── Moderation guard ───────────────────────────────────────────
        # "sem uso excessivo de X", "sem excesso de X", "sem exagero de X"
        # These mean moderation, NOT elimination → food is still consumed.
        _MODERATION_BRIDGE = {"excessivo", "excessiva", "excesso", "exagero",
                              "abuso", "exageracao", "exagerar"}
        for tok in doc:
            if (_norm(tok.text) == "sem" and tok.pos_ == "ADP"
                    and tok.i < ftok.i and ftok.i - tok.i <= 12):
                # Check tokens between "sem" and food for moderation language
                between = [_norm(t.text) for t in doc[tok.i + 1: ftok.i]]
                if any(b in _MODERATION_BRIDGE for b in between):
                    return frames  # empty — no negation frame

        # Pattern 1: "sem" is a direct child of food with dep=case
        for child in ftok.children:
            if child.dep_ == "case" and _norm(child.text) == "sem":
                # Check it's not property negation ("sem casca")
                if not self._is_property_negation(ftok, fpos):
                    frames.append(Frame(
                        type="negation",
                        food_norm=fpos.norm,
                        food_text=fpos.text,
                        trigger="sem",
                        trigger_lemma="sem",
                        detail="'sem' + food (direct)",
                    ))

        # Pattern 2: "sem" is nearby (within 8 tokens before food)
        #            and the food is in its syntactic scope
        for tok in doc:
            if (_norm(tok.text) == "sem" and tok.pos_ == "ADP"
                    and tok.i < ftok.i and ftok.i - tok.i <= 8):
                # Check if already found via dep pattern
                if any(f.trigger == "sem" for f in frames):
                    continue
                # Check scope: is food reachable from sem's head?
                sem_head = tok.head
                if ftok in sem_head.subtree:
                    if not self._is_property_negation(ftok, fpos):
                        frames.append(Frame(
                            type="negation",
                            food_norm=fpos.norm,
                            food_text=fpos.text,
                            trigger="sem",
                            trigger_lemma="sem",
                            detail="'sem' + food (scope)",
                        ))

        # Pattern 3: "sem adição de X" — bridge pattern
        for tok in doc:
            if (_norm(tok.text) == "sem" and tok.pos_ == "ADP"
                    and tok.i < ftok.i and ftok.i - tok.i <= 10):
                # Look for bridge noun between sem and food
                bridge_found = False
                for mid in doc[tok.i + 1: ftok.i]:
                    if _norm(mid.text) in {_norm(b) for b in SEM_BRIDGE_NOUNS}:
                        bridge_found = True
                        break
                if bridge_found:
                    if not any(f.detail.startswith("'sem'") for f in frames):
                        frames.append(Frame(
                            type="negation",
                            food_norm=fpos.norm,
                            food_text=fpos.text,
                            trigger="sem",
                            trigger_lemma="sem",
                            detail=f"'sem' + bridge + food",
                        ))

        return frames

    def _is_property_negation(self, ftok: Token, fpos: FoodPosition) -> bool:
        """
        Check if 'sem' negates a property of the food, not the food itself.
        E.g., "tomate sem sementes" — sem applies to sementes, not tomate.
        """
        # Check if the food text matches a blocked complement
        if _norm(fpos.text) in {_norm(w) for w in SEM_BLOCKED_COMPLEMENTS}:
            return True
        if _norm(ftok.text) in {_norm(w) for w in SEM_BLOCKED_COMPLEMENTS}:
            return True
        return False

    def _check_passive_negation(
        self, ftok: Token, fpos: FoodPosition, doc: Doc
    ) -> list[Frame]:
        """Detect passive negation patterns: 'deve ser evitado', 'é inadequado'."""
        frames: list[Frame] = []
        sent_text = ftok.sent.text if ftok.sent else str(doc)

        if PASSIVE_NEGATION_RE.search(sent_text):
            # Check proximity: passive negation marker within ±6 tokens of food
            for match in PASSIVE_NEGATION_RE.finditer(sent_text):
                # Find token positions of the match
                match_start = match.start() + (ftok.sent.start_char if ftok.sent else 0)
                food_start = fpos.start
                # Simple proximity check
                if abs(match_start - food_start) < 200:
                    frames.append(Frame(
                        type="negation",
                        food_norm=fpos.norm,
                        food_text=fpos.text,
                        trigger=match.group(),
                        trigger_lemma=match.group().lower(),
                        detail=f"passive negation: '{match.group()}'",
                    ))
                    break  # one passive negation per food per sentence

        return frames

    def _check_descriptive_context(
        self, ftok: Token, fpos: FoodPosition
    ) -> list[Frame]:
        """Detect biomarker/descriptive contexts: 'açúcar no sangue', 'nível de açúcar'."""
        frames: list[Frame] = []

        # Pattern 1: food → nmod → biomarker noun
        #   "açúcar no sangue" = açúcar has nmod child "sangue"
        for child in ftok.children:
            if child.dep_ in ("nmod", "appos") and _norm(child.text) in {_norm(b) for b in BIOMARKER_NOUNS}:
                frames.append(Frame(
                    type="descriptive",
                    food_norm=fpos.norm,
                    food_text=fpos.text,
                    trigger=child.text,
                    trigger_lemma=child.lemma_.lower(),
                    detail=f"food → nmod → biomarker '{child.text}'",
                ))

        # Pattern 2: biomarker → de → food
        #   "nível de açúcar" = nível has nmod child "açúcar"
        head = ftok.head
        if head != ftok and _norm(head.text) in {_norm(b) for b in BIOMARKER_NOUNS}:
            frames.append(Frame(
                type="descriptive",
                food_norm=fpos.norm,
                food_text=fpos.text,
                trigger=head.text,
                trigger_lemma=head.lemma_.lower(),
                detail=f"biomarker '{head.text}' → de → food",
            ))

        # Pattern 2b: consumption/control noun → de → food
        #   "consumo de açúcar", "excesso de sal", "restrição de mel"
        #   Structural: food is governed by an intake/control noun → descriptive.
        #   Walks up to 2 hops (to handle "controle do consumo de açúcar").
        #
        #   GUARD: If the control noun's own head is an action/substitution
        #   verb, the food is actually being USED ("adoçada com uma pequena
        #   quantidade de mel", "Ajustar a quantidade de mel").
        #   In that case, do NOT mark as descriptive.
        _cc_norms = {_norm(n) for n in CONSUMPTION_CONTROL_NOUNS}
        current = ftok
        for _depth in range(3):
            ancestor = current.head
            if ancestor == current:
                break
            if _norm(ancestor.text) in _cc_norms:
                # Check: is this control noun governed by an ACTIVE action verb?
                # "adoçada com uma quantidade de mel" → action verb → NOT descriptive
                # "desde que seja feito com moderação" → passive/conditional → descriptive OK
                ctrl_head = ancestor.head
                action_governed = False
                if ctrl_head != ancestor:
                    cat = self._get_verb_category(ctrl_head)
                    if cat in ("action", "substitution"):
                        # Passive voice check: if the verb has aux:pass child
                        # ("seja feito", "foi preparado"), it's not an active
                        # instruction — let the descriptive frame through.
                        is_passive = any(
                            c.dep_ == "aux:pass" for c in ctrl_head.children
                        )
                        if not is_passive:
                            action_governed = True
                if not action_governed:
                    frames.append(Frame(
                        type="descriptive",
                        food_norm=fpos.norm,
                        food_text=fpos.text,
                        trigger=ancestor.text,
                        trigger_lemma=ancestor.lemma_.lower(),
                        detail=f"control/consumption noun '{ancestor.text}' governs food (depth={_depth})",
                    ))
                break
            current = ancestor

        # Pattern 3: food + "naturais/natural" as substance-class descriptor
        # ONLY fires when the food surface form is PLURAL ("açúcares naturais",
        # "gorduras naturais") — indicates a substance category, not a product.
        # "iogurte natural" is a product name/variant → NOT descriptive.
        for child in ftok.children:
            if child.dep_ in ("amod", "appos") and _norm(child.text) in {"natural", "naturais", "naturales"}:
                surface = _norm(fpos.text)
                is_plural = surface.endswith(("es", "is", "s")) and len(surface) > len(fpos.norm)
                if is_plural:
                    frames.append(Frame(
                        type="descriptive",
                        food_norm=fpos.norm,
                        food_text=fpos.text,
                        trigger=child.text,
                        trigger_lemma=child.lemma_.lower(),
                        detail=f"food + natural descriptor",
                    ))

        return frames

    # ── Regex-based detection ───────────────────────────────────────────

    def _detect_regex_frames(
        self, text: str, food_positions: list[FoodPosition]
    ) -> list[Frame]:
        """
        Detect multi-word patterns that are hard to express as dep patterns.
        These are structural patterns, not food-specific.

        IMPORTANT: ``text_n`` uses ``lexicon_builder.norm`` — the **same**
        normaliser that entity_extractor uses — so ``fp.start / fp.end``
        indices are valid offsets into ``text_n``.
        """
        frames: list[Frame] = []
        text_n = _lex_norm(text)          # same norm as entity_extractor

        for fp in food_positions:
            # ── Substitution markers: "ao invés de X" ──
            for match in SUBSTITUTION_MARKERS_RE.finditer(text_n):
                marker_end = match.end()
                # Check if food appears within 60 chars after the marker
                if 0 <= fp.start - marker_end <= 60:
                    frames.append(Frame(
                        type="substitution",
                        food_norm=fp.norm,
                        food_text=fp.text,
                        role="replaced",
                        trigger=match.group(),
                        trigger_lemma=match.group().lower(),
                        detail=f"substitution marker '{match.group()}' before food",
                    ))

            # ── Multi-word negation: "livre de X", "ausência de X" ──
            for match in MULTI_WORD_NEG_RE.finditer(text_n):
                marker_end = match.end()
                if 0 <= fp.start - marker_end <= 40:
                    frames.append(Frame(
                        type="negation",
                        food_norm=fp.norm,
                        food_text=fp.text,
                        trigger=match.group(),
                        trigger_lemma=match.group().lower(),
                        detail=f"multi-word negation '{match.group()}'",
                    ))

            # ── Comparative: "melhor que X" ──
            for match in COMPARATIVE_RE.finditer(text_n):
                marker_end = match.end()
                if 0 <= fp.start - marker_end <= 30:
                    frames.append(Frame(
                        type="comparative",
                        food_norm=fp.norm,
                        food_text=fp.text,
                        trigger=match.group(),
                        trigger_lemma=match.group().lower(),
                        detail=f"comparative '{match.group()}'",
                    ))

            # ── Conditional negation: "desde que não" ... food ──
            cond_neg_re = re.compile(
                r"\b(desde\s+que\s+n[ãa]o|contanto\s+que\s+n[ãa]o|"
                r"a\s+menos\s+que\s+n[ãa]o)\b",
                re.IGNORECASE,
            )
            for match in cond_neg_re.finditer(text_n):
                marker_start = match.start()
                if 0 <= fp.start - marker_start <= 120:
                    frames.append(Frame(
                        type="negation",
                        food_norm=fp.norm,
                        food_text=fp.text,
                        trigger=match.group(),
                        trigger_lemma=match.group().lower(),
                        detail=f"conditional negation '{match.group()}'",
                    ))

            # ── "exceto X" ──
            exceto_re = re.compile(
                r"\b(exceto|com\s+exce[çc][ãa]o\s+de)\b",
                re.IGNORECASE,
            )
            for match in exceto_re.finditer(text_n):
                marker_end = match.end()
                if 0 <= fp.start - marker_end <= 60:
                    frames.append(Frame(
                        type="negation",
                        food_norm=fp.norm,
                        food_text=fp.text,
                        trigger=match.group(),
                        trigger_lemma=match.group().lower(),
                        detail=f"exception marker '{match.group()}'",
                    ))

            # ── Condition-description: "alergia ao X" ──────────────────
            # Position-INDEPENDENT: search the full normalised text for
            # "medical-term + prep + food" anywhere.  This handles cases
            # where the food's first occurrence (taken by entity_extractor
            # dedup) is far from the condition phrase.
            # No hardcoded food names — the automaton supplies them.
            fp_re = _plural_re(_norm(fp.text))
            # Allow optional plural suffix so singular "ovo" matches
            # "ovos" or "amendoins" in the condition phrase.
            cond_food_re = re.compile(
                r"\b(?:"
                r"alergia|alergic[oa]s?"
                r"|intolerancia"
                r"|sensibilidade"
                r"|hipersensibilidade"
                r")\s+ao?s?\s+"
                + fp_re + r"\b",
                re.IGNORECASE,
            )
            if cond_food_re.search(text_n):
                frames.append(Frame(
                    type="descriptive",
                    food_norm=fp.norm,
                    food_text=fp.text,
                    trigger="condition_description",
                    trigger_lemma="condition_description",
                    detail=f"condition-description context for '{fp.norm}'",
                ))

            # ── Fix 3: Category/collective mentions ────────────────────
            # When the surface text is a *plural* form followed by a
            # collective qualifier, this is a substance-class warning
            # ("açúcares simples", "gorduras saturadas"), not a food.
            # Structural: plural suffix + qualifier adjective.
            # text_n already uses _lex_norm (same as entity_extractor)
            # so fp.start / fp.end are valid offsets.
            surface_at = text_n[fp.start:fp.end]
            after_food = text_n[fp.end:fp.end + 25].strip()
            is_plural_surface = (
                len(surface_at) > len(fp.norm)
                and surface_at.endswith(("es", "is", "s"))
            )
            _COLLECTIVE_QUALIFIERS_RE = re.compile(
                r"^(?:simples|refinad[oa]s?|adicionad[oa]s?"
                r"|natural|naturais|totais"
                r"|minerais|saturad[oa]s?|trans"
                r"|insoluveis|soluveis)\b"
            )
            if is_plural_surface and _COLLECTIVE_QUALIFIERS_RE.match(after_food):
                frames.append(Frame(
                    type="descriptive",
                    food_norm=fp.norm,
                    food_text=fp.text,
                    trigger="category_collective",
                    trigger_lemma="category_collective",
                    detail=f"plural collective: '{surface_at} {after_food.split()[0]}'",
                ))

            # ── Fix 4: Biomarker / clinical context ────────────────────
            # Position-independent patterns that indicate the food is
            # mentioned in a clinical/physiological context, not as an
            # ingredient.  Structural shapes: clinical-noun + prep + food.
            # No hardcoded food names — uses food_escaped from automaton.
            # fp_re (with PT plural morphology) set above via _plural_re.
            biomarker_patterns = [
                # "açúcar no sangue", "sal no sangue"
                rf"\b{fp_re}\s+no\s+sangue\b",
                # "nível/níveis/teor de açúcar"
                rf"\b(?:nivel|niveis|teor)\s+de\s+{fp_re}\b",
                # "controle/monitorar/regular do/da/de açúcar"
                rf"\b(?:control\w*|monitor\w*|regul\w*)\s+(?:do|da|de|dos|das)\s+(?:consumo\s+de\s+)?{fp_re}\b",
                # "consumo/ingestão de açúcar" near restriction verbs
                rf"\b(?:consumo|ingestao|uso)\s+(?:de|do|da)\s+{fp_re}\b",
                # REMOVED: "reduzir/restringir/limitar o X" pattern.
                # Moderation verbs imply the food IS being consumed in
                # reduced quantity — should be a violation, not descriptive.
                # "X sanguíneo", "regulação do X sanguíneo"
                rf"\b{fp_re}\s+sanguine\w*\b",
            ]
            if any(re.search(p, text_n) for p in biomarker_patterns):
                # Don't override if an explicit elimination is present
                # ("sem adição de açúcar para manter a glicemia" → negation wins)
                if not any(f.type == "negation" and f.food_norm == fp.norm for f in frames):
                    frames.append(Frame(
                        type="descriptive",
                        food_norm=fp.norm,
                        food_text=fp.text,
                        trigger="biomarker_clinical",
                        trigger_lemma="biomarker_clinical",
                        detail=f"biomarker/clinical context for '{fp.norm}'",
                    ))

            # ── Fix 5: Cross-contamination / trace context ─────────────
            # WordNet-derived vocabulary (see constants above).
            # Food is mentioned as a contamination RISK, not an ingredient.
            # Cooking/mixing verbs are excluded — they indicate actual use.
            #
            # Patterns derived from melhores_extratores_logica.py:
            #   _base_patterns, is_cross_contamination_context,
            #   _amendoim_extra_patterns, is_peanut_processing_context,
            #   is_eggs_absence_context — all rewritten structurally.
            _TS = _TRACE_NOUN_STEMS
            _CS = _CONTAM_NOUN_STEMS
            _FS = _FACILITY_NOUN_STEMS
            _HS = _HANDLING_VERB_STEMS
            _DS = _DERIVATIVE_STEMS
            cross_contam_patterns = [
                # A: trace/residue + de + food
                #    "traços de amendoim", "vestígios de ovo"
                rf"\b(?:{_TS})\s+de\s+{fp_re}\b",
                # B: contamination/exposure/risk noun + (cruzada)? + prep + food
                #    "contaminação com amendoins", "risco de exposição a amendoim"
                rf"\b(?:{_CS})\s+(?:cruzad\w*\s+)?(?:com|por|de|a)\s+{fp_re}\b",
                # C: facility + que + (não)? + handling-verb + food
                #    "fábrica que não processa amendoins"
                rf"\b(?:{_FS})\s+que\s+(?:nao\s+)?(?:{_HS})\w*"
                rf"\s+(?:(?:produt\w*|aliment\w*)\s+(?:com\s+|contendo\s+)?)?{fp_re}\b",
                # D: sem + trace/contam/risk + (de)? + food
                #    "sem traços de amendoim", "sem contaminação de ovo"
                rf"\bsem\s+(?:(?:o\s+)?risco\s+de\s+)?(?:{_TS}|{_CS})\s+(?:de\s+)?{fp_re}\b",
                # E: não + handling-verb + (products)? + (com)? + food
                #    "não manipula produtos com amendoim"
                rf"\bnao\s+(?:{_HS})\w*\s+(?:(?:produt\w*|aliment\w*)\s+)?(?:com\s+|contendo\s+)?{fp_re}\b",
                # F: derivative/byproduct + de + food (in negation context)
                #    "sem derivados de amendoim", "não derivados de ovo"
                rf"\b(?:sem|nao)\s+(?:{_DS})\s+(?:de|do|da|dos|das)\s+{fp_re}\b",
                # G: shared production lines + food
                #    "linhas compartilhadas ... amendoim"
                rf"\blinhas?\s+compartilhad\w*\b[^.!?]{{0,80}}\b{fp_re}\b",
                # H: contato/exposicao + com/a + food
                #    "contato com amendoim", "exposição a amendoim"
                rf"\b(?:contato|exposicao)\s+(?:com|a)\s+{fp_re}\b",
                # I: label-reading + trace context
                #    "rotulo ... traços ... amendoim"
                rf"\b(?:{_LABEL_NOUN_STEMS})\b[^.!?]{{0,100}}\b(?:{_TS})\b[^.!?]{{0,50}}\b{fp_re}\b",
                # J: "pode conter" + food (allergen labelling)
                rf"\bpode[m]?\s+conter\s+(?:(?:{_TS})\s+de\s+)?{fp_re}\b",
            ]
            if any(re.search(p, text_n) for p in cross_contam_patterns):
                frames.append(Frame(
                    type="descriptive",
                    food_norm=fp.norm,
                    food_text=fp.text,
                    trigger="cross_contamination",
                    trigger_lemma="cross_contamination",
                    detail=f"cross-contamination/trace context for '{fp.norm}'",
                ))

            # ── Fix 6: Direct-negation regex patterns (position-independent)
            # Inspired by _DIRECT_NEGATION_PATTERNS from melhores_extratores_logica.
            # These catch cases where dep-based detection fails due to offset
            # misalignment (spaCy uses original text offsets, but food positions
            # are in normalised text space from entity_extractor).
            # All patterns use fp_re (PT plural morphology) — no hardcoded foods.
            _DS = _DERIVATIVE_STEMS
            direct_neg_patterns = [
                # ─ A: "sem food" (direct) ─────────────────────────────
                # "sem amendoim", "sem sal", "sem açúcar"
                # Skip if already caught by dep-based sem detection above.
                rf"\bsem\s+(?:o\s+|a\s+|os\s+|as\s+)?{fp_re}\b",
                # ─ B: "não contém/contenham/conter/contenha food" ──────
                # "não contém amendoim", "que não contenham amendoins"
                rf"\bnao\s+(?:contem|contenham|conter|contenha|possui|possuem"
                rf"|inclui|incluir|leva|levar|tem|tenha|ha)\s+"
                rf"(?:o\s+|a\s+|os\s+|as\s+)?{fp_re}\b",
                # ─ C: "não adicione/adicionar food" ───────────────────
                rf"\bnao\s+(?:adicione|adicionar|use|usar|utilize|utilizar"
                rf"|consuma|consumir)\b[^.!?]{{0,40}}\b{fp_re}\b",
                # ─ D: "isento/isenção de food" ────────────────────────
                rf"\bisent\w*\s+de\s+{fp_re}\b",
                rf"\bisencao\s+de\s+{fp_re}\b",
                # ─ E: "ausência de food" ──────────────────────────────
                rf"\bausencia\s+de\s+{fp_re}\b",
                rf"\bfalta\s+de\s+{fp_re}\b",
                # ─ F: "nada de food" / "zero food" ────────────────────
                rf"\bnada\s+de\s+{fp_re}\b",
                rf"\bzero\s+{fp_re}\b",
                # ─ G: "sem food ou (seus) derivados" ──────────────────
                rf"\bsem\s+(?:o\s+|a\s+)?{fp_re}\s+ou\s+(?:seus?\s+)?(?:{_DS})\b",
                # ─ H: "sem derivados de food" ────────────────────────
                rf"\bsem\s+(?:{_DS})\s+(?:de|do|da|dos|das)\s+{fp_re}\b",
                # ─ I: "não contém food ou derivados" ─────────────────
                rf"\bnao\s+(?:contem|contenham|conter|contenha)\s+"
                rf"(?:o\s+|a\s+|os\s+|as\s+)?{fp_re}\s+ou\s+(?:seus?\s+)?(?:{_DS})\b",
                # ─ J: List-extended negation ──────────────────────────
                # "livre de X ou food", "não contém X nem food"
                rf"\blivr\w*\s+de\s+(?:\w+\s+(?:e|ou|nem)\s+)+{fp_re}\b",
                rf"\bnao\s+(?:contem|contenham|conter)\s+"
                rf"(?:\w+\s+(?:e|ou|nem)\s+)+{fp_re}\b",
                rf"\bisent\w*\s+de\s+(?:\w+\s+(?:e|ou|nem)\s+)+{fp_re}\b",
                # ─ K: "verificar/garantir/certificar que não contém food"
                # "ler rótulos para verificar se contém food"
                rf"\b(?:verific\w+|certific\w+|garantir|checar|conferir)"
                rf"\b[^.!?]{{0,60}}\bnao\s+(?:contem|contenham|contenha)"
                rf"\s+(?:o\s+|a\s+)?{fp_re}\b",
                rf"\b(?:verific\w+|certific\w+|garantir)\b[^.!?]{{0,60}}"
                rf"\b(?:contem|contenham|contenha|contendo)\s+"
                rf"(?:o\s+|a\s+)?{fp_re}\b",
                # ─ L: "evitar food" / "ao evitar food" ────────────────
                rf"\bevit\w*\s+(?:o\s+|a\s+|os\s+|as\s+)?(?:uso|consumo|ingestao)?"
                rf"\s*(?:de\s+|do\s+|da\s+)?{fp_re}\b",
                # ─ M: "food não é recomendado/está restrito" ──────────
                rf"\b{fp_re}\b[^.!?]{{0,40}}\bnao\s+(?:e|sao)\s+recomendad\w*\b",
                rf"\b{fp_re}\b[^.!?]{{0,40}}\bproibid\w*\b",
                rf"\b{fp_re}\b[^.!?]{{0,40}}\bcontraindicad\w*\b",
                # ─ N: "exceto food" (position-independent) ────────────
                # Broadens the offset-limited exceto_re above.
                rf"\bexceto\s+(?:o\s+|a\s+|os\s+|as\s+)?{fp_re}\b",
                rf"\bcom\s+excecao\s+de\s+{fp_re}\b",
            ]
            if any(re.search(p, text_n) for p in direct_neg_patterns):
                # Only add if no negation frame was already found for this food
                if not any(f.type == "negation" and f.food_norm == fp.norm
                           for f in frames):
                    frames.append(Frame(
                        type="negation",
                        food_norm=fp.norm,
                        food_text=fp.text,
                        trigger="direct_negation_regex",
                        trigger_lemma="direct_negation_regex",
                        detail=f"direct negation pattern for '{fp.norm}'",
                    ))

            # ── Fix 10c: Avoidance-context regex patterns ─────────────
            # Run on _norm(text) (preserves punctuation) so that
            # [^.!?\n] correctly stops at sentence boundaries.
            # These patterns are broader and need sentence boundary guards.
            if not any(f.type == "negation" and f.food_norm == fp.norm
                       for f in frames):
                text_punc = _norm(text)
                avoidance_context_patterns = [
                    # O: "evitar NOUN ... como ... food" (examples list)
                    # "evitar peixes altos em purinas, como sardinha, arenque"
                    rf"\bevit\w+\b[^.!?\n]{{0,80}}\bcomo\b[^.!?\n]{{0,80}}\b{fp_re}\b",
                    # P: "proibid/restrit/vetad + ... food" (same sentence)
                    # "proibidas! presentes em margarinas sólidas"
                    # Allow "!" within match (emphasis, not sentence end)
                    rf"\b(?:proibid|restrit|vetad|banid)\w*\b[^.\n]{{0,120}}\b{fp_re}\b",
                    # Q: "evitar + ... + food" (within same sentence, ≤80 chars)
                    # "evitar gorduras trans em margarinas"
                    rf"\bevit\w+\b[^.!?\n]{{0,80}}\b{fp_re}\b",
                ]
                if any(re.search(p, text_punc) for p in avoidance_context_patterns):
                    frames.append(Frame(
                        type="negation",
                        food_norm=fp.norm,
                        food_text=fp.text,
                        trigger="avoidance_context_regex",
                        trigger_lemma="avoidance_context_regex",
                        detail=f"avoidance context pattern for '{fp.norm}'",
                    ))

        return frames

    # ── Fix 10: Avoidance-header + colon context ────────────────────────

    def _check_avoidance_colon_context(
        self,
        text: str,
        food_positions: list[FoodPosition],
        existing_frames: list[Frame],
    ) -> list[Frame]:
        """Detect foods in avoidance lists after a colon (intra-sentence).

        Catches patterns like:
          "Gorduras a evitar: manteiga, banha, óleo de coco"
          "O que evitar: cerveja, vinho, cachaça"
          "Alimentos proibidos: ..."

        The avoidance verb/noun phrase appears before ":" and food items
        appear in the comma-separated list after the colon.
        Standard dep-based detection misses this because the verb is an
        acl modifier of a noun, not an ancestor of the food items.

        Uses _norm (accent-strip + lowercase) which preserves punctuation,
        NOT _lex_norm which strips colons/commas.
        """
        frames: list[Frame] = []
        # Use _norm (preserves punctuation) — NOT _lex_norm (strips it)
        text_n = _norm(text)

        # Already-negated foods — don't duplicate
        negated_norms = {
            f.food_norm for f in existing_frames if f.type == "negation"
        }

        # Find all avoidance-header positions in the text
        for match in AVOIDANCE_HEADER_RE.finditer(text_n):
            colon_end = match.end()  # position right after ":"

            # The "list zone" extends from the colon until a clear
            # sentence/section break (max 500 chars).
            zone_end = min(len(text_n), colon_end + 500)
            for terminator in ('\n\n', '.\n', '?\n', '!\n'):
                t_pos = text_n.find(terminator, colon_end)
                if t_pos != -1 and t_pos < zone_end:
                    zone_end = t_pos

            zone_text = text_n[colon_end:zone_end]

            # Check which foods appear in the zone text
            for fp in food_positions:
                if fp.norm in negated_norms:
                    continue
                # Search for the food name in the zone (can't use fp.start
                # because those offsets are in _lex_norm space)
                food_norm = _norm(fp.norm)
                if re.search(r'\b' + re.escape(food_norm) + r'\b', zone_text):
                    negated_norms.add(fp.norm)
                    header_text = match.group().strip().rstrip(':')
                    frames.append(Frame(
                        type="negation",
                        food_norm=fp.norm,
                        food_text=fp.text,
                        trigger=header_text,
                        trigger_lemma=header_text.lower(),
                        verb_category="avoidance",
                        detail=(
                            f"avoidance header+colon: '"
                            f"{header_text}' governs food after ':'"
                        ),
                    ))

        return frames

    # ── List-propagation ────────────────────────────────────────────────

    def _propagate_list_negation(
        self,
        doc: Doc,
        food_token_map: dict[FoodPosition, list[Token]],
    ) -> list[Frame]:
        """Propagate avoidance from a header sentence to list-item sentences.

        Pattern detected (structurally, via spaCy):
          Sent_i: contains an avoidance verb + ends with ":"
          Sent_i+1 … Sent_i+k: list items (no verb) containing food

        Uses verb_classifier (not raw regex) to identify avoidance verbs.
        Uses spaCy sentence segmentation (not line splitting).

        The propagation stops when a sentence has its own verb (= new
        section) or after a gap of 2+ empty/numberonly sentences.
        """
        frames: list[Frame] = []
        sents = list(doc.sents)
        if len(sents) < 2:
            return frames

        # Build a set of foods that already have negation frames
        # (from dep-parse or regex), so we don't duplicate
        already_negated: set[str] = set()

        # Pre-index: which FoodPositions are in which sentence?
        sent_foods: dict[int, list[FoodPosition]] = {}
        for fpos, ftokens in food_token_map.items():
            for ftok in ftokens:
                for si, sent in enumerate(sents):
                    if sent.start <= ftok.i < sent.end:
                        sent_foods.setdefault(si, []).append(fpos)
                        break

        for si, sent in enumerate(sents):
            # ── Is this an avoidance header? ──
            # Requirements:
            #   1. Has an avoidance verb (via verb_classifier)
            #   2. Ends with ":" (last non-space token is PUNCT ":")
            has_avoidance_verb = False
            avoidance_trigger = ""
            avoidance_tok = None
            for tok in sent:
                cat = self._get_verb_category(tok)
                if cat == "avoidance":
                    has_avoidance_verb = True
                    avoidance_trigger = tok.text
                    avoidance_tok = tok
                    break

            if not has_avoidance_verb:
                continue

            # Guard: if the avoidance verb already governs a specific
            # object (e.g. "evitar alimentos ultraprocessados"), the
            # list items are alternatives/recommendations — skip.
            # A specific object = obj with a restrictive modifier
            # (acl, acl:relcl, or a non-generic amod).
            if avoidance_tok is not None and self._verb_has_specific_object(avoidance_tok):
                continue

            # Check that ":" appears among the last tokens
            # (spaCy may absorb a list number like "1." into the same
            # sentence, so ":" may not be THE last token)
            last_tokens = [t for t in sent if t.text.strip()]
            has_colon = any(t.text == ":" for t in last_tokens[-4:]) if last_tokens else False

            # Fix 10b: Also accept short header-like sentences without
            # colons (e.g. "### O que evitar", "**Evite**").
            # Criteria: ≤ 8 non-space tokens AND has avoidance verb
            # AND the first following non-empty sentence looks like a
            # list item (starts with -, *, •, or a number).
            _LIST_MARKER_RE = re.compile(r"^\s*[-*•–—]\s|^\s*\d+[.)]\s")
            is_short_header = False
            if not has_colon and len(last_tokens) <= 8:
                # Peek at the next sentence to check for list markers
                for sj_peek in range(si + 1, min(si + 3, len(sents))):
                    peek_text = sents[sj_peek].text.strip()
                    if peek_text and _LIST_MARKER_RE.match(peek_text):
                        is_short_header = True
                        break
                    if peek_text:  # non-empty, non-list → not a header
                        break
            if not has_colon and not is_short_header:
                continue

            # ── Propagate to following list-item sentences ──
            gap = 0
            for sj in range(si + 1, len(sents)):
                following = sents[sj]
                text_stripped = following.text.strip()

                # Skip empty / number-only sentences ("2.", "3.")
                if not text_stripped or all(
                    t.pos_ in ("NUM", "PUNCT", "SPACE") for t in following
                ):
                    gap += 1
                    if gap >= 3:
                        break
                    continue

                # Stop if the sentence has a real verb — i.e. spaCy
                # recognises it as VERB/AUX + verb_classifier confirms.
                # Participles used as modifiers (dep=acl/amod, e.g.
                # "processado") are excluded.
                # We do NOT call _get_verb_category for PROPN/NOUN here,
                # because the imperative-as-PROPN fallback would wrongly
                # match food nouns that happen to be in the lexicon
                # (e.g. "Azeitonas" classified as action).
                has_own_verb = any(
                    t.pos_ in ("VERB", "AUX")
                    and t.dep_ not in ("acl", "amod")
                    and self.verb_classifier.classify(t.lemma_.lower())
                    for t in following
                )
                if has_own_verb:
                    break

                gap = 0  # reset gap counter

                # Any food in this sentence gets a negation frame
                for fpos in sent_foods.get(sj, []):
                    if fpos.norm not in already_negated:
                        already_negated.add(fpos.norm)
                        frames.append(Frame(
                            type="negation",
                            food_norm=fpos.norm,
                            food_text=fpos.text,
                            trigger=avoidance_trigger,
                            trigger_lemma=avoidance_trigger.lower(),
                            verb_category="avoidance",
                            detail=(
                                f"list propagation: avoidance verb "
                                f"'{avoidance_trigger}' in header sentence "
                                f"governs list item"
                            ),
                        ))

        return frames

    # Generic adjectives that don't restrict the object's meaning.
    _GENERIC_AMOD = frozenset({
        "seguinte", "seguintes", "algum", "alguns", "algumas",
        "outro", "outros", "outras", "certo", "certos", "certas",
        "determinado", "determinados", "determinadas",
        "possivel", "possiveis",
    })

    def _verb_has_specific_object(self, verb_tok: Token) -> bool:
        """Check if an avoidance verb governs a specific (non-generic) object.

        Returns True when the verb's direct object has restrictive
        modifiers — acl, acl:relcl, or a non-generic amod — meaning
        it specifies a food *category* (e.g. "evitar alimentos
        ultraprocessados").  In that case, list items after ":" are
        alternatives, not things to avoid.
        """
        for child in verb_tok.children:
            if child.dep_ != "obj":
                continue
            # Check children of the object noun
            for mod in child.children:
                if mod.dep_ in ("acl", "acl:relcl"):
                    return True
                if mod.dep_ == "amod":
                    if _norm(mod.lemma_) not in self._GENERIC_AMOD:
                        return True
            break  # only check first obj
        return False

    # ── Deduplication ───────────────────────────────────────────────────

    def _deduplicate(self, frames: list[Frame]) -> list[Frame]:
        """Keep first frame per (food, type, role) tuple."""
        seen: dict[tuple, Frame] = {}
        for f in frames:
            key = (f.food_norm, f.type, f.role)
            if key not in seen:
                seen[key] = f
        return list(seen.values())


# ═══════════════════════════════════════════════════════════════════════════
# CLI — quick test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading FrameDetector…")
    fd = FrameDetector.create()
    print(f"  Verb lexicon: {len(fd.verb_classifier)} entries")

    test_cases = [
        (
            "Evite mel e açúcar refinado. Prefira adoçantes naturais.",
            [FoodPosition("mel", "mel", 6, 9),
             FoodPosition("açúcar", "acucar", 12, 18)],
        ),
        (
            "Substitua o mel por stevia ou adoçante.",
            [FoodPosition("mel", "mel", 12, 15),
             FoodPosition("stevia", "stevia", 20, 26)],
        ),
        (
            "Não consuma leite nem derivados lácteos.",
            [FoodPosition("leite", "leite", 14, 19)],
        ),
        (
            "Ao invés de mel, use adoçante.",
            [FoodPosition("mel", "mel", 12, 15)],
        ),
        (
            "O açúcar no sangue deve ser monitorado.",
            [FoodPosition("açúcar", "acucar", 2, 8)],
        ),
        (
            "Sem adição de açúcar.",
            [FoodPosition("açúcar", "acucar", 14, 20)],
        ),
        (
            "Consuma frutas com baixo índice glicêmico.",
            [FoodPosition("frutas", "fruta", 8, 14)],
        ),
    ]

    for text, foods in test_cases:
        print(f'\n── "{text}" ──')
        frames = fd.detect(text, foods)
        if frames:
            for f in frames:
                print(f"  {f}")
        else:
            print("  (no safe frames → potential violation)")
