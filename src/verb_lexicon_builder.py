# -*- coding: utf-8 -*-
"""
verb_classifier.py — Layer 1: Knowledge-Driven Verb Semantics
═════════════════════════════════════════════════════════════════

Classifies Portuguese verb lemmas into semantic categories using an
ensemble of two external knowledge sources:

  1. **WordNet hypernym climbing** (NLTK Open Multilingual Wordnet)
     — Walks up the hypernym tree from PT synsets to find anchor concepts
     — High precision for verbs that WordNet covers well

  2. **Word-embedding cosine similarity** (spaCy pt_core_news_md)
     — Compares verb vectors against category centroids
     — High recall — provides signal even for verbs absent in WordNet

Neither source requires manual verb lists.  The only curated items are:
  • ~20 English WordNet anchor synsets (stable — English WN is frozen)
  • ~20 Portuguese seed lemmas for centroid computation
  • ~10 override entries for known edge cases

The build-time pipeline scans spaCy's full vocabulary (~20k words),
classifies every verb, and caches the result as ``data/verb_lexicon.pkl``.
Runtime is a single dict lookup: O(1).

Categories
──────────
  avoidance       — evitar, eliminar, excluir, proibir, …
  substitution    — substituir, trocar, …
  action          — consumir, comer, beber, adicionar, …
  recommendation  — recomendar, sugerir, indicar, …

Usage
─────
    from verb_classifier import VerbClassifier
    vc = VerbClassifier.load()           # loads cached lexicon
    print(vc.classify("evitar"))         # → "avoidance"
    print(vc.classify("cozinhar"))       # → "action"
    print(vc.classify("xyzverb"))        # → None
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — the only "curated" items in the entire module
# ═══════════════════════════════════════════════════════════════════════════

#: English WordNet synsets that define each semantic category.
#: These are "anchor concepts" — if a PT verb's hypernym chain passes
#: through any of these, we know its category.  ~20 synsets total.
WN_ANCHORS: dict[str, frozenset[str]] = {
    "avoidance": frozenset({
        "avoid.v.01", "avoid.v.02", "avoid.v.03",
        "prevent.v.01", "prevent.v.02",
        "refrain.v.01",
        "rid.v.02",
        "destroy.v.01",
        "remove.v.01",
        "forbid.v.01",
        "discontinue.v.01",
        "end.v.02",
        "obstruct.v.01",
        "reject.v.01",
        "refuse.v.01",
        "deny.v.01",
    }),
    "substitution": frozenset({
        "replace.v.01",
        "exchange.v.01",
        "substitute.v.01",
        "supplant.v.01",
    }),
    "action": frozenset({
        "consume.v.02", "consume.v.06",
        "eat.v.01",
        "drink.v.01",
        "add.v.01",
        "cook.v.01", "cook.v.03",
        "make.v.03",
        "prepare.v.01",
        "mix.v.01",
        "serve.v.06",
        "use.v.01",
        "ingest.v.01",
        "fry.v.01", "bake.v.01", "boil.v.01", "roast.v.01",
    }),
    "recommendation": frozenset({
        "propose.v.01",
        "recommend.v.01",
        "advise.v.01",
        "suggest.v.01",
        "counsel.v.01",
    }),
}

#: Portuguese seed lemmas for computing embedding centroids.
#: ~5-10 per category — just enough to define the vector-space region.
EMB_SEEDS: dict[str, list[str]] = {
    "avoidance": [
        "evitar", "eliminar", "excluir", "proibir", "remover",
        "retirar", "suspender", "impedir", "interromper",
    ],
    "substitution": [
        "substituir", "trocar", "alternar", "optar",
    ],
    "action": [
        "consumir", "comer", "beber", "adicionar", "incluir",
        "preparar", "cozinhar", "misturar", "temperar", "servir",
    ],
    "recommendation": [
        "recomendar", "sugerir", "indicar", "aconselhar",
    ],
}

#: Manual override for the ~10 verbs where the ensemble gets it wrong.
#: This is a *correction list*, not the primary classification.
OVERRIDES: dict[str, str] = {
    "dispensar": "avoidance",       # WN maps to "recommend", but PT usage is "dismiss/dispense with"
    "desaconselhar": "avoidance",   # Contains "aconselhar" → embedding pulls toward recommendation
    "restringir": "avoidance",      # "restrinja X" = avoid X entirely
    # Moderation verbs: imply REDUCED consumption, not elimination.
    # "Limite o açúcar" = still consuming → NOT a negation/safe context.
    "limitar": "moderation",
    "reduzir": "moderation",
    "diminuir": "moderation",
    "moderar": "moderation",
    "controlar": "moderation",
    "monitorar": "moderation",
    "aumentar": "action",           # WN maps to substitution, but in dietary = action
    "selecionar": "action",         # WN maps to substitution, but = choosing to use
}

CATEGORIES = ("avoidance", "substitution", "action", "recommendation", "moderation")

# ═══════════════════════════════════════════════════════════════════════════
# CORE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class VerbEntry:
    """Classification result for a single verb lemma."""
    lemma: str
    category: str               # avoidance | substitution | action | recommendation
    confidence: float           # combined score of best category
    margin: float               # gap between best and second-best
    source: str                 # "ensemble" | "override" | "wn_only" | "emb_only"


@dataclass
class VerbClassifier:
    """
    Verb semantic classifier backed by a pre-built lexicon.

    The lexicon is computed at build time by scanning WordNet + embeddings.
    At runtime, classification is a dict lookup.
    """
    lexicon: dict[str, VerbEntry] = field(default_factory=dict)
    _categories: tuple[str, ...] = CATEGORIES

    # ── Runtime API ─────────────────────────────────────────────────────

    def classify(self, lemma: str) -> Optional[str]:
        """Return the semantic category for a Portuguese verb lemma, or None."""
        entry = self.lexicon.get(lemma.lower())
        return entry.category if entry else None

    def get_entry(self, lemma: str) -> Optional[VerbEntry]:
        """Return full classification details, or None."""
        return self.lexicon.get(lemma.lower())

    def is_avoidance(self, lemma: str) -> bool:
        return self.classify(lemma) == "avoidance"

    def is_substitution(self, lemma: str) -> bool:
        return self.classify(lemma) == "substitution"

    def is_action(self, lemma: str) -> bool:
        return self.classify(lemma) == "action"

    def is_recommendation(self, lemma: str) -> bool:
        return self.classify(lemma) == "recommendation"

    def __len__(self) -> int:
        return len(self.lexicon)

    def summary(self) -> dict[str, int]:
        """Count entries per category."""
        counts: dict[str, int] = {c: 0 for c in self._categories}
        for entry in self.lexicon.values():
            counts[entry.category] = counts.get(entry.category, 0) + 1
        return counts

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: str | Path = "data/verb_lexicon.pkl") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize as plain dicts to avoid pickle class-reference issues
        data = {
            lemma: {
                "lemma": e.lemma,
                "category": e.category,
                "confidence": e.confidence,
                "margin": e.margin,
                "source": e.source,
            }
            for lemma, e in self.lexicon.items()
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved verb lexicon (%d entries) → %s", len(self.lexicon), path)

    @classmethod
    def load(cls, path: str | Path = "data/verb_lexicon.pkl") -> VerbClassifier:
        """Load a pre-built verb lexicon from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Verb lexicon not found at {path}. "
                f"Run `python verb_classifier.py` to build it."
            )
        with open(path, "rb") as f:
            raw = pickle.load(f)
        # Reconstruct VerbEntry objects from plain dicts
        lexicon = {}
        for lemma, d in raw.items():
            if isinstance(d, dict):
                lexicon[lemma] = VerbEntry(**d)
            else:
                # Already a VerbEntry (backward compat)
                lexicon[lemma] = d
        # Re-apply code-level OVERRIDES so changes take effect without rebuild
        _n_patched = 0
        for lemma, cat in OVERRIDES.items():
            existing = lexicon.get(lemma)
            if existing is None or existing.category != cat:
                lexicon[lemma] = VerbEntry(
                    lemma=lemma,
                    category=cat,
                    confidence=1.0,
                    margin=1.0,
                    source="override",
                )
                _n_patched += 1
        vc = cls(lexicon=lexicon)
        if _n_patched:
            logger.info("Patched %d verb entries with current overrides.", _n_patched)
        logger.info("Loaded verb lexicon (%d entries) from %s", len(vc), path)
        return vc

    # ── Build-time pipeline ─────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        *,
        wn_max_depth: int = 6,
        wn_weight: float = 0.4,
        emb_weight: float = 0.6,
        threshold: float = 0.30,
        save_path: str | Path = "data/verb_lexicon.pkl",
    ) -> VerbClassifier:
        """
        Build the verb lexicon from scratch using WordNet + embeddings.

        Steps:
          1. Load spaCy pt_core_news_md for word vectors
          2. Load NLTK WordNet for hypernym chains
          3. Compute embedding centroids from seed lemmas
          4. Scan all verbs in spaCy's vocabulary
          5. For each verb: ensemble(WN_score, EMB_score) → category
          6. Apply overrides for known edge cases
          7. Cache to disk
        """
        import spacy
        from nltk.corpus import wordnet as wn

        print("[verb_classifier] Building verb lexicon…")
        t0 = time.time()

        # 1. Load spaCy md model
        nlp = spacy.load("pt_core_news_md")
        print(f"  spaCy pt_core_news_md: {nlp.vocab.vectors.shape[0]} vectors")

        # 2. Compute embedding centroids
        centroids: dict[str, np.ndarray] = {}
        for cat, seeds in EMB_SEEDS.items():
            vecs = []
            for w in seeds:
                lex = nlp.vocab[w]
                if lex.has_vector:
                    vecs.append(lex.vector)
                else:
                    print(f"  WARNING: seed '{w}' has no vector")
            if vecs:
                centroids[cat] = np.mean(vecs, axis=0)
            else:
                raise ValueError(f"No vectors for category '{cat}' seeds")
        print(f"  Centroids computed for {len(centroids)} categories")

        # 3. Build WN scoring function (cached per lemma)
        def _wn_scores(lemma: str) -> dict[str, float]:
            scores = {c: 0.0 for c in CATEGORIES}
            synsets = wn.synsets(lemma, lang="por", pos=wn.VERB)
            if not synsets:
                return scores
            for ss in synsets:
                visited: set[str] = set()
                frontier = [(ss, 0)]
                while frontier:
                    cur, depth = frontier.pop(0)
                    if depth > wn_max_depth or cur.name() in visited:
                        continue
                    visited.add(cur.name())
                    for cat, anchors in WN_ANCHORS.items():
                        if cur.name() in anchors:
                            scores[cat] = max(scores[cat], 1.0 / (1 + depth))
                    for hyp in cur.hypernyms():
                        frontier.append((hyp, depth + 1))
            return scores

        # 4. Build EMB scoring function
        def _emb_scores(lemma: str) -> dict[str, float]:
            lex = nlp.vocab[lemma]
            if not lex.has_vector:
                return {c: 0.0 for c in CATEGORIES}
            return {c: max(0, _cosine(lex.vector, centroids[c])) for c in CATEGORIES}

        # 5. Ensemble classification
        def _classify_one(lemma: str) -> Optional[VerbEntry]:
            ws = _wn_scores(lemma)
            es = _emb_scores(lemma)
            combined = {
                c: wn_weight * ws[c] + emb_weight * es[c] for c in CATEGORIES
            }
            sorted_cats = sorted(CATEGORIES, key=lambda c: combined[c], reverse=True)
            best = sorted_cats[0]
            best_score = combined[best]
            second_score = combined[sorted_cats[1]]
            margin = best_score - second_score

            if best_score < threshold:
                return None

            source = "ensemble"
            if max(ws.values()) == 0:
                source = "emb_only"
            elif max(es.values()) == 0:
                source = "wn_only"

            return VerbEntry(
                lemma=lemma,
                category=best,
                confidence=best_score,
                margin=margin,
                source=source,
            )

        # 6. Scan all words in spaCy vocabulary that could be verbs
        #    We check: has vector + length >= 3 + all alpha
        #    Then classify via ensemble.  Also explicitly include all seeds + overrides.
        candidates: set[str] = set()

        # All words with vectors in spaCy
        for word in nlp.vocab.strings:
            lex = nlp.vocab[word]
            if lex.has_vector and len(word) >= 3 and word.isalpha() and word.islower():
                candidates.add(word)

        # Ensure all seeds and overrides are included
        for seeds in EMB_SEEDS.values():
            candidates.update(seeds)
        candidates.update(OVERRIDES.keys())

        print(f"  Scanning {len(candidates)} candidate words…")

        # Classify candidates
        lexicon: dict[str, VerbEntry] = {}
        wn_hits = 0

        for lemma in sorted(candidates):
            entry = _classify_one(lemma)
            if entry is not None:
                lexicon[lemma] = entry
                if entry.source != "emb_only":
                    wn_hits += 1

        print(f"  Ensemble classified: {len(lexicon)} verbs "
              f"(WN contributed to {wn_hits})")

        # 7. Apply overrides
        override_count = 0
        for lemma, cat in OVERRIDES.items():
            existing = lexicon.get(lemma)
            if existing is None or existing.category != cat:
                lexicon[lemma] = VerbEntry(
                    lemma=lemma,
                    category=cat,
                    confidence=1.0,
                    margin=1.0,
                    source="override",
                )
                override_count += 1

        print(f"  Overrides applied: {override_count}")

        # 8. Also add morphological variants for known verbs
        #    Portuguese conjugation: if "evitar" is avoidance, then
        #    "evite", "evitando", "evitou", "evitei" etc. should be too.
        #    We do this by checking: for each classified lemma, scan vocab
        #    for words starting with the lemma stem (first 4+ chars).
        #    Only assign if the word has a vector and is not already classified.
        stem_expansions = 0
        lemma_stems: dict[str, str] = {}   # stem → category

        for lemma, entry in list(lexicon.items()):
            # Use first N chars as stem (min 4 to avoid collisions)
            stem_len = max(4, len(lemma) - 2)
            stem = lemma[:stem_len]
            # Don't overwrite if shorter stem already exists for different cat
            if stem not in lemma_stems:
                lemma_stems[stem] = entry.category

        for word in candidates:
            if word in lexicon:
                continue
            for stem, cat in lemma_stems.items():
                if word.startswith(stem) and len(word) <= len(stem) + 6:
                    lex_w = nlp.vocab[word]
                    if lex_w.has_vector:
                        # Verify with embedding that it's in the right ballpark
                        es = _emb_scores(word)
                        if es[cat] >= es.get(
                            max(es, key=es.get), 0
                        ) * 0.85:  # within 85% of best
                            lexicon[word] = VerbEntry(
                                lemma=word,
                                category=cat,
                                confidence=0.8,
                                margin=0.0,
                                source="stem_expansion",
                            )
                            stem_expansions += 1
                    break

        print(f"  Stem expansions: {stem_expansions}")

        # Build and save
        vc = cls(lexicon=lexicon)
        vc.save(save_path)

        elapsed = time.time() - t0
        summary = vc.summary()
        print(f"\n  Final lexicon: {len(vc)} verbs in {elapsed:.1f}s")
        for cat, count in summary.items():
            print(f"    {cat:18s}: {count}")

        return vc


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--build" in sys.argv or not Path("data/verb_lexicon.pkl").exists():
        vc = VerbClassifier.build()
    else:
        vc = VerbClassifier.load()

    # Interactive test
    print("\n── Test verbs ──")
    test_words = [
        "evitar", "eliminar", "excluir", "proibir", "remover", "retirar",
        "suspender", "abandonar", "cessar", "impedir", "afastar",
        "interromper", "rejeitar", "dispensar", "desaconselhar", "abster",
        "substituir", "trocar", "optar", "preferir",
        "consumir", "comer", "beber", "adicionar", "incluir",
        "preparar", "servir", "temperar", "misturar", "cozinhar",
        "assar", "grelhar", "fritar", "ingerir", "tomar",
        "recomendar", "sugerir", "indicar", "aconselhar",
        "moderar", "limitar", "restringir", "controlar", "monitorar",
        "reduzir", "diminuir", "aumentar",
        # Conjugated forms (should work via stem expansion)
        "evite", "elimine", "substitua", "consuma", "cozinhe",
    ]

    for w in test_words:
        entry = vc.get_entry(w)
        if entry:
            print(f"  {w:20s} → {entry.category:15s} "
                  f"(conf={entry.confidence:.2f}, src={entry.source})")
        else:
            print(f"  {w:20s} → UNKNOWN")
