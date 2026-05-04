# -*- coding: utf-8 -*-
"""
lexicon_builder.py — Module 1: Build the Food Lexicon with Substance Flags
═══════════════════════════════════════════════════════════════════════════

Builds a comprehensive Brazilian Portuguese food lexicon from TBCA
(Tabela Brasileira de Composição de Alimentos), enriched with:

  • Nutritional data (carbs, fiber, sodium, fat, etc.) from TBCA product pages
  • Allergen flags (gluten, lactose, egg, peanut, nuts) from Open Food Facts
  • Derived `involves_substance` flags via nutrient thresholds

Outputs:
  data/tbca_foods.parquet   — raw TBCA foods with nutrients + flags
  data/lexicon.parquet      — expanded variant lexicon
  data/automaton.pkl        — serialised Aho-Corasick automaton for fast matching

Run once:
    python lexicon_builder.py --build --pages 300 --workers 16

Load and inspect:
    python lexicon_builder.py --info
"""

from __future__ import annotations

import logging
import pickle
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    import ahocorasick
except ImportError:
    ahocorasick = None  # type: ignore

logger = logging.getLogger(__name__)

# ─── Output paths ─────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
TBCA_PARQUET = DATA_DIR / "tbca_foods.parquet"
LEXICON_PARQUET = DATA_DIR / "lexicon.parquet"
AUTOMATON_PKL = DATA_DIR / "automaton.pkl"

# ─── Constants ────────────────────────────────────────────────────────────

TBCA_BASE_URL = "https://www.tbca.net.br/base-dados/composicao_alimentos.php"
TBCA_NUTRIENT_URL = (
    "https://tbca.net.br/base-dados-en/int_statistical_composition.php"
)
TBCA_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}
OFF_HEADERS = {"User-Agent": "food-research-bot/1.0"}

NOISE_PARTS = {"brasil", "dado importado"}

PREP_STOPWORDS = {
    "cozido", "cozida", "cozidos", "cozidas",
    "cru", "crua", "crus", "cruas",
    "refogado", "refogada", "refogados", "refogadas",
    "assado", "assada", "assados", "assadas",
    "grelhado", "grelhada", "grelhados", "grelhadas",
    "drenado", "drenada", "drenados", "drenadas",
    "congelado", "congelada", "congelados", "congeladas",
    "desidratado", "desidratada", "desidratados", "desidratadas",
    "caramelizado", "caramelizada", "caramelizados", "caramelizadas",
}

QUERY_STOPWORDS = {"caseiro", "caseira", "simples"}

# Words that should never appear at the END of a variant (dangling prepositions,
# conjunctions, articles). These are structural Portuguese function words.
TRAILING_STOPWORDS = {
    "de", "do", "da", "dos", "das",
    "com", "sem", "em", "no", "na", "nos", "nas",
    "ao", "aos", "a", "as", "o", "os",
    "e", "ou", "por", "para",
    "c", "s",  # abbreviations for com/sem in TBCA
}

# Tokens that mark a TBCA comma-part as a *qualifier* (preparation state,
# processing method, origin, etc.) rather than a specific food name.
# A comma-part is "qualifier-only" if ALL its tokens are in this set.
_QUALIFIER_TOKENS = frozenset({
    # Preparation states — skip BOTH standalone AND compound variants.
    # These describe HOW a food is prepared, not WHAT the food is.
    "cru", "crua", "crus", "cruas",
    "cozido", "cozida", "cozidos", "cozidas",
    "frito", "frita", "fritos", "fritas",
    "assado", "assada", "assados", "assadas",
    "grelhado", "grelhada", "grelhados", "grelhadas",
    "refogado", "refogada", "refogados", "refogadas",
    "torrado", "torrada", "torrados", "torradas",
    "defumado", "defumada", "moido", "moida",
    "drenado", "drenada", "caramelizado", "caramelizada",
    "temperado", "temperada",
    "gratinado", "gratinada",
    "empanado", "empanada", "milanesa",
    "caramelado", "caramelada",
    "fresco", "fresca", "frescos", "frescas",
    "texturizado", "texturizada",
    # Processing states
    "congelado", "congelada", "desidratado", "desidratada",
    "liofilizado", "liofilizada", "pasteurizado", "pasteurizada",
    "industrializado", "industrializada",
    "reconstituido", "reconstituida",
    "processado", "processada",
    "preparado", "preparada",
    # Origin / meta — never meaningful as food names
    "brasil", "dado", "importado", "importada",
    # Generic particles
    "fluido", "fluida", "po", "em", "na", "no",
})

# ── Non-standalone words ──
# Words that should NOT become standalone automaton entries, but ARE
# allowed in compound variants (e.g., "branco" → no, but "arroz branco"
# → yes).  This set is checked in two places:
#   1) _build_cross_category_foods() — prevents them from entering the
#      cross-category set.
#   2) The comma-part variant logic — skips standalone emission for
#      single-word parts in this set.
_NON_STANDALONE_WORDS = frozenset({
    # Group/container labels
    "mar",        # part of "frutos do mar"
    "grao",       # part of "grão de bico"
    "semente",    # part of "semente de chia"
    "folha",      # part of "folha de couve"
    "frutos",     # part of "frutos do mar"
    "conserva",   # part of "em conserva"
    "frios",      # generic deli category
    "casca",      # part of "casca de laranja"
    "forno",      # "de forno" is a preparation style
    # Descriptive adjectives — meaningful only in compounds
    "branco", "branca",
    "seco", "seca",
    "vegano", "vegana",
    "diet", "tradicional",
    "integral", "doce", "salgado", "salgada",
    "natural", "light", "zero", "dietetico", "dietetica",
    "maduro", "madura", "verde",
    "polpa", "caseiro", "caseira", "artesanal", "simples",
    "fatiado", "fatiada",
    "ensopado", "ensopada",
    "concentrado", "concentrada",
    "fundido", "fundida",
    "descafeinado", "descafeinada",
    "desengordurado", "desengordurada",
    "desnatado", "desnatada",
    "instantaneo", "instantanea",
    "frescal",
    # Too-generic categories — only useful as compounds
    "cha", "chas",        # "chá verde" yes, bare "chá" no
    "suco", "sucos",      # "suco de laranja" yes, bare "suco" no
    "bebida", "bebidas",  # "bebida alcoolica" yes, bare "bebida" no
    "fruta", "frutas",    # "fruta fresca" yes, bare "fruta" no
    "erva", "ervas",      # "erva doce" yes, bare "erva" no
    "legume", "legumes",  # "legume verde" yes, bare "legume" no
    "vegetal", "vegetais",# "vegetal cozido" yes, bare "vegetal" no
    "lanche", "lanches",  # "lanche natural" maybe, bare "lanche" no
    "oleo", "oleos",      # "óleo de oliva" yes, bare "óleo" no — avoids peanut FP
    "algodao",            # cotton — not a food item (TBCA cottonseed oil entry)
    # Food components — not standalone food items
    "proteina", "proteinas",
    "gordura", "gorduras",
    # Nationality / origin adjectives
    "italiano", "italiana", "japones", "japonesa",
    "americano", "americana", "paulista",
    "mineiro", "mineira",
})

# ── Non-food tokens: terms to REMOVE from the automaton entirely ──
# These are cooking methods, containers, anatomy terms, descriptive phrases,
# or other tokens that are NOT foods and cause false positives.
_NON_FOOD_TOKENS = frozenset({
    # Containers / packaging
    "lata", "latas",
    # Cooking method / descriptors
    "ralado", "ralada", "ralados", "raladas",
    "soluvel", "soluveis",
    # Anatomy / clinical terms (not food items)
    "musculo", "sangue",
    # Descriptive phrases misidentified as food
    "fonte de fibra",
    # Color/adjective tokens that are not standalone foods
    "preto", "branco", "verde",
    "preta", "pretas", "pretos",
    # Generic nutritional terms (not food items)
    "vitamina", "vitaminas",
    # Clinical/physiological terms (not food items)
    "glicose",
    # Cooking/preparation descriptors that match food entries incorrectly
    "recheado", "recheada", "recheados", "recheadas",
})

# ── Canonical overrides: fix wrong canonical mappings ──
# {variant_norm: correct_canonical}
_CANONICAL_OVERRIDES = {
    "file": "file",                         # filé (fillet), not generic "carne"
    "boi": "boi",                           # beef, not "almondega"
    "peito de peru": "peito de peru",       # turkey breast, not "pate"
    "salsinha e cebolinha": "salsinha e cebolinha",  # herbs, not "omelete"
    "leite de coco": "leite de coco",       # coconut milk, not "bombom"
    "cacao": "cacao",                       # cocoa, not "oleo"
    "peito": "peito",                       # breast cut, not generic "carne"
    "frango": "frango",                     # chicken, not generic "carne"
    "farinha de arroz": "farinha de arroz", # rice flour, not "biscoito"
    "polpa de fruta": "polpa de fruta",     # fruit pulp, not "iogurte"
    "espinafre refogado": "espinafre",      # sauteed spinach, not "salada"
    "floco": "floco",                       # flake (oat), not compound
    "massa": "massa",                       # pasta, not "cheese cake"
    "cenoura e pepino": "cenoura e pepino", # vegetables, not "sanduiche"
    "peito de frango grelhado": "peito de frango grelhado",  # grilled chicken breast, not "sanduiche"
    "file mignon": "file mignon",       # filet mignon, not "almondega"
    "carne": "carne",                     # generic meat, not "almondega"
    "carnes": "carne",                    # plural
    "abobrinha e cebola": "abobrinha e cebola",  # vegetables, not beef rib dish BRC0581F
    "galinha": "galinha",                 # chicken meat, not egg/bouillon
}

# ── Substance overrides: fix incorrect substance flags ──
# {variant_norm: frozenset_of_correct_substances}
# None means "remove all substances" (herbs/non-food items).
_SUBSTANCE_OVERRIDES = {
    "sal": frozenset({"sodium"}),                    # salt has 0 cholesterol
    "alecrim": frozenset(),                          # rosemary is an herb — no relevant restrictions
    "salsinha e cebolinha": frozenset(),              # herbs — no restrictions
    "leite de coco": frozenset({"saturated_fat"}),   # coconut milk: no lactose!
    "file": frozenset(),                             # filé: not inherently restricted
    "peito": frozenset(),                            # lean breast cut: not inherently restricted
    "peito de peru": frozenset(),                    # turkey breast: lean, not pate
    "cacao": frozenset({"caffeine"}),                # cocoa: caffeine, not "oleo"
    "efo": frozenset({"saturated_fat"}),             # keep: leaf lard is indeed high sat fat
    "tempero": frozenset({"sodium"}),                # seasonings: sodium only, not added_sugar
    "temperos": frozenset({"sodium"}),               # plural
    # Correct substance flags for foods with inherited contamination
    "frango": frozenset(),                           # chicken: lean meat, no inherent restriction
    "amendoa": frozenset({"nut"}),                   # almond: tree nut only, NO gluten
    "amendoas": frozenset({"nut"}),                  # plural
    "castanha": frozenset({"nut"}),                  # cashew/chestnut: tree nut only, NO gluten
    "castanhas": frozenset({"nut"}),                 # plural
    "farinha de arroz": frozenset(),                 # rice flour: gluten-free!
    "polpa de fruta": frozenset(),                   # fruit pulp: no lactose
    "tapioca": frozenset({"high_glycemic"}),         # tapioca: high GI but NO lactose
    "floco": frozenset({"insoluble_fiber"}),         # oat flake: fiber but NO lactose
    "flocos": frozenset({"insoluble_fiber"}),        # plural
    "espinafre refogado": frozenset(),               # spinach: no nut, no lactose
    "espinafre": frozenset(),                        # spinach: safe vegetable
    "massa": frozenset({"gluten"}),                  # pasta: gluten yes, but NOT trans_fat
    "pao de queijo": frozenset({"lactose"}),         # cheese bread: lactose yes, but NO gluten (tapioca-based)
    "torrada integral": frozenset({"gluten", "insoluble_fiber"}),  # whole grain toast: gluten+fiber, NOT lactose
    "torrada": frozenset({"gluten"}),                # toast: gluten only, not cholesterol/lactose
    "torradas": frozenset({"gluten"}),               # plural
    "fermento": frozenset(),                         # yeast: no restriction
    "suplemento": frozenset(),                       # supplement: generic, no inherent restriction
    "suplementos": frozenset(),                      # plural
    "caldo": frozenset({"sodium"}),                  # broth: sodium, not added_sugar
    "macarrao": frozenset({"gluten"}),               # pasta: gluten, not insoluble_fiber
    # Protein cuts: lean meats should not carry trans_fat
    "peito": frozenset(),                            # lean breast cut
    "peitos": frozenset(),                           # plural
    "sanduiche": frozenset({"gluten"}),              # sandwich: gluten from bread, not inherent cholesterol
    "sanduiches": frozenset({"gluten"}),             # plural
    "omelete": frozenset({"egg"}),                   # omelet: egg only, not trans_fat/lactose
    "alga": frozenset(),                             # seaweed: not high_glycemic
    "algas": frozenset(),                            # plural
    # Pork loin: lean meat — BRC0192F is smoked Canadian bacon;
    # plain lombo should NOT carry those processed-meat flags
    "lombo": frozenset(),
    # Bacalhau: BRC0545E is breaded/fried with egg — plain cod only has purine
    "bacalhau": frozenset({"purine"}),
    # Garlic/onion powder: retain fodmap but NOT high_glycemic
    "alho em po": frozenset({"fodmap", "potassium"}),
    "cebola em po": frozenset({"fodmap"}),
    # Vegetables: cenoura e pepino is NOT a sandwich
    "cenoura e pepino": frozenset(),
    # Salt: refined salt is still just sodium, NOT cholesterol
    "sal refinado": frozenset({"sodium"}),
    # Herb sauce: not inherently high saturated fat
    "molho de ervas": frozenset(),
    # Gengibre em pó: used in tiny quantities, not a standalone high-GI food
    "gengibre em po": frozenset(),
    # Plant-based cheese: no lactose (may contain nuts from cashew base)
    "queijo vegetal": frozenset({"nut"}),
    # "sais" matches "jardineira" in TBCA — only sodium is relevant
    "sais": frozenset({"sodium"}),
    # "cenoura crua" mapped to sandwich — just a vegetable
    "cenoura crua": frozenset(),
    # "peito de frango" mapped to generic carne with insoluble_fiber — chicken has no fiber
    "peito de frango": frozenset(),
    "peitos de frango": frozenset(),
    # "peito de frango grelhado" mapped to sanduiche in TBCA — just grilled chicken
    "peito de frango grelhado": frozenset(),
    # "carne"/"carnes" mapped to almondega (BRC0001F) with trans_fat — generic lean meat token
    "carne": frozenset({"saturated_fat", "cholesterol"}),
    "carnes": frozenset({"saturated_fat", "cholesterol"}),
    # "filé mignon" mapped to almondega — lean beef cut
    "file mignon": frozenset({"cholesterol"}),
    # "aveia" (oats): contains gluten (unless certified GF) + insoluble fiber.
    # Oats also have soluble fiber (beta-glucan) which is beneficial, but
    # the insoluble component is relevant for Crohn's/SIC.
    "aveia": frozenset({"gluten", "insoluble_fiber"}),
    # "sal grosso" (coarse salt): only sodium, NO cholesterol (TBCA BRC0018L error)
    "sal grosso": frozenset({"sodium"}),
    # "iogurte de coco" — when LLMs recommend this, it's plant-based (no lactose)
    "iogurte de coco": frozenset(),
    # Spices: used in tiny quantities (1-2g), per-100g carb values are irrelevant
    "cardamomo": frozenset(),
    "curcuma em po": frozenset(),
    # "galinha" (chicken meat): NOT an egg allergen
    "galinha": frozenset(),
    # "maionese sem ovos" — explicitly eggless mayo
    "maionese sem ovos": frozenset({"saturated_fat"}),
    # "abobrinha e cebola" — vegetables, not beef rib dish (BRC0581F)
    "abobrinha e cebola": frozenset(),
    # Alcoholic beverages: ensure they carry the 'alcohol' substance
    "cerveja": frozenset({"alcohol", "purine", "gluten"}),  # beer: alcohol + purine + gluten
    "cervejas": frozenset({"alcohol", "purine", "gluten"}),
    "chopp": frozenset({"alcohol", "purine", "gluten"}),
    "chope": frozenset({"alcohol", "purine", "gluten"}),
    "vodka": frozenset({"alcohol"}),
    "whisky": frozenset({"alcohol"}),
    "rum": frozenset({"alcohol"}),
    "gin": frozenset({"alcohol"}),
    "tequila": frozenset({"alcohol"}),
    "licor": frozenset({"alcohol", "added_sugar"}),
    "aguardente": frozenset({"alcohol"}),
    "cachaca": frozenset({"alcohol"}),
    "champanhe": frozenset({"alcohol"}),
    "espumante": frozenset({"alcohol"}),
    "sidra": frozenset({"alcohol"}),
    # ── False-positive fixes (from v3 analysis) ──
    # Açaí: no caffeine — wrongly inherited from TBCA mapping
    "acai": frozenset(),
    # Trigo sarraceno (buckwheat): NOT wheat — gluten-free, low-FODMAP pseudocereal
    "trigo sarraceno": frozenset(),
    # Arroz basmati: explicitly low-FODMAP rice
    "arroz basmati": frozenset(),
    # Sal rosa: salt has sodium only, NOT cholesterol (TBCA mapping error)
    "sal rosa": frozenset({"sodium"}),
    "sal rosa do himalaia": frozenset({"sodium"}),
    "sal do himalaia": frozenset({"sodium"}),
    # Iogurte: lactose only — no gluten (wrongly inherited from TBCA)
    "iogurte": frozenset({"lactose"}),
    "iogurtes": frozenset({"lactose"}),
    # Polvo: purine only — no lactose, no sodium/potassium at dietary concern levels
    "polvo": frozenset({"purine"}),
    # Algodão doce: cotton candy is high-sugar, but bare "algodao" is blocked
    # via _NON_STANDALONE_WORDS; keep the compound form correct
    "algodao doce": frozenset({"added_sugar", "high_glycemic"}),
    # Frozen (frozen yogurt): lactose from dairy, but no nut
    "frozen": frozenset({"lactose"}),
    # Nori (seaweed): per-serving sodium/phosphorus/potassium is negligible
    "nori": frozenset(),
    # Cebola caramelizada: caramelized onion — keep FODMAP but not sodium
    "cebola caramelizada": frozenset({"fodmap"}),
    # Panqueca: gluten from flour + egg, but NOT cholesterol/lactose per se
    "panqueca": frozenset({"gluten", "egg"}),
    "panquecas": frozenset({"gluten", "egg"}),
    # Bolo: sugar + gluten from flour; high_glycemic is debatable but keep
    "bolo": frozenset({"added_sugar", "gluten"}),
    "bolos": frozenset({"added_sugar", "gluten"}),
    # Gluten-free breads: pão de milho, pão de batata, pão de mandioca are
    # made from corn/potato/cassava flour — NO gluten
    "pao de milho": frozenset({"high_glycemic", "added_sugar"}),
    "pao de batata": frozenset({"high_glycemic"}),
    "pao de batata e polvilho doce": frozenset({"high_glycemic"}),
    "pao de mandioca e inhame": frozenset({"high_glycemic"}),
    # Frozens (plural): frozen yogurt — lactose only, no nut
    "frozens": frozenset({"lactose"}),
    # Iogurte simples / caseira: plain yogurt — lactose only
    "iogurte simples": frozenset({"lactose"}),
    "iogurte caseira": frozenset({"lactose"}),
    "iogurte de simples": frozenset({"lactose"}),
    "iogurte de caseira": frozenset({"lactose"}),
    # ── Recall fixes: correct incomplete substance flags ──
    # Refrigerante: TBCA has no sugar flag for carbonated drinks
    "refrigerante": frozenset({"added_sugar", "high_glycemic"}),
    "refrigerantes": frozenset({"added_sugar", "high_glycemic"}),
    # Linguiça: TBCA only has cholesterol; add sodium + saturated_fat
    "linguica": frozenset({"sodium", "saturated_fat", "cholesterol"}),
    "linguicas": frozenset({"sodium", "saturated_fat", "cholesterol"}),
    # Queijo: lactose + saturated_fat. Sodium removed — fresh cheeses
    # (cottage, ricota) are low-sodium and commonly recommended.
    "queijo": frozenset({"saturated_fat", "lactose"}),
    "queijos": frozenset({"saturated_fat", "lactose"}),
    # Manteiga: TBCA has trans_fat+lactose; should be saturated_fat+cholesterol+lactose
    "manteiga": frozenset({"saturated_fat", "cholesterol", "lactose"}),
    # Enlatados: TBCA has no flags; add sodium
    "enlatados": frozenset({"sodium"}),
    "enlatado": frozenset({"sodium"}),
    # Vitela: TBCA has no flags; add purine (organ-like red meat)
    "vitela": frozenset({"purine"}),
    # Achocolatado: TBCA has high_glycemic+potassium; add added_sugar+lactose
    "achocolatado": frozenset({"added_sugar", "high_glycemic", "lactose"}),
    # Manteiga de amendoim: peanut + nut (cross-reactivity with tree nuts)
    "manteiga de amendoim": frozenset({"peanut", "nut"}),
    # Frutos do mar: TBCA has no flags; add purine+cholesterol
    "frutos do mar": frozenset({"purine", "cholesterol"}),
    # Presunto: remove spurious added_sugar flag
    "presunto": frozenset({"sodium", "saturated_fat", "cholesterol"}),
    # ── Recall fixes round 2: substance gaps causing FNs ──
    # Bread: gluten + high_glycemic. TBCA sodium is 400-550 mg/100g,
    # below ANVISA 600 mg threshold — sodium removed (not in LISTA_HARD
    # for hypertension either).
    "pao": frozenset({"gluten", "high_glycemic"}),
    "paes": frozenset({"gluten", "high_glycemic"}),
    "pao integral": frozenset({"gluten"}),
    "pao de forma": frozenset({"gluten"}),
    # Sanduíche: bread + fillings → sodium + gluten
    "sanduiche": frozenset({"gluten", "sodium"}),
    "sanduiches": frozenset({"gluten", "sodium"}),
    # Bolinho/torta: pastries — sugar, gluten, saturated_fat
    "bolinho": frozenset({"added_sugar", "gluten", "saturated_fat"}),
    "bolinhos": frozenset({"added_sugar", "gluten", "saturated_fat"}),
    "torta": frozenset({"added_sugar", "gluten", "saturated_fat"}),
    "tortas": frozenset({"added_sugar", "gluten", "saturated_fat"}),
    "torta de frutas": frozenset({"added_sugar", "gluten"}),
    # Iogurte: keep lactose only — small-serving yogurt is low-FODMAP
    "iogurte": frozenset({"lactose"}),
    "iogurtes": frozenset({"lactose"}),
    "iogurte simples": frozenset({"lactose"}),
    "iogurte caseira": frozenset({"lactose"}),
    "iogurte de simples": frozenset({"lactose"}),
    "iogurte de caseira": frozenset({"lactose"}),
    # Brócolis: high-FODMAP (Monash certified)
    "brocolis": frozenset({"fodmap"}),
    "brocolil": frozenset({"fodmap"}),
    # Truta: moderate purine (already in clinical KB)
    "truta": frozenset({"purine"}),
    # Duck: high saturated_fat + cholesterol
    "carne de pato": frozenset({"saturated_fat", "cholesterol"}),
    # Amendoim: add nut flag — clinically grouped with tree nuts (cross-reactivity ~25-40%)
    "amendoim": frozenset({"peanut", "nut", "insoluble_fiber"}),
    # Seeds: phosphorus + potassium for CKD relevance
    "semente de chia": frozenset({"phosphorus", "potassium"}),
    "semente de abobora": frozenset({"potassium"}),
    "semente de girassol": frozenset({"phosphorus", "potassium"}),
    # ── False-positive fixes (from v5 human-annotation analysis) ──
    # Chá verde: kept as caffeine — green tea has ~30mg caffeine/cup, genuinely
    # restricted for GERD; human annotator confirmed 10/14 affected cases are TP.
    # Iogurte de soja: plant-based — no lactose (inherits wrongly from base "iogurte")
    "iogurte de soja": frozenset(),
    # "Assada no forno" (baked in oven): cooking method, not a food item.
    # TBCA maps it to a baked dish with cholesterol+sodium.
    "assada no forno": frozenset(),
    # Torta de abobrinha: homemade zucchini quiche — only gluten from crust.
    # TBCA maps to commercial version with saturated_fat, lactose, sodium, added_sugar.
    "torta de abobrinha": frozenset({"gluten"}),
}

# ── Custom food entries not present in any data source ──
# These are common foods mentioned by LLMs that don't appear in TBCA.
# Injected into the automaton at build time (Lever 4) and at load time.
_CUSTOM_FOODS: dict[str, frozenset[str]] = {
    "bacon": frozenset({"sodium", "saturated_fat"}),
    "picanha": frozenset({"saturated_fat", "cholesterol", "purine"}),
    "nata": frozenset({"saturated_fat", "lactose"}),
    "doce": frozenset({"added_sugar", "high_glycemic"}),
    "embutidos": frozenset({"sodium", "saturated_fat"}),
    "embutido": frozenset({"sodium", "saturated_fat"}),
    "conserva": frozenset({"sodium"}),
    "conservas": frozenset({"sodium"}),
    "ketchup": frozenset({"sodium", "added_sugar"}),
    "shoyu": frozenset({"sodium"}),
    "chocolate quente": frozenset({"added_sugar", "lactose"}),
    "manteiga clarificada": frozenset({"saturated_fat", "cholesterol"}),
    "mariscos": frozenset({"purine", "cholesterol"}),
    "carne vermelha": frozenset({"saturated_fat", "cholesterol", "purine"}),
    "carnes vermelhas": frozenset({"saturated_fat", "cholesterol", "purine"}),
    "leite de amendoim": frozenset({"peanut", "nut"}),
    "camarao": frozenset({"purine", "cholesterol"}),
    "vinho": frozenset({"alcohol"}),
    "vinho branco": frozenset({"alcohol"}),
    "vinho tinto": frozenset({"alcohol"}),
    "rose": frozenset({"alcohol"}),
}

# ── Herb/spice whitelist: substances forced to empty ──
_HERB_SPICE_TOKENS = frozenset({
    "alecrim", "salsinha", "cebolinha", "manjericao",
    "oregano", "cominho", "coentro", "salsa", "hortela",
    "louro", "tomilho", "endro", "manjerona", "estragao",
    "noz moscada", "acafrao", "curcuma", "gengibre",
    "canela", "cravo", "pimenta do reino",
    "paprica", "colorau", "chimichurri",
})


def _singularize(word: str) -> str:
    """Basic Portuguese plural → singular conversion."""
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
    # -es after consonant: remove just the -s (legumes→legume, tomates→tomate)
    # But -zes → -z (nozes→noz), -res → -r (flores→flor)
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


def _pluralize(word: str) -> str:
    """Basic Portuguese singular → plural conversion."""
    if len(word) <= 2:
        return word + "s"
    # Common -ão → -ães irregulars (pão→pães, cão→cães, alemão→alemães)
    if word in ("pao", "cao", "alemao", "capitao", "charlatao"):
        return word[:-2] + "aes"
    if word.endswith("ao"):
        return word[:-2] + "oes"
    if word.endswith("al") and len(word) >= 3:
        return word[:-1] + "is"
    if word.endswith("el") and len(word) >= 3:
        return word[:-1] + "is"
    if word.endswith("ol") and len(word) >= 3:
        return word[:-1] + "is"
    if word.endswith("il") and len(word) >= 3:
        return word[:-2] + "is"
    if word.endswith("m") and len(word) >= 3:
        return word[:-1] + "ns"
    if word.endswith(("r", "z")):
        return word + "es"
    if word.endswith("s"):
        return word  # already plural-looking
    return word + "s"


def plural_re(food_norm: str) -> str:
    """Build a regex fragment matching *food_norm* + its Portuguese plural.

    Uses :func:`_pluralize` for morphology, so rules are shared with
    the variant lexicon builder.  Returns a non-capturing group pattern
    that can be embedded in larger regexes.

    >>> plural_re('amendoim')   # -m → -ns
    '(?:amendoim|amendoins)'
    >>> plural_re('acucar')     # -r → -res
    '(?:acucar|acucares)'
    >>> plural_re('ovo')        # regular +s
    '(?:ovo|ovos)'
    """
    esc = re.escape(food_norm)
    pl = re.escape(_pluralize(food_norm))
    if esc == pl:
        return esc          # already looks plural or no change
    return rf"(?:{esc}|{pl})"


def _extract_first_comma_part(official_name: str) -> str:
    """Extract the core food name from the FIRST comma-separated part of
    the TBCA official name.

    TBCA names follow the pattern: "FoodName, qualifier1, qualifier2, ..."
    The first part before the first comma IS the actual food.

    Examples:
        "Banana, prata, crua"           → "banana"
        "Mel, de abelha"                → "mel"
        "Sangue, bovino, cru"           → "sangue"
        "Bem casado, c/ recheio de ..." → "bem casado"
        "Flor de abóbora, à milanesa"   → "flor de abobora"
    """
    s = re.sub(r"\(.*?\)", " ", str(official_name))
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return norm(official_name)
    first = _normalize_part_tokens(parts[0])
    for noise in NOISE_PARTS:
        first = re.sub(rf"\b{re.escape(noise)}\b", " ", first)
    return re.sub(r"\s+", " ", first).strip()


def _is_qualifier_part(part_norm: str) -> bool:
    """True if a TBCA comma-part is purely a qualifier (preparation state,
    processing method, origin, etc.) and should NOT produce standalone or
    compound food-name variants."""
    tokens = part_norm.split()
    if not tokens:
        return True
    # Parts starting with sem/com/sobre/para are recipe qualifiers ("c/ sal")
    if tokens[0] in ("sem", "com", "sobre", "para", "em", "na", "no", "a", "c", "s"):
        return True
    # If ALL tokens are known qualifier words → qualifier
    return all(t in _QUALIFIER_TOKENS for t in tokens)


def _extract_comma_parts(official_name: str) -> list[str]:
    """Split a TBCA official name by commas and normalise each part.

    Parenthetical content is removed first.  Each part is accent-stripped,
    lowercased, and cleaned of noise words.  Returns a list of non-empty
    normalised parts.

    Example:
        "Bebida alcóolica, Amarula, caseira"
        → ["bebida alcoolica", "amarula", "caseira"]
    """
    s = re.sub(r"\(.*?\)", " ", str(official_name))
    raw_parts = [p.strip() for p in s.split(",") if p.strip()]
    parts: list[str] = []
    for p in raw_parts:
        n = _normalize_part_tokens(p)
        for noise in NOISE_PARTS:
            n = re.sub(rf"\b{re.escape(noise)}\b", " ", n)
        n = re.sub(r"\s+", " ", n).strip()
        if n and len(n) >= 2:
            parts.append(n)
    return parts


def _is_dangling(variant: str) -> bool:
    """True if the variant starts or ends with a stopword (preposition, etc.)."""
    toks = variant.split()
    if not toks:
        return True
    # Trailing stopwords: "sopa de", "arroz com"
    if toks[-1] in TRAILING_STOPWORDS:
        return True
    # Leading stopwords: "de frango", "com sal"
    if toks[0] in TRAILING_STOPWORDS:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════

def norm(text: str) -> str:
    """Canonical normalisation: lowercase, strip accents, keep alphanumeric + space."""
    s = str(text).strip().lower()
    # Expand c/ and s/ abbreviations BEFORE stripping non-alphanumeric,
    # otherwise the slash is removed and we end up with bare 'c' or 's'.
    s = re.sub(r"\bc/\s*", "com ", s)
    s = re.sub(r"\bs/\s*", "sem ", s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _normalize_part_tokens(part: str) -> str:
    """Normalise a name part: expand abbreviations, strip accents, remove hyphens."""
    p = str(part)
    # Expand c/ and s/ abbreviations BEFORE norm() strips the slash
    p = re.sub(r"\bc/\s*", "com ", p)
    p = re.sub(r"\bs/\s*", "sem ", p)
    p = norm(p)
    p = p.replace("-", " ")
    return re.sub(r"\s+", " ", p).strip()


# ═══════════════════════════════════════════════════════════════════════════
# TBCA SCRAPING
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_page_text(page: int) -> str:
    """Fetch one listing page from the TBCA website."""
    url = f"{TBCA_BASE_URL}?pagina={page}"
    r = requests.get(url, headers=TBCA_HEADERS, timeout=30)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def _extract_tbca_pairs(html: str) -> list[tuple[str, str]]:
    """Extract (code, food_name) pairs from a TBCA listing page."""
    pairs: list[tuple[str, str]] = []
    soup = BeautifulSoup(html, "html.parser")
    strings = [s.strip() for s in soup.stripped_strings if s and s.strip()]
    code_pat = re.compile(r"^BRC[A-Z0-9]+$")
    seen: set[tuple[str, str]] = set()

    skip_labels = {
        "codigo", "nome", "nome cientifico", "grupo", "marca",
        "buscar", "tbca", "apoio",
    }

    for i, s in enumerate(strings[:-1]):
        if not code_pat.match(s):
            continue
        for j in range(i + 1, min(i + 6, len(strings))):
            nxt = strings[j].strip()
            if code_pat.match(nxt):
                break
            if nxt.lower() in skip_labels:
                continue
            if len(nxt) >= 3:
                key = (s, nxt)
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
                break

    if pairs:
        return pairs

    # Fallback: regex on raw HTML
    html_min = re.sub(r"\s+", " ", html)
    pat = re.compile(
        r">\s*(BRC[A-Z0-9]+)\s*<.*?>\s*([^<>]{3,}?)\s*<",
        flags=re.IGNORECASE,
    )
    for m in pat.finditer(html_min):
        code, name = m.group(1).strip(), m.group(2).strip()
        if code_pat.match(name) or name.lower() in skip_labels:
            continue
        key = (code, name)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    return pairs


def scrape_tbca(max_pages: int = 300) -> pd.DataFrame:
    """Scrape all food entries from TBCA. Returns a DataFrame with
    columns: tbca_code, official_name."""
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for page in tqdm(range(1, max_pages + 1), desc="Scraping TBCA"):
        try:
            html = _fetch_page_text(page)
        except Exception as exc:
            logger.warning("Page %d failed: %s", page, exc)
            continue

        pairs = _extract_tbca_pairs(html)
        if not pairs:
            logger.info("No items on page %d — stopping.", page)
            break

        for code, name in pairs:
            key = (code, name)
            if key not in seen:
                seen.add(key)
                rows.append({"tbca_code": code, "official_name": name})

    if not rows:
        raise RuntimeError(
            "No items scraped from TBCA. Check network / site availability."
        )

    df = pd.DataFrame(rows).drop_duplicates()
    df = df[df["official_name"].str.len() >= 3].copy()
    logger.info("Scraped %d unique TBCA foods.", len(df))
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# NAME PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def build_surface_name(name: str) -> str:
    """Normalise an official TBCA name into a clean surface form."""
    s = str(name).replace("...", " ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = norm(s)
    # Remove noise parts using word boundaries to avoid corrupting
    # words that contain the noise as a substring (e.g. "brasileiro")
    for x in NOISE_PARTS:
        s = re.sub(rf"\b{re.escape(x)}\b", " ", s)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return s.strip()
    return " ".join(_normalize_part_tokens(p) for p in parts).strip()


def build_query_name(surface_name: str) -> str:
    """Strip preparation descriptors from a surface name for matching."""
    s = _normalize_part_tokens(surface_name)
    toks = [t for t in s.split() if t not in QUERY_STOPWORDS]
    while toks and toks[-1] in PREP_STOPWORDS:
        toks.pop()
    return " ".join(toks).strip()


def build_base_food(surface_name: str) -> str:
    """Extract the core food term from a surface name."""
    s = _normalize_part_tokens(surface_name)
    toks = s.split()
    if not toks:
        return s

    # Cut at "com"/"sem" — those introduce preparation modifiers
    for stop in ("com", "sem"):
        if stop in toks:
            toks = toks[: toks.index(stop)]
    if not toks:
        return _normalize_part_tokens(surface_name)

    # If the second token is a preposition, we need the third to form
    # a complete compound ("mel de abelha", "oleo de soja").
    # Taking only 2 would produce dangling fragments like "mel de".
    prepositions = {"de", "do", "da", "dos", "das"}
    if len(toks) >= 3 and toks[1] in prepositions:
        return " ".join(toks[:3])
    if len(toks) >= 2 and toks[1] in prepositions:
        # Only 2 tokens and second is a preposition → just the head word
        return toks[0]
    if len(toks) >= 2:
        return " ".join(toks[:2])
    return toks[0]


def process_food_names(tbca_df: pd.DataFrame) -> pd.DataFrame:
    """Add surface_name, query_name, base_food columns.

    base_food is derived from the FIRST comma-separated part of the official
    TBCA name (not from the surface name), because TBCA uses commas to
    separate the food from qualifiers like cooking method, variety, etc.
    """
    df = tbca_df.copy()
    df["surface_name"] = df["official_name"].map(build_surface_name)
    df["query_name"] = df["surface_name"].map(build_query_name)
    df["base_food"] = df["official_name"].map(_extract_first_comma_part)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# NUTRIENT FETCHING (FROM TBCA PRODUCT PAGES)
# ═══════════════════════════════════════════════════════════════════════════

_NUTRIENT_PATTERNS: dict[str, list[str]] = {
    "carbohydrates_100g": [
        r"total carbohydrates.*?(tr|\d+(?:[.,]\d+)?)",
        r"carbohydrate.*?(tr|\d+(?:[.,]\d+)?)",
        r"carboidrato.*?(tr|\d+(?:[.,]\d+)?)",
    ],
    "fiber_100g": [
        r"fiber.*?(tr|\d+(?:[.,]\d+)?)",
        r"fibra.*?(tr|\d+(?:[.,]\d+)?)",
    ],
    "sodium_100g": [
        r"sodium.*?(tr|\d+(?:[.,]\d+)?)",
        r"s[oó]dio.*?(tr|\d+(?:[.,]\d+)?)",
    ],
    "protein_100g": [
        r"proteins.*?(tr|\d+(?:[.,]\d+)?)",
        r"protein.*?(tr|\d+(?:[.,]\d+)?)",
        r"prote[ií]na.*?(tr|\d+(?:[.,]\d+)?)",
    ],
    "total_fat_100g": [
        r"total lipids.*?(tr|\d+(?:[.,]\d+)?)",
        r"total lipid.*?(tr|\d+(?:[.,]\d+)?)",
        r"lipids.*?(tr|\d+(?:[.,]\d+)?)",
        r"gordura total.*?(tr|\d+(?:[.,]\d+)?)",
        r"gorduras totais.*?(tr|\d+(?:[.,]\d+)?)",
    ],
    "saturated_fat_100g": [
        r"total saturated fatty acids.*?(tr|\d+(?:[.,]\d+)?)",
        r"saturated fatty acids.*?(tr|\d+(?:[.,]\d+)?)",
        r"gordura saturada.*?(tr|\d+(?:[.,]\d+)?)",
    ],
    "sugars_100g": [
        r"added sugar.*?(tr|\d+(?:[.,]\d+)?)",
        r"total sugars.*?(tr|\d+(?:[.,]\d+)?)",
        r"sugars.*?(tr|\d+(?:[.,]\d+)?)",
        r"a[cç][uú]cares.*?(tr|\d+(?:[.,]\d+)?)",
    ],
    "potassium_100g": [
        r"potassium.*?(tr|\d+(?:[.,]\d+)?)",
        r"pot[aá]ssio.*?(tr|\d+(?:[.,]\d+)?)",
    ],
    "phosphorus_100g": [
        r"phosphor.*?(tr|\d+(?:[.,]\d+)?)",
        r"phosphorus.*?(tr|\d+(?:[.,]\d+)?)",
        r"f[oó]sforo.*?(tr|\d+(?:[.,]\d+)?)",
    ],
}

NUTRIENT_COLUMNS = list(_NUTRIENT_PATTERNS.keys())


def _to_float(x: str) -> float | None:
    if x is None:
        return None
    x = str(x).strip().lower()
    if not x:
        return None
    if x == "tr":
        return 0.0
    x = x.replace("\xa0", " ").replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", x)
    return float(m.group(0)) if m else None


def fetch_nutrients_for_code(tbca_code: str) -> dict[str, float | None]:
    """Fetch nutrient values for a single TBCA code from its product page."""
    url = f"{TBCA_NUTRIENT_URL}?cod_produto={tbca_code}"
    nutrients: dict[str, float | None] = {k: None for k in NUTRIENT_COLUMNS}

    try:
        r = requests.get(url, headers=TBCA_HEADERS, timeout=30)
        r.raise_for_status()
    except Exception:
        return nutrients

    text = BeautifulSoup(r.text, "html.parser").get_text("\n", strip=True).lower()

    for key, pats in _NUTRIENT_PATTERNS.items():
        for pat in pats:
            m = re.search(pat, text, flags=re.I | re.S)
            if m:
                nutrients[key] = _to_float(m.group(1))
                break

    return nutrients


def fetch_nutrients_parallel(
    tbca_df: pd.DataFrame,
    max_workers: int = 16,
) -> pd.DataFrame:
    """Fetch nutrient data for all TBCA foods in parallel."""
    records: list[dict] = []
    rows = tbca_df.to_dict("records")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(fetch_nutrients_for_code, row["tbca_code"]): row
            for row in rows
        }
        for future in tqdm(
            as_completed(future_map), total=len(future_map),
            desc="Fetching nutrients",
        ):
            row = future_map[future]
            try:
                nutrients = future.result()
            except Exception:
                nutrients = {k: None for k in NUTRIENT_COLUMNS}
            records.append({**row, **nutrients})

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# ALLERGEN FLAG FETCHING (FROM OPEN FOOD FACTS)
# ═══════════════════════════════════════════════════════════════════════════

def _clean_name_for_off(name: str) -> str:
    """Simplify a TBCA name for Open Food Facts search."""
    s = norm(name)
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\bs/ [a-z]+\b", " ", s)
    s = re.sub(r"\bc/ [a-z]+\b", " ", s)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return " ".join(parts) if parts else s


def _query_off(query: str) -> list[dict]:
    """Query Open Food Facts search API for Brazilian products."""
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "search_terms": query,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 5,
        "countries_tags": "brazil",
    }
    try:
        r = requests.get(url, params=params, headers=OFF_HEADERS, timeout=15)
        r.raise_for_status()
        return r.json().get("products", [])
    except Exception:
        return []


def _extract_off_flags(product: dict) -> dict[str, bool]:
    """Extract allergen / label flags from an OFF product entry."""
    allergens = set(product.get("allergens_tags") or [])
    labels = set(product.get("labels_tags") or [])
    ingredients = norm(product.get("ingredients_text") or "")
    name = norm(
        product.get("product_name") or product.get("product_name_pt") or ""
    )

    def _has(words: list[str]) -> bool:
        return any(w in ingredients or w in name for w in words)

    gluten_free = "en:gluten-free" in labels or _has(
        ["sem gluten", "gluten free"]
    )
    lactose_free = "en:lactose-free" in labels or _has(
        ["sem lactose", "zero lactose"]
    )

    return {
        "gluten_flag": (
            False
            if gluten_free
            else (
                "en:gluten" in allergens
                or _has(["trigo", "cevada", "centeio", "malte"])
            )
        ),
        "lactose_flag": (
            False
            if lactose_free
            else ("en:milk" in allergens or _has(["leite", "queijo", "lactose"]))
        ),
        "egg_flag": "en:eggs" in allergens or _has(["ovo"]),
        "peanut_flag": "en:peanuts" in allergens or _has(["amendoim"]),
        "nuts_flag": "en:nuts" in allergens or _has(
            ["castanha", "noz", "amendoa"]
        ),
    }

ALLERGEN_COLUMNS = [
    "gluten_flag", "lactose_flag", "egg_flag", "peanut_flag", "nuts_flag",
]


def fetch_allergen_flags(
    tbca_df: pd.DataFrame,
    delay: float = 0.2,
) -> pd.DataFrame:
    """Enrich the TBCA DataFrame with allergen flags from Open Food Facts.

    This is slow (~0.2 s per food to respect rate limits).
    Pass the column 'surface_name' or 'official_name' for the query.
    """
    records: list[dict] = []
    null_flags = {k: None for k in ALLERGEN_COLUMNS}

    for row in tqdm(
        tbca_df.to_dict("records"), desc="Fetching allergen flags (OFF)"
    ):
        query = _clean_name_for_off(
            row.get("surface_name") or row.get("official_name", "")
        )
        products = _query_off(query)

        if products:
            flags = _extract_off_flags(products[0])
        else:
            flags = null_flags

        records.append({**row, **flags})
        time.sleep(delay)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# SUBSTANCE FLAG DERIVATION
# ═══════════════════════════════════════════════════════════════════════════

# Maps substance name → callable(row_dict) → bool | None
# None means "unknown" (nutrient data missing); True/False are definitive.

def _flag_added_sugar(r: dict) -> bool | None:
    # ANVISA IN 75/2020 Anexo XV: "ALTO EM" = ≥15g added sugar / 100g
    v = r.get("sugars_100g")
    return None if v is None else v >= 15

def _flag_high_glycemic(r: dict) -> bool | None:
    # High net carbs: flag only foods with ≥40g net carbs per 100g
    # (pure sugar ~100, honey ~80, white bread ~50, rice ~28 → safe)
    carbs = r.get("carbohydrates_100g")
    if carbs is None:
        return None
    fiber = r.get("fiber_100g") or 0
    return (carbs - fiber) >= 40

def _flag_sodium(r: dict) -> bool | None:
    # ANVISA IN 75/2020 Anexo XV: "ALTO EM" = ≥600mg sodium / 100g
    v = r.get("sodium_100g")
    return None if v is None else v >= 600

def _flag_saturated_fat(r: dict) -> bool | None:
    # ANVISA IN 75/2020 Anexo XV: "ALTO EM" = ≥6g sat fat / 100g
    # Anexo XVI exemptions (olive oil, nuts, eggs, etc.) handled separately
    v = r.get("saturated_fat_100g")
    return None if v is None else v >= 6

def _flag_potassium(r: dict) -> bool | None:
    v = r.get("potassium_100g")
    return None if v is None else v > 400

def _flag_phosphorus(r: dict) -> bool | None:
    v = r.get("phosphorus_100g")
    return None if v is None else v > 300

def _flag_gluten(r: dict) -> bool | None:
    return r.get("gluten_flag")   # from OFF; may be None

def _flag_lactose(r: dict) -> bool | None:
    return r.get("lactose_flag")

def _flag_peanut(r: dict) -> bool | None:
    return r.get("peanut_flag")

def _flag_nut(r: dict) -> bool | None:
    return r.get("nuts_flag")

def _flag_egg(r: dict) -> bool | None:
    return r.get("egg_flag")

def _flag_alcohol(r: dict) -> bool | None:
    return r.get("alcohol_flag")

def _flag_caffeine(r: dict) -> bool | None:
    return r.get("caffeine_flag")


SUBSTANCE_FLAG_FNS: dict[str, Any] = {
    "added_sugar":   _flag_added_sugar,
    "high_glycemic": _flag_high_glycemic,
    "sodium":        _flag_sodium,
    "saturated_fat": _flag_saturated_fat,
    "potassium":     _flag_potassium,
    "phosphorus":    _flag_phosphorus,
    "gluten":        _flag_gluten,
    "lactose":       _flag_lactose,
    "peanut":        _flag_peanut,
    "nut":           _flag_nut,
    "egg":           _flag_egg,
    "alcohol":       _flag_alcohol,
    "caffeine":      _flag_caffeine,
}

# Substances that cannot be derived from available data (extend later)
UNDERIVABLE_SUBSTANCES = {
    "uric_acid",   # purine serves as proxy for uric_acid via hyperuricemia
}

# ── External substance flags (USDA + clinical KB) ──────────────────────
try:
    from external_substances import EXTERNAL_FLAG_FNS as _EXT_FLAGS
    SUBSTANCE_FLAG_FNS.update(_EXT_FLAGS)
except ImportError:
    logger.warning("external_substances not available — skipping external flags")
    _EXT_FLAGS = {}

# ─── Name-based allergen/substance heuristics ────────────────────────────
# When Open Food Facts returns no match (flag = None), we fall back to
# the TBCA official name itself.  Crucially, "s/ lactose" and "s/ glúten"
# in the name explicitly OVERRIDE to False.

_LACTOSE_POSITIVE_KW = re.compile(
    # Match only foods that are INHERENTLY dairy-based.
    # Exclude "leite de coco", "leite de soja" etc. via negative lookahead.
    r"\b(com lactose|leite(?! de (?:coco|soja|amendoa|aveia|arroz|amend))"
    r"|queijo|iogurte|requeijao|nata"
    r"|creme de leite|manteiga|coalhada|ricota"
    r"|mucarela|mussarela|provolone|parmesao|gorgonzola"
    r"|brie|cheddar|gouda|cottage|mascarpone"
    r"|cream cheese|chantilly)\b"
)
_LACTOSE_NEGATIVE_KW = re.compile(
    r"\b(sem lactose|zero lactose)\b"
)
_GLUTEN_POSITIVE_KW = re.compile(
    # Only match actual gluten-containing GRAINS, not product types that
    # *might* be made from wheat.  "pão sem glúten" exists, so "pão" alone
    # is not evidence of gluten.
    r"\b(com gluten|com glutem|trigo|cevada|centeio|malte|farinha de trigo|"
    r"farinha de centeio|farinha de cevada)\b"
)
_GLUTEN_NEGATIVE_KW = re.compile(
    r"\b(sem gluten|sem glutem)\b"
)
_ALCOHOL_KW = re.compile(
    r"\b(alcoolica|alcoolico|cerveja|vinho|vodka|rum|whisky|"
    r"cachaca|aguardente|amarula|licor|champanhe|espumante|"
    r"conhaque|gim|tequila|sake|sidra|sangria|caipirinha)\b"
)
_CAFFEINE_KW = re.compile(
    # "café" alone matches "café da manhã" (breakfast) → FP.
    # Require café to NOT be followed by " da manha".
    r"\b(cafe(?! da manha)|cafeina|espresso|cappuccino|"
    r"energetico|energetica|cha preto|cha mate|guarana)\b"
)
_PEANUT_KW = re.compile(
    r"\b(amendoim|pasta de amendoim|manteiga de amendoim)\b"
)
_NUT_KW = re.compile(
    r"\b(castanha|noz|nozes|macadamia|pistache|amandoa|amendoa|"
    r"avel[ãa]|pecã|pecan)\b"
)
_EGG_KW = re.compile(
    r"\b(ovo|ovos|omelete|albumina|gema|clara de ovo)\b"
)


def _apply_name_based_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in None allergen flags using heuristics from the food name.

    Rules:
      • If name says "s/ lactose" or "sem lactose" → lactose_flag = False
      • Elif name mentions dairy keywords → lactose_flag = True (if still None)
      • If name says "s/ glúten" or "sem glúten" → gluten_flag = False
      • Elif name mentions wheat/gluten keywords → gluten_flag = True (if still None)
      • If name mentions alcohol keywords → tag 'alcohol' (via a new column)
      • If name mentions caffeine keywords → tag 'caffeine'

    Only fills None/NaN values — never overwrites definitive True/False from OFF.
    """
    result = df.copy()

    # Ensure flag columns exist
    for col in ("alcohol_flag", "caffeine_flag", "peanut_flag", "nuts_flag", "egg_flag"):
        if col not in result.columns:
            result[col] = None

    for idx, row in result.iterrows():
        name_n = norm(row.get("official_name", ""))

        # ── Lactose ──
        if pd.isna(row.get("lactose_flag")):
            if _LACTOSE_NEGATIVE_KW.search(name_n):
                result.at[idx, "lactose_flag"] = False
            elif _LACTOSE_POSITIVE_KW.search(name_n):
                result.at[idx, "lactose_flag"] = True

        # ── Gluten ──
        if pd.isna(row.get("gluten_flag")):
            if _GLUTEN_NEGATIVE_KW.search(name_n):
                result.at[idx, "gluten_flag"] = False
            elif _GLUTEN_POSITIVE_KW.search(name_n):
                result.at[idx, "gluten_flag"] = True

        # ── Peanut ──
        if pd.isna(row.get("peanut_flag")) and _PEANUT_KW.search(name_n):
            result.at[idx, "peanut_flag"] = True

        # ── Nuts (tree nuts) ──
        if pd.isna(row.get("nuts_flag")) and _NUT_KW.search(name_n):
            result.at[idx, "nuts_flag"] = True

        # ── Egg ──
        if pd.isna(row.get("egg_flag")) and _EGG_KW.search(name_n):
            result.at[idx, "egg_flag"] = True

        # ── Alcohol ──
        if pd.isna(row.get("alcohol_flag")) and _ALCOHOL_KW.search(name_n):
            result.at[idx, "alcohol_flag"] = True

        # ── Caffeine ──
        if pd.isna(row.get("caffeine_flag")) and _CAFFEINE_KW.search(name_n):
            result.at[idx, "caffeine_flag"] = True

    return result


def derive_substance_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'substances' column: list[str] of substance names that the food
    involves (only those where the flag is True).

    Unknown (None) flags are treated as *not* involving the substance
    (conservative = fewer false positives for restriction).

    Note: ANVISA IN 75/2020 Anexo XVI exemptions for saturated_fat and
    sodium are applied based on the **base_food head word** (the first
    word of the base_food column), NOT the full official_name.  This
    avoids wrongly exempting compound foods like "bolinho de bacalhau"
    just because "bacalhau" appears in the description — the head word
    "bolinho" is not a natural food, so the exemption does not apply.
    """
    result = df.copy()

    # ANVISA Anexo XVI: natural food categories exempt from sat_fat/sodium flags.
    # Checked against base_food HEAD WORD only (first word).
    #
    # Cholesterol exemption rationale (AHA/ACC 2019):
    #   • Dietary cholesterol from natural foods (fish, eggs, lean meat) does
    #     not significantly raise serum cholesterol in most individuals.
    #   • Fish is *recommended* ≥2 servings/week for cardiovascular health
    #     (omega-3 benefits outweigh dietary cholesterol).
    #   • USDA lookup returns ONE median cholesterol value per base_food,
    #     applied uniformly to all variants — chicken gets the same 62 mg as
    #     fatty beef cuts, which is incorrect.  Exempting cholesterol for the
    #     whole natural-food set avoids these false positives.
    _ANVISA_EXEMPT_HEADS = frozenset({
        "azeite", "oleo",                              # oils
        "amendoim", "castanha", "noz", "macadamia",    # nuts/peanuts
        "pistache", "amandoa", "avela",                 # nuts cont.
        "semente", "gergelim", "linhaca", "chia", "girassol",  # seeds
        "carne", "frango", "porco", "cordeiro",         # fresh meat
        "peixe", "salmao", "atum", "bacalhau",          # fish
        "ovo", "omelete",                               # eggs
        "leite", "queijo", "iogurte", "manteiga",       # dairy
        "caju", "coco", "abacate",                      # naturally fatty
    })
    _ANVISA_EXEMPT_SUBSTANCES = {"saturated_fat", "sodium", "cholesterol"}

    # ── Hard data corrections ─────────────────────────────────────
    # USDA lookup returned incorrect cholesterol for plant-derived foods.
    # Honey is 100 % plant-based → 0 mg cholesterol (USDA FDC NDB 19296).
    # Plain bread (pão francês) has 0 mg cholesterol unless enriched with
    # eggs/butter, but the USDA median includes enriched variants.
    # Checked against head word (first word of base_food) so "pao frances",
    # "pao de alho", etc. are all covered.
    _CHOLESTEROL_ZERO_HEADS: frozenset[str] = frozenset({
        "mel",            # honey — plant product, 0 mg cholesterol
        "pao",            # plain bread — 0 mg unless enriched
    })

    def _compute_substances(row: dict) -> list[str]:
        subs: list[str] = []
        # Exempt check: only the HEAD word of base_food (first word)
        bf = norm(str(row.get("base_food", "")))
        head_word = bf.split()[0] if bf else ""
        is_natural = head_word in _ANVISA_EXEMPT_HEADS

        # Hard override: zero out cholesterol for known-incorrect entries
        force_no_cholesterol = head_word in _CHOLESTEROL_ZERO_HEADS

        for name, fn in SUBSTANCE_FLAG_FNS.items():
            # Skip sat_fat/sodium/cholesterol for natural foods (ANVISA Anexo XVI)
            if is_natural and name in _ANVISA_EXEMPT_SUBSTANCES:
                continue
            # Skip cholesterol for plant-derived foods with wrong USDA data
            if force_no_cholesterol and name == "cholesterol":
                continue
            val = fn(row)
            if val is True:
                subs.append(name)
        return subs

    records = result.to_dict("records")
    result["substances"] = [_compute_substances(r) for r in records]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-CATEGORY FOOD WORDS
# ═══════════════════════════════════════════════════════════════════════════

def _build_cross_category_foods(df: pd.DataFrame, min_categories: int = 2) -> frozenset[str]:
    """Identify single words that appear across multiple TBCA food categories.

    A word is considered a real standalone food if it appears as:
      (a) a 2nd+ comma-part in entries from ≥ *min_categories* distinct
          first-part categories, OR
      (b) an ingredient inside a compound first-part (e.g., "arroz com
          **frango**") across ≥ *min_categories* different lead categories.

    Words in ``_QUALIFIER_TOKENS`` or ``_NON_STANDALONE_WORDS`` are excluded.

    Returns a frozenset of normalised single words (e.g., "frango",
    "salmao", "castanha") that may safely be emitted as standalone variants.
    """
    from collections import defaultdict
    word_cats: dict[str, set[str]] = defaultdict(set)

    for _, row in df.iterrows():
        official = row["official_name"]
        fp_full = norm(_extract_first_comma_part(official))
        lead = fp_full.split()[0] if fp_full else ""

        # (a) 2nd+ comma-parts
        parts = _extract_comma_parts(official)
        if len(parts) >= 2:
            for i in range(1, len(parts)):
                p = parts[i]
                if not p or _is_qualifier_part(p) or " " in p:
                    continue
                word_cats[p].add(lead)

        # (b) Ingredients embedded in compound first-parts
        tokens = fp_full.split()
        for i, t in enumerate(tokens):
            if t in ("com", "de", "do", "da") and i + 1 < len(tokens) and i > 0:
                ingredient = tokens[i + 1]
                category = tokens[0]
                if len(ingredient) >= 3:
                    word_cats[ingredient].add(category)

    # Filter: keep only words that cross the threshold and are not modifiers
    blocked = _QUALIFIER_TOKENS | _NON_STANDALONE_WORDS
    result = frozenset(
        w for w, cats in word_cats.items()
        if len(cats) >= min_categories and w not in blocked
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# VARIANT LEXICON GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def build_variant_lexicon(df: pd.DataFrame) -> pd.DataFrame:
    """From the enriched TBCA DataFrame, generate all surface-form variants
    for Aho-Corasick matching. Each row is one (food, variant) pair."""

    # Pre-compute the set of single words that may appear standalone.
    cross_cat_foods = _build_cross_category_foods(df)
    logger.info("Cross-category standalone foods: %d words", len(cross_cat_foods))

    rows: list[dict] = []

    for rec in df.to_dict("records"):
        code = rec["tbca_code"]
        official = rec["official_name"]
        surface = rec.get("surface_name", "")
        query = rec.get("query_name", "")
        base = rec.get("base_food", "")
        substances = rec.get("substances", [])

        variants: set[str] = set()
        for name in (surface, query, base):
            if name:
                variants.add(_normalize_part_tokens(name))

        # ── Comma-part variants ──
        # TBCA names follow "Category, Specific/Variety, Qualifier, ..."
        # Three tiers of comma-part handling:
        #
        #  1. PREPARATION qualifiers (_QUALIFIER_TOKENS): skip entirely.
        #     "cru", "cozido", "frito" etc. are cooking methods, not
        #     food identifiers.
        #
        #  2. DESCRIPTIVE modifiers (_NON_STANDALONE_WORDS): skip standalone
        #     but generate compounds.  "branco" alone is not a food, but
        #     "arroz branco" and "feijão branco" are valid identifiers.
        #
        #  3. FOOD words: standalone (if multi-word or in cross_cat_foods)
        #     AND compounds.
        comma_parts = _extract_comma_parts(official)
        if len(comma_parts) >= 2:
            for i in range(1, len(comma_parts)):
                p = comma_parts[i]
                if not p:
                    continue

                # Tier 1: preparation qualifier → skip everything
                if _is_qualifier_part(p):
                    continue

                # Tier 2 & 3: standalone emission
                is_non_standalone = (
                    " " not in p and p in _NON_STANDALONE_WORDS
                )
                if not is_non_standalone:
                    if " " in p:
                        variants.add(p)
                    elif p in cross_cat_foods:
                        variants.add(p)

                # Tier 2 & 3: compound emission (always, for both
                # descriptive modifiers and food words)
                if comma_parts[0]:
                    variants.add(f"{comma_parts[0]} {p}")
                    # With "de" preposition ("bolo de banana",
                    # "leite de cabra", "suco de acai").
                    # Skip if it would produce "de de", "de com", "de sem"
                    # or if the category already ends with "com"/"sem".
                    p_first = p.split()[0] if p.split() else ""
                    cat_last = comma_parts[0].split()[-1] if comma_parts[0].split() else ""
                    if p_first not in ("de", "do", "da", "dos", "das", "com", "sem") \
                       and cat_last not in ("com", "sem"):
                        variants.add(f"{comma_parts[0]} de {p}")

        # ── Singular / plural expansion for single-word variants ──
        # Ensures "fruta" and "frutas", "cereal" and "cereais", etc.
        # are both in the automaton so either form in text matches.
        expanded: set[str] = set()
        for v in variants:
            expanded.add(v)
            if " " not in v and len(v) >= 3:
                sg = _singularize(v)
                pl = _pluralize(v)
                if sg != v and len(sg) >= 3:
                    expanded.add(sg)
                if pl != v and len(pl) >= 3:
                    expanded.add(pl)
        variants = expanded

        # ── Post-filter ──
        # Remove variants that are:
        #   a) dangling (start/end with preposition/article),
        #   b) single-word _NON_STANDALONE (catches base_food/surface_name
        #      variants like "cha" or "suco" that slipped through above).
        for variant in variants:
            variant = re.sub(r"\s+", " ", variant).strip()
            if not variant or len(variant) < 3:
                continue
            if _is_dangling(variant):
                continue
            if " " not in variant and variant in _NON_STANDALONE_WORDS:
                continue
            rows.append({
                "tbca_code": code,
                "official_name": official,
                "surface_name": surface,
                "base_food": base,
                "variant": variant,
                "variant_norm": _normalize_part_tokens(variant),
                "substances": substances,
            })

    result = pd.DataFrame(rows)

    # Per-entry dedup: same code + same variant
    result = result.drop_duplicates(subset=["tbca_code", "variant_norm"])

    # Global dedup: when multiple TBCA entries share the same variant_norm
    # (e.g. 86 entries all produce variant "abobrinha"), we need ONE row
    # with the most REPRESENTATIVE substance flags.
    #
    # Strategy: MAJORITY VOTE among PRIMARY entries only.
    #
    # A "primary" entry is one where the variant comes from the food's
    # base_food (first comma part), not from a sub-ingredient in a
    # compound dish.  E.g. for variant "azeitona":
    #   PRIMARY:   "Azeitona, preta, conserva" (base_food starts with azeitona)
    #   SECONDARY: "Salada, brócolis, queijo, azeitona" (azeitona is sub-ingredient)
    #
    # If primary entries exist, we vote ONLY among them.
    # If not (only secondary), we vote among all entries.
    #
    # A substance is kept if >50% of the voting entries have it.
    from collections import Counter as _Counter
    variant_groups = result.groupby("variant_norm")
    variant_substances: dict[str, list[str]] = {}

    for vn, group in variant_groups:
        # Identify primary entries (base_food starts with or matches the variant)
        vn_first = vn.split()[0] if vn else vn
        primary_mask = group["base_food"].apply(
            lambda bf: isinstance(bf, str) and (
                bf == vn or bf.startswith(vn_first)
            )
        )
        voters = group[primary_mask] if primary_mask.any() else group
        n = len(voters)

        sub_counts = _Counter()
        for subs in voters["substances"]:
            if isinstance(subs, list):
                for s in subs:
                    sub_counts[s] += 1
        # Keep substances present in > 50% of voting entries
        majority_subs = [s for s, c in sub_counts.items() if c > n * 0.5]
        variant_substances[vn] = majority_subs

    # Dedup: keep first row per variant_norm, then replace its substances
    result = (
        result
        .drop_duplicates(subset=["variant_norm"], keep="first")
        .reset_index(drop=True)
    )
    result["substances"] = result["variant_norm"].map(variant_substances)

    return result


def build_productive_variants(lexicon_df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic compound variants (e.g. 'leite de soja',
    'iogurte de coco') that are common in Brazilian Portuguese but
    may not appear as separate TBCA entries.

    Synthetic variants have empty substance lists (cannot determine
    nutritional content from the name alone). Longest-match in the
    automaton ensures these take priority over their base components.
    """
    existing = set(lexicon_df["variant_norm"].dropna().astype(str).map(norm))
    bases = existing.copy()
    rows: list[dict] = []

    def _add(phrase: str) -> None:
        n = norm(phrase)
        if n and n not in existing:
            rows.append({
                "tbca_code": None,
                "official_name": None,
                "surface_name": phrase,
                "base_food": phrase,
                "variant": phrase,
                "variant_norm": n,
                "substances": [],  # unknown — conservative
            })
            existing.add(n)

    nut_seeds = {
        "amendoim", "amendoa", "castanha", "noz", "chia",
        "abobora", "soja", "coco", "arroz", "aveia", "quinoa", "canola",
    } & bases

    fruits = {
        "banana", "manga", "mirtilo", "abacate", "damasco", "cranberry",
    } & bases

    cereals = {"aveia", "quinoa", "arroz", "milho"} & bases

    # Compound patterns
    for x in nut_seeds:
        _add(f"leite de {x}")
    for x in {"soja", "coco"} & nut_seeds:
        _add(f"iogurte de {x}")
    for x in {"amendoim", "abacate"} & (nut_seeds | fruits):
        _add(f"manteiga de {x}")
    for x in {"chia", "abobora"} & bases:
        _add(f"semente de {x}")
    for x in cereals:
        _add(f"{x} em flocos")
    for x in {"canola", "soja"} & nut_seeds:
        _add(f"oleo de {x}")

    # Descriptive food terms
    for phrase in [
        "cafe preto", "cha herbal", "cha preto",
        "cha de camomila", "cha de erva doce", "cha de hibisco",
        "cha de hortelã", "cha de gengibre",
        "suco natural", "suco de fruta",
        "carne magra", "peixe branco", "arroz branco",
        "oleo de oliva", "azeite de oliva",
    ]:
        _add(phrase)

    return pd.DataFrame(rows).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# AHO-CORASICK AUTOMATON
# ═══════════════════════════════════════════════════════════════════════════

def build_automaton(lexicon_df: pd.DataFrame):
    """Build an Aho-Corasick automaton from the variant lexicon.

    Each entry in the automaton maps a normalised variant string to a
    payload dict with:
      - matched_norm   : the normalised variant
      - canonical      : base_food or surface_name (for display)
      - tbca_code      : TBCA code (may be None for synthetics)
      - substances     : frozenset of substance strings
    """
    if ahocorasick is None:
        raise ImportError(
            "pyahocorasick is required. Install: pip install pyahocorasick"
        )

    # ── Pre-pass: collect all known base_food names ──────────────────
    # Used to fix misassignments where a standalone variant (e.g. "noz")
    # gets canonical from a compound food ("bolo") instead of the actual
    # food it names.  E.g. TBCA "Bolo, trigo, com recheio, nozes" yields
    # variant "nozes" with base_food="bolo", but "noz" is a real
    # base_food from "Noz, crua".  We remap to the correct one.
    known_base_foods: set[str] = set()
    for bf in lexicon_df["base_food"].dropna().unique():
        if isinstance(bf, str) and bf:
            known_base_foods.add(bf)

    # Lever 2a: Pre-compute base_food → INTERSECTION of substance flags.
    # When we remap a variant's canonical, we also need its correct
    # substance flags (not the ones from the wrong compound food).
    # Uses INTERSECTION (same strategy as the automaton's merge) across
    # all primary entries.  This ensures only substances common to ALL
    # variants of the food are kept — e.g. "noz crua" has no gluten
    # but "noz pecan" does → intersection drops gluten for bare "noz".
    _base_food_subs: dict[str, frozenset[str]] = {}
    for bf, grp in lexicon_df.groupby("base_food"):
        if not isinstance(bf, str) or not bf:
            continue
        # Only use entries whose variant starts with the base_food word
        bf_first = bf.split()[0]
        primary = grp[grp["variant_norm"].apply(
            lambda v: isinstance(v, str) and v.startswith(bf_first)
        )]
        voters = primary if len(primary) > 0 else grp
        # Intersection of all substance sets
        result_subs: frozenset[str] | None = None
        for subs in voters["substances"]:
            if isinstance(subs, list):
                s = frozenset(subs)
                result_subs = s if result_subs is None else (result_subs & s)
        _base_food_subs[bf] = result_subs if result_subs else frozenset()

    A = ahocorasick.Automaton()

    # Count removals for logging
    _n_blocked = 0
    _n_canonical_fixed = 0
    _n_substance_fixed = 0
    _n_herb_fixed = 0

    for rec in lexicon_df.to_dict("records"):
        term = rec.get("variant_norm")
        if not term or not isinstance(term, str):
            continue

        # ── Block non-food tokens entirely ──────────────────────────
        if term in _NON_FOOD_TOKENS:
            _n_blocked += 1
            continue

        substances_raw = rec.get("substances", [])
        if isinstance(substances_raw, str):
            # Handle parquet round-trip (stored as string repr)
            import ast
            try:
                substances_raw = ast.literal_eval(substances_raw)
            except Exception:
                substances_raw = []

        new_substances = frozenset(substances_raw)

        # If this variant already exists in the automaton, MERGE substances
        # via INTERSECTION rather than union.  This ensures that only
        # substances common across all TBCA entries are kept.  E.g.
        # "abobrinha" has 86 entries: only the milanesa variant has
        # gluten, so intersection correctly drops it.
        if term in A:
            existing = A.get(term)
            new_substances = existing["substances"] & new_substances

        # Determine canonical name
        canonical = rec.get("base_food") or rec.get("surface_name") or term

        # ── Lever 1: Fix standalone variant → wrong canonical ───────
        # When a standalone variant (no spaces) or its singular/plural
        # form IS a known base_food, but the current canonical comes
        # from a compound food, remap to the correct base_food.
        # E.g. variant "nozes" with canonical "bolo" → remap to "noz"
        # because base_food "noz" exists (from "Noz, crua").
        if " " not in term and canonical != term:
            # Check: is the term (or its singular) a known base_food?
            term_sg = _singularize(term)
            correct_bf = None
            if term in known_base_foods:
                correct_bf = term
            elif term_sg in known_base_foods:
                correct_bf = term_sg
            if correct_bf:
                canonical = correct_bf
                # Lever 2a: Also fix substance flags to match the
                # correct base_food's majority-voted substances.
                if correct_bf in _base_food_subs:
                    new_substances = _base_food_subs[correct_bf]

        payload = {
            "matched_norm": term,
            "canonical": canonical,
            "tbca_code": rec.get("tbca_code"),
            "substances": new_substances,
            "base_food": rec.get("base_food", ""),
        }

        # ── Apply canonical overrides ───────────────────────────────
        if term in _CANONICAL_OVERRIDES:
            payload["canonical"] = _CANONICAL_OVERRIDES[term]
            _n_canonical_fixed += 1

        # ── Apply substance overrides ───────────────────────────────
        if term in _SUBSTANCE_OVERRIDES:
            payload["substances"] = _SUBSTANCE_OVERRIDES[term]
            _n_substance_fixed += 1

        # ── Herb/spice whitelist: clear all restrictions ────────────
        if term in _HERB_SPICE_TOKENS:
            payload["substances"] = frozenset()
            _n_herb_fixed += 1

        A.add_word(term, payload)

    # ── Lever 3: Inject clinical substance flags for standalone foods ──
    # Foods like "atum", "salmao", "cavala" may only appear as variants
    # of compound TBCA entries (e.g. "cuscuz de atum") and thus lack
    # purine/fodmap flags.  If a standalone term (no spaces) is in the
    # clinical KB sets, inject the missing substance flag.
    try:
        from external_substances import _HIGH_PURINE_HEADS, _HIGH_FODMAP_HEADS
    except ImportError:
        _HIGH_PURINE_HEADS = frozenset()
        _HIGH_FODMAP_HEADS = frozenset()

    for term in list(_HIGH_PURINE_HEADS | _HIGH_FODMAP_HEADS):
        if " " in term:
            continue
        if term in A:
            existing = A.get(term)
            subs = set(existing["substances"])
            changed = False
            if term in _HIGH_PURINE_HEADS and "purine" not in subs:
                subs.add("purine")
                changed = True
            if term in _HIGH_FODMAP_HEADS and "fodmap" not in subs:
                subs.add("fodmap")
                changed = True
            if changed:
                existing["substances"] = frozenset(subs)
                # Fix canonical if it was inherited from a compound
                # (e.g. "atum" canonical="cuscuz" from "cuscuz de atum")
                if existing["canonical"] != term and term not in known_base_foods:
                    existing["canonical"] = term
                A.add_word(term, existing)
        else:
            # Term not in automaton at all — add it as a standalone entry
            subs: set[str] = set()
            if term in _HIGH_PURINE_HEADS:
                subs.add("purine")
            if term in _HIGH_FODMAP_HEADS:
                subs.add("fodmap")
            A.add_word(term, {
                "matched_norm": term,
                "canonical": term,
                "tbca_code": None,
                "substances": frozenset(subs),
            })

    A.make_automaton()

    # ── Lever 4: Inject custom food entries not in any data source ─────
    # Common foods mentioned by LLMs that don't appear in TBCA/FoodOn.
    _n_custom = 0
    for term, subs in _CUSTOM_FOODS.items():
        if term not in A:
            A.add_word(term, {
                "matched_norm": term,
                "canonical": term,
                "tbca_code": None,
                "substances": subs,
                "base_food": term,
            })
            _n_custom += 1
        else:
            # Overwrite incomplete substances from Lever 3
            # Also fix base_food so variant-collision guard passes
            existing = A.get(term)
            changed = False
            if existing["substances"] != subs:
                existing["substances"] = subs
                changed = True
            if existing.get("base_food") != term:
                existing["base_food"] = term
                changed = True
            if changed:
                A.add_word(term, existing)
                _n_custom += 1
    if _n_custom > 0:
        A.make_automaton()
        logger.info("Lever 4: injected/updated %d custom food entries.", _n_custom)

    # ── Final pass: apply substance overrides to any entries that were
    # injected by Lever 3 (purine/fodmap) or Lever 4 after the main loop
    # This ensures tokens like "cerveja" get their full substance set
    # even if they weren't in the original lexicon_df.
    _n_post_override = 0
    for term, override_subs in _SUBSTANCE_OVERRIDES.items():
        if term in A:
            existing = A.get(term)
            if existing["substances"] != override_subs:
                existing["substances"] = override_subs
                A.add_word(term, existing)
                _n_post_override += 1
    if _n_post_override > 0:
        A.make_automaton()

    logger.info(
        "Automaton built: %d entries (blocked=%d, canonical_fixed=%d, "
        "substance_fixed=%d, herb_cleared=%d).",
        len(A), _n_blocked, _n_canonical_fixed,
        _n_substance_fixed, _n_herb_fixed,
    )
    return A


# ═══════════════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════

def save_lexicon(
    tbca_df: pd.DataFrame,
    lexicon_df: pd.DataFrame,
    automaton,
    output_dir: Path = DATA_DIR,
) -> None:
    """Persist all artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert substance lists to strings for parquet compatibility
    tbca_save = tbca_df.copy()
    if "substances" in tbca_save.columns:
        tbca_save["substances"] = tbca_save["substances"].apply(str)
    tbca_save.to_parquet(output_dir / "tbca_foods.parquet", index=False)

    lex_save = lexicon_df.copy()
    if "substances" in lex_save.columns:
        lex_save["substances"] = lex_save["substances"].apply(str)
    lex_save.to_parquet(output_dir / "lexicon.parquet", index=False)

    with open(output_dir / "automaton.pkl", "wb") as f:
        pickle.dump(automaton, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        "Saved: %d TBCA foods, %d lexicon variants, automaton → %s",
        len(tbca_df), len(lexicon_df), output_dir,
    )


def load_lexicon(
    output_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Load lexicon artifacts from disk.

    Returns (tbca_df, lexicon_df, automaton).
    """
    import ast

    tbca_df = pd.read_parquet(output_dir / "tbca_foods.parquet")
    if "substances" in tbca_df.columns:
        tbca_df["substances"] = tbca_df["substances"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    lexicon_df = pd.read_parquet(output_dir / "lexicon.parquet")
    if "substances" in lexicon_df.columns:
        lexicon_df["substances"] = lexicon_df["substances"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    with open(output_dir / "automaton.pkl", "rb") as f:
        automaton = pickle.load(f)

    # ── Apply substance overrides at load time ──────────────────────
    # The pickled automaton may be stale (built before latest overrides).
    # Re-apply _SUBSTANCE_OVERRIDES so code-level fixes take effect
    # immediately without requiring a full automaton rebuild.
    #
    # Phase 0: inject custom food entries not in any data source.
    _n_patched = 0
    for term, subs in _CUSTOM_FOODS.items():
        if term not in automaton:
            automaton.add_word(term, {
                "matched_norm": term,
                "canonical": term,
                "tbca_code": None,
                "substances": subs,
                "base_food": term,
            })
            _n_patched += 1
        else:
            # Term may exist from Lever 3 (purine/fodmap) with incomplete
            # substances — always overwrite with the full custom set.
            existing = automaton.get(term)
            changed = False
            if existing["substances"] != subs:
                existing["substances"] = subs
                changed = True
            if existing.get("base_food") != term:
                existing["base_food"] = term
                changed = True
            if changed:
                automaton.add_word(term, existing)
                _n_patched += 1

    # Phase 1: patch base terms, their plurals, and record removals.
    _removed_by_base: dict[str, frozenset] = {}  # base_term → removed substances
    for term, override_subs in _SUBSTANCE_OVERRIDES.items():
        if term in automaton:
            existing = automaton.get(term)
            old_subs = existing["substances"]
            removed = old_subs - override_subs
            if removed:
                _removed_by_base[term] = removed
            if old_subs != override_subs:
                existing["substances"] = override_subs
                automaton.add_word(term, existing)
                _n_patched += 1
        # Also patch common plural forms not explicitly in overrides
        if not term.endswith("s"):
            _plural_candidates = [term + "s"]
        else:
            _plural_candidates = []
        if term.endswith("ao"):
            _plural_candidates.extend([term[:-2] + "oes", term[:-2] + "aes"])
        if term.endswith("al"):
            _plural_candidates.append(term[:-2] + "ais")
        if term.endswith("el"):
            _plural_candidates.append(term[:-2] + "eis")
        for pl in _plural_candidates:
            if pl in automaton and pl not in _SUBSTANCE_OVERRIDES:
                pl_entry = automaton.get(pl)
                if pl_entry["substances"] != override_subs:
                    pl_entry["substances"] = override_subs
                    automaton.add_word(pl, pl_entry)
                    _n_patched += 1

    # Phase 2: propagate removals to compound entries.
    # E.g. if "iogurte" override removed "gluten", then "iogurte simples"
    # (which inherited "gluten" from the same TBCA base) also loses it.
    # Compounds with genuinely added substances (e.g. "iogurte com granola"
    # → nut from granola) are handled by an exception list.
    _COMPOUND_KEEP_EXCEPTIONS: dict[str, dict[str, set[str]]] = {
        # If compound key contains any of these tokens, KEEP these substances
        "iogurte": {
            "nut": {"granola", "castanha", "nozes", "amendoa", "amendoim"},
            "gluten": {"granola", "aveia", "flocos", "cereal"},
        },
        "frozen": {
            "nut": {"castanha", "nozes", "amendoa", "amendoim"},
        },
    }
    if _removed_by_base:
        for key, payload in automaton.items():
            # Skip entries already handled by exact overrides
            if key in _SUBSTANCE_OVERRIDES:
                continue
            for base_term, removed_subs in _removed_by_base.items():
                prefix = base_term + " "
                if not (key.startswith(prefix) or key.startswith(base_term + "s ")):
                    continue
                current_subs = payload.get("substances", frozenset())
                to_remove = set()
                exceptions = _COMPOUND_KEEP_EXCEPTIONS.get(base_term, {})
                for sub in removed_subs:
                    # Check if compound has a token that legitimately adds this substance
                    keep_tokens = exceptions.get(sub, set())
                    if keep_tokens and any(tok in key for tok in keep_tokens):
                        continue  # keep this substance for this compound
                    if sub in current_subs:
                        to_remove.add(sub)
                if to_remove:
                    payload["substances"] = current_subs - to_remove
                    automaton.add_word(key, payload)
                    _n_patched += 1

    # Phase 3: propagate base-override substances to compounds.
    # If the override says "bolo" should have {added_sugar, gluten},
    # then "bolo de chocolate" should also have added_sugar (unless it's
    # a "diet"/"sem acucar" variant).
    _SUBSTANCE_EXCLUDE_TOKENS: dict[str, set[str]] = {
        "added_sugar": {"diet", "dietetico", "light", "zero", "sem acucar"},
        "high_glycemic": {"diet", "dietetico", "light", "zero", "sem acucar"},
        "gluten": {"sem gluten"},
        "lactose": {"sem lactose"},
    }
    _phase3_candidates: list[tuple[str, dict]] = []
    for term, override_subs in _SUBSTANCE_OVERRIDES.items():
        if " " in term:
            continue  # only propagate from single-word base terms
        prefix = term + " "
        prefix_pl = term + "s "
        for key, payload in automaton.items():
            if key in _SUBSTANCE_OVERRIDES:
                continue
            if not (key.startswith(prefix) or key.startswith(prefix_pl)):
                continue
            current_subs = payload.get("substances", frozenset())
            missing = override_subs - current_subs
            if not missing:
                continue
            to_add = set()
            for sub in missing:
                excl = _SUBSTANCE_EXCLUDE_TOKENS.get(sub, set())
                if excl and any(tok in key for tok in excl):
                    continue
                to_add.add(sub)
            if to_add:
                _phase3_candidates.append((key, current_subs | frozenset(to_add)))
    for key, new_subs in _phase3_candidates:
        entry = automaton.get(key)
        entry["substances"] = new_subs
        automaton.add_word(key, entry)
        _n_patched += 1

    # Also apply _NON_FOOD_TOKENS removals and _HERB_SPICE_TOKENS clearing
    for term in _HERB_SPICE_TOKENS:
        if term in automaton:
            existing = automaton.get(term)
            if existing["substances"]:
                existing["substances"] = frozenset()
                automaton.add_word(term, existing)
                _n_patched += 1
    if _n_patched > 0:
        automaton.make_automaton()
        logger.info("Patched %d automaton entries with current overrides.", _n_patched)

    logger.info(
        "Loaded: %d TBCA foods, %d lexicon variants, automaton from %s",
        len(tbca_df), len(lexicon_df), output_dir,
    )
    return tbca_df, lexicon_df, automaton


# ═══════════════════════════════════════════════════════════════════════════
# FULL BUILD PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def build_full_lexicon(
    *,
    max_pages: int = 300,
    max_workers: int = 16,
    fetch_allergens: bool = False,
    off_delay: float = 0.2,
    save: bool = True,
    output_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Run the complete lexicon build pipeline:

    1. Scrape TBCA
    2. Process food names
    3. Fetch nutrient data (parallel)
    4. (Optional) Fetch allergen flags from Open Food Facts
    5. Derive involves_substance flags
    6. Build variant lexicon + productive variants
    7. Build Aho-Corasick automaton
    8. Save to disk

    Returns (tbca_df, lexicon_df, automaton).
    """
    print("=" * 60)
    print("  LEXICON BUILDER — Full Build Pipeline")
    print("=" * 60)

    # Step 1: Scrape
    print("\n[1/7] Scraping TBCA…")
    tbca_df = scrape_tbca(max_pages=max_pages)
    print(f"       → {len(tbca_df)} foods scraped.")

    # Step 2: Names
    print("\n[2/7] Processing food names…")
    tbca_df = process_food_names(tbca_df)

    # Step 3: Nutrients
    print(f"\n[3/7] Fetching nutrients ({max_workers} workers)…")
    tbca_df = fetch_nutrients_parallel(tbca_df, max_workers=max_workers)
    n_with_nutrients = tbca_df[NUTRIENT_COLUMNS].notna().any(axis=1).sum()
    print(f"       → {n_with_nutrients}/{len(tbca_df)} foods with nutrient data.")

    # Step 4: Allergens (optional)
    if fetch_allergens:
        print(f"\n[4/8] Fetching allergen flags from Open Food Facts…")
        tbca_df = fetch_allergen_flags(tbca_df, delay=off_delay)
        n_with_flags = tbca_df[ALLERGEN_COLUMNS].notna().any(axis=1).sum()
        print(f"       → {n_with_flags}/{len(tbca_df)} foods with allergen data.")
    else:
        print("\n[4/8] Skipping allergen flags (use --allergens to enable).")
        for col in ALLERGEN_COLUMNS:
            if col not in tbca_df.columns:
                tbca_df[col] = None

    # Step 5: Name-based flag heuristics (fills None flags from food name)
    print("\n[5/8] Applying name-based substance heuristics…")
    tbca_df = _apply_name_based_flags(tbca_df)
    n_name_filled = (
        tbca_df[["lactose_flag", "gluten_flag"]].notna().any(axis=1).sum()
        - (tbca_df[ALLERGEN_COLUMNS].notna().any(axis=1).sum() if fetch_allergens else 0)
    )
    print(f"       → Filled flags for {n_name_filled} foods from name heuristics.")

    # Step 6: Substance flags
    print("\n[6/8] Deriving substance flags…")
    tbca_df = derive_substance_flags(tbca_df)
    n_flagged = tbca_df["substances"].apply(bool).sum()
    print(f"       → {n_flagged}/{len(tbca_df)} foods with ≥1 substance flag.")

    # Step 7: Variant lexicon
    print("\n[7/8] Building variant lexicon…")
    lexicon_df = build_variant_lexicon(tbca_df)
    productive = build_productive_variants(lexicon_df)
    if not productive.empty:
        lexicon_df = (
            pd.concat([lexicon_df, productive], ignore_index=True)
            .drop_duplicates(subset=["variant_norm"])
            .reset_index(drop=True)
        )
    print(f"       → {len(lexicon_df)} total variants.")

    # Step 8: Automaton
    print("\n[8/8] Building Aho-Corasick automaton…")
    auto = build_automaton(lexicon_df)
    print(f"       → Automaton ready ({len(auto)} patterns).")

    # Save
    if save:
        save_lexicon(tbca_df, lexicon_df, auto, output_dir=output_dir)
        print(f"\n✓ Artifacts saved to {output_dir}/")

    print("\n" + "=" * 60)
    print("  BUILD COMPLETE")
    print("=" * 60)

    return tbca_df, lexicon_df, auto


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Build the food lexicon with substance flags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--build", action="store_true",
        help="Run the full build pipeline (scrape + nutrients + automaton).",
    )
    ap.add_argument(
        "--rebuild", action="store_true",
        help="Re-derive flags and rebuild lexicon+automaton from existing parquet (no scraping).",
    )
    ap.add_argument(
        "--info", action="store_true",
        help="Load and print summary of the saved lexicon.",
    )
    ap.add_argument("--pages", type=int, default=300, help="Max TBCA pages.")
    ap.add_argument("--workers", type=int, default=16, help="Parallel workers.")
    ap.add_argument(
        "--allergens", action="store_true",
        help="Also fetch allergen flags from Open Food Facts (slow).",
    )
    ap.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {DATA_DIR}).",
    )
    ap.add_argument(
        "--no-usda", action="store_true",
        help="Skip USDA API calls during rebuild (use clinical KB only).",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    out = Path(args.output_dir) if args.output_dir else DATA_DIR

    if args.build:
        build_full_lexicon(
            max_pages=args.pages,
            max_workers=args.workers,
            fetch_allergens=args.allergens,
            output_dir=out,
        )

    elif args.rebuild:
        print("[1/6] Loading existing TBCA parquet…")
        tbca_df = pd.read_parquet(out / "tbca_foods.parquet")
        drop_cols = [c for c in ["substances", "cholesterol_mg", "trans_fat_g",
                                 "total_fiber_g", "insoluble_fiber_g",
                                 "purine_flag", "fodmap_flag",
                                 "cholesterol_flag", "trans_fat_flag",
                                 "insoluble_fiber_flag"]
                     if c in tbca_df.columns]
        if drop_cols:
            tbca_df = tbca_df.drop(columns=drop_cols)
        print(f"  {len(tbca_df)} foods")

        print("[2/6] Applying name-based allergen flags…")
        tbca_df = _apply_name_based_flags(tbca_df)

        print("[3/6] Enriching with external substances…")
        try:
            from external_substances import enrich_substances
            use_usda = not getattr(args, 'no_usda', False)
            tbca_df = enrich_substances(tbca_df, use_usda=use_usda)
        except ImportError:
            print("  external_substances.py not found — skipping.")

        print("[4/6] Deriving substance flags…")
        tbca_df = derive_substance_flags(tbca_df)

        print("[5/6] Building variant lexicon…")
        lexicon_df = build_variant_lexicon(tbca_df)
        productive = build_productive_variants(lexicon_df)
        if not productive.empty:
            lexicon_df = (
                pd.concat([lexicon_df, productive], ignore_index=True)
                .drop_duplicates(subset=["variant_norm"])
                .reset_index(drop=True)
            )
        print(f"  {len(lexicon_df)} variants")

        print("[6/6] Building automaton…")
        auto = build_automaton(lexicon_df)
        save_lexicon(tbca_df, lexicon_df, auto, output_dir=out)
        print(f"✓ Done. {len(auto)} automaton entries saved to {out}/")

    elif args.info:
        tbca_df, lexicon_df, automaton = load_lexicon(out)
        print(f"TBCA foods     : {len(tbca_df)}")
        print(f"Lexicon variants: {len(lexicon_df)}")
        print(f"Automaton size : {len(automaton)} patterns")
        print(f"\nNutrient coverage:")
        for col in NUTRIENT_COLUMNS:
            pct = tbca_df[col].notna().mean() * 100
            print(f"  {col:25s}: {pct:5.1f}%")
        print(f"\nSubstance distribution:")
        from collections import Counter
        ctr = Counter()
        for subs in tbca_df["substances"]:
            for s in subs:
                ctr[s] += 1
        for sub, cnt in ctr.most_common():
            print(f"  {sub:20s}: {cnt:5d} foods")

    else:
        ap.print_help()
