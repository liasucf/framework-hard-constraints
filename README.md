# Framework for Assessing Hard-Constraint Compliance in LLMs

Anonymous supplementary material for BRACIS 2026 submission.

## Overview

This repository contains the complete implementation, data, and prompts for a framework that evaluates whether Large Language Models comply with hard constraints in safety-critical domains. The framework is instantiated on a **Brazilian Portuguese clinical nutrition benchmark** comprising 19 chronic disease management scenarios.

## Repository Structure

```
├── README.md
├── run_violation_detection.py              # Entry point: parallel violation detection pipeline
│
├── prompts/                                # Prompt templates
│   ├── quality_judge_prompt.txt            # Quality assessment LLM-as-Judge (Section 4.3)
│   └── llm_based_evaluator_prompt.txt      # LLM-based evaluator baseline: food extraction with safe-context filtering (Section 5.3)
│
├── data/
│   ├── benchmark/                          # Benchmark corpus (Section 5.1)
│   │   ├── benchmark_dataset.csv           # 19 scenarios: conditions, queries, restricted food lists
│   │   ├── scenario_queries.csv            # Scenarios: health condition, dietary restriction, user query
│   │   └── prompting_strategies.csv        # All prompting strategy templates (5 strategies x 2 emphasis x 19 conditions)
│   │
│   ├── domain_kb/                          # Domain Knowledge Base (Section 4, Figure 1)
│   │   ├── tbca_foods.parquet              # TBCA food composition data (11,849 entries)
│   │   ├── food_entity_lexicon.parquet     # Food entity lexicon (Aho-Corasick matching)
│   │   ├── verb_lexicon.pkl                # Verb lexicon (Algorithm 1 output)
│   │   ├── involves_substance_facts.json   # involves_substance(a,s) ground facts
│   │   └── foodon_allergen_foods.json      # FoodOn-derived allergen flags
│   │
│   └── results/                            # Evaluation results (Section 5)
│       ├── human_annotation_475.csv              # Human-annotated validation set (Table 2)
│       ├── human_annotation_agreement_50.csv     # Agreement cases validation (50 samples)
│       ├── pviol_pquality_logical_module.csv     # P(R_viol) per model (logical module)
│       └── pviol_pquality_llm_evaluator.csv      # P(R_viol) per model (LLM-based evaluator)
│
└── src/                                    # Violation detection pipeline modules
    ├── entity_extractor.py                 # Entity identification + substance linking (Aho-Corasick)
    ├── lexicon_builder.py                  # Builds food entity lexicon from TBCA
    ├── verb_lexicon_builder.py             # Verb lexicon construction (Algorithm 1)
    ├── linguistic_relation_classifier.py   # Linguistic relation classification (safe-context detection)
    ├── violation_detection_pipeline.py     # Orchestrates predicate grounding -> logical inference
    ├── asp_inference_engine.py             # ASP/Clingo solver interface
    └── asp_rules.py                        # ASP rule definitions (Eqs. 1-3)
```

## Components (mapped to paper sections)

| Paper Section | Component | Repository Location |
|---|---|---|
| 4.1 Predicate Grounding | Entity identification and substance linking | `src/entity_extractor.py` |
| 4.1 Predicate Grounding | Verb lexicon construction (Algorithm 1) | `src/verb_lexicon_builder.py` |
| 4.1 Predicate Grounding | Linguistic relation classification | `src/linguistic_relation_classifier.py` |
| 4.1 Predicate Grounding | Food entity lexicon builder | `src/lexicon_builder.py` |
| 4.2 Logical Inference | ASP rules (Eqs. 1-3) | `src/asp_rules.py` |
| 4.2 Logical Inference | Clingo solver interface | `src/asp_inference_engine.py` |
| 4.2 Logical Inference | Violation detection orchestrator | `src/violation_detection_pipeline.py` |
| 4.3 Quality Assessment | LLM-as-Judge prompt | `prompts/quality_judge_prompt.txt` |
| 5.1 Instantiation | Domain KB (TBCA + USDA + FoodOn) | `data/domain_kb/` |
| 5.1 Instantiation | Benchmark scenarios (19 conditions) | `data/benchmark/` |
| 5.2 Experimental Setup | Prompting strategies (CoT/CoVe/SC/SR/MAI) | `data/benchmark/prompting_strategies.csv` |
| 5.3 Validation | Human annotation data | `data/results/human_annotation_475.csv` |
| 5.3 Validation | LLM-based evaluator prompt | `prompts/llm_based_evaluator_prompt.txt` |
| Table 3a | Conditions and restricted substances | `data/benchmark/benchmark_dataset.csv` |
| Figure 2 | Safety-quality trade-off data | `data/results/pviol_pquality_logical_module.csv` |
| -- | Parallel batch runner | `run_violation_detection.py` |

## Data Sources

- **TBCA** (Tabela Brasileira de Composicao de Alimentos): Food composition with nutrient concentrations
- **USDA FoodData Central** (SR Legacy 2018-04): Supplementary nutrient data
- **FoodOn**: Food ontology for allergen category hierarchies
- **MONDO/DOID**: Standardised disease identifiers
- **Open Multilingual Wordnet**: Semantic relatedness for verb lexicon construction
- **Clinical Guidelines**: ANVISA IN 75/2020, KDIGO, AHA, WHO, ADA (threshold definitions)

## Predicate Vocabulary

The framework uses first-order logic predicates (Table 1 in paper):

**Static (domain KB):** `has_condition(p,d)`, `condition_restricts(d,s)`, `involves_substance(a,s)`, `restr_type(d,s,t)`, `cross_cont_risk(a,s)`

**Extracted per response:** `recommends_consumption(r,a)`, `safe_context(r,a,c)`, `quantity_mentioned(r,a)`

**Derived by inference:** `restricted_item(p,a)`, `within_threshold(r,a)`, `unsafe(r,a)`, `inadequate_rec(r)`

## Prompting Strategies

Five strategies evaluated (x2 emphasis levels x 19 conditions):

| Strategy | Key Mechanism |
|---|---|
| **CoT** (Chain-of-Thought) | Step-by-step reasoning before answering |
| **CoVe** (Chain-of-Verification) | Self-generated verification questions |
| **SC** (Self-Consistency) | Multiple reasoning paths, majority vote |
| **SR** (Self-Refine) | Iterative self-improvement via feedback |
| **MAI** (Maieutic Prompting) | Recursive hypothesis tree with consistency checks |

## Requirements

- Python 3.10+
- spaCy (Portuguese model: `pt_core_news_lg`)
- Clingo (ASP solver)
- pandas, numpy, scikit-learn
- pyarrow (for parquet files)
- nltk + Open Multilingual Wordnet

## Quick Start

```bash
# Run the parallel violation detection pipeline on model responses
python run_violation_detection.py --workers 6 --output results.csv
```

See `run_violation_detection.py` for full usage and CLI options.

## License

This material is provided for anonymous peer review purposes.
