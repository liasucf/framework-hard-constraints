#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_foodon_safe_parallel.py — Parallel FoodOn + Safe-Context pipeline
═══════════════════════════════════════════════════════════════════════

Same logic as run_foodon_safe_batch.py but uses multiprocessing for ~3x speedup.
Each worker loads its own spaCy model + automaton.

Usage:
    python run_foodon_safe_parallel.py [--workers 6] [--output results_foodon_safe_final.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import pandas as pd

csv.field_size_limit(sys.maxsize)
sys.path.insert(0, str(Path(__file__).parent))


def build_id_to_condition(path: str = "restricoes_atualizado.csv") -> dict[int, str]:
    df = pd.read_csv(path)
    id_map = {}
    for _, row in df.iterrows():
        cid = int(row["ID"])
        cond = str(row["CONDIÇAO_DE_SAUDE"]).split("\n")[0].strip()
        id_map[cid] = cond
    return id_map


def _worker_init():
    """Each worker loads its own resources (not picklable)."""
    global _automaton, _pipeline
    from lexicon_builder import load_lexicon
    from violation_detection_pipeline import SafeContextPipeline
    _, _, _automaton = load_lexicon()
    _pipeline = SafeContextPipeline.create()


def _process_rows(args):
    """Process a batch of (text, condition, row_idx) tuples."""
    from entity_extractor import extract_restricted_items
    global _automaton, _pipeline

    rows_batch, id_map = args
    results = []

    for text, row_id in rows_batch:
        condition = ""
        try:
            cid = int(row_id)
            condition = id_map.get(cid, "")
        except (ValueError, TypeError):
            pass

        if not condition or not text:
            results.append({
                "has_restricted": False, "has_violation": False,
                "n_restricted": 0, "n_violations": 0,
                "foods_restricted": "[]", "foods_safe": "[]",
                "foods_violation": "[]", "safe_labels": "{}",
            })
            continue

        try:
            all_foods, restricted_items = extract_restricted_items(
                text, condition, _automaton
            )
        except ValueError:
            results.append({
                "has_restricted": False, "has_violation": False,
                "n_restricted": 0, "n_violations": 0,
                "foods_restricted": "[]", "foods_safe": "[]",
                "foods_violation": "[]", "safe_labels": "{}",
            })
            continue

        if not restricted_items:
            results.append({
                "has_restricted": False, "has_violation": False,
                "n_restricted": 0, "n_violations": 0,
                "foods_restricted": "[]", "foods_safe": "[]",
                "foods_violation": "[]", "safe_labels": "{}",
            })
            continue

        # Safe-context classification
        classif = _pipeline.classify_response(text, restricted_items)
        foods_safe = [r.food for r in classif if not r.is_violation]
        foods_viol = [r.food for r in classif if r.is_violation]
        labels = {r.food: r.label for r in classif}

        results.append({
            "has_restricted": True,
            "has_violation": len(foods_viol) > 0,
            "n_restricted": len(restricted_items),
            "n_violations": len(foods_viol),
            "foods_restricted": str([r.food.canonical for r in restricted_items]),
            "foods_safe": str(foods_safe),
            "foods_violation": str(foods_viol),
            "safe_labels": str(labels),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="todos_experimentos.csv")
    parser.add_argument("--output", default="results_foodon_safe_final.csv")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=3000,
                        help="Rows per read chunk (split across workers)")
    parser.add_argument("--skip-rows", type=int, default=0,
                        help="Skip first N data rows (for resuming)")
    args = parser.parse_args()

    print(f"Starting parallel pipeline ({args.workers} workers)…")
    t0_total = time.time()

    id_map = build_id_to_condition()
    print(f"  Disease map: {len(id_map)} IDs")

    output_path = Path(args.output)
    header_written = args.skip_rows > 0
    rows_done = args.skip_rows
    rows_with_violations = 0
    t_start = time.time()

    # Create process pool with initializer
    pool = Pool(args.workers, initializer=_worker_init)
    print(f"  Pool ready ({args.workers} workers loaded)")

    result_keys = [
        "has_restricted", "has_violation",
        "n_restricted", "n_violations",
        "foods_restricted", "foods_safe", "foods_violation", "safe_labels",
    ]

    try:
        skip_range = range(1, args.skip_rows + 1) if args.skip_rows else None
        if args.skip_rows:
            print(f"  Resuming from row {args.skip_rows}…")
        for chunk_df in pd.read_csv(
            args.input,
            chunksize=args.chunk_size,
            dtype=str,
            keep_default_na=False,
            skiprows=skip_range,
        ):
            # Prepare row data
            text_data = []
            for _, row in chunk_df.iterrows():
                text = row.get("TEXTO_CLEAN", "") or row.get("TEXTO", "")
                row_id = row.get("ID", "")
                text_data.append((text, row_id))

            # Split across workers
            n = len(text_data)
            per_worker = max(1, n // args.workers)
            batches = []
            for i in range(0, n, per_worker):
                batches.append((text_data[i:i + per_worker], id_map))

            # Process in parallel
            worker_results = pool.map(_process_rows, batches)

            # Flatten results
            all_results = []
            for wr in worker_results:
                all_results.extend(wr)

            # Attach to dataframe
            for k in result_keys:
                chunk_df[k] = [r[k] for r in all_results]

            # Write
            chunk_df.to_csv(
                output_path,
                mode="a" if header_written else "w",
                header=not header_written,
                index=False,
            )
            header_written = True
            rows_done += len(chunk_df)

            viols_in_chunk = sum(1 for r in all_results if r["has_violation"])
            rows_with_violations += viols_in_chunk

            elapsed = time.time() - t_start
            rate = rows_done / elapsed if elapsed > 0 else 0
            eta = (271700 - rows_done) / rate / 60 if rate > 0 else 0
            viol_pct = rows_with_violations * 100 / rows_done

            print(
                f"  {rows_done:>7d} rows | "
                f"violations={rows_with_violations} ({viol_pct:.1f}%) | "
                f"{rate:.0f} rows/s | ETA ~{eta:.0f}min",
                flush=True,
            )

    finally:
        pool.close()
        pool.join()

    elapsed = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"DONE: {rows_done} rows in {elapsed:.0f}s ({rows_done/elapsed:.0f} rows/s)")
    print(f"  Violations: {rows_with_violations} ({rows_with_violations*100/rows_done:.1f}%)")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
