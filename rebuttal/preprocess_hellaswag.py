#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_processed(rec: Dict[str, Any], fallback_id: int):
    endings = rec.get("endings") or []
    if len(endings) != 4:
        return None, "bad_endings"

    label = rec.get("label")
    if label is None:
        return None, "missing_label"
    try:
        gold = int(label)
    except Exception:
        return None, "bad_label"
    if gold not in (0, 1, 2, 3):
        return None, "bad_label_range"

    ctx = (rec.get("ctx") or "").strip()
    if not ctx:
        ctx_a = (rec.get("ctx_a") or "").strip()
        ctx_b = (rec.get("ctx_b") or "").strip()
        ctx = (ctx_a + " " + ctx_b).strip()
    if not ctx:
        return None, "empty_context"

    out = {
        "sample_id": rec.get("ind", fallback_id),
        "context": ctx,
        "endings": [str(x) for x in endings],
        "gold": str(gold),
        "split": rec.get("split", ""),
        "split_type": rec.get("split_type", ""),
        "activity_label": rec.get("activity_label", ""),
        "source_id": rec.get("source_id", ""),
    }
    return out, None


def main():
    ap = argparse.ArgumentParser(description="Preprocess HellaSwag into a steering-ready JSONL format.")
    ap.add_argument("--input", required=True, help="Path to raw HellaSwag jsonl")
    ap.add_argument("--output", required=True, help="Path to processed jsonl")
    ap.add_argument("--max_samples", type=int, default=0, help="Keep first N valid samples; 0 means all")
    args = ap.parse_args()

    raw = read_jsonl(args.input)
    processed = []
    skipped = {
        "bad_endings": 0,
        "missing_label": 0,
        "bad_label": 0,
        "bad_label_range": 0,
        "empty_context": 0,
    }

    for i, rec in enumerate(raw):
        row, reason = to_processed(rec, i)
        if row is None:
            skipped[reason] += 1
            continue
        processed.append(row)
        if args.max_samples and len(processed) >= args.max_samples:
            break

    write_jsonl(args.output, processed)

    print(f"Input records: {len(raw)}")
    print(f"Output records: {len(processed)}")
    print("Skipped breakdown:", skipped)
    print(f"Saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

