#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_record(rec: Dict[str, Any], idx: int, require_label: bool) -> Tuple[Optional[Dict[str, Any]], str]:
    context = str(rec.get("context", "")).strip()
    question = str(rec.get("question", "")).strip()
    answers = rec.get("answers")
    if not context:
        return None, "empty_context"
    if not question:
        return None, "empty_question"
    if not isinstance(answers, list) or len(answers) != 4:
        return None, "bad_answers"

    out = {
        "sample_id": rec.get("id_string", f"sample_{idx}"),
        "context": context,
        "question": question,
        "options": [str(x) for x in answers],
    }

    if require_label:
        label = rec.get("label", None)
        if label is None:
            return None, "missing_label"
        try:
            gold = int(label)
        except Exception:
            return None, "bad_label"
        if gold not in (0, 1, 2, 3):
            return None, "label_out_of_range"
        out["gold"] = gold

    if "question_type" in rec:
        out["question_type"] = rec["question_type"]
    return out, "ok"


def main():
    ap = argparse.ArgumentParser(description="Preprocess ReClor JSON into flat JSONL.")
    ap.add_argument("--input", required=True, help="Path to raw ReClor split JSON")
    ap.add_argument("--output", required=True, help="Path to output processed JSONL")
    ap.add_argument("--max_samples", type=int, default=0, help="Keep first N valid samples; 0 means all")
    ap.add_argument("--allow_unlabeled", action="store_true", help="Allow records without labels (for test split)")
    args = ap.parse_args()

    raw = read_json(args.input)
    require_label = not args.allow_unlabeled
    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}

    for i, rec in enumerate(raw):
        out, status = normalize_record(rec, i, require_label=require_label)
        if out is None:
            skipped[status] = skipped.get(status, 0) + 1
            continue
        rows.append(out)
        if args.max_samples and len(rows) >= args.max_samples:
            break

    write_jsonl(args.output, rows)

    print(f"Input records: {len(raw)}")
    print(f"Output records: {len(rows)}")
    print(f"Require label: {require_label}")
    if skipped:
        print("Skipped:", skipped)
    print(f"Saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

