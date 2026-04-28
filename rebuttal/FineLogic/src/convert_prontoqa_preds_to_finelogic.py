#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert ProntoQA prediction JSONL into FineLogic eval_step input JSON format.

Input row schema (expected):
  {"story_id","gold","pred","text","gen", ...}

Output item schema:
  {
    "problem": {
      "input": "...",
      "proof_label": "__PROVED__/__DISPROVED__",
      "original_data": {"steps": int}
    },
    "responses": [
      {"model":"...", "prompt_style":"cot", "response":"...", "success": true}
    ],
    "meta": {...}
  }
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_prontoqa_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_view_text(rec: Dict[str, Any], view_name: str) -> Optional[str]:
    pair = rec.get("pair") or []
    for item in pair:
        if item.get("view", "") == view_name:
            return item.get("text", "")
    return None


def estimate_steps_from_with_proof(text: str) -> Optional[int]:
    """
    Heuristic for ProntoQA NL/FOL *_with_proof text:
    count non-empty proof lines between question line and "The query is ...".
    """
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines()]
    if not lines:
        return None

    q_idx = None
    end_idx = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if q_idx is None and ("true or false:" in low or "question:" in low):
            q_idx = i
        if "the query is" in low:
            end_idx = i
            break

    if q_idx is None or end_idx is None or end_idx <= q_idx:
        return None

    body = [ln for ln in lines[q_idx + 1 : end_idx] if ln]
    return len(body) if body else None


def gold_to_marker(gold: Any) -> str:
    g = str(gold).strip().lower()
    if g in {"true", "t", "__proved__", "proved"}:
        return "__PROVED__"
    if g in {"false", "f", "__disproved__", "disproved"}:
        return "__DISPROVED__"
    return "__UNKNOWN__"


def build_story_steps_lookup(prontoqa_json: str, steps_view: str) -> Dict[str, Optional[int]]:
    if not prontoqa_json:
        return {}
    data = read_prontoqa_json(prontoqa_json)
    lookup = {}
    for rec in data:
        sid = rec.get("story_id")
        if sid is None:
            continue
        txt = get_view_text(rec, steps_view)
        lookup[sid] = estimate_steps_from_with_proof(txt or "")
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_preds", required=True, help="ProntoQA preds JSONL")
    ap.add_argument("--output_json", required=True, help="FineLogic-format JSON (list)")
    ap.add_argument("--model_name", default="llama31_8b")
    ap.add_argument("--prompt_style", default="cot")
    ap.add_argument("--prontoqa_json", default="", help="Optional original ProntoQA JSON for step-count estimation")
    ap.add_argument(
        "--steps_view",
        default="NL_with_proof",
        choices=["NL_with_proof", "NL_without_proof", "FOL_with_proof", "FOL_without_proof"],
        help="Which view to use when estimating ground-truth step count",
    )
    ap.add_argument(
        "--default_steps",
        type=int,
        default=-1,
        help="Fallback step count when not estimable from source data",
    )
    args = ap.parse_args()

    preds = read_jsonl(args.input_preds)
    steps_lookup = build_story_steps_lookup(args.prontoqa_json, args.steps_view)

    out = []
    missing_steps = 0
    empty_gen = 0
    for i, r in enumerate(preds):
        sid = r.get("story_id", f"sample_{i}")
        text = str(r.get("text", ""))
        gen = str(r.get("gen", ""))
        gold = r.get("gold", "")

        steps = steps_lookup.get(sid, None)
        if steps is None:
            steps = int(args.default_steps)
            missing_steps += 1

        success = bool(gen.strip())
        if not success:
            empty_gen += 1

        item = {
            "problem": {
                "input": text,
                "proof_label": gold_to_marker(gold),
                "original_data": {"steps": steps},
            },
            "responses": [
                {
                    "model": args.model_name,
                    "prompt_style": args.prompt_style,
                    "response": gen,
                    "success": success,
                }
            ],
            "meta": {
                "story_id": sid,
                "gold": r.get("gold"),
                "pred": r.get("pred"),
            },
        }
        out.append(item)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Input preds: {len(preds)}")
    print(f"Output items: {len(out)}")
    print(f"Missing/estimated steps: {missing_steps}")
    print(f"Empty generations: {empty_gen}")
    print(f"Saved: {os.path.abspath(args.output_json)}")


if __name__ == "__main__":
    main()

