#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean FineLogic-format response fields by removing echoed chat/prompt prefixes.

Input/Output format:
  JSON list of items with:
    item["responses"][i]["response"] as raw text.
"""

import argparse
import json
import os
from typing import Any, Dict, List


PROMPT_END_MARKERS = [
    "Truth value: <True|False>",
    "Answer: <A|B|C|D>",
]


def clean_response_text(text: str) -> str:
    if not text:
        return text

    out = text

    # Keep only model continuation after the known prompt tail if present.
    for m in PROMPT_END_MARKERS:
        idx = out.rfind(m)
        if idx != -1:
            out = out[idx + len(m) :]
            break

    # Common chat-template separators that can leak into decoded text.
    for sep in ["assistant\n", "assistant ", "<|assistant|>", "<|im_start|>assistant"]:
        idx = out.find(sep)
        if idx != -1:
            out = out[idx + len(sep) :]
            break

    return out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="Input FineLogic-format JSON")
    ap.add_argument("--output_json", required=True, help="Output cleaned JSON")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    changed = 0
    empty_after = 0
    for item in data:
        responses = item.get("responses") or []
        for r in responses:
            old = str(r.get("response", ""))
            new = clean_response_text(old)
            if new != old:
                changed += 1
            r["response"] = new
            r["success"] = bool(new.strip())
            if not r["success"]:
                empty_after += 1

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Input items: {len(data)}")
    print(f"Responses changed: {changed}")
    print(f"Empty responses after cleaning: {empty_after}")
    print(f"Saved: {os.path.abspath(args.output_json)}")


if __name__ == "__main__":
    main()

