#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Consistency Chain-of-Thought Baseline for ProntoQA (三分类: True/False/Uncertain)
OWA (Open-World Assumption) version
"""

import argparse, os, csv, json, sys, re
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional

import torch
import numpy as np
import random
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- ProntoQA OWA Prompt (三分类：True/False/Uncertain) ----
PROMPT_TEMPLATE = """You are a careful logician. Use classical deductive reasoning.

{text}

Instructions:
- First, reason step by step (MAXIMUM 25 steps).
- Then, on the last line, output exactly:
Truth value: <True|False|Uncertain>
- IMPORTANT: If you cannot complete reasoning in 25 steps, you MUST output your best judgment anyway on the last line.
"""

VAL_RE = re.compile(r"Truth\s*value\s*:\s*(True|False|Uncertain)", re.IGNORECASE)


def norm_truth(x: str):
    if x is None: return None
    t = x.strip().lower()
    if t in {"true", "t"}: return "True"
    if t in {"false", "f"}: return "False"
    if t in {"uncertain", "unknown", "u"}: return "Uncertain"
    m = VAL_RE.search(x)
    if m: return norm_truth(m.group(1))
    return None


def parse_truth(text: str):
    lines = [ln.strip() for ln in text.splitlines() if "truth value" in ln.lower()]
    if lines:
        m = VAL_RE.search(lines[-1])
        if m: return norm_truth(m.group(1))
    m = list(VAL_RE.finditer(text))
    return norm_truth(m[-1].group(1)) if m else None


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(text=text.strip())


def read_prontoqa_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_view_text(rec: Dict[str, Any], view_name: str) -> str:
    pair = rec.get("pair") or []
    for item in pair:
        if item.get("view", "") == view_name:
            return item.get("text", "")
    return None


def primary_device(model):
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for dev in set(model.hf_device_map.values()):
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
    if hasattr(model, "device"):
        return model.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def majority_vote(preds: List[Optional[str]], valid_labels: List[str], min_valid_ratio: float = 0.5) -> Optional[str]:
    valid_preds = [p for p in preds if p in valid_labels]
    if len(valid_preds) <= len(preds) * min_valid_ratio:
        return None
    counter = Counter(valid_preds)
    return counter.most_common(1)[0][0]


def ensure_dir(p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)


def run_self_consistency(mdl, tok, device, samples, max_new_tokens, num_samples, temperature, top_p, min_valid_ratio):
    results = []
    correct = 0
    parse_failed = 0
    per_class = defaultdict(lambda: {"tp": 0, "total": 0})
    labels = ["True", "False", "Uncertain"]
    all_preds = labels + ["PARSE_FAILED"]
    cm = {a: {b: 0 for b in all_preds} for a in labels}

    for s in tqdm(samples, desc="Self-consistency evaluation"):
        ptxt = build_prompt(s["text"])
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            msgs = [
                {"role": "system", "content": "You are a helpful reasoning assistant."},
                {"role": "user", "content": ptxt}
            ]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        else:
            prompt = ptxt

        enc = tok(prompt, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            try:
                outputs = mdl.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_samples,
                    pad_token_id=tok.pad_token_id,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                outputs_list = []
                for _ in range(num_samples):
                    out_single = mdl.generate(
                        **enc, max_new_tokens=max_new_tokens, do_sample=True,
                        temperature=temperature, top_p=top_p, num_return_sequences=1,
                        pad_token_id=tok.pad_token_id,
                    )
                    outputs_list.append(out_single)
                outputs = torch.cat(outputs_list, dim=0)

        generated_texts = tok.batch_decode(outputs, skip_special_tokens=True)

        path_preds = [parse_truth(text) for text in generated_texts]
        final_pred = majority_vote(path_preds, labels, min_valid_ratio)
        gold = s["gold"]
        valid_count = sum(1 for p in path_preds if p in labels)

        if final_pred is None:
            parse_failed += 1
            per_class[gold]["total"] += 1
            final_pred_str = "PARSE_FAILED"
        else:
            hit = int(final_pred == gold)
            correct += hit
            per_class[gold]["total"] += 1
            if hit: per_class[gold]["tp"] += 1
            final_pred_str = final_pred

        cm[gold][final_pred_str] += 1
        results.append({
            "story_id": s["story_id"], "gold": gold, "pred": final_pred_str,
            "text": s["text"], "num_paths": num_samples, "valid_paths": valid_count,
            "path_preds": path_preds,
        })

    total = len(samples)
    parsed = total - parse_failed
    acc = correct / parsed if parsed else 0.0
    acc_with_failed = correct / total if total else 0.0

    tp = {l: cm[l][l] for l in labels}
    fp = {l: sum(cm[g][l] for g in labels if g != l) for l in labels}
    fn = {l: sum(cm[l][p] for p in all_preds if p != l) for l in labels}
    prec = {l: (tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0) for l in labels}
    rec = {l: (tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0) for l in labels}
    f1 = {l: (2 * prec[l] * rec[l] / (prec[l] + rec[l]) if (prec[l] + rec[l]) > 0 else 0.0) for l in labels}
    macro_f1 = sum(f1.values()) / len(labels)

    return {
        "total": total, "parsed": parsed, "parse_failed": parse_failed,
        "accuracy": acc, "accuracy_with_failed": acc_with_failed, "macro_f1": macro_f1,
        "cm": cm, "per_class": {l: {"prec": prec[l], "rec": rec[l], "f1": f1[l], "n": per_class[l]["total"]} for l in labels},
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--file", type=str, required=True)
    ap.add_argument("--view", type=str, required=True,
                    choices=["NL_with_proof", "NL_without_proof", "FOL_with_proof", "FOL_without_proof"])
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    ap.add_argument("--num_paths", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--min_valid_ratio", type=float, default=0.5)
    ap.add_argument("--out_csv", type=str, default="self_consistency_prontoqa_owa.csv")
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()

    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    data = read_prontoqa_json(args.file)
    samples = []
    for rec in data:
        story_id = rec.get("story_id", "unknown")
        view_text = get_view_text(rec, args.view)
        if view_text is None: continue
        gold_label = rec.get("label")
        if gold_label is None: continue
        gold = norm_truth(gold_label)
        if gold is None: continue
        samples.append({"story_id": story_id, "text": view_text, "gold": gold})

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
    if not samples:
        print("No valid samples.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(samples)} samples (view: {args.view})")

    load_kwargs = {}
    if args.use_4bit: load_kwargs["load_in_4bit"] = True
    if args.use_8bit: load_kwargs["load_in_8bit"] = True

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=dtype,
            attn_implementation="sdpa", low_cpu_mem_usage=True, **load_kwargs
        )
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=dtype,
            low_cpu_mem_usage=True, **load_kwargs
        )

    mdl.eval()
    mdl.generation_config.pad_token_id = tok.pad_token_id
    mdl.generation_config.eos_token_id = tok.eos_token_id
    mdl.generation_config.use_cache = True

    device = primary_device(mdl)
    print(f"Device: {device}, SC config: paths={args.num_paths}, T={args.temperature}")

    out = run_self_consistency(
        mdl, tok, device, samples, args.max_new_tokens,
        args.num_paths, args.temperature, args.top_p, args.min_valid_ratio
    )

    print(f"\nAccuracy: {out['accuracy']:.4f}, Macro F1: {out['macro_f1']:.4f}")

    ensure_dir(args.out_csv)
    first_write = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if first_write:
            w.writerow([
                "num_paths", "temperature", "top_p", "min_valid_ratio", "view",
                "total", "parsed", "parse_failed", "accuracy", "accuracy_with_failed", "macro_f1",
                "cm_T_T", "cm_T_F", "cm_T_U", "cm_T_PARSE_FAILED",
                "cm_F_T", "cm_F_F", "cm_F_U", "cm_F_PARSE_FAILED",
                "cm_U_T", "cm_U_F", "cm_U_U", "cm_U_PARSE_FAILED",
                "prec_T", "rec_T", "f1_T", "n_T",
                "prec_F", "rec_F", "f1_F", "n_F",
                "prec_U", "rec_U", "f1_U", "n_U",
                "model", "file"
            ])
        cm, per = out["cm"], out["per_class"]
        w.writerow([
            args.num_paths, args.temperature, args.top_p, args.min_valid_ratio, args.view,
            out["total"], out["parsed"], out["parse_failed"],
            out["accuracy"], out["accuracy_with_failed"], out["macro_f1"],
            cm["True"]["True"], cm["True"]["False"], cm["True"]["Uncertain"], cm["True"]["PARSE_FAILED"],
            cm["False"]["True"], cm["False"]["False"], cm["False"]["Uncertain"], cm["False"]["PARSE_FAILED"],
            cm["Uncertain"]["True"], cm["Uncertain"]["False"], cm["Uncertain"]["Uncertain"], cm["Uncertain"]["PARSE_FAILED"],
            per["True"]["prec"], per["True"]["rec"], per["True"]["f1"], per["True"]["n"],
            per["False"]["prec"], per["False"]["rec"], per["False"]["f1"], per["False"]["n"],
            per["Uncertain"]["prec"], per["Uncertain"]["rec"], per["Uncertain"]["f1"], per["Uncertain"]["n"],
            args.model, os.path.abspath(args.file)
        ])

    print(f"Results saved to: {args.out_csv}")

    if args.save_preds:
        pred_path = args.out_csv.replace(".csv", "_preds.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for r in out["results"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Predictions saved to: {pred_path}")


if __name__ == "__main__":
    main()
