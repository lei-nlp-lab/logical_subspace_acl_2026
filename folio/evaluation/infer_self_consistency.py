#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Consistency Chain-of-Thought Baseline for FOLIO

Reference: Wang et al. (2022) "Self-Consistency Improves Chain of Thought Reasoning in Language Models"

Key idea:
1. Sample multiple reasoning paths using temperature > 0
2. Parse the final answer from each path
3. Take majority vote among all answers
"""

import argparse, os, csv, json, sys
from collections import defaultdict, Counter
from typing import List, Dict, Optional

import torch
import numpy as np
import random
from tqdm import tqdm

try:
    from evaluation.steering_infer import (
        build_prompt, read_jsonl, parse_truth,
        primary_device, norm_truth
    )
except Exception:
    from steering_infer import (
        build_prompt, read_jsonl, parse_truth,
        primary_device, norm_truth
    )

from transformers import AutoTokenizer, AutoModelForCausalLM


def majority_vote(
    preds: List[Optional[str]],
    valid_labels: List[str],
    min_valid_ratio: float = 0.5
) -> Optional[str]:
    """
    Take majority vote from a list of predictions.
    Only count valid (non-None) predictions.

    Args:
        preds: List of predictions (may contain None for parse failures)
        valid_labels: List of valid label strings
        min_valid_ratio: Minimum ratio of valid predictions required (default 0.5)

    Returns:
        The majority vote result, or None if:
        - No valid predictions exist
        - Valid predictions ratio < min_valid_ratio
    """
    valid_preds = [p for p in preds if p in valid_labels]

    # If valid paths < 50%, treat as PARSE_FAILED
    if len(valid_preds) <= len(preds) * min_valid_ratio:
        return None

    counter = Counter(valid_preds)
    # Return the most common prediction; ties broken arbitrarily
    return counter.most_common(1)[0][0]


def ensure_dir(p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)


def run_self_consistency(
    mdl, tok, device, samples,
    max_new_tokens: int,
    num_samples: int,
    temperature: float,
    top_p: float,
    min_valid_ratio: float = 0.5,
):
    """
    Run self-consistency evaluation.

    For each sample:
    1. Generate `num_samples` reasoning paths with sampling
    2. Parse answer from each path
    3. Take majority vote

    Returns result dict with accuracy, macro_f1, confusion matrix, etc.
    """
    results = []
    correct = 0
    parse_failed = 0
    per_class = defaultdict(lambda: {"tp": 0, "total": 0})
    labels = ["True", "False", "Unknown"]
    all_preds = labels + ["PARSE_FAILED"]
    cm = {a: {b: 0 for b in all_preds} for a in labels}

    for s in tqdm(samples, desc="Self-consistency evaluation"):
        # Build prompt
        ptxt = build_prompt(s["prem"], s["concl"])
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            msgs = [
                {"role": "system", "content": "You are a helpful reasoning assistant."},
                {"role": "user", "content": ptxt}
            ]
            chat_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            prompt = chat_text
        else:
            prompt = ptxt

        # Encode
        enc = tok(prompt, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        # Generate multiple paths
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
                # OOM fallback: generate paths one by one
                torch.cuda.empty_cache()
                outputs_list = []
                for _ in range(num_samples):
                    out_single = mdl.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=1,
                        pad_token_id=tok.pad_token_id,
                    )
                    outputs_list.append(out_single)
                outputs = torch.cat(outputs_list, dim=0)

        # Decode all paths
        generated_texts = tok.batch_decode(outputs, skip_special_tokens=True)

        # Parse answers from each path
        path_preds = []
        for text in generated_texts:
            pred = parse_truth(text)
            path_preds.append(pred)

        # Majority vote (with min_valid_ratio check)
        final_pred = majority_vote(path_preds, labels, min_valid_ratio)
        gold = s["gold"]

        # Count valid predictions per path
        valid_count = sum(1 for p in path_preds if p in labels)

        if final_pred is None:
            parse_failed += 1
            per_class[gold]["total"] += 1
            final_pred_str = "PARSE_FAILED"
        else:
            hit = int(final_pred == gold)
            correct += hit
            per_class[gold]["total"] += 1
            if hit:
                per_class[gold]["tp"] += 1
            final_pred_str = final_pred

        cm[gold][final_pred_str] += 1

        results.append({
            "gold": gold,
            "pred": final_pred_str,
            "prem": s["prem"],
            "hypo": s["concl"],
            "num_paths": num_samples,
            "valid_paths": valid_count,
            "path_preds": path_preds,
            "path_texts": generated_texts,
        })

    # Compute metrics
    total = len(samples)
    parsed = total - parse_failed
    acc = correct / parsed if parsed else 0.0
    acc_with_failed = correct / total if total else 0.0

    # Per-class precision/recall/F1 + macro F1
    tp = {l: cm[l][l] for l in labels}
    fp = {l: sum(cm[g][l] for g in labels if g != l) for l in labels}
    fn = {l: sum(cm[l][p] for p in all_preds if p != l) for l in labels}

    prec = {l: (tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0) for l in labels}
    rec = {l: (tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0) for l in labels}
    f1 = {l: (2 * prec[l] * rec[l] / (prec[l] + rec[l]) if (prec[l] + rec[l]) > 0 else 0.0) for l in labels}
    macro_f1 = sum(f1.values()) / len(labels)

    return {
        "total": int(total),
        "parsed": int(parsed),
        "parse_failed": int(parse_failed),
        "accuracy": float(acc),
        "accuracy_with_failed": float(acc_with_failed),
        "macro_f1": float(macro_f1),
        "cm": cm,
        "per_class": {l: {"prec": prec[l], "rec": rec[l], "f1": f1[l], "n": per_class[l]["total"]} for l in labels},
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    # Data and model
    ap.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    ap.add_argument("--file", type=str, required=True, help="FOLIO jsonl file")
    # Generation
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=0, help="Max data samples to evaluate (0=all)")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    # Self-consistency parameters
    ap.add_argument("--num_paths", type=int, default=5, help="Number of reasoning paths to sample")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    ap.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    ap.add_argument("--min_valid_ratio", type=float, default=0.5, help="Min ratio of valid paths for majority vote (default 0.5)")
    # Output
    ap.add_argument("--out_csv", type=str, default="self_consistency_results.csv")
    ap.add_argument("--save_preds", action="store_true", help="Save per-sample predictions to jsonl")
    args = ap.parse_args()

    # Set random seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Load data
    rows = read_jsonl(args.file)

    def get_field(d, candidates, default=None):
        for k in candidates:
            if k in d:
                return d[k]
        return default

    samples = []
    for r in rows:
        prem = get_field(r, ["premises", "Premises - NL", "premises_nl", "premises-nl"])
        concl = get_field(r, ["conclusion", "Conclusions - NL", "conclusion_nl", "conclusion-nl"])
        lab = get_field(r, ["label", "Truth Values", "truth", "gold_label"])
        if isinstance(prem, list):
            prem = " ".join(prem)
        if prem is None or concl is None or lab is None:
            continue
        gold = norm_truth(str(lab))
        if gold is None:
            continue
        samples.append({"prem": prem, "concl": concl, "gold": gold})

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
    if not samples:
        print("No valid samples.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(samples)} valid samples from {args.file}")

    # Load model
    load_kwargs = {}
    if args.use_4bit:
        load_kwargs["load_in_4bit"] = True
    if args.use_8bit:
        load_kwargs["load_in_8bit"] = True

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
    print(f"Primary device: {device}")
    print(f"Self-consistency config: num_paths={args.num_paths}, temperature={args.temperature}, top_p={args.top_p}, min_valid_ratio={args.min_valid_ratio}")

    # Run evaluation
    out = run_self_consistency(
        mdl=mdl, tok=tok, device=device, samples=samples,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_paths,
        temperature=args.temperature,
        top_p=args.top_p,
        min_valid_ratio=args.min_valid_ratio,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Self-Consistency Results")
    print("=" * 60)
    print(f"Total samples: {out['total']}")
    print(f"Parsed: {out['parsed']}, Parse failed: {out['parse_failed']}")
    print(f"Accuracy: {out['accuracy']:.4f}")
    print(f"Accuracy (with failed): {out['accuracy_with_failed']:.4f}")
    print(f"Macro F1: {out['macro_f1']:.4f}")
    print("\nPer-class metrics:")
    for l in ["True", "False", "Unknown"]:
        pc = out["per_class"][l]
        print(f"  {l}: P={pc['prec']:.3f}, R={pc['rec']:.3f}, F1={pc['f1']:.3f}, n={pc['n']}")
    print("\nConfusion Matrix:")
    for gold in ["True", "False", "Unknown"]:
        row_str = " ".join(f"{out['cm'][gold][pred]:3d}" for pred in ["True", "False", "Unknown", "PARSE_FAILED"])
        print(f"  {gold:8s}: {row_str}")

    # Save CSV
    ensure_dir(args.out_csv)
    first_write = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if first_write:
            w.writerow([
                "num_paths", "temperature", "top_p", "min_valid_ratio",
                "total", "parsed", "parse_failed", "accuracy", "accuracy_with_failed", "macro_f1",
                "cm_T_T", "cm_T_F", "cm_T_U", "cm_T_PARSE_FAILED",
                "cm_F_T", "cm_F_F", "cm_F_U", "cm_F_PARSE_FAILED",
                "cm_U_T", "cm_U_F", "cm_U_U", "cm_U_PARSE_FAILED",
                "prec_T", "rec_T", "f1_T", "n_T",
                "prec_F", "rec_F", "f1_F", "n_F",
                "prec_U", "rec_U", "f1_U", "n_U",
                "model", "file"
            ])

        cm = out["cm"]
        per = out["per_class"]
        w.writerow([
            args.num_paths, args.temperature, args.top_p, args.min_valid_ratio,
            out["total"], out["parsed"], out["parse_failed"],
            out["accuracy"], out["accuracy_with_failed"], out["macro_f1"],
            cm["True"]["True"], cm["True"]["False"], cm["True"]["Unknown"], cm["True"]["PARSE_FAILED"],
            cm["False"]["True"], cm["False"]["False"], cm["False"]["Unknown"], cm["False"]["PARSE_FAILED"],
            cm["Unknown"]["True"], cm["Unknown"]["False"], cm["Unknown"]["Unknown"], cm["Unknown"]["PARSE_FAILED"],
            per["True"]["prec"], per["True"]["rec"], per["True"]["f1"], per["True"]["n"],
            per["False"]["prec"], per["False"]["rec"], per["False"]["f1"], per["False"]["n"],
            per["Unknown"]["prec"], per["Unknown"]["rec"], per["Unknown"]["f1"], per["Unknown"]["n"],
            args.model, os.path.abspath(args.file)
        ])

    print(f"\nResults saved to: {os.path.abspath(args.out_csv)}")

    # Save predictions
    if args.save_preds:
        pred_path = args.out_csv.replace(".csv", "_preds.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for r in out["results"]:
                # Don't save full path_texts to reduce file size
                r_save = {k: v for k, v in r.items() if k != "path_texts"}
                f.write(json.dumps(r_save, ensure_ascii=False) + "\n")
        print(f"Predictions saved to: {os.path.abspath(pred_path)}")


if __name__ == "__main__":
    main()
