#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProntoQA 3-shot + normalized steering evaluation.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ========= steering helpers =========
EVAL_MULTI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation_multi"))
if EVAL_MULTI_DIR not in sys.path:
    sys.path.append(EVAL_MULTI_DIR)

from steering_infer_normalized import HFSteererNormalized as HFSteerer
from steering_infer import primary_device, find_decoder_layers


# ========= 3-shot examples =========
FEW_SHOT_EXAMPLES = [
    {
        "label": "True",
        "text": """Premises:
Each tumpus is bright. Tumpuses are yumpuses. Tumpuses are gorpuses. Every yumpus is hot. Yumpuses are sterpuses. Each yumpus is a brimpus. Sterpuses are rainy. Sterpuses are jompuses. Shumpuses are small. Each sterpus is a lorpus. Each jompus is mean. Each jompus is a rompus. Jompuses are impuses. Every rompus is not small. Rompuses are dumpuses. Rompuses are zumpuses. Zumpuses are shy. Every impus is liquid. Lorpuses are moderate. Each brimpus is fruity. Every gorpus is blue. Every wumpus is melodic. Every wumpus is a lempus. Polly is a tumpus. Polly is a wumpus.

True or false: Polly is not small.

Polly is a tumpus.
Tumpuses are yumpuses.
Polly is a yumpus.
Yumpuses are sterpuses.
Polly is a sterpus.
Sterpuses are jompuses.
Polly is a jompus.
Each jompus is a rompus.
Polly is a rompus.
Every rompus is not small.
Polly is not small.

The query is True."""
    },
    {
        "label": "False",
        "text": """Premises:
Impuses are sunny. Impuses are rompuses. Impuses are grimpuses. Rompuses are floral. Rompuses are numpuses. Rompuses are shumpuses. Each numpus is liquid. Numpuses are dumpuses. Numpuses are yumpuses. Dumpuses are large. Every dumpus is a brimpus. Every dumpus is a tumpus. Every brimpus is not transparent. Brimpuses are lorpuses. Each wumpus is transparent. Brimpuses are gorpuses. Each gorpus is muffled. Tumpuses are not angry. Each yumpus is bitter. Every shumpus is not brown. Every grimpus is not dull. Sterpuses are not moderate. Every sterpus is a vumpus. Sam is an impus. Sam is a sterpus.

True or false: Sam is transparent.

Sam is an impus.
Impuses are rompuses.
Sam is a rompus.
Rompuses are numpuses.
Sam is a numpus.
Numpuses are dumpuses.
Sam is a dumpus.
Every dumpus is a brimpus.
Sam is a brimpus.
Every brimpus is not transparent.
Sam is not transparent.

The query is False."""
    },
    {
        "label": "True",
        "text": """Premises:
Lempuses are not hot. Each impus is bitter. Each impus is a rompus. Impuses are numpuses. Each rompus is not bright. Rompuses are grimpuses. Rompuses are gorpuses. Grimpuses are fast. Each grimpus is a yumpus. Every grimpus is a zumpus. Every yumpus is large. Each yumpus is a brimpus. Yumpuses are lorpuses. Every brimpus is hot. Each brimpus is a vumpus. Every brimpus is a jompus. Jompuses are shy. Lorpuses are not opaque. Each zumpus is muffled. Every gorpus is red. Every numpus is kind. Sterpuses are orange. Every sterpus is a tumpus. Max is an impus. Max is a sterpus.

True or false: Max is a jompus.

Max is an impus.
Each impus is a rompus.
Max is a rompus.
Rompuses are grimpuses.
Max is a grimpus.
Each grimpus is a yumpus.
Max is a yumpus.
Each yumpus is a brimpus.
Max is a brimpus.
Every brimpus is a jompus.
Max is a jompus.

The query is True."""
    },
]

PROMPT_TEMPLATE = """You are a careful logician. Use classical deductive reasoning.

{examples}
=== Your Turn ===
{text}

Instructions:
- First, reason step by step.
- Then, on the last line, output exactly:
Truth value: <True|False>
"""

VAL_RE = re.compile(r"Truth\s*value\s*:\s*(True|False)", re.IGNORECASE)


def build_few_shot_prompt(text: str) -> str:
    examples_str = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_str += f"=== Example {i} ===\n"
        examples_str += ex["text"] + "\n"
        examples_str += f"\nTruth value: {ex['label']}\n\n"
    return PROMPT_TEMPLATE.format(examples=examples_str, text=text.strip())


def norm_truth(x: str):
    if x is None:
        return None
    t = x.strip().lower()
    if t in {"true", "t"}:
        return "True"
    if t in {"false", "f"}:
        return "False"
    m = VAL_RE.search(x)
    if m:
        return norm_truth(m.group(1))
    return None


def parse_truth(text: str):
    lines = [ln.strip() for ln in text.splitlines() if "truth value" in ln.lower()]
    if lines:
        m = VAL_RE.search(lines[-1])
        if m:
            return norm_truth(m.group(1))
    m = list(VAL_RE.finditer(text))
    return norm_truth(m[-1].group(1)) if m else None


def read_prontoqa_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_view_text(rec: Dict[str, Any], view_name: str) -> str:
    pair = rec.get("pair") or []
    for item in pair:
        if item.get("view", "") == view_name:
            return item.get("text", "")
    return None


def batched(it, bs):
    for i in range(0, len(it), bs):
        yield it[i : i + bs]


def ensure_dir(p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)


def run_eval_once(
    mdl,
    tok,
    device,
    samples,
    max_new_tokens,
    batch_size,
    svcca_pt,
    layer,
    lam,
    use_projectors,
    top_k,
    corr_min,
    anchor,
    window,
    enable_thinking=False,
):
    steerer = None
    if lam != 0:
        steerer = HFSteerer(
            model=mdl,
            svcca_pt=svcca_pt,
            layers=[layer],
            lambdas={layer: float(lam)},
            use_projectors=use_projectors,
            top_k=top_k,
            corr_min=corr_min,
            anchor=anchor,
            window=window,
        )

    results = []
    correct = 0
    parse_failed = 0
    per_class = defaultdict(lambda: {"tp": 0, "total": 0})
    labels = ["True", "False"]
    all_preds = labels + ["PARSE_FAILED"]
    cm = {a: {b: 0 for b in all_preds} for a in labels}

    bs = max(1, batch_size)
    for batch in batched(samples, bs):
        prompts = []
        for s in batch:
            ptxt = build_few_shot_prompt(s["text"])
            if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
                msgs = [
                    {"role": "system", "content": "You are a helpful reasoning assistant."},
                    {"role": "user", "content": ptxt},
                ]
                chat_kwargs = dict(add_generation_prompt=True, tokenize=False)
                chat_kwargs["enable_thinking"] = bool(enable_thinking)
                try:
                    chat_text = tok.apply_chat_template(msgs, **chat_kwargs)
                except TypeError:
                    chat_kwargs.pop("enable_thinking", None)
                    chat_text = tok.apply_chat_template(msgs, **chat_kwargs)
                prompts.append(chat_text)
            else:
                prompts.append(ptxt)

        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        texts = tok.batch_decode(out, skip_special_tokens=True)

        for text, s in zip(texts, batch):
            pred = parse_truth(text)
            gold = s["gold"]
            if pred is None:
                parse_failed += 1
                per_class[gold]["total"] += 1
                pred = "PARSE_FAILED"
            else:
                hit = int(pred == gold)
                correct += hit
                per_class[gold]["total"] += 1
                if hit:
                    per_class[gold]["tp"] += 1
            cm[gold][pred] += 1
            results.append({"story_id": s["story_id"], "gold": gold, "pred": pred, "text": s["text"], "gen": text})

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

    if steerer is not None:
        steerer.close()

    return {
        "lambda": float(lam),
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
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--file", type=str, required=True, help="ProntoQA JSON file")
    ap.add_argument("--svcca_pt", type=str, required=True, help="SVCCA results")
    ap.add_argument("--layer_start", type=int, required=True)
    ap.add_argument("--layer_end", type=int, required=True)
    ap.add_argument(
        "--view",
        type=str,
        default="NL_without_proof",
        choices=["NL_with_proof", "NL_without_proof", "FOL_with_proof", "FOL_without_proof"],
    )
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")

    ap.add_argument("--use_projectors", action="store_true")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--corr_min", type=float, default=0.0)
    ap.add_argument("--anchor", choices=["last", "all"], default="all")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--enable_thinking", action="store_true")

    ap.add_argument("--lam_start", type=float, default=0.0)
    ap.add_argument("--lam_end", type=float, default=1.0)
    ap.add_argument("--lam_step", type=float, default=0.1)

    ap.add_argument("--out_csv", type=str, default="prontoqa_3shot_steering.csv")
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    data = read_prontoqa_json(args.file)
    print(f"Loaded {len(data)} samples from {args.file}")
    print(f"Using view: {args.view}")

    samples = []
    for item in data:
        story_id = item.get("story_id", "unknown")
        gold_label = item.get("label")
        view_text = get_view_text(item, args.view)
        if view_text is None or gold_label is None:
            continue
        gold = norm_truth(gold_label)
        if gold is None:
            continue
        samples.append({"story_id": story_id, "text": view_text, "gold": gold})

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[: args.max_samples]
    if not samples:
        print("No valid samples.", file=sys.stderr)
        sys.exit(1)
    print(f"Prepared {len(samples)} valid samples")

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
            args.model,
            device_map="auto",
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            **load_kwargs,
        )
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )

    mdl.eval()
    for module in mdl.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    mdl.generation_config.pad_token_id = tok.pad_token_id
    mdl.generation_config.eos_token_id = tok.eos_token_id
    mdl.generation_config.use_cache = True

    device = primary_device(mdl)
    print("Primary device:", device)

    layer_list, _ = find_decoder_layers(mdl)
    total_layers = len(layer_list)
    if not (0 <= args.layer_start < total_layers):
        print(f"[error] layer_start {args.layer_start} out of range (0..{total_layers - 1})", file=sys.stderr)
        sys.exit(1)
    if not (0 <= args.layer_end < total_layers):
        print(f"[error] layer_end {args.layer_end} out of range (0..{total_layers - 1})", file=sys.stderr)
        sys.exit(1)
    if args.layer_start > args.layer_end:
        print(f"[error] layer_start ({args.layer_start}) > layer_end ({args.layer_end})", file=sys.stderr)
        sys.exit(1)

    layers_to_scan = list(range(args.layer_start, args.layer_end + 1))
    lam_vals = []
    lam = args.lam_start
    while lam <= args.lam_end + 1e-9:
        lam_vals.append(round(lam, 6))
        lam += args.lam_step
    lam_vals = sorted(set(lam_vals))

    ensure_dir(args.out_csv)
    first_write = not os.path.exists(args.out_csv)
    fcsv = open(args.out_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(fcsv)
    if first_write:
        w.writerow(
            [
                "layer",
                "lambda",
                "total",
                "parsed",
                "parse_failed",
                "accuracy",
                "accuracy_with_failed",
                "macro_f1",
                "cm_T_T",
                "cm_T_F",
                "cm_T_PARSE_FAILED",
                "cm_F_T",
                "cm_F_F",
                "cm_F_PARSE_FAILED",
                "prec_T",
                "rec_T",
                "f1_T",
                "n_T",
                "prec_F",
                "rec_F",
                "f1_F",
                "n_F",
                "model",
                "file",
                "view",
                "svcca_pt",
                "useP",
                "top_k",
                "corr_min",
                "anchor",
                "window",
                "prompt_style",
            ]
        )
        fcsv.flush()

    for layer in tqdm(layers_to_scan, desc="Scanning layers"):
        print(f"\nProcessing Layer {layer}")
        for lam in tqdm(lam_vals, desc=f"Layer {layer} lambda sweep", leave=False):
            out = run_eval_once(
                mdl=mdl,
                tok=tok,
                device=device,
                samples=samples,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                svcca_pt=args.svcca_pt,
                layer=layer,
                lam=lam,
                use_projectors=args.use_projectors,
                top_k=args.top_k,
                corr_min=args.corr_min,
                anchor=args.anchor,
                window=args.window,
                enable_thinking=args.enable_thinking,
            )
            cm = out["cm"]
            per = out["per_class"]
            row = [
                layer,
                lam,
                out["total"],
                out["parsed"],
                out["parse_failed"],
                out["accuracy"],
                out["accuracy_with_failed"],
                out["macro_f1"],
                cm["True"]["True"],
                cm["True"]["False"],
                cm["True"]["PARSE_FAILED"],
                cm["False"]["True"],
                cm["False"]["False"],
                cm["False"]["PARSE_FAILED"],
                per["True"]["prec"],
                per["True"]["rec"],
                per["True"]["f1"],
                per["True"]["n"],
                per["False"]["prec"],
                per["False"]["rec"],
                per["False"]["f1"],
                per["False"]["n"],
                args.model,
                os.path.abspath(args.file),
                args.view,
                os.path.abspath(args.svcca_pt),
                int(args.use_projectors),
                int(args.top_k),
                float(args.corr_min),
                args.anchor,
                int(args.window),
                "3shot",
            ]
            w.writerow(row)
            fcsv.flush()

            if args.save_preds:
                pred_dir = os.path.join(os.path.dirname(args.out_csv) or ".", "preds_3shot")
                os.makedirs(pred_dir, exist_ok=True)
                pth = os.path.join(pred_dir, f"preds_3shot_layer{layer}_lambda_{lam:.2f}.jsonl")
                with open(pth, "w", encoding="utf-8") as fp:
                    for r in out["results"]:
                        fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    fcsv.close()
    print(f"\nResults saved to: {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()

