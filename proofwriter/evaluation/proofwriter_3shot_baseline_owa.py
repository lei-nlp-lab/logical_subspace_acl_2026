#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProofWriter OWA 3-shot baseline evaluation (no steering)
Ternary classification: True/False/Uncertain
"""

import argparse, os, csv, json, sys, re
from collections import defaultdict
from typing import List, Dict, Any

import torch
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Few-shot Examples for OWA (True/False/Uncertain)
# =========================
FEW_SHOT_EXAMPLES = [
    {
        "label": "True",
        "text": """Facts:
Anne is blue.
Anne is round.
Anne is smart.
Anne is young.
Dave is big.
Dave is blue.
Dave is nice.
Nice, round things are green.
Nice things are big.
Blue things are young.

Rules:
If Anne is blue and Anne is young then Anne is round.
All green, nice things are smart.
All big, young things are green.
If something is big then it is young.
All nice things are blue.

Query: Anne is round.

Proof:
Anne is round is stated as a fact.

The query is true."""
    },
    {
        "label": "False",
        "text": """Facts:
Charlie is cold.
Charlie is green.
Charlie is kind.
Charlie is rough.
Gary is green.
Gary is kind.
Gary is round.
Cold, nice people are round.
Nice, young people are kind.
Round, cold people are young.

Rules:
If someone is rough then they are kind.
If Gary is green and Gary is young then Gary is kind.
If someone is round then they are nice.
If someone is kind and young then they are rough.
All nice people are cold.

Query: Gary is not cold.

Proof:
Gary is round. If someone is round then they are nice. Therefore Gary is nice.
Gary is nice. All nice people are cold. Therefore Gary is cold.
The query states Gary is not cold, but we proved Gary is cold.

The query is false."""
    },
    {
        "label": "Uncertain",
        "text": """Facts:
The bald eagle is red.
The bear is young.
The cow is green.
The lion sees the bear.
Green people are blue.

Rules:
If someone is blue then they see the bear.
If the cow is blue then the cow likes the lion.
If someone likes the lion then they see the cow.
If someone likes the cow and the cow is red then they see the bald eagle.
If someone likes the cow and the cow visits the bald eagle then the cow likes the bald eagle.

Query: The bald eagle does not like the lion.

Proof:
The cow is green. Green people are blue. Therefore the cow is blue.
The cow is blue. If the cow is blue then the cow likes the lion. Therefore the cow likes the lion.
The cow likes the lion. If someone likes the lion then they see the cow. Therefore the cow sees the cow.
We cannot derive whether the bald eagle likes or does not like the lion from the given facts and rules.

The query is uncertain."""
    }
]

# =========================
# Prompt Template
# =========================
PROMPT_TEMPLATE = """You are a careful logician. Use classical deductive reasoning.

{examples}
=== Your Turn ===
{text}

Instructions:
- First, reason step by step.
- Then, on the last line, output exactly:
Truth value: <True|False|Uncertain>
"""

def build_few_shot_prompt(text: str) -> str:
    """Build prompt with 3-shot examples"""
    examples_str = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_str += f"=== Example {i} ===\n"
        examples_str += ex['text'] + "\n"
        examples_str += f"\nTruth value: {ex['label']}\n\n"

    return PROMPT_TEMPLATE.format(examples=examples_str, text=text.strip())

# =========================
# Parsing
# =========================
VAL_RE = re.compile(r"Truth\s*value\s*:\s*(True|False|Uncertain)", re.IGNORECASE)

def norm_truth(x: str):
    if x is None: return None
    t = x.strip().lower()
    if t in {"true","t"}: return "True"
    if t in {"false","f"}: return "False"
    if t in {"uncertain","u","unknown"}: return "Uncertain"
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

# =========================
# Utils
# =========================
def read_proofwriter_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_view_text(rec: Dict[str, Any], view_name: str) -> str:
    pair = rec.get("pair") or []
    for item in pair:
        if item.get("view", "") == view_name:
            return item.get("text", "")
    return None

def batched(it, bs):
    for i in range(0, len(it), bs):
        yield it[i:i+bs]

def primary_device(model):
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for dev in set(model.hf_device_map.values()):
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
    if hasattr(model, "device"):
        return model.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--file", type=str, required=True, help="ProofWriter JSON file")
    ap.add_argument("--view", type=str, default="NL_without_proof",
                    choices=["NL_with_proof", "NL_without_proof"])
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    ap.add_argument("--output", type=str, default="proofwriter_owa_3shot_preds.jsonl")
    ap.add_argument("--out_csv", type=str, default="proofwriter_owa_3shot.csv")
    args = ap.parse_args()

    # Set random seed
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Read data
    data = read_proofwriter_json(args.file)
    print(f"Loaded {len(data)} samples from {args.file}")
    print(f"Using view: {args.view}")

    samples = []
    for item in data:
        story_id = item.get("story_id", "")
        gold_label = item.get("label", "")
        view_text = get_view_text(item, args.view)

        if view_text is None:
            continue

        gold = norm_truth(str(gold_label))
        if gold is None:
            continue

        samples.append({
            "story_id": story_id,
            "text": view_text,
            "gold": gold
        })

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
    if not samples:
        print("No valid samples.", file=sys.stderr)
        sys.exit(1)

    print(f"Prepared {len(samples)} valid samples")

    # Load model
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
    print("Primary device:", device)

    # Evaluation
    results = []
    correct = 0
    parse_failed = 0
    labels = ["True", "False", "Uncertain"]
    all_preds = labels + ["PARSE_FAILED"]
    per_class = defaultdict(lambda: {"tp":0,"total":0})
    cm = {a:{b:0 for b in all_preds} for a in labels}

    bs = max(1, args.batch_size)
    for batch in tqdm(list(batched(samples, bs)), desc="Evaluating OWA 3-shot"):
        prompts = []
        for s in batch:
            ptxt = build_few_shot_prompt(s["text"])
            if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
                msgs = [{"role":"system","content":"You are a helpful reasoning assistant."},
                        {"role":"user","content":ptxt}]
                chat_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                prompts.append(chat_text)
            else:
                prompts.append(ptxt)

        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            out = mdl.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
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
                if hit: per_class[gold]["tp"] += 1
            cm[gold][pred] += 1
            results.append({"story_id": s["story_id"], "gold": gold, "pred": pred, "gen": text})

    # Metrics
    total = len(samples)
    parsed = total - parse_failed
    acc = correct / parsed if parsed else 0.0
    acc_with_failed = correct / total if total else 0.0

    tp = {l:cm[l][l] for l in labels}
    fp = {l:sum(cm[g][l] for g in labels if g!=l) for l in labels}
    fn = {l:sum(cm[l][p] for p in all_preds if p!=l) for l in labels}

    prec = {l:(tp[l]/(tp[l]+fp[l]) if (tp[l]+fp[l])>0 else 0.0) for l in labels}
    rec  = {l:(tp[l]/(tp[l]+fn[l]) if (tp[l]+fn[l])>0 else 0.0) for l in labels}
    f1   = {l:(2*prec[l]*rec[l]/(prec[l]+rec[l]) if (prec[l]+rec[l])>0 else 0.0) for l in labels}
    macro_f1 = sum(f1.values())/len(labels)

    # Print results
    print(f"\n{'='*60}")
    print(f"ProofWriter OWA 3-Shot Baseline Results")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"View: {args.view}")
    print(f"Total={total}  Parsed={parsed}  Parse_Failed={parse_failed}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Accuracy (with failed): {acc_with_failed*100:.2f}%")
    print(f"Macro F1: {macro_f1*100:.2f}%")
    print(f"\nPer-class:")
    for l in labels:
        print(f"  {l}: Prec={prec[l]*100:.1f}% Rec={rec[l]*100:.1f}% F1={f1[l]*100:.1f}%")

    print(f"\nConfusion Matrix:")
    print("         " + " ".join(f"{p:>12s}" for p in all_preds))
    for g in labels:
        print(f"  {g:9s}: " + " ".join(f"{cm[g][p]:>12d}" for p in all_preds))

    # Save predictions
    ensure_dir(args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved predictions to {args.output}")

    # Save CSV summary
    ensure_dir(args.out_csv)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","view","total","parsed","parse_failed","accuracy","accuracy_with_failed","macro_f1",
                    "prec_T","rec_T","f1_T","prec_F","rec_F","f1_F","prec_U","rec_U","f1_U"])
        w.writerow([args.model, args.view, total, parsed, parse_failed, acc, acc_with_failed, macro_f1,
                    prec["True"], rec["True"], f1["True"],
                    prec["False"], rec["False"], f1["False"],
                    prec["Uncertain"], rec["Uncertain"], f1["Uncertain"]])
    print(f"Saved summary to {args.out_csv}")

if __name__ == "__main__":
    main()
