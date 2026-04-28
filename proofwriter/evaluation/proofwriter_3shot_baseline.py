#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProofWriter CWA 3-shot baseline evaluation (no steering)
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
# Few-shot Examples (from NL_with_proof view)
# =========================
FEW_SHOT_EXAMPLES = [
    {
        "label": "True",
        "text": """Facts:
Bob is big.
Erin is big.
Erin is cold.
Erin is red.
Fiona is red.
Harry is big.
Harry is blue.
Harry is nice.
Harry is quiet.
Harry is young.

Rules:
Young, big people are nice.
All red, young people are big.
If someone is red then they are quiet.
Quiet, red people are blue.
If someone is big and nice then they are quiet.
If Fiona is cold then Fiona is blue.
If Fiona is young and Fiona is red then Fiona is cold.
All red, blue people are young.

Query: Fiona is nice.

Proof:
Fiona is red. If someone is red then they are quiet. Therefore fiona is quiet.
Fiona is quiet and Fiona is red. Quiet, red people are blue. Therefore fiona is blue.
Fiona is red and Fiona is blue. All red, blue people are young. Therefore fiona is young.
Fiona is red and Fiona is young. All red, young people are big. Therefore fiona is big.
Fiona is young and Fiona is big. Young, big people are nice. Therefore fiona is nice.

The query is true."""
    },
    {
        "label": "False",
        "text": """Facts:
Charlie is cold.
Charlie is round.
Charlie is young.
Erin is big.
Gary is big.
Gary is young.
Harry is big.
Harry is cold.
Harry is quiet.
Harry is red.
Harry is round.
Harry is young.

Rules:
If Harry is big and Harry is red then Harry is round.
All quiet things are young.
Rough, young things are red.
If Erin is big then Erin is young.
All big, cold things are round.
If Charlie is rough then Charlie is round.
Big, round things are rough.
Big, young things are cold.

Query: Erin is not red.

Proof:
Erin is big. If Erin is big then Erin is young. Therefore erin is young.
Erin is big and Erin is young. Big, young things are cold. Therefore erin is cold.
Erin is big and Erin is cold. All big, cold things are round. Therefore erin is round.
Erin is big and Erin is round. Big, round things are rough. Therefore erin is rough.
Erin is rough and Erin is young. Rough, young things are red. Therefore erin is red.

The query is false."""
    },
    {
        "label": "True",
        "text": """Facts:
The bear is blue.
The bear is nice.
The bear sees the tiger.
The dog needs the bear.
The dog sees the tiger.
The rabbit needs the dog.
The tiger is rough.

Rules:
If someone is nice then they are rough.
If someone is rough then they see the rabbit.
All rough people are kind.
If someone chases the rabbit and the rabbit is kind then they need the dog.
If someone is green then they are nice.
If someone is kind and they chase the bear then they chase the tiger.
If someone chases the bear then they chase the rabbit.
If the bear sees the rabbit then the rabbit is green.
If the rabbit needs the dog then the rabbit sees the tiger.

Query: The rabbit is rough.

Proof:
The bear is nice. If someone is nice then they are rough. Therefore the bear is rough.
The bear is rough. If someone is rough then they see the rabbit. Therefore the bear sees the rabbit.
The bear sees the rabbit. If the bear sees the rabbit then the rabbit is green. Therefore the rabbit is green.
The rabbit is green. If someone is green then they are nice. Therefore the rabbit is nice.
The rabbit is nice. If someone is nice then they are rough. Therefore the rabbit is rough.

The query is true."""
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
Truth value: <True|False>
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
VAL_RE = re.compile(r"Truth\s*value\s*:\s*(True|False)", re.IGNORECASE)

def norm_truth(x: str):
    if x is None: return None
    t = x.strip().lower()
    if t in {"true","t"}: return "True"
    if t in {"false","f"}: return "False"
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
    ap.add_argument("--output", type=str, default="proofwriter_3shot_preds.jsonl")
    ap.add_argument("--out_csv", type=str, default="proofwriter_3shot.csv")
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
    per_class = defaultdict(lambda: {"tp":0,"total":0})
    labels = ["True","False"]
    all_preds = labels + ["PARSE_FAILED"]
    cm = {a:{b:0 for b in all_preds} for a in labels}

    bs = max(1, args.batch_size)
    for batch in tqdm(list(batched(samples, bs)), desc="Evaluating 3-shot"):
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
    print(f"ProofWriter CWA 3-Shot Baseline Results")
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
        print(f"  {g:7s}: " + " ".join(f"{cm[g][p]:>12d}" for p in all_preds))

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
                    "prec_T","rec_T","f1_T","prec_F","rec_F","f1_F"])
        w.writerow([args.model, args.view, total, parsed, parse_failed, acc, acc_with_failed, macro_f1,
                    prec["True"], rec["True"], f1["True"],
                    prec["False"], rec["False"], f1["False"]])
    print(f"Saved summary to {args.out_csv}")

if __name__ == "__main__":
    main()
