#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FOLIO 3-shot baseline evaluation (no steering)
"""

import argparse, os, csv, json, sys, re
from collections import defaultdict
from typing import List, Dict

import torch
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Few-shot Examples
# =========================
FEW_SHOT_EXAMPLES = [
    {
        "premises": """All people who regularly drink coffee are dependent on caffeine.
People regularly drink coffee, or they don't want to be addicted to caffeine, or both.
No one who doesn't want to be addicted to caffeine is unaware that caffeine is a drug.
Rina is either a student who is unaware that caffeine is a drug, or she is not a student and is she aware that caffeine is a drug.
Rina is either a student who is dependent on caffeine, or she is not a student and not dependent on caffeine.""",
        "conclusion": "Rina doesn't want to be addicted to caffeine or is unaware that caffeine is a drug.",
        "label": "True",
        "reasoning": """Let's analyze step by step:
1. From premise 5: Rina is either (student AND dependent on caffeine) OR (not student AND not dependent on caffeine).
2. From premise 4: Rina is either (student AND unaware caffeine is a drug) OR (not student AND aware caffeine is a drug).
3. Case 1: If Rina is a student, then she is dependent on caffeine (from 1) and unaware that caffeine is a drug (from 2).
   - Being unaware that caffeine is a drug satisfies the conclusion.
4. Case 2: If Rina is not a student, then she is not dependent on caffeine (from 1) and aware that caffeine is a drug (from 2).
   - From premise 1: If not dependent on caffeine, then not regularly drinking coffee.
   - From premise 2: If not regularly drinking coffee, then doesn't want to be addicted to caffeine.
   - This satisfies the conclusion.
In both cases, the conclusion holds."""
    },
    {
        "premises": """All eels are fish.
No fish are plants.
Everything displayed in the collection is either a plant or an animal.
All multicellular animals are not bacteria.
All animals displayed in the collection are multicellular.
A sea eel is displayed in the collection.
The sea eel is an eel or an animal or not a plant.""",
        "conclusion": "The sea eel is bacteria.",
        "label": "False",
        "reasoning": """Let's analyze step by step:
1. From premise 6: The sea eel is displayed in the collection.
2. From premise 3: Since it's displayed, the sea eel is either a plant or an animal.
3. From premise 7: The sea eel is an eel or an animal or not a plant.
4. From premise 1: All eels are fish.
5. From premise 2: No fish are plants. So if the sea eel is an eel, it's a fish, thus not a plant.
6. From premise 3: Since it's displayed and not a plant, the sea eel must be an animal.
7. From premise 5: All animals displayed in the collection are multicellular.
8. So the sea eel is a multicellular animal.
9. From premise 4: All multicellular animals are not bacteria.
10. Therefore, the sea eel is NOT bacteria."""
    },
    {
        "premises": """Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
Any choral conductor is a musician.
Some musicians love music.
Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.""",
        "conclusion": "Miroslav Venhoda loved music.",
        "label": "Uncertain",
        "reasoning": """Let's analyze step by step:
1. From premise 1: Miroslav Venhoda was a choral conductor.
2. From premise 2: Any choral conductor is a musician. So Miroslav Venhoda is a musician.
3. From premise 3: Some musicians love music. This means at least one musician loves music, but not necessarily all.
4. We know Miroslav Venhoda is a musician, but we cannot conclude whether he specifically loves music.
5. The premises don't provide enough information to determine if Miroslav Venhoda loved music.
Therefore, the truth value is uncertain."""
    }
]

# =========================
# Prompt Templates
# =========================
def build_few_shot_prompt(premises: str, conclusion: str) -> str:
    """Build prompt with 3-shot examples"""
    prompt = "You are a careful logician. Use classical deductive reasoning.\n\n"

    # Add few-shot examples
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        prompt += f"=== Example {i} ===\n"
        prompt += f"Premises:\n{ex['premises']}\n\n"
        prompt += f"Hypothesis:\n{ex['conclusion']}\n\n"
        prompt += f"Reasoning:\n{ex['reasoning']}\n\n"
        prompt += f"Truth value: {ex['label']}\n\n"

    # Add test example
    prompt += "=== Your Turn ===\n"
    prompt += f"Premises:\n{premises.strip()}\n\n"
    prompt += f"Hypothesis:\n{conclusion.strip()}\n\n"
    prompt += "Instructions:\n"
    prompt += "- First, reason step by step.\n"
    prompt += "- Then, on the last line, output exactly:\n"
    prompt += "Truth value: <True|False|Uncertain>\n"

    return prompt

# =========================
# Parsing
# =========================
VAL_RE = re.compile(r"Truth\s*value\s*:\s*(True|False|Unknown|Uncertain)", re.IGNORECASE)

def norm_truth(x: str):
    if x is None: return None
    t = x.strip().lower()
    if t in {"true","t"}: return "True"
    if t in {"false","f"}: return "False"
    if t in {"unknown","uncertain","u"}: return "Unknown"
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
def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

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
    ap.add_argument("--file", type=str, required=True, help="FOLIO jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    ap.add_argument("--output", type=str, default="folio_3shot_baseline.jsonl")
    ap.add_argument("--out_csv", type=str, default="folio_3shot_baseline.csv")
    args = ap.parse_args()

    # Set random seed
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Read data
    rows = read_jsonl(args.file)
    def get_field(d, candidates, default=None):
        for k in candidates:
            if k in d: return d[k]
        return default

    samples = []
    for r in rows:
        prem = get_field(r, ["premises","Premises - NL","premises_nl","premises-nl"])
        concl = get_field(r, ["conclusion","Conclusions - NL","conclusion_nl","conclusion-nl"])
        lab = get_field(r, ["label","Truth Values","truth","gold_label"])
        if isinstance(prem, list): prem = " ".join(prem)
        if prem is None or concl is None or lab is None: continue
        gold = norm_truth(str(lab))
        if gold is None: continue
        samples.append({"prem": prem, "concl": concl, "gold": gold})

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
    if not samples:
        print("No valid samples.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(samples)} samples")

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
    labels = ["True","False","Unknown"]
    all_preds = labels + ["PARSE_FAILED"]
    cm = {a:{b:0 for b in all_preds} for a in labels}

    bs = max(1, args.batch_size)
    for batch in tqdm(list(batched(samples, bs)), desc="Evaluating 3-shot"):
        prompts = []
        for s in batch:
            ptxt = build_few_shot_prompt(s["prem"], s["concl"])
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
            results.append({"gold": gold, "pred": pred, "prem": s["prem"], "concl": s["concl"], "gen": text})

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
    print(f"FOLIO 3-Shot Baseline Results")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
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
        w.writerow(["model","total","parsed","parse_failed","accuracy","accuracy_with_failed","macro_f1",
                    "prec_T","rec_T","f1_T","prec_F","rec_F","f1_F","prec_U","rec_U","f1_U"])
        w.writerow([args.model, total, parsed, parse_failed, acc, acc_with_failed, macro_f1,
                    prec["True"], rec["True"], f1["True"],
                    prec["False"], rec["False"], f1["False"],
                    prec["Unknown"], rec["Unknown"], f1["Unknown"]])
    print(f"Saved summary to {args.out_csv}")

if __name__ == "__main__":
    main()
