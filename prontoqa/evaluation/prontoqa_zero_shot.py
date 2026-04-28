#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re, sys
from collections import defaultdict
from typing import List, Dict

import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ---- Prompt ----
PROMPT_TEMPLATE = """You are a careful logician. Use classical deductive reasoning.

{text}

Instructions:
- First, reason step by step.
- Then, on the last line, output exactly:
Truth value: <True|False>
"""

VAL_RE = re.compile(r"Truth\s*value\s*:\s*(True|False)", re.IGNORECASE)

# ---- Utils ----
def norm_truth(x: str):
    if x is None: return None
    t = x.strip().lower()
    if t in {"true","t"}: return "True"
    if t in {"false","f"}: return "False"
    m = VAL_RE.search(x)
    if m: return norm_truth(m.group(1))
    return None

def parse_truth(text: str):
    # 先找含 truth value 的行，取最后一行再跑正则
    lines = [ln.strip() for ln in text.splitlines() if "truth value" in ln.lower()]
    if lines:
        m = VAL_RE.search(lines[-1])
        if m: return norm_truth(m.group(1))
    m = list(VAL_RE.finditer(text))
    return norm_truth(m[-1].group(1)) if m else None

def build_prompt(text: str) -> str:
    """ProntoQA 的 text 已经包含了 premises 和 question，直接包装即可"""
    return PROMPT_TEMPLATE.format(text=text.strip())

def read_prontoqa_json(path: str) -> List[Dict]:
    """读取 ProntoQA JSON 格式数据"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def batched(it, bs):
    for i in range(0, len(it), bs):
        yield it[i:i+bs]

def primary_device(model):
    # 对 device_map=auto 的模型，取一个 cuda 设备；否则退回 model.device / cpu
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for dev in set(model.hf_device_map.values()):
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
    if hasattr(model, "device"):
        return model.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--file", type=str, required=True,
                    help="Path to ProntoQA JSON file (e.g., 5hop_0shot_noadj_processed_test.json)")
    ap.add_argument("--view", type=str, default="NL_without_proof",
                    choices=["NL_with_proof", "NL_without_proof", "FOL_with_proof", "FOL_without_proof"],
                    help="Which view to use for evaluation")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    ap.add_argument("--output", type=str, default="prontoqa_preds.jsonl")
    args = ap.parse_args()

    # 设置随机种子以确保可复现性
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # 读取数据
    data = read_prontoqa_json(args.file)
    print(f"Loaded {len(data)} samples from {args.file}")
    print(f"Using view: {args.view}")

    if args.max_samples and args.max_samples < len(data):
        data = data[:args.max_samples]

    # 提取样本
    samples = []
    for item in data:
        story_id = item["story_id"]
        gold_label = item["label"]

        # 找到对应的 view
        view_text = None
        for view_item in item["pair"]:
            if view_item["view"] == args.view:
                view_text = view_item["text"]
                break

        if view_text is None:
            print(f"Warning: view {args.view} not found in {story_id}", file=sys.stderr)
            continue

        # 归一化标签
        gold = norm_truth(gold_label)
        if gold is None:
            print(f"Warning: invalid label '{gold_label}' in {story_id}", file=sys.stderr)
            continue

        samples.append({
            "story_id": story_id,
            "text": view_text,
            "gold": gold
        })

    if not samples:
        print("No valid samples parsed. Check your data file.", file=sys.stderr)
        sys.exit(1)

    print(f"Prepared {len(samples)} valid samples")

    # 模型与 tokenizer
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
            attn_implementation="flash_attention_2", low_cpu_mem_usage=True, **load_kwargs
        )
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=dtype,
            attn_implementation="sdpa", low_cpu_mem_usage=True, **load_kwargs
        )

    mdl.generation_config.pad_token_id = tok.pad_token_id
    mdl.generation_config.eos_token_id = tok.eos_token_id
    mdl.generation_config.use_cache = True

    # 设备检查
    pdev = primary_device(mdl)
    print("Primary device:", pdev)
    print("cuda_available:", torch.cuda.is_available(), "torch:", torch.__version__, "built_cuda:", torch.version.cuda)

    # 轻微加速
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # 推理
    results = []
    correct = 0
    parse_failed = 0
    per_class = defaultdict(lambda: {"tp":0,"total":0})
    bs = max(1, args.batch_size)

    for batch in tqdm(list(batched(samples, bs)), ncols=100, desc="Evaluating"):
        prompts = []
        for s in batch:
            ptxt = build_prompt(s["text"])
            if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
                msgs = [{"role":"system","content":"You are a helpful reasoning assistant."},
                        {"role":"user","content":ptxt}]
                chat_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                prompts.append(chat_text)
            else:
                prompts.append(ptxt)

        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(pdev) for k, v in enc.items()}

        with torch.inference_mode():
            try:
                out = mdl.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # OOM 时把 batch 切半重试
                outs = []
                half = len(prompts)//2 or 1
                for sub in (prompts[:half], prompts[half:]):
                    enc_sub = tok(sub, return_tensors="pt", padding=True, truncation=True)
                    enc_sub = {k:v.to(pdev) for k,v in enc_sub.items()}
                    outs.append(mdl.generate(**enc_sub, max_new_tokens=args.max_new_tokens, do_sample=False))
                out = torch.cat(outs, dim=0)

        texts = tok.batch_decode(out, skip_special_tokens=True)
        for text, s in zip(texts, batch):
            pred = parse_truth(text)
            if pred is None:
                parse_failed += 1
                per_class[s["gold"]]["total"] += 1
                pred = "PARSE_FAILED"
            else:
                hit = int(pred == s["gold"])
                correct += hit
                per_class[s["gold"]]["total"] += 1
                if hit: per_class[s["gold"]]["tp"] += 1
            results.append({
                "story_id": s["story_id"],
                "gold": s["gold"],
                "pred": pred,
                "gen": text
            })

    # 指标
    total = len(samples)
    parsed = total - parse_failed
    acc = correct / parsed if parsed else 0.0
    acc_with_failed = correct / total if total else 0.0

    print(f"\n{'='*60}")
    print(f"ProntoQA Zero-Shot Evaluation Results")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"View: {args.view}")
    print(f"Data: {args.file}")
    print(f"\nTotal={total}  Parsed={parsed}  Parse_Failed={parse_failed} ({parse_failed/total*100:.1f}%)")
    print(f"Accuracy (on parsed samples)={acc*100:.2f}%")
    print(f"Accuracy (treat parse_failed as wrong): {acc_with_failed*100:.2f}%")

    print(f"\nPer-class accuracy:")
    for c in ["True","False"]:
        if per_class[c]["total"]>0:
            pc = per_class[c]["tp"]/per_class[c]["total"]*100
            print(f"  {c:7s}: {pc:.2f}%  (n={per_class[c]['total']})")

    # 混淆矩阵与宏 F1
    labels = ["True","False"]
    all_preds = labels + ["PARSE_FAILED"]
    cm = {a:{b:0 for b in all_preds} for a in labels}
    for r in results: cm[r["gold"]][r["pred"]] += 1

    print("\nConfusion matrix (gold x pred):")
    print("         " + " ".join(f"{p:>12s}" for p in all_preds))
    for g in labels:
        print(f"  {g:7s}: " + " ".join(f"{cm[g][p]:>12d}" for p in all_preds))

    tp = {l:cm[l][l] for l in labels}
    fp = {l:sum(cm[g][l] for g in labels if g!=l) for l in labels}
    fn = {l:sum(cm[l][p] for p in all_preds if p!=l) for l in labels}

    prec = {l:(tp[l]/(tp[l]+fp[l]) if tp[l]+fp[l]>0 else 0.0) for l in labels}
    rec  = {l:(tp[l]/(tp[l]+fn[l]) if tp[l]+fn[l]>0 else 0.0) for l in labels}
    f1   = {l:(2*prec[l]*rec[l]/(prec[l]+rec[l]) if (prec[l]+rec[l])>0 else 0.0) for l in labels}
    macro_f1 = sum(f1.values())/len(labels)

    print(f"\nMacro-F1 = {macro_f1*100:.2f}%")

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved predictions to {args.output}")

if __name__ == "__main__":
    main()
