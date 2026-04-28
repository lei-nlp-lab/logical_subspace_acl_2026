#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct Answer baseline for ProofWriter OWA - No CoT, No steering
直接回答 baseline：不使用 Chain-of-Thought，让模型直接输出答案
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


# Direct answer prompt - 不要求 CoT，直接回答 (Ternary: True/False/Uncertain)
DIRECT_PROMPT_TEMPLATE = """Given the facts, rules, and query, answer with ONLY one word: True, False, or Uncertain.

{text}

Answer:"""


def build_direct_prompt(text: str) -> str:
    return DIRECT_PROMPT_TEMPLATE.format(text=text.strip())


def parse_direct_answer(text: str):
    """解析直接回答的输出 (Ternary: True/False/Uncertain)

    所有 prompt 都以 "Answer:" 结尾，模型回答紧跟其后
    """
    # 统一用 "Answer:" 分割，取最后一部分
    if "answer:" in text.lower():
        response = text.lower().split("answer:")[-1].strip()
    else:
        response = text.lower().strip()

    # 遍历词汇，跳过 chat 标记词 (assistant/model)
    for word in response.split():
        word_clean = re.sub(r'[^\w]', '', word)
        if word_clean in {"true", "t", "yes"}:
            return "True"
        if word_clean in {"false", "f", "no"}:
            return "False"
        if word_clean in {"uncertain", "unknown", "u"}:
            return "Unknown"
        if word_clean in {"assistant", "model", ""}:
            continue
        break  # 遇到其他词停止

    # fallback: 搜索整个 response
    if "true" in response:
        return "True"
    if "false" in response:
        return "False"
    if "uncertain" in response or "unknown" in response:
        return "Unknown"

    return None


def norm_truth(x: str):
    """ProofWriter OWA: True/False/Unknown"""
    if x is None:
        return None
    t = str(x).strip().lower()
    if t in {"true", "t", "yes", "1"}:
        return "True"
    if t in {"false", "f", "no", "0"}:
        return "False"
    if t in {"unknown", "uncertain", "u"}:
        return "Unknown"
    return None


def read_prontoqa_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_view_text(rec: Dict[str, Any], view_name: str) -> str:
    for item in rec.get("pair", []):
        if item.get("view", "") == view_name:
            return item.get("text", "")
    return ""


def primary_device(model):
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for dev in set(model.hf_device_map.values()):
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
    if hasattr(model, "device"):
        return model.device
    return torch.device("cpu")


def batched(it, bs):
    for i in range(0, len(it), bs):
        yield it[i:i+bs]


def ensure_dir(p):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)


def run_direct_answer_eval(mdl, tok, device, samples, max_new_tokens, batch_size):
    """运行直接回答评测 (Ternary)"""
    results = []
    correct = 0
    parse_failed = 0
    per_class = defaultdict(lambda: {"tp": 0, "total": 0})
    labels = ["True", "False", "Unknown"]
    all_preds = labels + ["PARSE_FAILED"]
    cm = {a: {b: 0 for b in all_preds} for a in labels}

    def encode_batch(prompts: List[str]):
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
        return enc

    def generate_texts(enc):
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        texts = tok.batch_decode(out, skip_special_tokens=True)
        return texts

    bs = max(1, batch_size)
    for batch in tqdm(batched(samples, bs), total=(len(samples) + bs - 1) // bs, desc="Evaluating"):
        prompts = []
        for s in batch:
            ptxt = build_direct_prompt(s["text"])
            if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
                msgs = [
                    {"role": "system", "content": "You are a logical reasoning assistant. Answer directly without explanation."},
                    {"role": "user", "content": ptxt}
                ]
                chat_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                prompts.append(chat_text)
            else:
                prompts.append(ptxt)
        enc = encode_batch(prompts)
        texts = generate_texts(enc)

        for text, s in zip(texts, batch):
            pred = parse_direct_answer(text)
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
            results.append({
                "story_id": s.get("story_id", ""),
                "gold": gold, "pred": pred,
                "text": s["text"], "gen": text
            })

    # 汇总指标
    total = len(samples)
    parsed = total - parse_failed
    acc = correct / parsed if parsed else 0.0
    acc_with_failed = correct / total if total else 0.0

    # per-class prec/rec/f1 + macroF1
    tp = {l: cm[l][l] for l in labels}
    fp = {l: sum(cm[g][l] for g in labels if g != l) for l in labels}
    fn = {l: sum(cm[l][p] for p in all_preds if p != l) for l in labels}

    prec = {l: (tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0) for l in labels}
    rec = {l: (tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0) for l in labels}
    f1 = {l: (2 * prec[l] * rec[l] / (prec[l] + rec[l]) if (prec[l] + rec[l]) > 0 else 0.0) for l in labels}
    macro_f1 = sum(f1.values()) / len(labels)

    out = {
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
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    ap.add_argument("--file", type=str, required=True, help="ProofWriter JSON file")
    ap.add_argument("--view", type=str, default="NL_without_proof")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    ap.add_argument("--out_csv", type=str, default="direct_answer_owa.csv")
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()

    # 设置随机种子
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"Warning: Could not enable deterministic algorithms: {e}")

    # 读取数据
    data = read_prontoqa_json(args.file)
    samples = []
    for rec in data:
        view_text = get_view_text(rec, args.view)
        if not view_text:
            continue
        gold = norm_truth(rec.get("label"))
        if gold is None:
            continue
        samples.append({
            "story_id": rec.get("story_id", ""),
            "text": view_text,
            "gold": gold
        })

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
    if not samples:
        print("No valid samples.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(samples)} valid samples from {args.file}")
    print(f"Using view: {args.view}")

    # 模型与 tokenizer
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
    for module in mdl.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    mdl.generation_config.pad_token_id = tok.pad_token_id
    mdl.generation_config.eos_token_id = tok.eos_token_id
    mdl.generation_config.use_cache = True

    device = primary_device(mdl)
    print(f"Primary device: {device}")
    print(f"Model: {args.model}")

    # 运行评测
    out = run_direct_answer_eval(
        mdl=mdl, tok=tok, device=device, samples=samples,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
    )

    # 输出结果
    print(f"\n{'='*60}")
    print("Direct Answer Results - ProofWriter OWA (No CoT)")
    print(f"{'='*60}")
    print(f"Total: {out['total']}, Parsed: {out['parsed']}, Parse Failed: {out['parse_failed']}")
    print(f"Accuracy: {out['accuracy']:.4f}")
    print(f"Accuracy (with failed): {out['accuracy_with_failed']:.4f}")
    print(f"Macro F1: {out['macro_f1']:.4f}")
    print(f"\nPer-class metrics:")
    for l in ["True", "False", "Unknown"]:
        p = out['per_class'][l]
        print(f"  {l}: prec={p['prec']:.4f}, rec={p['rec']:.4f}, f1={p['f1']:.4f}, n={p['n']}")

    # 写 CSV
    ensure_dir(args.out_csv)
    first_write = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        if first_write:
            w.writerow([
                "view", "total", "parsed", "parse_failed", "accuracy", "accuracy_with_failed", "macro_f1",
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
        row = [
            args.view,
            out["total"], out["parsed"], out["parse_failed"],
            out["accuracy"], out["accuracy_with_failed"], out["macro_f1"],
            cm["True"]["True"], cm["True"]["False"], cm["True"]["Unknown"], cm["True"]["PARSE_FAILED"],
            cm["False"]["True"], cm["False"]["False"], cm["False"]["Unknown"], cm["False"]["PARSE_FAILED"],
            cm["Unknown"]["True"], cm["Unknown"]["False"], cm["Unknown"]["Unknown"], cm["Unknown"]["PARSE_FAILED"],
            per["True"]["prec"], per["True"]["rec"], per["True"]["f1"], per["True"]["n"],
            per["False"]["prec"], per["False"]["rec"], per["False"]["f1"], per["False"]["n"],
            per["Unknown"]["prec"], per["Unknown"]["rec"], per["Unknown"]["f1"], per["Unknown"]["n"],
            args.model, os.path.abspath(args.file)
        ]
        w.writerow(row)

    print(f"\nResults saved to: {os.path.abspath(args.out_csv)}")

    # 保存逐样本预测
    if args.save_preds:
        pred_path = args.out_csv.replace(".csv", "_preds.jsonl")
        with open(pred_path, "w", encoding="utf-8") as fp:
            for r in out["results"]:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Predictions saved to: {os.path.abspath(pred_path)}")


if __name__ == "__main__":
    main()
