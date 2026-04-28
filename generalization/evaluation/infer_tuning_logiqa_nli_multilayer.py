#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogiQA NLI steering evaluation - 跨数据集迁移实验
使用 ProntoQA 训练的 SVCCA 在 LogiQA NLI 任务上做 steering

NLI 任务：给定前提 (major_premise + minor_premise)，判断结论 (conclusion) 是否成立
标签：entailed / not entailed (二分类，类似 ProntoQA 的 True/False)
"""

import argparse, os, csv, json, sys, re
from collections import defaultdict
from typing import List, Dict, Any

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------- import helpers from steering_infer_normalized --------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "proofwriter", "evaluation"))
try:
    from steering_infer_normalized import HFSteererNormalized as HFSteerer
    from steering_infer import primary_device, find_decoder_layers
except Exception as e:
    print(f"Error importing steering modules: {e}", file=sys.stderr)
    sys.exit(1)

from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- LogiQA NLI Prompt ----
PROMPT_TEMPLATE = """You are a careful logician. Use classical deductive reasoning.

Premises:
{premises}

Conclusion:
{conclusion}

Instructions:
- First, reason step by step about whether the conclusion follows from the premises.
- Then, on the last line, output exactly:
Answer: <Entailed|Not Entailed>
"""

# 解析答案的正则
ANSWER_RE = re.compile(r"Answer\s*:\s*(Entailed|Not\s*Entailed)", re.IGNORECASE)

# ---- NLI Utils ----
LABELS = ["Entailed", "Not Entailed"]

def norm_label(x: str):
    """标准化标签"""
    if x is None:
        return None
    t = x.strip().lower()
    if t in {"entailed", "yes", "true"}:
        return "Entailed"
    if t in {"not entailed", "not_entailed", "no", "false"}:
        return "Not Entailed"
    return None

def parse_answer(text: str):
    """从生成文本中解析答案 Entailed/Not Entailed"""
    # 先找含 "answer" 的行，取最后一行
    lines = [ln.strip() for ln in text.splitlines() if "answer" in ln.lower()]
    if lines:
        m = ANSWER_RE.search(lines[-1])
        if m:
            return norm_label(m.group(1))
    # fallback: 全文搜索
    matches = list(ANSWER_RE.finditer(text))
    if matches:
        return norm_label(matches[-1].group(1))
    # 再 fallback: 找 entailed 或 not entailed
    text_lower = text.lower()
    if "not entailed" in text_lower:
        return "Not Entailed"
    if "entailed" in text_lower:
        return "Entailed"
    return None

def build_prompt(sample: Dict[str, Any]) -> str:
    """构建 NLI prompt"""
    # major_premise 可能是 list 或 string
    major = sample["major_premise"]
    if isinstance(major, list):
        major = " ".join(major)
    minor = sample["minor_premise"]

    # 组合前提
    premises_parts = []
    if major and major.strip():
        premises_parts.append(major.strip())
    if minor and minor.strip():
        premises_parts.append(minor.strip())
    premises = "\n".join(premises_parts) if premises_parts else "(No premises given)"

    return PROMPT_TEMPLATE.format(
        premises=premises,
        conclusion=sample["conclusion"].strip(),
    )

def batched(it, bs):
    for i in range(0, len(it), bs):
        yield it[i:i+bs]

def ensure_dir(p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

def read_nli_jsonl(path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    """读取 LogiQA NLI JSONL 格式数据"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # 检查必要字段
            if "label" not in rec or "conclusion" not in rec:
                continue
            # 标准化 label
            gold_label = norm_label(rec["label"])
            if gold_label is None:
                continue
            samples.append({
                "id": len(samples),
                "major_premise": rec.get("major_premise", ""),
                "minor_premise": rec.get("minor_premise", ""),
                "conclusion": rec["conclusion"],
                "gold": gold_label,
            })
            if max_samples > 0 and len(samples) >= max_samples:
                break
    return samples

def run_eval_once(mdl, tok, device, samples, max_new_tokens, batch_size,
                  svcca_pt, layer, lam, use_projectors, top_k, corr_min,
                  anchor, window):
    """
    运行一次评测（给定单层与 λ）
    """
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
    all_preds = LABELS + ["PARSE_FAILED"]
    cm = {a: {b: 0 for b in all_preds} for a in LABELS}

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
    for batch in batched(samples, bs):
        prompts = []
        for s in batch:
            ptxt = build_prompt(s)
            if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
                msgs = [{"role": "system", "content": "You are a helpful reasoning assistant."},
                        {"role": "user", "content": ptxt}]
                chat_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                prompts.append(chat_text)
            else:
                prompts.append(ptxt)
        enc = encode_batch(prompts)
        texts = generate_texts(enc)

        for text, s in zip(texts, batch):
            pred = parse_answer(text)
            gold = s["gold"]
            if pred is None:
                parse_failed += 1
                per_class[gold]["total"] += 1
                pred_for_cm = "PARSE_FAILED"
            else:
                hit = int(pred == gold)
                correct += hit
                per_class[gold]["total"] += 1
                if hit:
                    per_class[gold]["tp"] += 1
                pred_for_cm = pred
            cm[gold][pred_for_cm] += 1
            results.append({
                "id": s["id"],
                "gold": gold,
                "pred": pred,
                "major_premise": s["major_premise"],
                "minor_premise": s["minor_premise"],
                "conclusion": s["conclusion"],
                "gen": text
            })

    total = len(samples)
    parsed = total - parse_failed
    acc = correct / parsed if parsed else 0.0
    acc_with_failed = correct / total if total else 0.0

    # per-class metrics
    tp = {l: cm[l][l] for l in LABELS}
    fp = {l: sum(cm[g][l] for g in LABELS if g != l) for l in LABELS}
    fn = {l: sum(cm[l][p] for p in all_preds if p != l) for l in LABELS}

    prec = {l: (tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0) for l in LABELS}
    rec = {l: (tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0) for l in LABELS}
    f1 = {l: (2 * prec[l] * rec[l] / (prec[l] + rec[l]) if (prec[l] + rec[l]) > 0 else 0.0) for l in LABELS}
    macro_f1 = sum(f1.values()) / len(LABELS)

    if steerer is not None:
        steerer.close()

    out = {
        "lambda": float(lam),
        "total": int(total),
        "parsed": int(parsed),
        "parse_failed": int(parse_failed),
        "accuracy": float(acc),
        "accuracy_with_failed": float(acc_with_failed),
        "macro_f1": float(macro_f1),
        "cm": cm,
        "per_class": {l: {"prec": prec[l], "rec": rec[l], "f1": f1[l], "n": per_class[l]["total"]} for l in LABELS},
        "results": results,
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--file", type=str, required=True, help="LogiQA NLI JSONL 文件")
    ap.add_argument("--svcca_pt", type=str, required=True, help="SVCCA 结果（从 ProntoQA 训练得到）")
    ap.add_argument("--layer_start", type=int, required=True, help="起始层号")
    ap.add_argument("--layer_end", type=int, required=True, help="结束层号（包含）")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=1500, help="最大样本数，默认 500")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    ap.add_argument("--use_projectors", action="store_true")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--corr_min", type=float, default=0.0)
    ap.add_argument("--anchor", choices=["last", "all"], default="all")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--lam_start", type=float, default=0.02)
    ap.add_argument("--lam_end", type=float, default=0.16)
    ap.add_argument("--lam_step", type=float, default=0.02)
    ap.add_argument("--out_csv", type=str, default="sweep_lambda_metrics_logiqa_nli.csv")
    ap.add_argument("--outdir_fig", type=str, default="figs_logiqa_nli")
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()

    # 随机种子
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
    samples = read_nli_jsonl(args.file, args.max_samples)
    print(f"Loaded {len(samples)} samples from {args.file}")

    if not samples:
        print("No valid samples found.", file=sys.stderr)
        sys.exit(1)

    # 统计标签分布
    label_counts = defaultdict(int)
    for s in samples:
        label_counts[s["gold"]] += 1
    print(f"Label distribution: {dict(label_counts)}")

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
    print("Primary device:", device)

    layer_list, fam = find_decoder_layers(mdl)
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
    print(f"Will scan layers: {layers_to_scan}")

    # 准备 CSV
    ensure_dir(args.out_csv)
    first_write = not os.path.exists(args.out_csv)
    fcsv = open(args.out_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(fcsv)
    if first_write:
        w.writerow([
            "layer", "lambda", "total", "parsed", "parse_failed", "accuracy", "accuracy_with_failed", "macro_f1",
            "cm_E_E", "cm_E_NE", "cm_E_PARSE_FAILED",
            "cm_NE_E", "cm_NE_NE", "cm_NE_PARSE_FAILED",
            "prec_E", "rec_E", "f1_E", "n_E",
            "prec_NE", "rec_NE", "f1_NE", "n_NE",
            "model", "file", "svcca_pt", "useP", "top_k", "corr_min", "anchor", "window"
        ])
        fcsv.flush()

    # lambda 列表
    lam_vals = []
    lam = args.lam_start
    while lam <= args.lam_end + 1e-9:
        lam_vals.append(round(lam, 6))
        lam += args.lam_step
    lam_vals = sorted(set(lam_vals))

    os.makedirs(args.outdir_fig, exist_ok=True)

    for layer in tqdm(layers_to_scan, desc="Scanning layers"):
        print(f"\n{'=' * 60}")
        print(f"Processing Layer {layer}")
        print(f"{'=' * 60}")

        layer_accs = []
        layer_mfs = []

        for lam in tqdm(lam_vals, desc=f"Layer {layer} λ sweep", leave=False):
            out = run_eval_once(
                mdl=mdl, tok=tok, device=device, samples=samples,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                svcca_pt=args.svcca_pt, layer=layer, lam=lam,
                use_projectors=args.use_projectors, top_k=args.top_k, corr_min=args.corr_min,
                anchor=args.anchor, window=args.window
            )
            layer_accs.append(out["accuracy"])
            layer_mfs.append(out["macro_f1"])

            cm = out["cm"]
            per = out["per_class"]

            row = [
                layer, lam, out["total"], out["parsed"], out["parse_failed"],
                out["accuracy"], out["accuracy_with_failed"], out["macro_f1"],
                cm["Entailed"]["Entailed"], cm["Entailed"]["Not Entailed"], cm["Entailed"]["PARSE_FAILED"],
                cm["Not Entailed"]["Entailed"], cm["Not Entailed"]["Not Entailed"], cm["Not Entailed"]["PARSE_FAILED"],
                per["Entailed"]["prec"], per["Entailed"]["rec"], per["Entailed"]["f1"], per["Entailed"]["n"],
                per["Not Entailed"]["prec"], per["Not Entailed"]["rec"], per["Not Entailed"]["f1"], per["Not Entailed"]["n"],
                args.model, os.path.abspath(args.file), os.path.abspath(args.svcca_pt),
                int(args.use_projectors), int(args.top_k), float(args.corr_min),
                args.anchor, int(args.window)
            ]
            w.writerow(row)
            fcsv.flush()

            if args.save_preds:
                pred_dir = os.path.join(os.path.dirname(args.out_csv) or ".", "preds_nli")
                os.makedirs(pred_dir, exist_ok=True)
                pth = os.path.join(pred_dir, f"preds_layer{layer}_lambda_{lam:.2f}.jsonl")
                with open(pth, "w", encoding="utf-8") as fp:
                    for r in out["results"]:
                        fp.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 图表
        plt.figure(figsize=(7, 4))
        plt.plot(lam_vals, layer_accs, marker="o", linewidth=1.8)
        plt.xlabel("λ")
        plt.ylabel("Accuracy")
        plt.title(f"LogiQA NLI Accuracy vs λ (layer {layer})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir_fig, f"accuracy_vs_lambda_layer{layer}.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(lam_vals, layer_mfs, marker="s", linewidth=1.8)
        plt.xlabel("λ")
        plt.ylabel("Macro F1")
        plt.title(f"LogiQA NLI Macro F1 vs λ (layer {layer})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir_fig, f"macroF1_vs_lambda_layer{layer}.png"), dpi=160)
        plt.close()

        print(f"Layer {layer} completed. Best accuracy: {max(layer_accs):.4f} at λ={lam_vals[layer_accs.index(max(layer_accs))]}")

    fcsv.close()

    print(f"\n{'=' * 60}")
    print("All layers completed!")
    print(f"{'=' * 60}")
    print(f"Results saved to: {os.path.abspath(args.out_csv)}")
    print(f"Figures saved to: {os.path.abspath(args.outdir_fig)}")


if __name__ == "__main__":
    main()
