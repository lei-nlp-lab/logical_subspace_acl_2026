#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogiQA steering evaluation - 跨数据集迁移实验
使用 ProntoQA 训练的 SVCCA 在 LogiQA 多选题上做 steering

主要改动：
1. 数据格式：LogiQA 是 4 选项多选题（A/B/C/D），不是 True/False
2. Prompt：多选题格式
3. 解析逻辑：解析 A/B/C/D
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

# ---- LogiQA Prompt (多选题格式) ----
PROMPT_TEMPLATE = """You are a careful logician. Use classical deductive reasoning.

Passage:
{text}

Question: {question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Instructions:
- First, reason step by step.
- Then, on the last line, output exactly:
Answer: <A|B|C|D>
"""

# 解析答案的正则：匹配 "Answer: A" 或 "answer: B" 等
ANSWER_RE = re.compile(r"Answer\s*:\s*([A-Da-d])", re.IGNORECASE)

# ---- LogiQA Utils ----
LABELS = ["A", "B", "C", "D"]
INDEX_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
LABEL_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}

def norm_answer(x: str):
    """标准化答案为大写 A/B/C/D"""
    if x is None:
        return None
    t = x.strip().upper()
    if t in LABELS:
        return t
    return None

def parse_answer(text: str):
    """从生成文本中解析答案 A/B/C/D"""
    # 先找含 "answer" 的行，取最后一行
    lines = [ln.strip() for ln in text.splitlines() if "answer" in ln.lower()]
    if lines:
        m = ANSWER_RE.search(lines[-1])
        if m:
            return norm_answer(m.group(1))
    # fallback: 全文搜索
    matches = list(ANSWER_RE.finditer(text))
    if matches:
        return norm_answer(matches[-1].group(1))
    return None

def build_prompt(sample: Dict[str, Any]) -> str:
    """构建 LogiQA 多选题 prompt"""
    options = sample["options"]
    return PROMPT_TEMPLATE.format(
        text=sample["text"].strip(),
        question=sample["question"].strip(),
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
    )

def batched(it, bs):
    for i in range(0, len(it), bs):
        yield it[i:i+bs]

def ensure_dir(p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

def read_logiqa_jsonl(path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    """读取 LogiQA JSONL 格式数据（每行一个 JSON）"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # 检查必要字段
            if "text" not in rec or "question" not in rec or "options" not in rec or "answer" not in rec:
                continue
            if len(rec["options"]) != 4:
                continue
            # 转换 answer（0-3 索引）为 A-D
            gold_idx = rec["answer"]
            gold_label = INDEX_TO_LABEL.get(gold_idx)
            if gold_label is None:
                continue
            samples.append({
                "id": rec.get("id", len(samples)),
                "text": rec["text"],
                "question": rec["question"],
                "options": rec["options"],
                "gold": gold_label,
                "type": rec.get("type", {}),
            })
            if max_samples > 0 and len(samples) >= max_samples:
                break
    return samples

def run_eval_once(mdl, tok, device, samples, max_new_tokens, batch_size,
                  svcca_pt, layer, lam, use_projectors, top_k, corr_min,
                  anchor, window):
    """
    运行一次评测（给定单层与 λ）。返回结果字典
    """
    # 注册/关闭 steerer：lam=0 时不加 hook（基线），lam!=0 时 steer
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

    # 评测循环
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
                "text": s["text"],
                "question": s["question"],
                "options": s["options"],
                "gen": text
            })

    # 汇总指标
    total = len(samples)
    parsed = total - parse_failed
    acc = correct / parsed if parsed else 0.0
    acc_with_failed = correct / total if total else 0.0

    # per-class prec/rec/f1 + macroF1
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
    # 数据与模型
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--file", type=str, required=True, help="LogiQA JSONL 文件（如 test.txt）")
    ap.add_argument("--svcca_pt", type=str, required=True, help="SVCCA 结果（从 ProntoQA 训练得到）")
    ap.add_argument("--layer_start", type=int, required=True, help="起始层号")
    ap.add_argument("--layer_end", type=int, required=True, help="结束层号（包含）")
    # 生成与批量
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=1571, help="最大样本数，默认 500")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    # Steering 细节
    ap.add_argument("--use_projectors", action="store_true")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--corr_min", type=float, default=0.0)
    ap.add_argument("--anchor", choices=["last", "all"], default="all")
    ap.add_argument("--window", type=int, default=1)
    # Lambda 扫描
    ap.add_argument("--lam_start", type=float, default=0.04)
    ap.add_argument("--lam_end", type=float, default=0.14)
    ap.add_argument("--lam_step", type=float, default=0.02)
    # 输出
    ap.add_argument("--out_csv", type=str, default="sweep_lambda_metrics_logiqa.csv")
    ap.add_argument("--outdir_fig", type=str, default="figs_logiqa")
    ap.add_argument("--save_preds", action="store_true", help="把每个 λ 的逐样本生成另存 jsonl")
    args = ap.parse_args()

    # 设置随机种子以确保可复现性
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 更严格的确定性设置
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"Warning: Could not enable deterministic algorithms: {e}")

    # 读取数据
    samples = read_logiqa_jsonl(args.file, args.max_samples)
    print(f"Loaded {len(samples)} samples from {args.file}")

    if not samples:
        print("No valid samples found.", file=sys.stderr)
        sys.exit(1)

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
        # 优先使用 sdpa（更确定性）而非 flash_attention_2
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
    # 显式关闭所有 dropout
    for module in mdl.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    mdl.generation_config.pad_token_id = tok.pad_token_id
    mdl.generation_config.eos_token_id = tok.eos_token_id
    mdl.generation_config.use_cache = True

    device = primary_device(mdl)
    print("Primary device:", device)

    # 检查层号范围
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
            "cm_A_A", "cm_A_B", "cm_A_C", "cm_A_D", "cm_A_PARSE_FAILED",
            "cm_B_A", "cm_B_B", "cm_B_C", "cm_B_D", "cm_B_PARSE_FAILED",
            "cm_C_A", "cm_C_B", "cm_C_C", "cm_C_D", "cm_C_PARSE_FAILED",
            "cm_D_A", "cm_D_B", "cm_D_C", "cm_D_D", "cm_D_PARSE_FAILED",
            "prec_A", "rec_A", "f1_A", "n_A",
            "prec_B", "rec_B", "f1_B", "n_B",
            "prec_C", "rec_C", "f1_C", "n_C",
            "prec_D", "rec_D", "f1_D", "n_D",
            "model", "file", "svcca_pt", "useP", "top_k", "corr_min", "anchor", "window"
        ])
        fcsv.flush()

    # 生成 lambda 列表
    lam_vals = []
    lam = args.lam_start
    while lam <= args.lam_end + 1e-9:
        lam_vals.append(round(lam, 6))
        lam += args.lam_step
    lam_vals = sorted(set(lam_vals))

    # 创建输出图表目录
    os.makedirs(args.outdir_fig, exist_ok=True)

    # 遍历每一层
    for layer in tqdm(layers_to_scan, desc="Scanning layers"):
        print(f"\n{'=' * 60}")
        print(f"Processing Layer {layer}")
        print(f"{'=' * 60}")

        layer_accs = []
        layer_mfs = []

        # 对当前层扫描所有 lambda 值
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
                # confusion matrix
                cm["A"]["A"], cm["A"]["B"], cm["A"]["C"], cm["A"]["D"], cm["A"]["PARSE_FAILED"],
                cm["B"]["A"], cm["B"]["B"], cm["B"]["C"], cm["B"]["D"], cm["B"]["PARSE_FAILED"],
                cm["C"]["A"], cm["C"]["B"], cm["C"]["C"], cm["C"]["D"], cm["C"]["PARSE_FAILED"],
                cm["D"]["A"], cm["D"]["B"], cm["D"]["C"], cm["D"]["D"], cm["D"]["PARSE_FAILED"],
                # per-class metrics
                per["A"]["prec"], per["A"]["rec"], per["A"]["f1"], per["A"]["n"],
                per["B"]["prec"], per["B"]["rec"], per["B"]["f1"], per["B"]["n"],
                per["C"]["prec"], per["C"]["rec"], per["C"]["f1"], per["C"]["n"],
                per["D"]["prec"], per["D"]["rec"], per["D"]["f1"], per["D"]["n"],
                # meta
                args.model, os.path.abspath(args.file), os.path.abspath(args.svcca_pt),
                int(args.use_projectors), int(args.top_k), float(args.corr_min),
                args.anchor, int(args.window)
            ]
            w.writerow(row)
            fcsv.flush()

            if args.save_preds:
                pred_dir = os.path.join(os.path.dirname(args.out_csv) or ".", "preds")
                os.makedirs(pred_dir, exist_ok=True)
                pth = os.path.join(pred_dir, f"preds_layer{layer}_lambda_{lam:.2f}.jsonl")
                with open(pth, "w", encoding="utf-8") as fp:
                    for r in out["results"]:
                        fp.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 为当前层生成图表
        plt.figure(figsize=(7, 4))
        plt.plot(lam_vals, layer_accs, marker="o", linewidth=1.8)
        plt.xlabel("λ")
        plt.ylabel("Accuracy")
        plt.title(f"LogiQA Accuracy vs λ (layer {layer})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        f1_path = os.path.join(args.outdir_fig, f"accuracy_vs_lambda_layer{layer}.png")
        plt.savefig(f1_path, dpi=160)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(lam_vals, layer_mfs, marker="s", linewidth=1.8)
        plt.xlabel("λ")
        plt.ylabel("Macro F1")
        plt.title(f"LogiQA Macro F1 vs λ (layer {layer})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        f2_path = os.path.join(args.outdir_fig, f"macroF1_vs_lambda_layer{layer}.png")
        plt.savefig(f2_path, dpi=160)
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
