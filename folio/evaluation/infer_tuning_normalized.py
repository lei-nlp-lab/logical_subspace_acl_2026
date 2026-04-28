#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalized steering version - 应用方案1：归一化 steering direction
主要修改：使用 HFSteererNormalized 替代 HFSteerer
"""

import argparse, os, csv, json, sys
from collections import defaultdict
from typing import List, Dict

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------- import helpers from steering_infer_normalized --------
try:
    from evaluation.steering_infer_normalized import HFSteererNormalized as HFSteerer
    from evaluation.steering_infer import (
        build_prompt, read_jsonl, parse_truth,
        primary_device, find_decoder_layers, norm_truth
    )
except Exception:
    from steering_infer_normalized import HFSteererNormalized as HFSteerer
    from steering_infer import (
        build_prompt, read_jsonl, parse_truth,
        primary_device, find_decoder_layers, norm_truth
    )

from transformers import AutoTokenizer, AutoModelForCausalLM

def batched(it, bs):
    for i in range(0, len(it), bs):
        yield it[i:i+bs]

def ensure_dir(p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

def run_eval_once(mdl, tok, device, samples, max_new_tokens, batch_size,
                  svcca_pt, layer, lam, use_projectors, top_k, corr_min,
                  anchor, window):
    """
    运行一次评测（给定单层与 λ）。返回结果字典：
      {
        'lambda': lam,
        'total': N, 'accuracy': acc, 'macro_f1': macro_f1,
        'cm': {gold->{pred->cnt}},
        'per_class': {label: {'prec':..., 'rec':..., 'f1':..., 'n':...}}
      }
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
    per_class = defaultdict(lambda: {"tp":0,"total":0})
    labels = ["True","False","Unknown"]
    all_preds = labels + ["PARSE_FAILED"]
    cm = {a:{b:0 for b in all_preds} for a in labels}

    # 预先把 prompts 编码好（可复用 attention_mask 给 hook）
    def encode_batch(prompts: List[str]):
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
        return enc

    # 生成
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
            ptxt = build_prompt(s["prem"], s["concl"])
            if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
                msgs = [{"role":"system","content":"You are a helpful reasoning assistant."},
                        {"role":"user","content":ptxt}]
                chat_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                prompts.append(chat_text)
            else:
                prompts.append(ptxt)
        enc = encode_batch(prompts)
        texts = generate_texts(enc)

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
            results.append({"gold": gold, "pred": pred, "prem": s["prem"], "hypo": s["concl"], "gen": text})

    # 汇总指标
    total = len(samples)
    parsed = total - parse_failed
    acc = correct / parsed if parsed else 0.0
    acc_with_failed = correct / total if total else 0.0
    # per-class prec/rec/f1 + macroF1
    # 先构建 tp/fp/fn
    tp = {l:cm[l][l] for l in labels}
    fp = {l:sum(cm[g][l] for g in labels if g!=l) for l in labels}
    fn = {l:sum(cm[l][p] for p in all_preds if p!=l) for l in labels}  # 包含 PARSE_FAILED

    prec = {l:(tp[l]/(tp[l]+fp[l]) if (tp[l]+fp[l])>0 else 0.0) for l in labels}
    rec  = {l:(tp[l]/(tp[l]+fn[l]) if (tp[l]+fn[l])>0 else 0.0) for l in labels}
    f1   = {l:(2*prec[l]*rec[l]/(prec[l]+rec[l]) if (prec[l]+rec[l])>0 else 0.0) for l in labels}
    macro_f1 = sum(f1.values())/len(labels)

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
        "per_class": {l: {"prec": prec[l], "rec": rec[l], "f1": f1[l], "n": per_class[l]["total"]} for l in labels},
        "results": results,  # 如需另存原始生成，可用
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    # 数据与模型
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--file", type=str, required=True, help="FOLIO jsonl（含 premises/conclusion/label）")
    ap.add_argument("--svcca_pt", type=str, required=True, help="SVCCA 结果（含 bases/projectors/corrs）")
    ap.add_argument("--layer", type=int, required=True, help="待扫描的层号（单层）")
    # 生成与批量
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    # Steering 细节
    ap.add_argument("--use_projectors", action="store_true")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--corr_min", type=float, default=0.0)
    ap.add_argument("--anchor", choices=["last","all"], default="all")
    ap.add_argument("--window", type=int, default=1)
    # Lambda 扫描
    ap.add_argument("--lam_start", type=float, default=0.0)
    ap.add_argument("--lam_end", type=float, default=1.0)
    ap.add_argument("--lam_step", type=float, default=0.1)
    # 输出
    ap.add_argument("--out_csv", type=str, default="sweep_lambda_metrics_normalized.csv")
    ap.add_argument("--outdir_fig", type=str, default="figs_val_normalized")
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
    rows = read_jsonl(args.file)
    def get_field(d, candidates, default=None):
        for k in candidates:
            if k in d: return d[k]
        return default

    samples = []
    for r in rows:
        prem = get_field(r, ["premises","Premises - NL","premises_nl","premises-nl"])
        concl = get_field(r, ["conclusion","Conclusions - NL","conclusion_nl","conclusion-nl"])
        lab  = get_field(r, ["label","Truth Values","truth","gold_label"])
        if isinstance(prem, list): prem = " ".join(prem)
        if prem is None or concl is None or lab is None: continue
        gold = norm_truth(str(lab))
        if gold is None: continue
        samples.append({"prem": prem, "concl": concl, "gold": gold})

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
    if not samples:
        print("No valid samples.", file=sys.stderr); sys.exit(1)

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
    # 显式关闭所有 dropout（双重保险）
    for module in mdl.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    mdl.generation_config.pad_token_id = tok.pad_token_id
    mdl.generation_config.eos_token_id = tok.eos_token_id
    mdl.generation_config.use_cache = True

    device = primary_device(mdl)
    print("Primary device:", device)

    # 检查层号
    layer_list, _ = find_decoder_layers(mdl)
    if not (0 <= args.layer < len(layer_list)):
        print(f"[error] layer out of range (0..{len(layer_list)-1})", file=sys.stderr); sys.exit(1)

    # 准备 CSV
    ensure_dir(args.out_csv)
    first_write = not os.path.exists(args.out_csv)
    fcsv = open(args.out_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(fcsv)
    if first_write:
        w.writerow([
            "lambda","total","parsed","parse_failed","accuracy","accuracy_with_failed","macro_f1",
            "cm_T_T","cm_T_F","cm_T_U","cm_T_PARSE_FAILED",
            "cm_F_T","cm_F_F","cm_F_U","cm_F_PARSE_FAILED",
            "cm_U_T","cm_U_F","cm_U_U","cm_U_PARSE_FAILED",
            "prec_T","rec_T","f1_T","n_T",
            "prec_F","rec_F","f1_F","n_F",
            "prec_U","rec_U","f1_U","n_U",
            "model","file","svcca_pt","layer","useP","top_k","corr_min","anchor","window"
        ])
        fcsv.flush()

    # 生成 lambda 列表
    vals = []
    lam = args.lam_start
    # 避免浮点误差，多加一丢丢
    while lam <= args.lam_end + 1e-9:
        vals.append(round(lam, 6))
        lam += args.lam_step
    vals = sorted(set(vals))

    accs, mfs = [], []

    # 扫描
    for lam in tqdm(vals, desc=f"Sweeping layer {args.layer} λ (normalized)"):
        out = run_eval_once(
            mdl=mdl, tok=tok, device=device, samples=samples,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
            svcca_pt=args.svcca_pt, layer=args.layer, lam=lam,
            use_projectors=args.use_projectors, top_k=args.top_k, corr_min=args.corr_min,
            anchor=args.anchor, window=args.window
        )
        accs.append(out["accuracy"])
        mfs.append(out["macro_f1"])

        cm = out["cm"]
        def cm_ij(i,j): return cm[i][j]
        per = out["per_class"]

        row = [
            lam, out["total"], out["parsed"], out["parse_failed"],
            out["accuracy"], out["accuracy_with_failed"], out["macro_f1"],
            cm_ij("True","True"), cm_ij("True","False"), cm_ij("True","Unknown"), cm_ij("True","PARSE_FAILED"),
            cm_ij("False","True"), cm_ij("False","False"), cm_ij("False","Unknown"), cm_ij("False","PARSE_FAILED"),
            cm_ij("Unknown","True"), cm_ij("Unknown","False"), cm_ij("Unknown","Unknown"), cm_ij("Unknown","PARSE_FAILED"),
            per["True"]["prec"], per["True"]["rec"], per["True"]["f1"], per["True"]["n"],
            per["False"]["prec"], per["False"]["rec"], per["False"]["f1"], per["False"]["n"],
            per["Unknown"]["prec"], per["Unknown"]["rec"], per["Unknown"]["f1"], per["Unknown"]["n"],
            args.model, os.path.abspath(args.file), os.path.abspath(args.svcca_pt),
            args.layer, int(args.use_projectors), int(args.top_k), float(args.corr_min),
            args.anchor, int(args.window)
        ]
        w.writerow(row); fcsv.flush()

        if args.save_preds:
            # 逐样本生成另存 jsonl，按 λ 区分文件
            pth = os.path.join(os.path.dirname(args.out_csv) or ".", f"preds_lambda_{lam:.2f}.jsonl")
            with open(pth, "w", encoding="utf-8") as fp:
                for r in out["results"]:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    fcsv.close()

    # 画图
    os.makedirs(args.outdir_fig, exist_ok=True)

    plt.figure(figsize=(7,4))
    plt.plot(vals, accs, marker="o", linewidth=1.8)
    plt.xlabel("λ"); plt.ylabel("Accuracy"); plt.title(f"Accuracy vs λ (layer {args.layer}, normalized)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    f1 = os.path.join(args.outdir_fig, f"accuracy_vs_lambda_layer{args.layer}.png")
    plt.savefig(f1, dpi=160); plt.close()

    plt.figure(figsize=(7,4))
    plt.plot(vals, mfs, marker="s", linewidth=1.8)
    plt.xlabel("λ"); plt.ylabel("Macro F1"); plt.title(f"Macro F1 vs λ (layer {args.layer}, normalized)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    f2 = os.path.join(args.outdir_fig, f"macroF1_vs_lambda_layer{args.layer}.png")
    plt.savefig(f2, dpi=160); plt.close()

    print("Saved:")
    print(" -", os.path.abspath(args.out_csv))
    print(" -", os.path.abspath(f1))
    print(" -", os.path.abspath(f2))

if __name__ == "__main__":
    main()
