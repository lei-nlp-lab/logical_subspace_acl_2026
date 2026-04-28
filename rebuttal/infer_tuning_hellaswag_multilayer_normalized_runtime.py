#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HellaSwag evaluation with normalized steering and runtime throughput metrics.

Design goals:
1) Reuse the same steering implementation as ProntoQA (HFSteererNormalized).
2) Evaluate side effects on a non-logic task (HellaSwag multiple-choice).
3) Keep output format close to existing prontoqa runtime CSV, with tokens/s.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Reuse steering implementation from prontoqa/evaluation
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PRONTOQA_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "prontoqa", "evaluation"))
if PRONTOQA_DIR not in sys.path:
    sys.path.append(PRONTOQA_DIR)

from steering_infer_normalized import HFSteererNormalized as HFSteerer
from steering_infer import primary_device, find_decoder_layers


PROMPT_TEMPLATE = """You are a careful reasoner for commonsense completion.

Context:
{context}

Candidate continuations:
0. {opt0}
1. {opt1}
2. {opt2}
3. {opt3}

Instructions:
- First, reason step by step about which option best completes the context.
- Then, on the last line, output exactly:
Answer: <0|1|2|3>
"""

ANS_RE = re.compile(r"Answer\s*:\s*([0-3]|[A-D])", re.IGNORECASE)


def norm_choice(x: str):
    if x is None:
        return None
    t = x.strip().upper()
    if t in {"0", "1", "2", "3"}:
        return t
    if t in {"A", "B", "C", "D"}:
        return str(ord(t) - ord("A"))
    m = ANS_RE.search(x)
    if m:
        return norm_choice(m.group(1))
    return None


def parse_choice(text: str):
    lines = [ln.strip() for ln in text.splitlines() if "answer" in ln.lower()]
    if lines:
        m = ANS_RE.search(lines[-1])
        if m:
            return norm_choice(m.group(1))
    m = list(ANS_RE.finditer(text))
    return norm_choice(m[-1].group(1)) if m else None


def build_prompt(context: str, endings: List[str]) -> str:
    return PROMPT_TEMPLATE.format(
        context=context.strip(),
        opt0=endings[0].strip(),
        opt1=endings[1].strip(),
        opt2=endings[2].strip(),
        opt3=endings[3].strip(),
    )


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def batched(it, bs):
    for i in range(0, len(it), bs):
        yield it[i : i + bs]


def ensure_dir(p: str):
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

    labels = ["0", "1", "2", "3"]
    all_preds = labels + ["PARSE_FAILED"]
    cm = {g: {p: 0 for p in all_preds} for g in labels}
    per_class = defaultdict(lambda: {"tp": 0, "total": 0})

    results = []
    correct = 0
    parse_failed = 0
    gen_tokens_total = 0
    gen_time_s = 0.0

    def encode_batch(prompts: List[str]):
        return tok(prompts, return_tensors="pt", padding=True, truncation=True)

    def generate_texts(enc):
        nonlocal gen_tokens_total, gen_time_s
        enc = {k: v.to(device) for k, v in enc.items()}
        input_lens = enc["attention_mask"].sum(dim=1)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        gen_time_s += time.perf_counter() - t0

        if tok.pad_token_id is not None:
            out_lens = (out != tok.pad_token_id).sum(dim=1)
        else:
            out_lens = torch.full((out.shape[0],), out.shape[1], device=out.device)
        new_lens = (out_lens - input_lens).clamp(min=0)
        gen_tokens_total += int(new_lens.sum().item())

        return tok.batch_decode(out, skip_special_tokens=True)

    bs = max(1, batch_size)
    for batch in batched(samples, bs):
        prompts = []
        for s in batch:
            ptxt = build_prompt(s["context"], s["endings"])
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

        enc = encode_batch(prompts)
        texts = generate_texts(enc)

        for text, s in zip(texts, batch):
            pred = parse_choice(text)
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
            results.append(
                {
                    "sample_id": s["sample_id"],
                    "gold": gold,
                    "pred": pred,
                    "context": s["context"],
                    "endings": s["endings"],
                    "gen": text,
                }
            )

    total = len(samples)
    parsed = total - parse_failed
    acc = correct / parsed if parsed else 0.0
    acc_with_failed = correct / total if total else 0.0

    tp = {l: cm[l][l] for l in labels}
    fp = {l: sum(cm[g][l] for g in labels if g != l) for l in labels}
    fn = {l: sum(cm[l][p] for p in all_preds if p != l) for l in labels}

    prec = {l: (tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0) for l in labels}
    rec = {l: (tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0) for l in labels}
    f1 = {
        l: (2 * prec[l] * rec[l] / (prec[l] + rec[l]) if (prec[l] + rec[l]) > 0 else 0.0)
        for l in labels
    }
    macro_f1 = sum(f1.values()) / len(labels)

    if steerer is not None:
        steerer.close()

    return {
        "lambda": float(lam),
        "total": int(total),
        "parsed": int(parsed),
        "parse_failed": int(parse_failed),
        "gen_tokens_total": int(gen_tokens_total),
        "gen_time_s": float(gen_time_s),
        "tokens_per_s": float(gen_tokens_total / gen_time_s) if gen_time_s > 0 else 0.0,
        "accuracy": float(acc),
        "accuracy_with_failed": float(acc_with_failed),
        "macro_f1": float(macro_f1),
        "cm": cm,
        "per_class": {
            l: {"prec": prec[l], "rec": rec[l], "f1": f1[l], "n": per_class[l]["total"]}
            for l in labels
        },
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--file", type=str, required=True, help="HellaSwag jsonl (prefer val split)")
    ap.add_argument("--svcca_pt", type=str, required=True, help="Subspace from logic task")
    ap.add_argument("--layer_start", type=int, required=True)
    ap.add_argument("--layer_end", type=int, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    ap.add_argument("--use_projectors", action="store_true")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--corr_min", type=float, default=0.0)
    ap.add_argument("--anchor", choices=["last", "all"], default="all")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode (default: False).",
    )
    ap.add_argument("--lam_start", type=float, default=0.0)
    ap.add_argument("--lam_end", type=float, default=0.0)
    ap.add_argument("--lam_step", type=float, default=0.02)
    ap.add_argument("--out_csv", type=str, default="sweep_lambda_metrics_hellaswag_multilayer.csv")
    ap.add_argument("--outdir_fig", type=str, default="figs_hellaswag_multilayer")
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
    except Exception as e:
        print(f"Warning: Could not enable deterministic algorithms: {e}")

    data = read_jsonl(args.file)
    print(f"Loaded {len(data)} records from {args.file}")

    samples = []
    for i, rec in enumerate(data):
        endings = rec.get("endings") or []
        label = rec.get("label")
        if len(endings) != 4:
            continue
        if label is None:
            continue
        try:
            gold_idx = int(label)
        except Exception:
            continue
        if gold_idx not in (0, 1, 2, 3):
            continue
        context = (rec.get("ctx") or "").strip()
        if not context:
            ctx_a = (rec.get("ctx_a") or "").strip()
            ctx_b = (rec.get("ctx_b") or "").strip()
            context = (ctx_a + " " + ctx_b).strip()
        if not context:
            continue
        samples.append(
            {
                "sample_id": rec.get("ind", i),
                "context": context,
                "endings": endings,
                "gold": str(gold_idx),
                "split_type": rec.get("split_type", ""),
            }
        )

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[: args.max_samples]
    if not samples:
        print("No valid labeled samples found.", file=sys.stderr)
        sys.exit(1)
    print(f"Extracted {len(samples)} valid HellaSwag samples")

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
            args.model, device_map="auto", torch_dtype=dtype, low_cpu_mem_usage=True, **load_kwargs
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
        print(f"[error] layer_start {args.layer_start} out of range (0..{total_layers-1})", file=sys.stderr)
        sys.exit(1)
    if not (0 <= args.layer_end < total_layers):
        print(f"[error] layer_end {args.layer_end} out of range (0..{total_layers-1})", file=sys.stderr)
        sys.exit(1)
    if args.layer_start > args.layer_end:
        print(f"[error] layer_start ({args.layer_start}) > layer_end ({args.layer_end})", file=sys.stderr)
        sys.exit(1)
    layers_to_scan = list(range(args.layer_start, args.layer_end + 1))
    print(f"Will scan layers: {layers_to_scan}")

    ensure_dir(args.out_csv)
    first_write = not os.path.exists(args.out_csv)
    fcsv = open(args.out_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(fcsv)
    if first_write:
        header = [
            "layer",
            "lambda",
            "total",
            "parsed",
            "parse_failed",
            "accuracy",
            "accuracy_with_failed",
            "macro_f1",
            "gen_tokens_total",
            "gen_time_s",
            "tokens_per_s",
        ]
        for g in ["0", "1", "2", "3"]:
            for p in ["0", "1", "2", "3", "PARSE_FAILED"]:
                header.append(f"cm_{g}_{p}")
        for l in ["0", "1", "2", "3"]:
            header.extend([f"prec_{l}", f"rec_{l}", f"f1_{l}", f"n_{l}"])
        header.extend(
            [
                "model",
                "file",
                "svcca_pt",
                "useP",
                "top_k",
                "corr_min",
                "anchor",
                "window",
            ]
        )
        w.writerow(header)
        fcsv.flush()

    lam_vals = []
    lam = args.lam_start
    while lam <= args.lam_end + 1e-9:
        lam_vals.append(round(lam, 6))
        lam += args.lam_step
    lam_vals = sorted(set(lam_vals))

    os.makedirs(args.outdir_fig, exist_ok=True)

    for layer in tqdm(layers_to_scan, desc="Scanning layers"):
        print(f"\n{'='*60}")
        print(f"Processing Layer {layer}")
        print(f"{'='*60}")

        layer_accs = []
        layer_mfs = []

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
            layer_accs.append(out["accuracy"])
            layer_mfs.append(out["macro_f1"])

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
                out["gen_tokens_total"],
                out["gen_time_s"],
                out["tokens_per_s"],
            ]
            for g in ["0", "1", "2", "3"]:
                for p in ["0", "1", "2", "3", "PARSE_FAILED"]:
                    row.append(cm[g][p])
            for l in ["0", "1", "2", "3"]:
                row.extend([per[l]["prec"], per[l]["rec"], per[l]["f1"], per[l]["n"]])
            row.extend(
                [
                    args.model,
                    os.path.abspath(args.file),
                    os.path.abspath(args.svcca_pt),
                    int(args.use_projectors),
                    int(args.top_k),
                    float(args.corr_min),
                    args.anchor,
                    int(args.window),
                ]
            )
            w.writerow(row)
            fcsv.flush()

            if args.save_preds:
                pred_dir = os.path.join(os.path.dirname(args.out_csv) or ".", "preds_hellaswag")
                os.makedirs(pred_dir, exist_ok=True)
                pth = os.path.join(pred_dir, f"preds_layer{layer}_lambda_{lam:.2f}.jsonl")
                with open(pth, "w", encoding="utf-8") as fp:
                    for r in out["results"]:
                        fp.write(json.dumps(r, ensure_ascii=False) + "\n")

        plt.figure(figsize=(7, 4))
        plt.plot(lam_vals, layer_accs, marker="o", linewidth=1.8)
        plt.xlabel("lambda")
        plt.ylabel("Accuracy")
        plt.title(f"HellaSwag Accuracy vs lambda (layer {layer})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        f1 = os.path.join(args.outdir_fig, f"hellaswag_accuracy_vs_lambda_layer{layer}.png")
        plt.savefig(f1, dpi=160)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(lam_vals, layer_mfs, marker="s", linewidth=1.8)
        plt.xlabel("lambda")
        plt.ylabel("Macro F1")
        plt.title(f"HellaSwag Macro F1 vs lambda (layer {layer})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        f2 = os.path.join(args.outdir_fig, f"hellaswag_macroF1_vs_lambda_layer{layer}.png")
        plt.savefig(f2, dpi=160)
        plt.close()

        best_idx = int(np.argmax(layer_accs))
        print(f"Layer {layer} completed. Best accuracy: {layer_accs[best_idx]:.4f} at lambda={lam_vals[best_idx]}")

    fcsv.close()
    print(f"\n{'='*60}")
    print("All layers completed!")
    print(f"{'='*60}")
    print(f"Results saved to: {os.path.abspath(args.out_csv)}")
    print(f"Figures saved to: {os.path.abspath(args.outdir_fig)}")


if __name__ == "__main__":
    main()

