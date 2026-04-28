#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多层 AUC 分析：计算每一层 CoT 能量与正确性的 ROC-AUC，画折线图

横坐标：层 index
纵坐标：该层的 AUC
"""

import argparse
import json
import os
import re
import csv

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM


# ---- CoT 定位 ----
COT_START_PATTERN = re.compile(r"Truth value: <True\|False>assistant\n\n", re.IGNORECASE)


def extract_cot_text(gen_text: str) -> str:
    """从生成文本中提取 CoT 部分"""
    match = COT_START_PATTERN.search(gen_text)
    if match:
        return gen_text[match.end():]
    alt_match = re.search(r"assistant\n\n", gen_text)
    if alt_match:
        return gen_text[alt_match.end():]
    return None


# ---- 模型激活捕获 ----
def find_decoder_layers(model):
    """返回 decoder layers 列表"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError("Unrecognized model architecture")


class MultiLayerCapture:
    """捕获多层的输入激活"""

    def __init__(self, model, layer_indices: list):
        self.model = model
        self.layer_indices = layer_indices
        self.layers = find_decoder_layers(model)
        self.activations = {}  # layer_idx -> tensor
        self.handles = []

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            hidden_states = input[0]
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            self.activations[layer_idx] = hidden_states.detach().cpu()
        return hook

    def register(self):
        for idx in self.layer_indices:
            h = self.layers[idx].register_forward_hook(self._make_hook(idx))
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def capture(self, input_ids, attention_mask=None):
        """运行模型并返回所有层的激活 {layer_idx: [1, S, D]}"""
        self.activations.clear()
        self.register()
        try:
            with torch.inference_mode():
                self.model(input_ids=input_ids, attention_mask=attention_mask)
            return dict(self.activations)
        finally:
            self.remove()


def compute_energy_ratio(residual: torch.Tensor, U: torch.Tensor) -> float:
    """计算能量比例"""
    if residual.dim() == 3:
        residual = residual.squeeze(0)  # [S, D]

    residual = residual.float()
    U = U.float()

    r_norm_sq = (residual ** 2).sum(dim=1)  # [S]
    projected = residual @ U  # [S, k]
    proj_norm_sq = (projected ** 2).sum(dim=1)  # [S]
    energy_ratio = proj_norm_sq / (r_norm_sq + 1e-10)  # [S]

    return energy_ratio.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_jsonl", required=True, help="预测结果 jsonl 文件")
    ap.add_argument("--model", required=True, help="HuggingFace 模型名")
    ap.add_argument("--cca_pt", required=True, help="CCA 结果文件（含 bases）")
    ap.add_argument("--layer_start", type=int, default=0, help="起始层")
    ap.add_argument("--layer_end", type=int, default=-1, help="结束层（-1 表示最后一层）")
    ap.add_argument("--out_dir", default=".", help="输出目录")
    ap.add_argument("--max_samples", type=int, default=0, help="最多处理多少样本（0=全部）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 加载 CCA bases ----
    print(f"Loading CCA bases from {args.cca_pt}")
    cca_obj = torch.load(args.cca_pt, map_location="cpu")
    bases = cca_obj["bases"]

    available_layers = sorted(bases.keys())
    print(f"Available layers in CCA: {available_layers}")

    # ---- 加载模型 ----
    print(f"Loading model: {args.model}")
    load_kwargs = {}
    if args.use_4bit:
        load_kwargs["load_in_4bit"] = True
    if args.use_8bit:
        load_kwargs["load_in_8bit"] = True

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **load_kwargs
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # 确定层范围
    n_layers = len(find_decoder_layers(model))
    layer_end = args.layer_end if args.layer_end >= 0 else n_layers - 1
    layer_indices = [l for l in range(args.layer_start, layer_end + 1) if l in bases]
    print(f"Will analyze layers: {layer_indices}")

    if not layer_indices:
        print("Error: No valid layers to analyze")
        return

    # 创建多层激活捕获器
    capture = MultiLayerCapture(model, layer_indices)

    # ---- 加载预测结果 ----
    print(f"Loading predictions from {args.preds_jsonl}")
    samples = []
    with open(args.preds_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[:args.max_samples]

    print(f"Loaded {len(samples)} samples")

    # ---- 预处理：提取 CoT 并 tokenize ----
    print("Preprocessing samples...")
    valid_samples = []
    for sample in samples:
        gen = sample.get("gen", "")
        cot_text = extract_cot_text(gen)
        if cot_text and len(cot_text.strip()) > 0:
            enc = tokenizer(cot_text, return_tensors="pt", add_special_tokens=False)
            if enc["input_ids"].shape[1] > 0:
                gold = sample.get("gold")
                pred = sample.get("pred")
                y_i = 1 if pred == gold else 0
                valid_samples.append({
                    "story_id": sample.get("story_id"),
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc.get("attention_mask"),
                    "y": y_i,
                })

    print(f"Valid samples: {len(valid_samples)}")

    if len(valid_samples) == 0:
        print("No valid samples to analyze.")
        return

    # ---- 计算每层的能量分数 ----
    # 存储：{layer: [z_scores]}
    layer_z_scores = {l: [] for l in layer_indices}
    y_labels = []

    for sample in tqdm(valid_samples, desc="Computing energy scores"):
        input_ids = sample["input_ids"].to(device)
        attention_mask = sample["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 捕获所有层的激活
        try:
            activations = capture.capture(input_ids, attention_mask)
        except Exception as e:
            print(f"Warning: Failed for {sample['story_id']}: {e}")
            continue

        # 计算每层的能量
        for layer_idx in layer_indices:
            if layer_idx in activations:
                U = bases[layer_idx]
                z = compute_energy_ratio(activations[layer_idx], U)
                layer_z_scores[layer_idx].append(z)

        y_labels.append(sample["y"])

    y_labels = np.array(y_labels)
    n_correct = y_labels.sum()
    n_wrong = len(y_labels) - n_correct
    print(f"\nCorrect: {n_correct}, Wrong: {n_wrong}")

    # ---- 计算每层的 AUC ----
    layer_aucs = {}
    for layer_idx in layer_indices:
        z_scores = np.array(layer_z_scores[layer_idx])
        if len(z_scores) == len(y_labels) and n_correct > 0 and n_wrong > 0:
            auc = roc_auc_score(y_labels, z_scores)
            layer_aucs[layer_idx] = auc
            print(f"Layer {layer_idx:2d}: AUC = {auc:.4f}")
        else:
            layer_aucs[layer_idx] = None
            print(f"Layer {layer_idx:2d}: AUC = N/A")

    # ---- 画折线图 ----
    valid_layers = [l for l in layer_indices if layer_aucs[l] is not None]
    valid_aucs = [layer_aucs[l] for l in valid_layers]

    plt.figure(figsize=(10, 6))
    plt.plot(valid_layers, valid_aucs, marker='o', linewidth=2, markersize=6)
    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random (AUC=0.5)')

    # 标记最大值
    if valid_aucs:
        max_auc = max(valid_aucs)
        max_layer = valid_layers[valid_aucs.index(max_auc)]
        plt.scatter([max_layer], [max_auc], color='red', s=100, zorder=5)
        plt.annotate(f'Max: {max_auc:.4f}\n(Layer {max_layer})',
                     xy=(max_layer, max_auc),
                     xytext=(max_layer + 1, max_auc + 0.01),
                     fontsize=10,
                     arrowprops=dict(arrowstyle='->', color='red'))

    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("ROC-AUC", fontsize=12)
    plt.title("CoT Energy vs Correctness: AUC by Layer", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()

    fig_path = os.path.join(args.out_dir, "auc_by_layer.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved figure to {fig_path}")
    plt.close()

    # ---- 保存结果 ----
    csv_path = os.path.join(args.out_dir, "auc_by_layer.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "auc", "mean_energy_correct", "mean_energy_wrong"])
        for layer_idx in layer_indices:
            z = np.array(layer_z_scores[layer_idx])
            mean_correct = z[y_labels == 1].mean() if n_correct > 0 else None
            mean_wrong = z[y_labels == 0].mean() if n_wrong > 0 else None
            w.writerow([layer_idx, layer_aucs[layer_idx], mean_correct, mean_wrong])
    print(f"Saved CSV to {csv_path}")

    # 保存 JSON 汇总
    stats_path = os.path.join(args.out_dir, "auc_by_layer_stats.json")
    stats = {
        "n_samples": len(valid_samples),
        "n_correct": int(n_correct),
        "n_wrong": int(n_wrong),
        "layer_aucs": {str(k): v for k, v in layer_aucs.items()},
        "best_layer": int(max_layer) if valid_aucs else None,
        "best_auc": float(max_auc) if valid_aucs else None,
        "model": args.model,
        "cca_pt": args.cca_pt,
        "preds_jsonl": args.preds_jsonl,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
