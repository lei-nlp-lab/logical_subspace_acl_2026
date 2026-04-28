#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-Direction Token Analysis for ProntoQA Chain-of-Thought

对每个 canonical 方向 u_j，计算：
    s_{i,j} = (r_i^T u_j)^2 / ||r_i||^2
表示 "token i 的残差里，有多大比例的能量是沿着方向 j 的"

输出：
1. 每个方向的 top-K 触发 token
2. 每个方向的词类分布统计
3. 方向的语义标签建议
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ============ ProntoQA 词汇表 ============

ENTITY_NAMES = {
    "fae", "rex", "sally", "max", "alex", "sam", "polly", "stella", "wren"
}

CONCEPT_NAMES = {
    # 主要概念（单数）
    "wumpus", "yumpus", "zumpus", "dumpus", "rompus", "numpus", "tumpus",
    "vumpus", "impus", "jompus", "gorpus", "shumpus", "lempus", "sterpus",
    "grimpus", "lorpus", "brimpus",
    # 干扰概念
    "timpus", "yimpus", "rempus", "fompus", "worpus", "terpus", "gerpus",
    "kerpus", "scrompus", "zhorpus", "bompus", "jelpus", "felpus", "chorpus",
    "hilpus", "storpus", "yerpus", "boompus", "gwompus", "rorpus", "quimpus",
    # 复数形式
    "wumpuses", "yumpuses", "zumpuses", "dumpuses", "rompuses", "numpuses",
    "tumpuses", "vumpuses", "impuses", "jompuses", "gorpuses", "shumpuses",
    "lempuses", "sterpuses", "grimpuses", "lorpuses", "brimpuses",
}

PROPERTIES = {
    "blue", "red", "brown", "orange",
    "small", "large",
    "metallic", "wooden", "luminous", "liquid",
    "transparent", "opaque",
    "nervous", "happy", "feisty", "shy",
    "bright", "dull",
    "sweet", "sour", "spicy", "bitter",
    "floral", "fruity", "earthy",
    "hot", "cold", "temperate",
    "kind", "mean", "angry", "amenable", "aggressive",
    "melodic", "muffled", "discordant", "loud",
    "slow", "moderate", "fast",
    "windy", "sunny", "overcast", "rainy", "snowy",
}

QUANTIFIERS = {"each", "every", "all", "a", "an", "no", "everything", "that"}
COPULA = {"is", "are"}
NEGATION = {"not"}
STRUCTURE = {"true", "false", "the", "query", "or", "and", "premises",
             "assume", "this", "contradicts", "with", "fact"}


def classify_word(word: str) -> str:
    """对完整的 word 进行分类"""
    clean = word.lower().strip()
    clean = re.sub(r'[^\w]', '', clean)

    if not clean:
        return "Punctuation"

    if clean in NEGATION:
        return "Negation"
    if clean in QUANTIFIERS:
        return "Quantifier"
    if clean in COPULA:
        return "Copula"
    if clean in ENTITY_NAMES:
        return "Entity"
    if clean in CONCEPT_NAMES:
        return "Concept"
    if clean in PROPERTIES:
        return "Property"
    if clean in STRUCTURE:
        return "Structure"

    return "Other"


# ============ 模型相关 ============

def find_decoder_layers(model):
    """返回 [layer_modules], family_tag"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers), "llama"
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers), "gpt-neox"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h), "gpt2"
    raise RuntimeError("Unrecognized model architecture")


class ActivationCapture:
    """捕获指定层的激活"""

    def __init__(self, model, layer_idx: int, device: str = "cuda"):
        self.model = model
        self.layer_idx = layer_idx
        self.device = device
        self.layers, self.family = find_decoder_layers(model)
        self.activation = None
        self.handle = None

    def _hook(self, module, input, output):
        hidden_states = input[0]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        self.activation = hidden_states.detach().cpu()

    def register(self):
        self.handle = self.layers[self.layer_idx].register_forward_hook(self._hook)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

    def capture(self, input_ids, attention_mask=None):
        self.activation = None
        self.register()
        try:
            with torch.inference_mode():
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return self.activation
        finally:
            self.remove()


# ============ Word-Token 对齐 ============

def get_word_token_mapping(tokenizer, token_ids: List[int]) -> List[Tuple[str, int, int]]:
    """
    将 token 序列合并成 words，返回每个 word 对应的 token span。
    返回: [(word, start_token_idx, end_token_idx), ...]
    """
    # 使用 convert_ids_to_tokens 获取原始 token 字符串（保留 ▁）
    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    # 同时获取 decoded 版本用于拼接
    decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]

    words = []
    current_word_tokens = []
    current_start = 0

    for i, (raw_tok, dec_tok) in enumerate(zip(raw_tokens, decoded_tokens)):
        # 检查 raw token 中的词边界标记
        is_word_boundary = (
            raw_tok.startswith('▁') or
            raw_tok.startswith('\u2581') or
            raw_tok.startswith('Ġ') or
            raw_tok.startswith(' ') or
            raw_tok.startswith('\n') or
            (len(raw_tok) == 1 and raw_tok in '.,;:!?()[]{}"\'-')
        )

        if is_word_boundary and current_word_tokens:
            word_text = ''.join(current_word_tokens)
            word_text = word_text.replace('▁', '').replace('\u2581', '').replace('Ġ', '').strip()
            word_text = word_text.split('\n')[0].strip()
            if word_text:
                words.append((word_text, current_start, i))
            current_word_tokens = [dec_tok]
            current_start = i
        else:
            current_word_tokens.append(dec_tok)

    if current_word_tokens:
        word_text = ''.join(current_word_tokens)
        word_text = word_text.replace('▁', '').replace('\u2581', '').replace('Ġ', '').strip()
        word_text = word_text.split('\n')[0].strip()
        if word_text:
            words.append((word_text, current_start, len(raw_tokens)))

    return words


def compute_per_direction_scores(residual: torch.Tensor, U: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    计算每个 token 在每个方向上的能量占比

    Args:
        residual: [seq_len, D] token 残差
        U: [D, k] canonical 方向基

    Returns:
        scores: [seq_len, k]  s_{i,j} = (r_i^T u_j)^2 / ||r_i||^2
    """
    # [seq_len, k] 投影系数
    proj_coeffs = residual @ U

    # [seq_len, k] 每个方向的能量
    proj_sq = proj_coeffs ** 2

    # [seq_len] 每个 token 的总能量
    resid_norm_sq = (residual ** 2).sum(dim=1, keepdim=True) + eps

    # [seq_len, k] 归一化得到能量占比
    scores = proj_sq / resid_norm_sq

    return scores


def suggest_direction_label(category_dist: Dict[str, int], top_words: List[str]) -> str:
    """根据词类分布和 top words 建议方向的语义标签"""
    total = sum(category_dist.values())
    if total == 0:
        return "Unknown"

    # 计算各类占比
    ratios = {cat: cnt / total for cat, cnt in category_dist.items()}

    # 规则判断
    if ratios.get("Negation", 0) > 0.3:
        return "Negation"
    if ratios.get("Quantifier", 0) > 0.3:
        return "Quantifier"
    if ratios.get("Copula", 0) > 0.4:
        return "Copula/Linking"
    if ratios.get("Entity", 0) > 0.3:
        return "Entity"
    if ratios.get("Concept", 0) > 0.3:
        return "Concept"
    if ratios.get("Property", 0) > 0.3:
        return "Property"
    if ratios.get("Structure", 0) > 0.3:
        return "Structure"

    # 如果没有明显主导类别，看 top words
    top_words_lower = [w.lower() for w in top_words[:10]]
    if any(w in ["not", "no", "never", "cannot"] for w in top_words_lower):
        return "Negation-related"
    if any(w in ["if", "then", "implies", "therefore", "because"] for w in top_words_lower):
        return "Conditional/Inference"
    if any(w in ["all", "every", "each", "some", "any"] for w in top_words_lower):
        return "Quantifier-related"

    # 找占比最高的非 Other 类别
    non_other = {k: v for k, v in ratios.items() if k != "Other" and k != "Punctuation"}
    if non_other:
        best = max(non_other, key=non_other.get)
        if non_other[best] > 0.15:
            return f"{best}-leaning"

    return "Mixed/General"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ProntoQA JSON 文件路径")
    ap.add_argument("--svcca_pt", required=True, help="SVCCA 结果 .pt 文件")
    ap.add_argument("--layer", type=int, required=True, help="分析哪一层")
    ap.add_argument("--model", required=True, help="HuggingFace 模型名")
    ap.add_argument("--max_samples", type=int, default=500, help="最多处理多少样本")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out_dir", default="./per_direction_results", help="输出目录")
    ap.add_argument("--top_k_tokens", type=int, default=50, help="每个方向显示 top K 触发 token")
    ap.add_argument("--top_k_directions", type=int, default=0, help="只分析前 K 个方向；0=全部")
    ap.add_argument("--sort_by_corr", action="store_true", help="按 canonical correlation 排序方向")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 加载 CCA 基 ----
    print(f"Loading SVCCA from {args.svcca_pt}")
    svcca = torch.load(args.svcca_pt, map_location="cpu")
    bases = svcca.get("bases", {})
    corrs = svcca.get("corrs", {})

    layer_key = args.layer
    if layer_key not in bases:
        layer_key = str(args.layer)
    if layer_key not in bases:
        layer_key = int(args.layer)
    if layer_key not in bases:
        raise ValueError(f"Layer {args.layer} not found in SVCCA bases. Available: {list(bases.keys())}")

    U = bases[layer_key].float()  # [D, k]
    cvec = corrs.get(layer_key)
    if cvec is not None:
        cvec = cvec.float()

    D, k_total = U.shape
    print(f"Loaded U: D={D}, k={k_total}")

    # 可选：按 correlation 排序
    if args.sort_by_corr and cvec is not None:
        sort_idx = torch.argsort(cvec, descending=True)
        U = U[:, sort_idx]
        cvec = cvec[sort_idx]
        print("Sorted directions by canonical correlation (descending)")

    # 可选：只分析前 K 个方向
    if args.top_k_directions > 0 and args.top_k_directions < k_total:
        U = U[:, :args.top_k_directions]
        if cvec is not None:
            cvec = cvec[:args.top_k_directions]
        k_total = args.top_k_directions
        print(f"Analyzing only top {k_total} directions")

    # ---- 加载模型 ----
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if args.device == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    capture = ActivationCapture(model, args.layer, args.device)

    # ---- 加载数据 ----
    print(f"Loading data from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    # ---- 收集每个方向、每个词类的 scores ----
    # direction_category_scores[j][category] = [score1, score2, ...]
    direction_category_scores: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    # 同时收集 top tokens
    direction_all_scores: Dict[int, List[Tuple[str, float, str]]] = defaultdict(list)

    n_processed = 0
    examples = list(data.values())[:args.max_samples]
    pbar = tqdm(examples, desc="Processing examples")

    for example in pbar:
        test_example = example.get("test_example", {})
        chain_of_thought = test_example.get("chain_of_thought", [])

        if not chain_of_thought:
            continue

        cot_text = " ".join(chain_of_thought)

        # Tokenize
        tokens = tokenizer(cot_text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(args.device)

        # 获取激活
        activation = capture.capture(input_ids)
        if activation is None:
            continue

        # [seq_len, D]
        activation = activation[0, :, :].float()

        # 计算 per-direction scores: [seq_len, k]
        scores = compute_per_direction_scores(activation, U)

        # 获取 token ids 和 word mapping
        token_ids = tokens["input_ids"][0].tolist()
        word_mapping = get_word_token_mapping(tokenizer, token_ids)

        # 对每个 word，计算在每个方向上的平均 score
        for word, start_idx, end_idx in word_mapping:
            if start_idx >= scores.shape[0] or end_idx > scores.shape[0]:
                continue

            category = classify_word(word)

            # [n_tokens, k] -> [k] 平均
            word_scores = scores[start_idx:end_idx, :].mean(dim=0)

            for j in range(k_total):
                score_val = word_scores[j].item()
                direction_category_scores[j][category].append(score_val)
                direction_all_scores[j].append((word, score_val, category))

        n_processed += 1
        pbar.set_postfix({"processed": n_processed})

    print(f"\nProcessed {n_processed} examples")

    # ---- 分析每个方向 ----
    print("\n" + "=" * 80)
    print("Per-Direction Analysis (Mean ± Std by Category)")
    print("=" * 80)

    categories = ["Negation", "Quantifier", "Copula", "Entity", "Concept",
                  "Property", "Structure", "Punctuation", "Other"]

    direction_summaries = []

    for j in range(k_total):
        cat_scores = direction_category_scores[j]
        all_scores = direction_all_scores[j]

        if not all_scores:
            continue

        # 计算每个词类的 mean, std, sem, count
        cat_stats = {}
        for cat in categories:
            scores_list = cat_scores.get(cat, [])
            if scores_list:
                arr = np.array(scores_list)
                cat_stats[cat] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "sem": float(np.std(arr) / np.sqrt(len(arr))),
                    "count": len(arr),
                }
            else:
                cat_stats[cat] = {"mean": 0.0, "std": 0.0, "sem": 0.0, "count": 0}

        # Top K tokens
        sorted_scores = sorted(all_scores, key=lambda x: -x[1])
        top_tokens = sorted_scores[:args.top_k_tokens]

        # 找 mean 最高的词类（排除 Other 和 Punctuation）
        meaningful_cats = ["Negation", "Quantifier", "Copula", "Entity", "Concept", "Property", "Structure"]
        best_cat = max(meaningful_cats, key=lambda c: cat_stats[c]["mean"] if cat_stats[c]["count"] > 0 else -1)
        best_mean = cat_stats[best_cat]["mean"]

        # 建议标签
        if best_mean > 0 and cat_stats[best_cat]["count"] >= 5:
            suggested_label = f"{best_cat} (mean={best_mean:.4f})"
        else:
            suggested_label = "Mixed/General"

        # 打印
        corr_str = f", corr={cvec[j].item():.4f}" if cvec is not None else ""
        print(f"\n[Direction {j}]{corr_str}")
        print(f"  Category stats (mean ± std, count):")
        for cat in categories:
            s = cat_stats[cat]
            if s["count"] > 0:
                print(f"    {cat:12s}: {s['mean']:.4f} ± {s['std']:.4f} (n={s['count']})")
        print(f"  Suggested: {suggested_label}")
        print(f"  Top 10 tokens: {', '.join([f'{w}({sc:.4f})' for w, sc, _ in top_tokens[:10]])}")

        # 保存摘要
        direction_summaries.append({
            "direction": j,
            "correlation": cvec[j].item() if cvec is not None else None,
            "suggested_label": suggested_label,
            "category_stats": cat_stats,
            "top_tokens": [(w, s, c) for w, s, c in top_tokens[:args.top_k_tokens]],
        })

    # ---- 保存结果 ----

    # 1. 保存每个方向的词类统计 CSV (mean, std, sem, count)
    stats_csv = os.path.join(args.out_dir, f"direction_category_stats_layer{args.layer}.csv")
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["direction", "correlation", "suggested_label"]
        for cat in categories:
            header.extend([f"{cat}_mean", f"{cat}_std", f"{cat}_sem", f"{cat}_count"])
        w.writerow(header)

        for s in direction_summaries:
            row = [
                s["direction"],
                f"{s['correlation']:.4f}" if s['correlation'] is not None else "",
                s["suggested_label"],
            ]
            for cat in categories:
                cs = s["category_stats"][cat]
                row.extend([f"{cs['mean']:.6f}", f"{cs['std']:.6f}", f"{cs['sem']:.6f}", cs["count"]])
            w.writerow(row)
    print(f"\nSaved category stats to {stats_csv}")

    # 2. 保存每个方向的 top tokens（详细）
    details_csv = os.path.join(args.out_dir, f"direction_top_tokens_layer{args.layer}.csv")
    with open(details_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["direction", "correlation", "suggested_label", "rank", "word", "score", "category"])
        for s in direction_summaries:
            for rank, (word, score, cat) in enumerate(s["top_tokens"], start=1):
                w.writerow([
                    s["direction"],
                    f"{s['correlation']:.4f}" if s['correlation'] is not None else "",
                    s["suggested_label"],
                    rank,
                    word,
                    f"{score:.6f}",
                    cat,
                ])
    print(f"Saved top tokens details to {details_csv}")

    # 3. 保存完整 Python 对象
    pt_path = os.path.join(args.out_dir, f"direction_analysis_layer{args.layer}.pt")
    torch.save({
        "direction_summaries": direction_summaries,
        "layer": args.layer,
        "model": args.model,
        "svcca_pt": args.svcca_pt,
        "n_samples": n_processed,
        "top_k_tokens": args.top_k_tokens,
    }, pt_path)
    print(f"Saved full analysis to {pt_path}")

    # 4. 画热力图：方向 × 词类 (用 mean score)
    # 只保留关键词类，去掉 Punctuation 和 Other
    heatmap_categories = ["Negation", "Quantifier", "Copula", "Entity", "Concept", "Structure"]

    # 调整图的比例，让它更接近正方形
    fig_width = 8
    fig_height = max(6, k_total * 0.25)  # 缩小行高
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    heatmap_data = np.zeros((k_total, len(heatmap_categories)))
    for i, s in enumerate(direction_summaries):
        for ci, cat in enumerate(heatmap_categories):
            heatmap_data[i, ci] = s["category_stats"][cat]["mean"]

    # 对每行做 min-max 归一化，增强对比度
    row_min = heatmap_data.min(axis=1, keepdims=True)
    row_max = heatmap_data.max(axis=1, keepdims=True)
    heatmap_norm = (heatmap_data - row_min) / (row_max - row_min + 1e-10)

    # 使用更柔和的颜色方案，降低最大值
    im = ax.imshow(heatmap_norm, aspect='auto', cmap='OrRd', vmin=0, vmax=1.0)
    ax.set_xticks(range(len(heatmap_categories)))
    ax.set_xticklabels(heatmap_categories, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(k_total))

    # 标注 correlation
    ylabels = []
    for s in direction_summaries:
        corr_str = f"{s['correlation']:.2f}" if s['correlation'] is not None else ""
        ylabels.append(f"Dir {s['direction']} ({corr_str})")
    ax.set_yticklabels(ylabels, fontsize=8)

    ax.set_xlabel("Word Category", fontsize=11)
    ax.set_ylabel("Canonical Direction", fontsize=11)
    ax.set_title(f"Per-Direction Category Activation (Layer {args.layer})", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normalized Score", fontsize=10)
    plt.tight_layout()

    fig_path = os.path.join(args.out_dir, f"direction_heatmap_layer{args.layer}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {fig_path}")
    plt.close(fig)

    # 额外保存一个不归一化的版本（原始 mean）
    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
    im2 = ax2.imshow(heatmap_data, aspect='auto', cmap='OrRd')
    ax2.set_xticks(range(len(heatmap_categories)))
    ax2.set_xticklabels(heatmap_categories, rotation=30, ha='right', fontsize=10)
    ax2.set_yticks(range(k_total))
    ax2.set_yticklabels(ylabels, fontsize=8)
    ax2.set_xlabel("Word Category", fontsize=11)
    ax2.set_ylabel("Canonical Direction", fontsize=11)
    ax2.set_title(f"Per-Direction Category Activation (Layer {args.layer}, raw)", fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label("Mean Score", fontsize=10)
    plt.tight_layout()

    fig2_path = os.path.join(args.out_dir, f"direction_heatmap_raw_layer{args.layer}.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"Saved raw heatmap to {fig2_path}")
    plt.close(fig2)

    # 4b. 按 dominant category 分块排序的版本
    # 找每个方向的 dominant category
    def get_dominant_cat(s):
        best_cat = None
        best_mean = -1
        for cat in heatmap_categories:
            if s["category_stats"][cat]["mean"] > best_mean:
                best_mean = s["category_stats"][cat]["mean"]
                best_cat = cat
        return best_cat

    # 按 dominant category 分组，组内按 correlation 降序
    cat_order = ["Negation", "Quantifier", "Copula", "Entity", "Concept", "Structure"]
    sorted_summaries = []
    group_boundaries = []  # 记录每组的起始位置和类别名

    for cat in cat_order:
        group = [s for s in direction_summaries if get_dominant_cat(s) == cat]
        if group:  # 只处理非空组
            group.sort(key=lambda x: x["correlation"] if x["correlation"] else 0, reverse=True)
            group_boundaries.append((len(sorted_summaries), len(group), cat))
            sorted_summaries.extend(group)

    # 重新构建 heatmap 数据
    heatmap_sorted = np.zeros((len(sorted_summaries), len(heatmap_categories)))
    for i, s in enumerate(sorted_summaries):
        for ci, cat in enumerate(heatmap_categories):
            heatmap_sorted[i, ci] = s["category_stats"][cat]["mean"]

    # 行归一化
    row_min_s = heatmap_sorted.min(axis=1, keepdims=True)
    row_max_s = heatmap_sorted.max(axis=1, keepdims=True)
    heatmap_sorted_norm = (heatmap_sorted - row_min_s) / (row_max_s - row_min_s + 1e-10)

    # 画图
    fig_height_sorted = max(8, k_total * 0.32)
    fig3, ax3 = plt.subplots(figsize=(9, fig_height_sorted))

    im3 = ax3.imshow(heatmap_sorted_norm, aspect='auto', cmap='OrRd', vmin=0, vmax=1.0)
    ax3.set_xticks(range(len(heatmap_categories)))
    ax3.set_xticklabels(heatmap_categories, rotation=30, ha='right', fontsize=10)
    ax3.set_yticks(range(len(sorted_summaries)))

    # Y 轴标签
    ylabels_sorted = []
    for s in sorted_summaries:
        corr_str = f"{s['correlation']:.2f}" if s['correlation'] else ""
        ylabels_sorted.append(f"Dir {s['direction']} ({corr_str})")
    ax3.set_yticklabels(ylabels_sorted, fontsize=8)

    # 画分隔线和右侧标签
    for i, (start, count, cat) in enumerate(group_boundaries):
        # 分隔线（除了第一组）
        if start > 0:
            ax3.axhline(y=start - 0.5, color='black', linewidth=1.5, linestyle='-')

        # 右侧标签
        mid_y = start + count / 2 - 0.5
        ax3.text(len(heatmap_categories) - 0.5, mid_y, f"  {cat}", fontsize=9, va='center', ha='left',
                fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

    ax3.set_xlabel("Word Category", fontsize=11)
    ax3.set_ylabel("Canonical Direction", fontsize=10)
    ax3.set_title(f"Per-Direction Category Activation - Grouped by Dominant Category (Layer {args.layer})", fontsize=11)

    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, pad=0.15)
    cbar3.set_label("Normalized Score", fontsize=10)
    plt.tight_layout()

    fig3_path = os.path.join(args.out_dir, f"direction_heatmap_grouped_layer{args.layer}.png")
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"Saved grouped heatmap to {fig3_path}")

    # 打印分组信息
    print(f"  Group breakdown:")
    for start, count, cat in group_boundaries:
        print(f"    {cat}: {count} directions")

    plt.close(fig3)

    # 5. 画柱状图：每个方向各词类的 mean ± sem（前几个方向）
    n_show = min(8, k_total)  # 显示前 8 个方向
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    bar_categories = ["Negation", "Quantifier", "Copula", "Entity", "Concept", "Property"]

    for idx in range(n_show):
        ax = axes[idx]
        s = direction_summaries[idx]

        means = [s["category_stats"][c]["mean"] for c in bar_categories]
        sems = [s["category_stats"][c]["sem"] for c in bar_categories]

        x_pos = np.arange(len(bar_categories))
        ax.bar(x_pos, means, yerr=sems, capsize=3, alpha=0.8,
               color=['steelblue', 'steelblue', 'darkorange', 'forestgreen', 'forestgreen', 'forestgreen'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_categories, rotation=45, ha='right', fontsize=8)
        corr_str = f"corr={s['correlation']:.3f}" if s['correlation'] is not None else ""
        ax.set_title(f"Dir {s['direction']} ({corr_str})", fontsize=10)
        ax.set_ylabel("Mean Score", fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"Per-Direction Category Scores (Layer {args.layer})", fontsize=12)
    plt.tight_layout()

    bar_fig_path = os.path.join(args.out_dir, f"direction_bars_layer{args.layer}.png")
    plt.savefig(bar_fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved bar charts to {bar_fig_path}")

    # 6. 打印最终摘要
    print("\n" + "=" * 80)
    print("Direction Summary")
    print("=" * 80)
    for s in direction_summaries:
        corr_str = f"(corr={s['correlation']:.4f})" if s['correlation'] is not None else ""
        print(f"  Direction {s['direction']:3d} {corr_str}: {s['suggested_label']}")


if __name__ == "__main__":
    main()
