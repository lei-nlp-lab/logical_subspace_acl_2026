#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token-level Energy Analysis for ProntoQA Chain-of-Thought

直接分析 chain_of_thought 字段中的文本。
计算每个 word 在 CCA shared subspace 上的 energy：
    energy(word) = mean of ||r_i @ U||^2 / ||r_i||^2 for tokens in word
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import List, Tuple

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
    # 来自 run_experiment.py 的完整词表
    # 颜色
    "blue", "red", "brown", "orange",
    # 大小
    "small", "large",
    # 材质/状态
    "metallic", "wooden", "luminous", "liquid",
    # 透明度
    "transparent", "opaque",
    # 情绪/性格
    "nervous", "happy", "feisty", "shy",
    # 外观
    "bright", "dull",
    # 味道
    "sweet", "sour", "spicy", "bitter",
    # 气味
    "floral", "fruity", "earthy",
    # 温度
    "hot", "cold", "temperate",
    # 性格
    "kind", "mean", "angry", "amenable", "aggressive",
    # 声音
    "melodic", "muffled", "discordant", "loud",
    # 速度
    "slow", "moderate", "fast",
    # 天气
    "windy", "sunny", "overcast", "rainy", "snowy",
}

QUANTIFIERS = {"each", "every", "all", "a", "an", "no", "everything", "that"}
COPULA = {"is", "are"}
NEGATION = {"not"}
STRUCTURE = {"true", "false", "the", "query", "or", "and", "premises",
             "assume", "this", "contradicts", "with", "fact"}  # Composed 数据集额外词


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
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]

    words = []
    current_word_tokens = []
    current_start = 0

    for i, tok_str in enumerate(token_strs):
        is_word_boundary = (tok_str.startswith(' ') or tok_str.startswith('\n') or
                           tok_str.startswith('Ġ') or tok_str.startswith('▁') or
                           '\n' in tok_str)

        if is_word_boundary and current_word_tokens:
            word_text = ''.join(current_word_tokens).strip()
            word_text = word_text.split('\n')[0].strip()
            if word_text:
                words.append((word_text, current_start, i))
            current_word_tokens = [tok_str]
            current_start = i
        else:
            current_word_tokens.append(tok_str)

    if current_word_tokens:
        word_text = ''.join(current_word_tokens).strip()
        word_text = word_text.split('\n')[0].strip()
        if word_text:
            words.append((word_text, current_start, len(token_strs)))

    return words


def compute_token_energy(residual: torch.Tensor, U: torch.Tensor, normalize: bool = False, eps: float = 1e-12) -> torch.Tensor:
    """
    计算每个 token 的 energy

    normalize=False: energy(i) = ||r_i @ U||^2  (直接投影 norm)
    normalize=True:  energy(i) = ||r_i @ U||^2 / ||r_i||^2  (投影比例)
    """
    proj = residual @ U
    proj_norm_sq = (proj ** 2).sum(dim=1)

    if normalize:
        resid_norm_sq = (residual ** 2).sum(dim=1) + eps
        energy = proj_norm_sq / resid_norm_sq
    else:
        energy = proj_norm_sq

    return energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ProntoQA JSON 文件路径 (5hop_0shot_noadj.json)")
    ap.add_argument("--svcca_pt", required=True, help="SVCCA 结果 .pt 文件")
    ap.add_argument("--layer", type=int, required=True, help="分析哪一层")
    ap.add_argument("--model", required=True, help="HuggingFace 模型名")
    ap.add_argument("--max_samples", type=int, default=5000, help="最多处理多少样本")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out_dir", default="./token_energy_results", help="输出目录")
    ap.add_argument("--top_k", type=int, default=0, help="只用前 k 个 CCA 维度；0=全部")
    ap.add_argument("--normalize", action="store_true", help="使用归一化 energy (除以 ||r||^2)")
    ap.add_argument("--verbose_other", action="store_true", help="显示 Other 类别的词")
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

    U = bases[layer_key].float()
    cvec = corrs.get(layer_key)

    if args.top_k > 0 and args.top_k < U.shape[1]:
        if cvec is not None:
            idx = torch.argsort(cvec, descending=True)[:args.top_k]
            U = U[:, idx]
        else:
            U = U[:, :args.top_k]

    print(f"Using U with shape {tuple(U.shape)} (D={U.shape[0]}, k={U.shape[1]})")

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
        trust_remote_code=True,  # 支持 Phi-3 等模型
    )
    model.eval()

    capture = ActivationCapture(model, args.layer, args.device)

    # ---- 加载数据 ----
    print(f"Loading data from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    # ---- 收集 word-level energy ----
    category_energies = defaultdict(list)
    all_word_info = []
    other_words = defaultdict(int)

    n_processed = 0

    # 遍历所有 example
    examples = list(data.values())[:args.max_samples]
    pbar = tqdm(examples, desc="Processing examples")

    for example in pbar:
        test_example = example.get("test_example", {})
        chain_of_thought = test_example.get("chain_of_thought", [])

        if not chain_of_thought:
            continue

        # 把 chain_of_thought 拼成一个文本
        cot_text = " ".join(chain_of_thought)

        # Tokenize
        tokens = tokenizer(cot_text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(args.device)

        # 获取激活
        activation = capture.capture(input_ids)
        if activation is None:
            continue

        # [seq_len, hidden_dim]
        activation = activation[0, :, :].float()

        # 计算 token-level energy
        token_energy = compute_token_energy(activation, U, normalize=args.normalize)

        # 获取 token ids
        token_ids = tokens["input_ids"][0].tolist()

        # Word-token 对齐
        word_mapping = get_word_token_mapping(tokenizer, token_ids)

        # 计算 word-level energy
        for word, start_idx, end_idx in word_mapping:
            if start_idx >= len(token_energy) or end_idx > len(token_energy):
                continue

            word_energy = token_energy[start_idx:end_idx].mean().item()
            category = classify_word(word)
            category_energies[category].append(word_energy)

            all_word_info.append({
                "word": word,
                "category": category,
                "energy": word_energy,
                "n_tokens": end_idx - start_idx,
            })

            if category == "Other":
                other_words[word.lower()] += 1

        n_processed += 1
        pbar.set_postfix({"processed": n_processed})

    print(f"\nProcessed {n_processed} examples")

    # 显示 Other 词
    if args.verbose_other and other_words:
        print("\n=== Other Words (top 50) ===")
        sorted_other = sorted(other_words.items(), key=lambda x: -x[1])[:50]
        for w, cnt in sorted_other:
            print(f"  {w}: {cnt}")

    # ---- 统计 ----
    print("\n=== Category Statistics ===")
    category_stats = {}
    for cat in ["Negation", "Quantifier", "Copula", "Entity", "Concept", "Property", "Structure", "Punctuation", "Other"]:
        if cat in category_energies:
            energies = category_energies[cat]
            mean_e = np.mean(energies)
            std_e = np.std(energies)
            sem_e = std_e / np.sqrt(len(energies))
            count = len(energies)
            category_stats[cat] = {"mean": mean_e, "std": std_e, "sem": sem_e, "count": count}
            print(f"{cat:12s}: mean={mean_e:.4f}, std={std_e:.4f}, sem={sem_e:.4f}, count={count}")

    # ---- 画柱状图 ----
    if not category_stats:
        print("Warning: No words were classified. Cannot generate plot.")
        return

    # 排除 Punctuation 类别
    sorted_cats = sorted([c for c in category_stats.keys() if c != "Punctuation"],
                         key=lambda x: category_stats[x]["mean"], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(sorted_cats))
    means = [category_stats[c]["mean"] for c in sorted_cats]
    sems = [category_stats[c]["sem"] for c in sorted_cats]
    counts = [category_stats[c]["count"] for c in sorted_cats]

    colors = []
    for c in sorted_cats:
        if c in ["Negation", "Quantifier"]:
            colors.append("steelblue")
        elif c == "Copula":
            colors.append("darkorange")
        elif c in ["Entity", "Concept", "Property"]:
            colors.append("forestgreen")
        elif c in ["Structure", "Punctuation"]:
            colors.append("lightgray")
        else:
            colors.append("gray")

    bars = ax.bar(x_pos, means, yerr=sems, capsize=5, color=colors, alpha=0.8)

    ax.set_xlabel("Word Category", fontsize=12)
    ax.set_ylabel("Mean Energy (projection ratio)", fontsize=12)
    ax.set_title(f"Word-level Energy by Category (Layer {args.layer}, k={U.shape[1]})", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{c}\n(n={counts[i]})" for i, c in enumerate(sorted_cats)], fontsize=10)

    for i, (bar, m) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sems[i] + 0.002,
                f"{m:.3f}", ha='center', va='bottom', fontsize=9)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    fig_path = os.path.join(args.out_dir, f"word_energy_cot_layer{args.layer}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to {fig_path}")

    # ---- 保存详细数据 ----
    csv_path = os.path.join(args.out_dir, f"word_energy_cot_details_layer{args.layer}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["word", "category", "energy", "n_tokens"])
        w.writeheader()
        w.writerows(all_word_info)
    print(f"Saved details to {csv_path}")

    # 保存统计摘要
    stats_path = os.path.join(args.out_dir, f"category_stats_cot_layer{args.layer}.csv")
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "mean_energy", "std_energy", "sem_energy", "count"])
        for cat in sorted_cats:
            s = category_stats[cat]
            w.writerow([cat, s["mean"], s["std"], s["sem"], s["count"]])
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
