#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 ProntoQA 词汇的 tokenization 情况
看看：
1. 每个词被拆成几个 token
2. 有没有 token 冲突（同一个 token 属于多个 category）
"""

import argparse
from collections import defaultdict
from transformers import AutoTokenizer

# ============ ProntoQA 词汇表 ============

ENTITY_NAMES = [
    "Fae", "Rex", "Sally", "Max", "Alex", "Sam", "Polly", "Stella", "Wren"
]

CONCEPT_NAMES = [
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
]

PROPERTIES = [
    # 颜色
    "blue", "red", "brown", "orange",
    # 大小
    "small", "large",
    # 材质/状态
    "metallic", "wooden", "luminous", "liquid", "transparent", "opaque",
    # 情绪/性格
    "nervous", "happy", "shy", "feisty", "mean", "kind", "angry",
    # 外观
    "bright", "dull", "furry", "fruity", "floral",
    # 温度
    "hot", "cold", "warm",
    # 速度
    "slow", "moderate", "fast",
    # 天气
    "windy", "sunny", "overcast", "rainy", "snowy",
    # 其他
    "bitter", "sweet", "melodic", "muffled", "aggressive",
]

QUANTIFIERS = ["each", "every", "all", "a", "an", "no"]
COPULA = ["is", "are"]
NEGATION = ["not"]
STRUCTURE = ["true", "false", "the", "query", "or", "and", "Premises"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print()

    # 收集每个 category 的词汇和它们的 tokenization
    categories = {
        "Entity": ENTITY_NAMES,
        "Concept": CONCEPT_NAMES,
        "Property": PROPERTIES,
        "Quantifier": QUANTIFIERS,
        "Copula": COPULA,
        "Negation": NEGATION,
        "Structure": STRUCTURE,
    }

    # token_id -> [(category, word), ...]
    token_to_sources = defaultdict(list)

    print("=" * 60)
    print("Tokenization Results")
    print("=" * 60)

    for cat_name, words in categories.items():
        print(f"\n### {cat_name} ###")
        multi_token_words = []

        for word in words:
            # 测试两种情况：词首和词中
            # 词首（有空格前缀）
            token_ids_start = tokenizer.encode(f" {word}", add_special_tokens=False)
            # 词中（无空格前缀）- 用于某些 tokenizer
            token_ids_mid = tokenizer.encode(word, add_special_tokens=False)

            # 解码看看
            tokens_start = [tokenizer.decode([tid]) for tid in token_ids_start]
            tokens_mid = [tokenizer.decode([tid]) for tid in token_ids_mid]

            # 记录
            for tid in set(token_ids_start + token_ids_mid):
                token_to_sources[tid].append((cat_name, word))

            n_tokens = len(token_ids_start)
            if n_tokens > 1:
                multi_token_words.append((word, tokens_start, token_ids_start))
                print(f"  {word:20s} -> {n_tokens} tokens: {tokens_start}")
            else:
                print(f"  {word:20s} -> 1 token: {tokens_start}")

        if multi_token_words:
            print(f"\n  [!] {len(multi_token_words)} words split into multiple tokens")

    # 检查冲突
    print("\n" + "=" * 60)
    print("Token Conflicts (same token in multiple categories)")
    print("=" * 60)

    conflicts = []
    for tid, sources in token_to_sources.items():
        # 提取唯一的 category
        cats = set(cat for cat, word in sources)
        if len(cats) > 1:
            token_str = tokenizer.decode([tid])
            conflicts.append((tid, token_str, sources))

    if conflicts:
        print(f"\nFound {len(conflicts)} conflicting tokens:\n")
        for tid, token_str, sources in conflicts:
            print(f"  Token '{token_str}' (id={tid}):")
            for cat, word in sources:
                print(f"    - {cat}: {word}")
            print()
    else:
        print("\nNo conflicts found! Each token belongs to only one category.")

    # 统计
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    total_words = sum(len(words) for words in categories.values())
    total_tokens = len(token_to_sources)
    print(f"Total words in vocabulary: {total_words}")
    print(f"Total unique tokens used: {total_tokens}")
    print(f"Conflicting tokens: {len(conflicts)}")

    # 显示每个 category 的 token 覆盖
    print("\nTokens per category:")
    for cat_name, words in categories.items():
        cat_tokens = set()
        for word in words:
            tids = tokenizer.encode(f" {word}", add_special_tokens=False)
            cat_tokens.update(tids)
        print(f"  {cat_name:12s}: {len(words):3d} words -> {len(cat_tokens):3d} unique tokens")


if __name__ == "__main__":
    main()
