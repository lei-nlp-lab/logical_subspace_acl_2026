#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 ProntoQA 预测结果中正确和错误回答的平均推理步骤数
"""

import argparse
import json
import re
from collections import defaultdict


def count_reasoning_steps(gen_text: str) -> int:
    """
    统计推理过程中的步骤数

    推理步骤通常以数字开头，如：
    1. Max is a wumpus. (Given)
    2. Wumpuses are vumpuses. (Given)
    ...
    """
    if not gen_text:
        return 0

    # 方法1：匹配 "数字." 或 "数字)" 开头的行
    # 例如 "1. xxx" 或 "1) xxx"
    step_pattern = r'^\s*(\d+)[.\)]\s+'

    lines = gen_text.split('\n')
    step_numbers = []

    for line in lines:
        match = re.match(step_pattern, line)
        if match:
            step_numbers.append(int(match.group(1)))

    if step_numbers:
        # 返回最大的步骤编号（因为有些推理可能跳过编号）
        return max(step_numbers)

    # 方法2：如果没有找到编号，尝试统计含有推理关键词的句子
    # reasoning_keywords = ["since", "therefore", "so", "thus", "because", "by"]
    # 但这可能不准确，先返回 0
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="预测结果 .jsonl 文件")
    ap.add_argument("--verbose", action="store_true", help="显示详细信息")
    args = ap.parse_args()

    # 统计
    stats = {
        "correct": {"steps": [], "count": 0},
        "wrong": {"steps": [], "count": 0},
        "parse_failed": {"steps": [], "count": 0},
    }

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            gold = rec.get("gold", "").strip()
            pred = rec.get("pred", "").strip()
            gen = rec.get("gen", "")
            story_id = rec.get("story_id", "")

            # 统计步骤数
            n_steps = count_reasoning_steps(gen)

            # 分类
            if pred == "PARSE_FAILED":
                category = "parse_failed"
            elif pred == gold:
                category = "correct"
            else:
                category = "wrong"

            stats[category]["steps"].append(n_steps)
            stats[category]["count"] += 1

            if args.verbose:
                print(f"{story_id}: gold={gold}, pred={pred}, steps={n_steps}, category={category}")

    # 输出统计
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    total = sum(s["count"] for s in stats.values())
    print(f"Total samples: {total}")
    print()

    for cat in ["correct", "wrong", "parse_failed"]:
        s = stats[cat]
        if s["count"] > 0:
            avg_steps = sum(s["steps"]) / len(s["steps"])
            min_steps = min(s["steps"])
            max_steps = max(s["steps"])
            # 过滤掉 0 步的（无法解析）
            valid_steps = [x for x in s["steps"] if x > 0]
            if valid_steps:
                avg_valid = sum(valid_steps) / len(valid_steps)
            else:
                avg_valid = 0

            print(f"{cat.upper():12s}:")
            print(f"  Count: {s['count']} ({100*s['count']/total:.1f}%)")
            print(f"  Avg steps (all): {avg_steps:.2f}")
            print(f"  Avg steps (valid only, n={len(valid_steps)}): {avg_valid:.2f}")
            print(f"  Min steps: {min_steps}, Max steps: {max_steps}")
            print()

    # 对比正确 vs 错误
    if stats["correct"]["count"] > 0 and stats["wrong"]["count"] > 0:
        correct_valid = [x for x in stats["correct"]["steps"] if x > 0]
        wrong_valid = [x for x in stats["wrong"]["steps"] if x > 0]

        if correct_valid and wrong_valid:
            avg_correct = sum(correct_valid) / len(correct_valid)
            avg_wrong = sum(wrong_valid) / len(wrong_valid)

            print("=" * 50)
            print("Comparison (valid steps only)")
            print("=" * 50)
            print(f"Correct avg steps: {avg_correct:.2f} (n={len(correct_valid)})")
            print(f"Wrong avg steps:   {avg_wrong:.2f} (n={len(wrong_valid)})")
            print(f"Difference:        {avg_correct - avg_wrong:.2f}")


if __name__ == "__main__":
    main()
