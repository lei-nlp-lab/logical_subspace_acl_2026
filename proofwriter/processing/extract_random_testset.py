#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ProofWriter meta-test.jsonl 中随机抽取 N 个样本
不考虑 label 平衡和 depth 限制
"""

import json
import random
from pathlib import Path


def parse_theory_to_facts_and_rules(theory: str) -> tuple:
    """
    将 theory 文本拆分成 Facts 和 Rules
    """
    sentences = [s.strip() for s in theory.split('.') if s.strip()]

    facts = []
    rules = []

    for sentence in sentences:
        if sentence.lower().startswith('if ') or ' then ' in sentence.lower() or 'all ' in sentence.lower():
            rules.append(sentence + '.')
        else:
            facts.append(sentence + '.')

    return facts, rules


def format_text_without_proof(theory: str, question: str) -> str:
    """
    格式化成 without_proof 版本的文本
    """
    facts, rules = parse_theory_to_facts_and_rules(theory)

    text_parts = []

    # Facts
    text_parts.append("Facts:")
    for fact in facts:
        text_parts.append(fact)

    # Rules
    text_parts.append("\nRules:")
    for rule in rules:
        text_parts.append(rule)

    # Query
    text_parts.append(f"\nTrue or False or Uncertain: {question}")

    # 添加末尾的 \n\n 以保证 tokenization 一致性
    return '\n'.join(text_parts) + "\n\n"


def extract_random_testset(input_jsonl: str, output_json: str, n_samples: int = 500, seed: int = 42, balanced: bool = False):
    """
    从 meta-test.jsonl 随机抽取 n_samples 个样本

    Args:
        input_jsonl: meta-test.jsonl 路径
        output_json: 输出 JSON 文件路径
        n_samples: 抽取样本数（如果 balanced=True，会调整为最接近的 3 的倍数）
        seed: 随机种子
        balanced: 是否保持 label 平衡（True/False/Unknown 各占 1/3）
    """
    random.seed(seed)

    print(f"Reading from {input_jsonl}...")

    # 按 label 分类收集样本
    true_samples = []
    false_samples = []
    unknown_samples = []

    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                story_id = data.get('id')
                theory = data.get('theory', '')
                questions = data.get('questions', {})

                # 每个 story 可能有多个 question
                for qid, q in questions.items():
                    answer = q.get('answer')
                    qdep = q.get('QDep')
                    question_text = q.get('question')

                    sample = {
                        'story_id': f"{story_id}_{qid}",
                        'theory': theory,
                        'question': question_text,
                        'answer': answer,
                        'qdep': qdep
                    }

                    # 按 label 分类（不限制 depth）
                    if answer is True:
                        true_samples.append(sample)
                    elif answer is False:
                        false_samples.append(sample)
                    elif answer == "Unknown":
                        unknown_samples.append(sample)

    print(f"Total samples by label:")
    print(f"  True: {len(true_samples)}")
    print(f"  False: {len(false_samples)}")
    print(f"  Unknown: {len(unknown_samples)}")
    print(f"  Total: {len(true_samples) + len(false_samples) + len(unknown_samples)}")

    # 选择样本
    if balanced:
        # 平衡抽取：每个 label 抽取相同数量（不限制 depth）
        n_per_class = n_samples // 3
        print(f"\nBalanced sampling: {n_per_class} samples per class (any depth)")

        selected_true = random.sample(true_samples, min(n_per_class, len(true_samples)))
        selected_false = random.sample(false_samples, min(n_per_class, len(false_samples)))
        selected_unknown = random.sample(unknown_samples, min(n_per_class, len(unknown_samples)))

        selected = selected_true + selected_false + selected_unknown
        random.shuffle(selected)  # 打乱顺序

        print(f"Selected: {len(selected_true)} true + {len(selected_false)} false + {len(selected_unknown)} unknown = {len(selected)} total")
    else:
        # 完全随机抽取
        all_samples = true_samples + false_samples + unknown_samples
        print(f"\nRandom sampling: {n_samples} samples (label distribution may be unbalanced)")

        if len(all_samples) < n_samples:
            print(f"Warning: Only {len(all_samples)} samples available, less than requested {n_samples}")
            selected = all_samples
        else:
            selected = random.sample(all_samples, n_samples)

        print(f"Randomly selected {len(selected)} samples")

    # 统计 label 和 depth 分布
    label_counts = {}
    depth_counts = {}

    # 转换为统一格式
    output_data = []
    for sample in selected:
        answer = sample['answer']
        qdep = sample['qdep']

        # 标准化 label
        if answer is True:
            label = 'True'
        elif answer is False:
            label = 'False'
        elif answer == "Unknown":
            label = 'Unknown'
        else:
            label = str(answer).lower()

        # 统计
        label_counts[label] = label_counts.get(label, 0) + 1
        depth_counts[qdep] = depth_counts.get(qdep, 0) + 1

        # 生成 NL_without_proof 文本
        nl_text = format_text_without_proof(sample['theory'], sample['question'])

        # 构造输出格式
        output_sample = {
            "story_id": sample['story_id'],
            "label": label,
            "pair": [
                {
                    "view": "NL_without_proof",
                    "text": nl_text
                }
            ],
            "metadata": {
                "depth": qdep,
                "original_answer": answer
            }
        }
        output_data.append(output_sample)

    # 保存
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(output_data)} samples to {output_json}")
    print(f"\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        print(f"  {label}: {label_counts[label]}")

    print(f"\nDepth distribution:")
    for depth in sorted(depth_counts.keys()):
        print(f"  depth={depth}: {depth_counts[depth]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="随机抽取 ProofWriter 测试样本")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to ProofWriter meta-test.jsonl")
    parser.add_argument("--output", type=str,
                        default="proofwriter/data/test_500_random.json",
                        help="Output JSON path")
    parser.add_argument("--n", type=int, default=501,
                        help="抽取样本数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--balanced", action="store_true",
                        help="是否保持 label 平衡（True/False/Unknown 各占 1/3，不限制 depth）")

    args = parser.parse_args()

    extract_random_testset(args.input, args.output, args.n, args.seed, args.balanced)
