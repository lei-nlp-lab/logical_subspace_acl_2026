#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from typing import List, Dict, Any

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
    text_parts.append(f"\nQuery: {question}")

    # 添加末尾的 \n\n 以保证 tokenization 一致性
    return '\n'.join(text_parts) + "\n\n"

def extract_balanced_testset(input_jsonl: str, output_json: str,
                            n_per_class: int = 167):
    """
    从 meta-test.jsonl 中提取均衡的测试集

    Args:
        input_jsonl: 输入的 ProofWriter meta-test.jsonl 文件
        output_json: 输出的 JSON 文件
        n_per_class: 每类的样本数量（默认167，总共约500）
    """
    # 容器：按类别收集问题
    true_d5 = []   # True 且 QDep=5
    false_d5 = []  # False 且 QDep=5
    unknown_all = [] # Unknown，任意 QDep

    print(f"Reading from {input_jsonl}...")

    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            story_id = data.get('id')
            theory = data.get('theory', '')
            questions = data.get('questions', {})

            for qid, q in questions.items():
                answer = q.get('answer')
                qdep = q.get('QDep')
                question_text = q.get('question')

                # 构建样本
                sample = {
                    'story_id': f"{story_id}_{qid}",
                    'theory': theory,
                    'question': question_text,
                    'label': None,
                    'qdep': qdep
                }

                # 分类收集
                if answer is True and qdep == 5:
                    sample['label'] = 'True'
                    true_d5.append(sample)
                elif answer is False and qdep == 5:
                    sample['label'] = 'False'
                    false_d5.append(sample)
                elif answer == "Unknown":
                    sample['label'] = 'Unknown'
                    unknown_all.append(sample)

    print(f"Collected: True(D=5)={len(true_d5)}, False(D=5)={len(false_d5)}, Unknown(any)={len(unknown_all)}")

    # 随机采样
    random.seed(42)

    selected_true = random.sample(true_d5, min(n_per_class, len(true_d5)))
    selected_false = random.sample(false_d5, min(n_per_class, len(false_d5)))
    selected_unknown = random.sample(unknown_all, min(n_per_class, len(unknown_all)))

    print(f"Sampled: True={len(selected_true)}, False={len(selected_false)}, Unknown={len(selected_unknown)}")

    # 合并并打乱
    all_samples = selected_true + selected_false + selected_unknown
    random.shuffle(all_samples)

    # 转换成输出格式
    output_data = []

    for sample in all_samples:
        # 生成 NL_without_proof 格式的文本
        nl_text = format_text_without_proof(sample['theory'], sample['question'])

        item = {
            'story_id': sample['story_id'],
            'label': sample['label'],
            'pair': [
                {
                    'view': 'NL_without_proof',
                    'text': nl_text
                }
            ]
        }

        output_data.append(item)

    # 保存
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(output_data)} samples to {output_json}")

    # 统计最终分布
    label_counts = {}
    for item in output_data:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nFinal distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入的 meta-test.jsonl 文件")
    ap.add_argument("--output", required=True, help="输出的 JSON 文件")
    ap.add_argument("--n_per_class", type=int, default=167,
                   help="每类的样本数量（默认167，总共约500）")
    args = ap.parse_args()

    extract_balanced_testset(args.input, args.output, args.n_per_class)
