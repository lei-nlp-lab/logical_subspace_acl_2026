#!/usr/bin/env python3
"""
处理 ProofWriter 数据用于 CCA 实验

生成 4 个视角：
1. NL with proof: facts + rules + query + NL proof + answer
2. NL without proof: facts + rules + query
3. Symbolic with proof: facts + rules + query + symbolic proof + answer
4. Symbolic without proof: facts + rules + query
"""

import json
import sys


def process_example(item):
    """
    处理单个样本，生成包含 4 个文本视角的 pair

    Args:
        item: 包含 id, nl_conclusion, proof_text, answer, facts, rules, symbolic_proof 的字典

    Returns:
        包含 story_id, label, pair 的字典
    """
    # 提取数据
    story_id = item['id']
    query = item['nl_conclusion']  # 问题：例如 "Bob is red."
    answer = "True" if item['answer'] else "False"
    facts = item['facts']
    rules = item['rules']
    nl_proof = item['proof_text']  # 自然语言证明
    symbolic_proof = item.get('symbolic_proof', '')  # 符号化证明（可能为空）

    # 构造 premises（facts + rules）
    premises_text = "Facts:\n" + "\n".join(facts) + "\n\n"
    premises_text += "Rules:\n" + "\n".join(rules)

    # === NL with proof ===
    nl_with_proof = f"{premises_text}\n\nTrue or False: {query}\n\n{nl_proof}\n\nThe query is {answer}."

    # === NL without proof ===
    nl_without_proof = f"{premises_text}\n\nTrue or False: {query}\n\n"

    # === Symbolic with proof ===
    if symbolic_proof:
        symbolic_with_proof = f"{premises_text}\n\nTrue or False: {query}\n\n{symbolic_proof}\n\nThe query is {answer}."
    else:
        # 如果没有 symbolic_proof，使用空字符串
        symbolic_with_proof = ""

    # === Symbolic without proof ===
    symbolic_without_proof = f"{premises_text}\n\nTrue or False: {query}\n\n"

    return {
        "story_id": story_id,
        "label": answer,
        "pair": [
            {"view": "NL_with_proof", "text": nl_with_proof},
            {"view": "NL_without_proof", "text": nl_without_proof},
            {"view": "Symbolic_with_proof", "text": symbolic_with_proof},
            {"view": "Symbolic_without_proof", "text": symbolic_without_proof}
        ]
    }


def main(input_file, output_file):
    print(f"📖 读取文件: {input_file}")

    # 读取 JSONL 格式
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    print(f"   总样本数: {len(data)}")

    # 统计有/无 symbolic_proof 的数量
    with_symbolic = sum(1 for item in data if item.get('symbolic_proof', ''))
    without_symbolic = len(data) - with_symbolic
    print(f"   有 symbolic_proof: {with_symbolic}")
    print(f"   无 symbolic_proof: {without_symbolic}")
    print()

    # 处理每个样本
    processed_data = []
    for item in data:
        processed = process_example(item)
        processed_data.append(processed)

    print(f"✅ 处理完成，共 {len(processed_data)} 个样本")
    print()

    # 保存结果
    print(f"💾 保存到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("示例输出（第一个样本）")
    print("=" * 60)

    # 打印第一个样本作为示例
    example = processed_data[0]
    print(f"Story ID: {example['story_id']}")
    print(f"Label: {example['label']}")
    print()

    for view_data in example['pair']:
        view_name = view_data['view']
        text = view_data['text']
        print(f"--- {view_name} ---")
        if text:
            # 显示前 300 字符
            preview = text[:300] + "..." if len(text) > 300 else text
            print(preview)
        else:
            print("(空)")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python process_for_cca.py <input_file> <output_file>")
        print()
        print("示例:")
        print("  python process_for_cca.py test_2k_symbolic.jsonl test_2k_for_cca.json")
        print()
        print("输入格式: JSONL（每行一个 JSON 对象）")
        print("输出格式: JSON（包含 story_id, label, pair 的数组）")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
