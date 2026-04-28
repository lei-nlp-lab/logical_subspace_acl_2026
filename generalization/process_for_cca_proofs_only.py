import json
import sys

def process_example(example_id, example_data):
    """
    处理单个example（proofs-only模式，没有answer字段）
    生成包含4个文本的pair：
    - NL with proof (包含推理链)
    - NL without proof (只有premises和query)
    - FOL with proof (包含symbolic proof)
    - FOL without proof (只有symbolic formulas和query)

    注意：proofs-only模式下query是"Prove: ..."格式，answer默认为True
    """
    test = example_data['test_example']

    # 提取数据
    question = test['question']
    query = test['query']
    chain_of_thought = test['chain_of_thought']
    symbolic_formulas = test['symbolic_formulas']
    symbolic_proof = test['symbolic_proof']

    # === NL with proof ===
    nl_with_proof = f"Premises:\n{question}\n\n{query}\n\n"
    nl_with_proof += "\n".join(chain_of_thought)

    # === NL without proof ===
    nl_without_proof = f"Premises:\n{question}\n\n{query}\n\n"

    # === FOL with proof ===
    # symbolic_formulas 是列表，需要转成字符串
    if isinstance(symbolic_formulas, list):
        formulas_str = "\n".join(symbolic_formulas)
    else:
        formulas_str = str(symbolic_formulas)

    fol_with_proof = f"Premises:\n{formulas_str}\n\n{query}\n\n"
    fol_with_proof += "\n".join(symbolic_proof)

    # === FOL without proof ===
    fol_without_proof = f"Premises:\n{formulas_str}\n\n{query}\n\n"

    return {
        "story_id": example_id,
        "pair": [
            {"view": "NL_with_proof", "text": nl_with_proof},
            {"view": "NL_without_proof", "text": nl_without_proof},
            {"view": "FOL_with_proof", "text": fol_with_proof},
            {"view": "FOL_without_proof", "text": fol_without_proof}
        ]
    }

def main(input_file, output_file):
    print(f"读取文件: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"总example数: {len(data)}")

    processed_data = []
    for example_id, example_content in data.items():
        processed = process_example(example_id, example_content)
        processed_data.append(processed)

    print(f"处理完成，共 {len(processed_data)} 个样本")

    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)

    print(f"已保存到: {output_file}")

    # 打印第一个样本作为示例
    print("\n=== 第一个样本示例 ===")
    print(f"Story ID: {processed_data[0]['story_id']}")
    print(f"\n--- NL with proof (前500字符) ---")
    print(processed_data[0]['pair'][0]['text'][:500] + "...")
    print(f"\n--- FOL with proof (前500字符) ---")
    print(processed_data[0]['pair'][2]['text'][:500] + "...")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python process_for_cca_proofs_only.py <input_file> <output_file>")
        print("示例: python process_for_cca_proofs_only.py 5hop_Composed_0shot.json processed.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
