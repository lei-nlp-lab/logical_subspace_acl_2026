#!/usr/bin/env python3
"""
从 test.jsonl 中采样 2000 条数据（1000 true + 1000 false）
确保 false 的 proof_text 不与选中的 true 数据重复
"""

import json
import random
import argparse
from pathlib import Path


def sample_diverse_subset(input_file, output_file, n_true=1000, n_false=1000, seed=42):
    """
    采样多样化的子集

    Args:
        input_file: 输入 JSONL 文件
        output_file: 输出 JSONL 文件
        n_true: 采样 true 样本数量
        n_false: 采样 false 样本数量
        seed: 随机种子
    """
    random.seed(seed)

    # 读取数据
    print(f"📖 Reading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # 分离 true 和 false
    true_data = [item for item in data if item['answer'] == True]
    false_data = [item for item in data if item['answer'] == False]

    print(f"   Total: {len(data)} items")
    print(f"   True: {len(true_data)} items")
    print(f"   False: {len(false_data)} items")
    print()

    # 检查数量是否足够
    if len(true_data) < n_true:
        print(f"⚠️  Warning: Only {len(true_data)} true items available, requested {n_true}")
        n_true = len(true_data)

    if len(false_data) < n_false:
        print(f"⚠️  Warning: Only {len(false_data)} false items available, requested {n_false}")
        n_false = len(false_data)

    # 步骤 1: 随机选择 n_true 个 true 样本
    print(f"🎲 Step 1: Randomly sampling {n_true} true items...")
    selected_true = random.sample(true_data, n_true)

    # 收集已选 true 的 proof_text（用于去重）
    true_proofs = set(item['proof_text'] for item in selected_true)
    print(f"   Unique proofs in selected true items: {len(true_proofs)}")
    print()

    # 步骤 2: 从 false 中筛选出 proof_text 不重复的
    print(f"🔍 Step 2: Filtering false items with unique proofs...")
    unique_false = [
        item for item in false_data
        if item['proof_text'] not in true_proofs
    ]

    print(f"   False items with unique proofs: {len(unique_false)} / {len(false_data)}")
    print(f"   Duplicate proofs filtered out: {len(false_data) - len(unique_false)}")
    print()

    # 检查是否有足够的 unique false
    if len(unique_false) < n_false:
        print(f"⚠️  Warning: Only {len(unique_false)} unique false items available")
        print(f"   Requested {n_false}, will sample all available")
        n_false = len(unique_false)

    # 步骤 3: 从 unique false 中随机选择 n_false 个
    print(f"🎲 Step 3: Randomly sampling {n_false} unique false items...")
    selected_false = random.sample(unique_false, n_false)
    print()

    # 合并
    result = selected_true + selected_false

    # 打乱顺序（可选）
    random.shuffle(result)

    # 保存
    print(f"💾 Saving to {output_file}...")
    with open(output_file, 'w') as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print()
    print("=" * 60)
    print("✅ Sampling complete!")
    print("=" * 60)
    print(f"   True items: {len(selected_true)}")
    print(f"   False items: {len(selected_false)}")
    print(f"   Total: {len(result)}")
    print(f"   Output: {output_file}")
    print()

    # 验证：检查是否有重复的 proof
    all_proofs = [item['proof_text'] for item in result]
    unique_proofs_count = len(set(all_proofs))
    print(f"📊 Verification:")
    print(f"   Total proofs: {len(all_proofs)}")
    print(f"   Unique proofs: {unique_proofs_count}")

    if unique_proofs_count == len(all_proofs):
        print(f"   ✅ All proofs are unique!")
    else:
        duplicates = len(all_proofs) - unique_proofs_count
        print(f"   ⚠️  {duplicates} duplicate proofs found (should be 0)")


def main():
    parser = argparse.ArgumentParser(
        description='Sample diverse subset from ProofWriter data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:

  # Sample 1000 true + 1000 false (default)
  python sample_diverse_subset.py \\
    --input test.jsonl \\
    --output test_2k_diverse.jsonl

  # Custom sample size
  python sample_diverse_subset.py \\
    --input test.jsonl \\
    --output test_500_diverse.jsonl \\
    --n-true 250 \\
    --n-false 250

  # Use different random seed
  python sample_diverse_subset.py \\
    --input test.jsonl \\
    --output test_2k_diverse_seed123.jsonl \\
    --seed 123
        '''
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSONL file')
    parser.add_argument('--n-true', type=int, default=1000,
                       help='Number of true samples (default: 1000)')
    parser.add_argument('--n-false', type=int, default=1000,
                       help='Number of false samples (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.input).exists():
        print(f"❌ Error: Input file {args.input} not found")
        return

    sample_diverse_subset(
        args.input,
        args.output,
        args.n_true,
        args.n_false,
        args.seed
    )


if __name__ == '__main__':
    main()
