import json
import sys
import random

def split_dataset(input_file, train_size, val_size, test_size, output_prefix, random_seed=42):
    """
    将数据集分割为train/val/test三部分

    参数:
        input_file: 输入JSON文件路径
        train_size: 训练集大小
        val_size: 验证集大小
        test_size: 测试集大小
        output_prefix: 输出文件前缀（如 "5hop_0shot_noadj"）
        random_seed: 随机种子（用于打乱数据，设为None则不打乱）
    """
    print(f"读取文件: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    total_samples = len(data)
    required_samples = train_size + val_size + test_size

    print(f"总样本数: {total_samples}")
    print(f"需要样本数: {required_samples} (train: {train_size}, val: {val_size}, test: {test_size})")

    if required_samples > total_samples:
        print(f"❌ 错误: 需要 {required_samples} 个样本，但只有 {total_samples} 个")
        print(f"建议调整为: train={total_samples-val_size-test_size}, val={val_size}, test={test_size}")
        sys.exit(1)

    # 打乱数据（可选）
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(data)
        print(f"✅ 已使用随机种子 {random_seed} 打乱数据")
    else:
        print("ℹ️  未打乱数据，按原始顺序分割")

    # 分割数据
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:train_size + val_size + test_size]

    # 保存三个文件
    train_file = f"{output_prefix}_train.json"
    val_file = f"{output_prefix}_val.json"
    test_file = f"{output_prefix}_test.json"

    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"✅ 训练集已保存: {train_file} ({len(train_data)} 个样本)")

    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"✅ 验证集已保存: {val_file} ({len(val_data)} 个样本)")

    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"✅ 测试集已保存: {test_file} ({len(test_data)} 个样本)")

    print(f"\n总计: {len(train_data) + len(val_data) + len(test_data)} 个样本已分割完成")

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("用法: python split_dataset.py <input_file> <train_size> <val_size> <test_size> <output_prefix> [random_seed]")
        print("\n示例1 (打乱数据):")
        print("  python split_dataset.py 5hop_0shot_noadj_processed.json 1000 500 500 5hop_0shot_noadj_processed 42")
        print("\n示例2 (不打乱):")
        print("  python split_dataset.py 5hop_0shot_noadj_processed.json 1000 500 500 5hop_0shot_noadj_processed none")
        print("\n参数说明:")
        print("  - input_file: 输入JSON文件")
        print("  - train_size: 训练集样本数")
        print("  - val_size: 验证集样本数")
        print("  - test_size: 测试集样本数")
        print("  - output_prefix: 输出文件名前缀")
        print("  - random_seed: 随机种子（可选，默认42；输入'none'则不打乱）")
        sys.exit(1)

    input_file = sys.argv[1]
    train_size = int(sys.argv[2])
    val_size = int(sys.argv[3])
    test_size = int(sys.argv[4])
    output_prefix = sys.argv[5]

    # 处理随机种子参数
    if len(sys.argv) > 6:
        seed_arg = sys.argv[6].lower()
        random_seed = None if seed_arg == 'none' else int(seed_arg)
    else:
        random_seed = 42  # 默认随机种子

    split_dataset(input_file, train_size, val_size, test_size, output_prefix, random_seed)
