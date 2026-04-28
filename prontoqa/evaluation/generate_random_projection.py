#!/usr/bin/env python3
"""
生成随机投影矩阵用于消融实验 (Ablation Study)

对比：
- 真实 CCA 投影矩阵：从数据学习的有意义子空间
- 随机投影矩阵：随机生成的子空间（控制组）

如果随机矩阵也能带来类似提升，说明 CCA 学到的子空间可能没有实际意义。
"""

import argparse
import torch
import numpy as np
from pathlib import Path


def generate_random_orthogonal_matrix(dim, num_components, seed=42):
    """
    生成随机正交矩阵（类似 CCA 的 U 矩阵）

    Args:
        dim: 隐藏层维度（如 4096）
        num_components: 子空间维度（如 100）
        seed: 随机种子

    Returns:
        U: [dim, num_components] 正交矩阵
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 生成随机矩阵
    A = np.random.randn(dim, num_components)

    # QR 分解得到正交矩阵
    Q, R = np.linalg.qr(A)

    return torch.from_numpy(Q).float()


def generate_random_gaussian_matrix(dim, num_components, seed=42, scale=1.0):
    """
    生成随机高斯矩阵（非正交）

    Args:
        dim: 隐藏层维度
        num_components: 子空间维度
        seed: 随机种子
        scale: 方差缩放

    Returns:
        U: [dim, num_components] 高斯随机矩阵
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 标准正态分布
    U = torch.randn(dim, num_components) * scale

    return U


def generate_random_uniform_matrix(dim, num_components, seed=42, low=-1.0, high=1.0):
    """
    生成均匀分布随机矩阵

    Args:
        dim: 隐藏层维度
        num_components: 子空间维度
        seed: 随机种子
        low, high: 均匀分布范围

    Returns:
        U: [dim, num_components] 均匀分布矩阵
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    U = torch.rand(dim, num_components) * (high - low) + low

    return U


def load_real_svcca(svcca_pt_path):
    """加载真实的 SVCCA 结果以获取维度信息"""
    data = torch.load(svcca_pt_path, map_location='cpu')

    bases = data.get('bases', {})
    projectors = data.get('projectors', {})
    corrs = data.get('corrs', {})

    return bases, projectors, corrs


def create_random_svcca(real_svcca_pt, output_pt, method='orthogonal', seed=42, **kwargs):
    """
    基于真实 SVCCA 的维度信息，生成随机投影矩阵

    Args:
        real_svcca_pt: 真实 SVCCA 结果路径（用于获取维度）
        output_pt: 输出的随机矩阵路径
        method: 'orthogonal', 'gaussian', 'uniform'
        seed: 随机种子
        **kwargs: 传递给生成函数的额外参数
    """
    # 加载真实 SVCCA 以获取维度
    bases, projectors, corrs = load_real_svcca(real_svcca_pt)

    random_bases = {}
    random_projectors = {}
    random_corrs = {}

    # 为每一层生成随机矩阵
    for layer_id, U_real in bases.items():
        dim, num_components = U_real.shape
        print(f"Layer {layer_id}: dim={dim}, num_components={num_components}")

        # 生成随机 U
        if method == 'orthogonal':
            U_random = generate_random_orthogonal_matrix(dim, num_components, seed + layer_id)
        elif method == 'gaussian':
            scale = kwargs.get('scale', 1.0)
            U_random = generate_random_gaussian_matrix(dim, num_components, seed + layer_id, scale)
        elif method == 'uniform':
            low = kwargs.get('low', -1.0)
            high = kwargs.get('high', 1.0)
            U_random = generate_random_uniform_matrix(dim, num_components, seed + layer_id, low, high)
        else:
            raise ValueError(f"Unknown method: {method}")

        random_bases[layer_id] = U_random

        # 生成对应的 P = U @ U^T
        P_random = U_random @ U_random.transpose(-1, -2)
        random_projectors[layer_id] = P_random

        # 生成虚假的相关系数（全 1 或随机）
        if kwargs.get('fake_corrs', False):
            random_corrs[layer_id] = torch.ones(num_components)
        else:
            # 使用真实的相关系数（保持选择逻辑一致）
            if layer_id in corrs:
                random_corrs[layer_id] = corrs[layer_id]

    # 保存
    random_svcca_data = {
        'bases': random_bases,
        'projectors': random_projectors,
        'corrs': random_corrs,
        'cfg': {
            'method': method,
            'seed': seed,
            'note': f'Random projection for ablation study (method={method}, seed={seed})'
        }
    }

    torch.save(random_svcca_data, output_pt)
    print(f"\n✅ Saved random SVCCA to: {output_pt}")
    print(f"   Method: {method}")
    print(f"   Seed: {seed}")
    print(f"   Layers: {list(random_bases.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description='生成随机投影矩阵用于消融实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法：

  # 1. 生成随机正交矩阵（推荐，最接近 CCA）
  python generate_random_projection.py \\
    --real-svcca results/svcca_layer15.pt \\
    --output results/random_orthogonal_layer15.pt \\
    --method orthogonal \\
    --seed 42

  # 2. 生成高斯随机矩阵
  python generate_random_projection.py \\
    --real-svcca results/svcca_layer15.pt \\
    --output results/random_gaussian_layer15.pt \\
    --method gaussian \\
    --scale 0.1

  # 3. 生成均匀分布矩阵
  python generate_random_projection.py \\
    --real-svcca results/svcca_layer15.pt \\
    --output results/random_uniform_layer15.pt \\
    --method uniform \\
    --low -0.5 --high 0.5

然后用于 inference：

  # 真实 CCA
  python infer_tuning_prontoqa.py --svcca_pt results/svcca_layer15.pt --lambda 0.3

  # 随机矩阵（对照组）
  python infer_tuning_prontoqa.py --svcca_pt results/random_orthogonal_layer15.pt --lambda 0.3
        '''
    )

    parser.add_argument('--real-svcca', type=str, required=True,
                       help='真实 SVCCA 结果路径（用于获取维度信息）')
    parser.add_argument('--output', type=str, required=True,
                       help='输出的随机矩阵路径')
    parser.add_argument('--method', type=str, default='orthogonal',
                       choices=['orthogonal', 'gaussian', 'uniform'],
                       help='随机矩阵生成方法')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认：42）')

    # 高斯方法参数
    parser.add_argument('--scale', type=float, default=1.0,
                       help='高斯分布的标准差（method=gaussian 时使用）')

    # 均匀分布参数
    parser.add_argument('--low', type=float, default=-1.0,
                       help='均匀分布下界（method=uniform 时使用）')
    parser.add_argument('--high', type=float, default=1.0,
                       help='均匀分布上界（method=uniform 时使用）')

    # 其他选项
    parser.add_argument('--fake-corrs', action='store_true',
                       help='生成虚假的相关系数（全1），而不是复用真实的')

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.real_svcca).exists():
        print(f"❌ Error: {args.real_svcca} 不存在")
        return

    # 生成随机矩阵
    create_random_svcca(
        args.real_svcca,
        args.output,
        method=args.method,
        seed=args.seed,
        scale=args.scale,
        low=args.low,
        high=args.high,
        fake_corrs=args.fake_corrs
    )


if __name__ == '__main__':
    main()
