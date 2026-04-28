#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
McNemar 配对检验脚本
用于比较两个模型/配置在同一数据集上的预测结果是否有显著差异

使用方法:
    python mcnemar_test.py --baseline baseline_preds.jsonl --treatment steering_preds.jsonl

输入文件格式 (jsonl):
    {"story_id": "xxx", "gold": "True/False", "pred": "True/False", ...}
"""

import argparse
import json
import numpy as np
from scipy.stats import binomtest, chi2


def load_preds(path):
    """
    读取预测文件，返回 {story_id: is_correct} 字典
    """
    results = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            story_id = obj['story_id']
            gold = str(obj['gold'])
            pred = str(obj['pred'])
            correct = (gold == pred)
            results[story_id] = correct
    return results


def mcnemar_test(baseline_preds, treatment_preds, alpha=0.05):
    """
    执行 McNemar 配对检验

    Args:
        baseline_preds: {story_id: is_correct} baseline 预测结果
        treatment_preds: {story_id: is_correct} treatment 预测结果
        alpha: 显著性水平

    Returns:
        dict: 包含检验结果的字典
    """
    # 找到共同的 story_id
    common_ids = set(baseline_preds.keys()) & set(treatment_preds.keys())
    n = len(common_ids)

    if n == 0:
        raise ValueError("No common story_ids found between the two files")

    # 构建 2x2 列联表
    # a: 两个都对
    # b: baseline 对, treatment 错
    # c: baseline 错, treatment 对
    # d: 两个都错
    a = b = c = d = 0

    for sid in common_ids:
        b_correct = baseline_preds[sid]
        t_correct = treatment_preds[sid]

        if b_correct and t_correct:
            a += 1
        elif b_correct and not t_correct:
            b += 1
        elif not b_correct and t_correct:
            c += 1
        else:
            d += 1

    # 计算准确率
    baseline_acc = (a + b) / n
    treatment_acc = (a + c) / n
    diff = treatment_acc - baseline_acc

    # McNemar 检验
    if b + c == 0:
        p_value = 1.0
        chi2_stat = 0.0
        test_method = "N/A (b+c=0)"
    elif b + c < 25:
        # 使用精确二项检验
        result = binomtest(c, b + c, p=0.5, alternative='two-sided')
        p_value = result.pvalue
        chi2_stat = None
        test_method = "exact_binomial"
    else:
        # 使用带 Yates 连续性修正的 McNemar 检验
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)
        test_method = "mcnemar_yates"

    # 计算 95% 置信区间
    if b + c > 0:
        se = np.sqrt((b + c) - (b - c)**2 / n) / n
    else:
        se = 0.0
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se

    # 显著性判断
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "n.s."

    # 结论
    if p_value < alpha:
        if c > b:
            conclusion = "Treatment significantly better than Baseline"
        else:
            conclusion = "Baseline significantly better than Treatment"
    else:
        conclusion = "No significant difference"

    return {
        'n_samples': n,
        'contingency_table': {
            'both_correct': a,
            'baseline_only': b,
            'treatment_only': c,
            'both_wrong': d,
        },
        'baseline_acc': baseline_acc,
        'treatment_acc': treatment_acc,
        'diff': diff,
        'b': b,
        'c': c,
        'chi2_stat': chi2_stat,
        'p_value': p_value,
        'test_method': test_method,
        'significance': significance,
        'ci_95': (ci_low, ci_high),
        'conclusion': conclusion,
    }


def print_results(results, baseline_name="Baseline", treatment_name="Treatment"):
    """打印检验结果"""
    print("=" * 60)
    print("McNemar 配对检验结果")
    print("=" * 60)

    print(f"\n共同样本数: {results['n_samples']}")

    ct = results['contingency_table']
    print(f"\n{'2x2 列联表':=^50}")
    print(f"{'':20} {treatment_name}正确  {treatment_name}错误")
    print(f"{baseline_name}正确 {ct['both_correct']:>10} {ct['baseline_only']:>14}")
    print(f"{baseline_name}错误 {ct['treatment_only']:>10} {ct['both_wrong']:>14}")

    print(f"\n{'准确率':=^50}")
    print(f"{baseline_name} 准确率:  {results['baseline_acc']:.4f} ({ct['both_correct'] + ct['baseline_only']}/{results['n_samples']})")
    print(f"{treatment_name} 准确率:  {results['treatment_acc']:.4f} ({ct['both_correct'] + ct['treatment_only']}/{results['n_samples']})")
    print(f"差异:             {results['diff']:+.4f}")

    print(f"\n{'McNemar 检验':=^50}")
    print(f"b ({baseline_name}对, {treatment_name}错): {results['b']}")
    print(f"c ({baseline_name}错, {treatment_name}对): {results['c']}")
    print(f"检验方法: {results['test_method']}")
    if results['chi2_stat'] is not None:
        print(f"Chi-squared: {results['chi2_stat']:.4f}")
    print(f"p-value: {results['p_value']:.6f}")

    print(f"\n{'结论':=^50}")
    print(f"显著性: {results['significance']}")
    print(f"95% CI: [{results['ci_95'][0]:.4f}, {results['ci_95'][1]:.4f}]")
    print(f"结论: {results['conclusion']} (p={results['p_value']:.4f})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='McNemar 配对检验：比较两个模型预测结果的显著性差异',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  # 比较 steering 与 baseline
  python mcnemar_test.py \\
      --baseline 500_preds.jsonl \\
      --treatment ../outputs/analysis/preds/preds_layer25_lambda_0.30.jsonl

  # 指定名称和显著性水平
  python mcnemar_test.py \\
      --baseline baseline.jsonl \\
      --treatment steering.jsonl \\
      --baseline_name "No Steering" \\
      --treatment_name "CCA Steering" \\
      --alpha 0.01
        '''
    )

    parser.add_argument('--baseline', required=True,
                        help='Baseline 预测文件路径 (jsonl)')
    parser.add_argument('--treatment', required=True,
                        help='Treatment 预测文件路径 (jsonl)')
    parser.add_argument('--baseline_name', default='Baseline',
                        help='Baseline 显示名称 (默认: Baseline)')
    parser.add_argument('--treatment_name', default='Treatment',
                        help='Treatment 显示名称 (默认: Treatment)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='显著性水平 (默认: 0.05)')
    parser.add_argument('--output_json', type=str, default=None,
                        help='输出 JSON 文件路径 (可选)')

    args = parser.parse_args()

    # 加载预测结果
    print(f"Loading baseline: {args.baseline}")
    baseline_preds = load_preds(args.baseline)
    print(f"  -> {len(baseline_preds)} samples")

    print(f"Loading treatment: {args.treatment}")
    treatment_preds = load_preds(args.treatment)
    print(f"  -> {len(treatment_preds)} samples")

    # 执行检验
    results = mcnemar_test(baseline_preds, treatment_preds, alpha=args.alpha)

    # 打印结果
    print_results(results, args.baseline_name, args.treatment_name)

    # 可选：保存 JSON
    if args.output_json:
        # 转换 tuple 为 list 以便 JSON 序列化
        results_json = results.copy()
        results_json['ci_95'] = list(results['ci_95'])
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == '__main__':
    main()
