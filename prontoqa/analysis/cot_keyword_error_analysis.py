#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoT 关键词和错误模式详细分析脚本
分析 Steering 前后的推理逻辑关键词变化和错误模式分布

使用方法:
    python cot_keyword_error_analysis.py --baseline 500_preds.jsonl --steered preds/preds_layer16_lambda_0.06.jsonl
"""

import argparse
import json
import re
import numpy as np
from collections import Counter, defaultdict


def load_jsonl(path):
    """读取 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def is_correct(record):
    """判断预测是否正确"""
    pred = str(record['pred'])
    gold = str(record['gold'])
    if pred == 'PARSE_FAILED':
        return False
    return pred == gold


# 关键词分类
KEYWORD_CATEGORIES = {
    # 量词类
    'Quantifiers': {
        'every': r'\bevery\b',
        'each': r'\beach\b',
        'all': r'\ball\b',
        'some': r'\bsome\b',
        'no': r'\bno\b',
        'any': r'\bany\b',
    },
    # 系动词/判断类
    'Copula': {
        'is_a': r'\bis\s+a\b',
        'is_an': r'\bis\s+an\b',
        'is_not': r'\bis\s+not\b',
        'are': r'\bare\b',
        'are_not': r'\bare\s+not\b',
    },
    # 逻辑连接词
    'Logical_Connectives': {
        'therefore': r'\btherefore\b',
        'thus': r'\bthus\b',
        'hence': r'\bhence\b',
        'so': r'\bso\b',
        'because': r'\bbecause\b',
        'since': r'\bsince\b',
        'if': r'\bif\b',
        'then': r'\bthen\b',
    },
    # 推理动词
    'Reasoning_Verbs': {
        'means': r'\bmeans\b',
        'implies': r'\bimplies\b',
        'conclude': r'\bconclude\b',
        'know': r'\bknow\b',
        'given': r'\bgiven\b',
        'follows': r'\bfollows\b',
    },
    # 结论标记
    'Conclusion_Markers': {
        'true': r'\btrue\b',
        'false': r'\bfalse\b',
        'truth_value': r'truth\s*value',
        'answer': r'\banswer\b',
    },
    # 否定词
    'Negation': {
        'not': r'\bnot\b',
        'never': r'\bnever\b',
        'neither': r'\bneither\b',
        'cannot': r'\bcannot\b',
    },
}

# 错误模式名称映射
ERROR_PATTERN_NAMES = {
    'repetitive_loop': '循环推理',
    'truncated': '截断/不完整',
    'contradiction': '矛盾推理',
    'over_reasoning': '过度推理',
    'reasoning_gap': '推理跳跃',
    'negation_error': '否定处理错误',
    'quantifier_confusion': '量词混淆',
    'irrelevant_reasoning': '无关推理',
    'conclusion_mismatch': '结论不一致',
    'entity_confusion': '实体混淆',
}


def count_keywords(text):
    """统计各类关键词"""
    results = {}
    for category, keywords in KEYWORD_CATEGORIES.items():
        results[category] = {}
        for name, pattern in keywords.items():
            count = len(re.findall(pattern, text, re.IGNORECASE))
            results[category][name] = count
    return results


def detect_error_patterns(text, pred, gold):
    """检测各种错误模式"""
    patterns = {}

    # 1. 循环推理 (Repetitive Loop)
    loop_pattern = r'(.{20,60})\1{2,}'
    patterns['repetitive_loop'] = bool(re.search(loop_pattern, text))

    # 2. 截断/不完整输出 (Truncated)
    has_conclusion = bool(re.search(r'truth\s*value|the\s+answer\s+is|therefore.*(?:true|false)', text, re.I))
    patterns['truncated'] = not has_conclusion and len(text) > 500

    # 3. 矛盾推理 (Contradiction)
    patterns['contradiction'] = False
    is_statements = re.findall(r'(\w+)\s+is\s+(?:a\s+)?(\w+)', text, re.I)
    is_not_statements = re.findall(r'(\w+)\s+is\s+not\s+(?:a\s+)?(\w+)', text, re.I)
    for subj1, obj1 in is_statements:
        for subj2, obj2 in is_not_statements:
            if subj1.lower() == subj2.lower() and obj1.lower() == obj2.lower():
                patterns['contradiction'] = True
                break

    # 4. 过度推理 (Over-reasoning)
    step_count = len(re.findall(r'\b(?:is|are)\s+(?:a|an|not)\b', text, re.I))
    patterns['over_reasoning'] = step_count > 60

    # 5. 推理跳跃 (Reasoning Gap)
    therefore_matches = list(re.finditer(r'\btherefore\b', text, re.I))
    patterns['reasoning_gap'] = False
    for match in therefore_matches:
        start = max(0, match.start() - 100)
        context = text[start:match.start()]
        if not re.search(r'\b(?:is|are|since|because|given)\b', context, re.I):
            patterns['reasoning_gap'] = True
            break

    # 6. 错误否定处理 (Negation Error)
    double_neg = re.findall(r'not\s+(?:\w+\s+){0,3}not', text, re.I)
    patterns['negation_error'] = len(double_neg) > 2

    # 7. 量词混淆 (Quantifier Confusion)
    every_count = len(re.findall(r'\bevery\b', text, re.I))
    each_count = len(re.findall(r'\beach\b', text, re.I))
    all_count = len(re.findall(r'\ball\b', text, re.I))
    patterns['quantifier_confusion'] = (every_count > 10 and each_count > 10) or (all_count > 5 and every_count > 10)

    # 8. 无关推理 (Irrelevant Reasoning)
    patterns['irrelevant_reasoning'] = False

    # 9. 结论与推理不一致 (Conclusion Mismatch)
    if pred not in ['PARSE_FAILED', None]:
        last_true = text.rfind('true')
        last_false = text.rfind('false')
        if last_true > last_false and pred == 'False':
            patterns['conclusion_mismatch'] = True
        elif last_false > last_true and pred == 'True':
            patterns['conclusion_mismatch'] = True
        else:
            patterns['conclusion_mismatch'] = False
    else:
        patterns['conclusion_mismatch'] = False

    # 10. 实体混淆 (Entity Confusion)
    entities = re.findall(r'\b([A-Z][a-z]+)\b', text)
    entity_counts = Counter(entities)
    rare_entities = sum(1 for e, c in entity_counts.items() if c <= 2)
    patterns['entity_confusion'] = rare_entities > 10

    return patterns


def analyze_keywords(common_ids, baseline_dict, steered_dict):
    """分析关键词变化"""
    print("=" * 80)
    print("推理逻辑关键词详细分析")
    print("=" * 80)

    baseline_counts = defaultdict(lambda: defaultdict(int))
    steered_counts = defaultdict(lambda: defaultdict(int))

    for sid in common_ids:
        b_gen = baseline_dict[sid].get('gen', '')
        s_gen = steered_dict[sid].get('gen', '')

        b_kw = count_keywords(b_gen)
        s_kw = count_keywords(s_gen)

        for cat in KEYWORD_CATEGORIES:
            for kw in KEYWORD_CATEGORIES[cat]:
                baseline_counts[cat][kw] += b_kw[cat][kw]
                steered_counts[cat][kw] += s_kw[cat][kw]

    # 打印各类别详情
    for category in KEYWORD_CATEGORIES:
        print(f"\n--- {category} ---")
        print(f"{'关键词':<15} {'Baseline':>10} {'Steered':>10} {'变化':>10} {'变化%':>10}")
        print("-" * 58)

        cat_baseline_total = 0
        cat_steered_total = 0

        for kw in KEYWORD_CATEGORIES[category]:
            b_val = baseline_counts[category][kw]
            s_val = steered_counts[category][kw]
            diff = s_val - b_val
            pct = (diff / b_val * 100) if b_val > 0 else 0
            print(f"{kw:<15} {b_val:>10} {s_val:>10} {diff:>+10} {pct:>+9.1f}%")
            cat_baseline_total += b_val
            cat_steered_total += s_val

        cat_diff = cat_steered_total - cat_baseline_total
        cat_pct = (cat_diff / cat_baseline_total * 100) if cat_baseline_total > 0 else 0
        print("-" * 58)
        print(f"{'小计':<15} {cat_baseline_total:>10} {cat_steered_total:>10} {cat_diff:>+10} {cat_pct:>+9.1f}%")

    # 类别总结
    print("\n" + "=" * 80)
    print("关键词类别总结")
    print("=" * 80)
    print(f"{'类别':<25} {'Baseline':>10} {'Steered':>10} {'变化':>10} {'变化%':>10}")
    print("-" * 68)

    for category in KEYWORD_CATEGORIES:
        b_total = sum(baseline_counts[category].values())
        s_total = sum(steered_counts[category].values())
        diff = s_total - b_total
        pct = (diff / b_total * 100) if b_total > 0 else 0
        print(f"{category:<25} {b_total:>10} {s_total:>10} {diff:>+10} {pct:>+9.1f}%")

    return baseline_counts, steered_counts


def analyze_error_patterns(common_ids, baseline_dict, steered_dict):
    """分析错误模式"""
    print("\n" + "=" * 80)
    print("错误模式详细分析")
    print("=" * 80)

    baseline_fail_ids = [sid for sid in common_ids if not is_correct(baseline_dict[sid])]
    steered_fail_ids = [sid for sid in common_ids if not is_correct(steered_dict[sid])]

    print(f"\nBaseline 错误样本数: {len(baseline_fail_ids)}")
    print(f"Steered 错误样本数:  {len(steered_fail_ids)}")

    # 统计错误模式
    baseline_patterns = defaultdict(int)
    steered_patterns = defaultdict(int)

    for sid in baseline_fail_ids:
        rec = baseline_dict[sid]
        patterns = detect_error_patterns(rec.get('gen', ''), rec.get('pred'), rec.get('gold'))
        for p, detected in patterns.items():
            if detected:
                baseline_patterns[p] += 1

    for sid in steered_fail_ids:
        rec = steered_dict[sid]
        patterns = detect_error_patterns(rec.get('gen', ''), rec.get('pred'), rec.get('gold'))
        for p, detected in patterns.items():
            if detected:
                steered_patterns[p] += 1

    # 打印统计
    print("\n" + "=" * 80)
    print("错误模式统计")
    print("=" * 80)
    print(f"{'错误模式':<25} {'Baseline':>10} {'Steered':>10} {'变化':>10}")
    print("-" * 58)

    all_patterns = set(baseline_patterns.keys()) | set(steered_patterns.keys())
    for p in sorted(all_patterns):
        b_val = baseline_patterns.get(p, 0)
        s_val = steered_patterns.get(p, 0)
        diff = s_val - b_val
        name = ERROR_PATTERN_NAMES.get(p, p)
        print(f"{name:<25} {b_val:>10} {s_val:>10} {diff:>+10}")

    # 比例统计
    print("\n" + "=" * 80)
    print("错误模式比例 (占各自错误样本的%)")
    print("=" * 80)
    print(f"{'错误模式':<25} {'Baseline%':>12} {'Steered%':>12}")
    print("-" * 52)

    for p in sorted(all_patterns):
        b_val = baseline_patterns.get(p, 0)
        s_val = steered_patterns.get(p, 0)
        b_pct = b_val / len(baseline_fail_ids) * 100 if baseline_fail_ids else 0
        s_pct = s_val / len(steered_fail_ids) * 100 if steered_fail_ids else 0
        name = ERROR_PATTERN_NAMES.get(p, p)
        print(f"{name:<25} {b_pct:>11.1f}% {s_pct:>11.1f}%")

    # 被修正样本分析
    print("\n" + "=" * 80)
    print("被修正样本的错误模式分析")
    print("=" * 80)

    fixed_ids = [sid for sid in baseline_fail_ids if is_correct(steered_dict[sid])]
    print(f"被修正的样本数: {len(fixed_ids)}")

    fixed_patterns = defaultdict(int)
    for sid in fixed_ids:
        rec = baseline_dict[sid]
        patterns = detect_error_patterns(rec.get('gen', ''), rec.get('pred'), rec.get('gold'))
        for p, detected in patterns.items():
            if detected:
                fixed_patterns[p] += 1

    if fixed_ids:
        print(f"\n{'错误模式':<25} {'被修正样本中':>12} {'占比':>10}")
        print("-" * 50)
        for p in sorted(fixed_patterns.keys(), key=lambda x: -fixed_patterns[x]):
            name = ERROR_PATTERN_NAMES.get(p, p)
            count = fixed_patterns[p]
            pct = count / len(fixed_ids) * 100
            print(f"{name:<25} {count:>12} {pct:>9.1f}%")

    # 新引入错误分析
    print("\n" + "=" * 80)
    print("新引入错误样本的错误模式分析")
    print("=" * 80)

    baseline_correct_ids = [sid for sid in common_ids if is_correct(baseline_dict[sid])]
    broken_ids = [sid for sid in baseline_correct_ids if not is_correct(steered_dict[sid])]
    print(f"新引入错误的样本数: {len(broken_ids)}")

    broken_patterns = defaultdict(int)
    for sid in broken_ids:
        rec = steered_dict[sid]
        patterns = detect_error_patterns(rec.get('gen', ''), rec.get('pred'), rec.get('gold'))
        for p, detected in patterns.items():
            if detected:
                broken_patterns[p] += 1

    if broken_ids:
        print(f"\n{'错误模式':<25} {'新错误样本中':>12} {'占比':>10}")
        print("-" * 50)
        for p in sorted(broken_patterns.keys(), key=lambda x: -broken_patterns[x]):
            name = ERROR_PATTERN_NAMES.get(p, p)
            count = broken_patterns[p]
            pct = count / len(broken_ids) * 100
            print(f"{name:<25} {count:>12} {pct:>9.1f}%")

    return baseline_patterns, steered_patterns


def main():
    parser = argparse.ArgumentParser(
        description='CoT 关键词和错误模式详细分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  python cot_keyword_error_analysis.py \\
      --baseline 500_preds.jsonl \\
      --steered preds/preds_layer16_lambda_0.06.jsonl
        '''
    )

    parser.add_argument('--baseline', required=True,
                        help='Baseline 预测文件路径 (jsonl)')
    parser.add_argument('--steered', required=True,
                        help='Steered 预测文件路径 (jsonl)')
    parser.add_argument('--output_json', type=str, default=None,
                        help='输出 JSON 文件路径 (可选)')

    args = parser.parse_args()

    # 加载数据
    print(f"Loading baseline: {args.baseline}")
    baseline = load_jsonl(args.baseline)
    print(f"  -> {len(baseline)} samples")

    print(f"Loading steered: {args.steered}")
    steered = load_jsonl(args.steered)
    print(f"  -> {len(steered)} samples")

    baseline_dict = {b['story_id']: b for b in baseline}
    steered_dict = {s['story_id']: s for s in steered}
    common_ids = list(set(baseline_dict.keys()) & set(steered_dict.keys()))

    print(f"\n共同样本数: {len(common_ids)}")

    # 关键词分析
    baseline_kw, steered_kw = analyze_keywords(common_ids, baseline_dict, steered_dict)

    # 错误模式分析
    baseline_err, steered_err = analyze_error_patterns(common_ids, baseline_dict, steered_dict)

    # 可选：保存 JSON
    if args.output_json:
        results = {
            'n_samples': len(common_ids),
            'keyword_counts': {
                'baseline': {cat: dict(kws) for cat, kws in baseline_kw.items()},
                'steered': {cat: dict(kws) for cat, kws in steered_kw.items()},
            },
            'error_patterns': {
                'baseline': dict(baseline_err),
                'steered': dict(steered_err),
            },
        }

        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == '__main__':
    main()
