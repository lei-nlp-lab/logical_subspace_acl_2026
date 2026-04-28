#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
from typing import List, Dict, Any

import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm

def read_prontoqa_json(path: str):
    """读取 ProntoQA JSON 格式数据"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_view_pair(rec: Dict[str, Any], view_prefix: str):
    """
    从一条记录里提取指定视角的 with_proof 和 without_proof 文本

    Args:
        rec: 单条数据记录
        view_prefix: "NL" 或 "FOL"

    Returns:
        (text_with_proof, text_without_proof) 或 (None, None)
    """
    pair = rec.get("pair") or []
    text_with = None
    text_without = None

    with_view = f"{view_prefix}_with_proof"
    without_view = f"{view_prefix}_without_proof"

    for item in pair:
        view = item.get("view", "")
        if view == with_view:
            text_with = item.get("text", "")
        elif view == without_view:
            text_without = item.get("text", "")

    return text_with, text_without

def count_valid_pairs(data: List[Dict], view_prefix: str, max_pairs: int = 0) -> int:
    """预扫一遍，统计有效的 (with_proof, without_proof) 对数量"""
    cnt = 0
    for rec in data:
        text_with, text_without = get_view_pair(rec, view_prefix)
        if text_with and text_without:
            cnt += 1
            if max_pairs and cnt >= max_pairs:
                break
    return cnt

def extract_proof_residue(model, text_with_proof: str, text_without_proof: str,
                          prepend_bos: bool) -> torch.Tensor:
    """
    提取 proof 部分的 residual representation

    Args:
        model: HookedTransformer 模型
        text_with_proof: 完整文本（含推理证明）
        text_without_proof: 前缀文本（不含推理证明）
        prepend_bos: 是否添加 BOS token

    Returns:
        proof_residue: [L, D] 平均池化后的 proof 区域 residual (fp32)
    """
    # Tokenize 两个版本
    toks_with = model.to_tokens([text_with_proof], prepend_bos=prepend_bos, move_to_device=True)  # [1, S_with]
    toks_without = model.to_tokens([text_without_proof], prepend_bos=prepend_bos, move_to_device=True)  # [1, S_without]

    # 定位 proof 起始位置
    proof_start_idx = toks_without.shape[1]

    # ===== 验证：检查 tokenization 是否一致 =====
    if proof_start_idx > toks_with.shape[1]:
        raise ValueError(f"proof_start_idx ({proof_start_idx}) > full_length ({toks_with.shape[1]})")

    # 检查前缀是否匹配
    prefix_match = torch.equal(toks_with[0, :proof_start_idx], toks_without[0, :])

    if not prefix_match:
        # Tokenization 不一致！逐个比对找到真正的分界点
        mismatch_found = False
        for i in range(min(toks_with.shape[1], toks_without.shape[1])):
            if toks_with[0, i] != toks_without[0, i]:
                raise ValueError(f"Tokenization mismatch at position {i}: "
                               f"with={toks_with[0, i].item()}, without={toks_without[0, i].item()}")

        # 如果所有 token 都匹配，但长度不同
        raise ValueError(f"Tokenization length mismatch: without={toks_without.shape[1]}, "
                        f"with_prefix={proof_start_idx}")

    # 检查 proof 是否为空
    if proof_start_idx >= toks_with.shape[1]:
        raise ValueError(f"No proof tokens found (proof_start_idx={proof_start_idx}, "
                        f"full_length={toks_with.shape[1]})")

    # Run model on 完整文本
    with torch.inference_mode():
        _, cache = model.run_with_cache(toks_with, stop_at_layer=None)

    # 提取 resid_pre: [L, 1, S_with, D]
    resid_pre = cache.stack_activation("resid_pre")

    # 只取 proof 部分（从 proof_start_idx 到末尾）
    proof_residue = resid_pre[:, 0, proof_start_idx:, :]  # [L, S_proof, D]

    # 对 proof 部分做平均池化
    proof_mean = proof_residue.mean(dim=1)  # [L, D]

    return proof_mean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ProntoQA JSON 文件路径（如 5hop_0shot_noadj_processed_train.json）")
    ap.add_argument("--output_pt", required=True, help="输出 .pt 文件路径（保存 2×N×L×D 张量）")
    ap.add_argument("--model", required=True, help="TransformerLens 支持的 HF 模型名")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--prepend_bos", action="store_true", help="需要时在开头加 BOS")
    ap.add_argument("--max_pairs", type=int, default=0, help="仅处理前 N 对；0 表示全量")
    args = ap.parse_args()

    # 选择 dtype
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 加载模型
    print(f"Loading model: {args.model} on {args.device} dtype={dtype}")
    model = HookedTransformer.from_pretrained(
        model_name=args.model,
        device=args.device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    L = model.cfg.n_layers
    D = model.cfg.d_model
    print(f"Model loaded. Layers={L}, d_model={D}, n_ctx={model.cfg.n_ctx}")

    # 读取数据
    print(f"Loading data from {args.input}")
    data = read_prontoqa_json(args.input)
    print(f"Loaded {len(data)} samples")

    # 预估 NL 和 FOL 的有效对数量
    nl_count = count_valid_pairs(data, "NL", args.max_pairs)
    fol_count = count_valid_pairs(data, "FOL", args.max_pairs)
    print(f"Valid pairs: NL={nl_count}, FOL={fol_count}")

    if nl_count == 0 or fol_count == 0:
        raise SystemExit("No valid NL/FOL proof pairs found in input.")

    # 容器：按视角分别收集 [L, D]，最后拼成 [2, N, L, D]
    nl_list: List[torch.Tensor] = []
    fol_list: List[torch.Tensor] = []
    story_ids: List[Any] = []
    labels: List[Any] = []

    # 处理每个样本
    n_processed = 0
    n_skipped = 0

    pbar = tqdm(data, ncols=100, desc="Extracting CoT residuals")

    for rec in pbar:
        # 检查是否达到最大数量
        if args.max_pairs and n_processed >= args.max_pairs:
            break

        # 提取 NL 和 FOL 的 proof 对
        nl_with, nl_without = get_view_pair(rec, "NL")
        fol_with, fol_without = get_view_pair(rec, "FOL")

        if not (nl_with and nl_without and fol_with and fol_without):
            n_skipped += 1
            pbar.set_postfix({"processed": n_processed, "skipped": n_skipped})
            continue

        try:
            # 提取 NL proof residue
            nl_residue = extract_proof_residue(model, nl_with, nl_without, args.prepend_bos)

            # 提取 FOL proof residue
            fol_residue = extract_proof_residue(model, fol_with, fol_without, args.prepend_bos)

            # 保存
            nl_list.append(nl_residue.cpu())   # [L, D]
            fol_list.append(fol_residue.cpu()) # [L, D]

            story_ids.append(rec.get("story_id"))
            labels.append(rec.get("label"))

            n_processed += 1
            pbar.set_postfix({"processed": n_processed, "skipped": n_skipped})

        except Exception as e:
            print(f"\nWarning: Failed to process {rec.get('story_id')}: {e}")
            n_skipped += 1
            pbar.set_postfix({"processed": n_processed, "skipped": n_skipped})
            continue

    pbar.close()

    if n_processed == 0:
        raise SystemExit("No samples were successfully processed.")

    print(f"\nProcessed {n_processed} samples, skipped {n_skipped}")

    # 拼成 [2, N, L, D]
    NL = torch.stack(nl_list, dim=0)    # [N, L, D]
    FOL = torch.stack(fol_list, dim=0)  # [N, L, D]
    emb = torch.stack([NL, FOL], dim=0) # [2, N, L, D]

    print(f"Final tensor shape: {tuple(emb.shape)}  (expect 2×N×L×D)")

    out_obj = {
        "tensor": emb,                      # [2, N, L, D]
        "views": ["NL", "FOL"],
        "story_ids": story_ids,             # len = N（与 emb[:, i] 对齐）
        "labels": labels,                   # len = N
        "model_name": args.model,
        "config": {
            "n_layers": L,
            "d_model": D,
            "prepend_bos": args.prepend_bos,
            "pooling": "mean_over_proof",
            "dataset": "ProntoQA",
            "extraction_method": "CoT_only",
        },
        "note": "tensor[0,i,:,:] = NL CoT residue of pair i; tensor[1,i,:,:] = FOL CoT residue of pair i (averaged over proof tokens only)",
    }

    os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
    torch.save(out_obj, args.output_pt)
    print(f"Saved to {args.output_pt}")

if __name__ == "__main__":
    main()
