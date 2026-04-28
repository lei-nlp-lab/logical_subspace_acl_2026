#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
from typing import List, Dict, Any

import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

def get_pair_texts(rec: Dict[str, Any]):
    """从一条记录里取 NL / FOL 文本；若缺任一视角返回 (None, None)"""
    pair = rec.get("pair") or []
    text_nl, text_fol = None, None
    for item in pair:
        v = (item.get("view") or "").upper()
        if v == "NL":
            text_nl = item.get("text", "")
        elif v == "FOL":
            text_fol = item.get("text", "")
    return text_nl, text_fol

def count_valid_pairs(path: str, max_pairs: int = 0) -> int:
    """预扫一遍，统计 (NL, FOL) 都存在的 pair 数量。"""
    cnt = 0
    for rec in read_jsonl(path):
        nl, fol = get_pair_texts(rec)
        if nl and fol:
            cnt += 1
            if max_pairs and cnt >= max_pairs:
                break
    return cnt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="成对 JSONL（每行含 story_id/label/pair[NL,FOL]）")
    ap.add_argument("--output_pt", required=True, help="输出 .pt 文件路径（保存 2×N×L×D 张量）")
    ap.add_argument("--model", required=True, help="TransformerLens 支持的 HF 模型名")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--prepend_bos", action="store_true", help="需要时在开头加 BOS")
    ap.add_argument("--max_pairs", type=int, default=0, help="仅处理前 N 对；0 表示全量")
    ap.add_argument("--tail_window", type=int, default=8, help="每条样本末尾用于池化的有效 token 数（左 pad 友好）")
    args = ap.parse_args()

    # 选择 dtype
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 加载模型（使用传入的 --model）
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
    pad_id = getattr(model.tokenizer, "pad_token_id", None)
    print(f"Model loaded. Layers={L}, d_model={D}, pad_id={pad_id}, n_ctx={model.cfg.n_ctx}")
    print(f"Tail-window pooling enabled: last {args.tail_window} tokens per example.")

    # 预估总数并创建进度条
    total_pairs = count_valid_pairs(args.input, args.max_pairs)
    if total_pairs == 0:
        raise SystemExit("No valid NL/FOL pairs found in input.")
    pbar = tqdm(total=total_pairs, ncols=100, desc="Extracting residual (tail-window means)")

    # 容器：按视角分别收集 [L, D]，最后拼成 [2, N, L, D]
    nl_list: List[torch.Tensor] = []
    fol_list: List[torch.Tensor] = []
    story_ids: List[Any] = []
    labels: List[Any] = []

    n_pairs = 0
    for rec in read_jsonl(args.input):
        text_nl, text_fol = get_pair_texts(rec)
        if not text_nl or not text_fol:
            continue

        texts = [text_nl, text_fol]  # batch=2，顺序固定：NL, FOL
        toks = model.to_tokens(texts, prepend_bos=args.prepend_bos, move_to_device=True)  # [2, S]

        with torch.inference_mode():
            _, cache = model.run_with_cache(toks, stop_at_layer=None)

        # resid_pre: [L, B, S, D]
        resid_pre = cache.stack_activation("resid_pre")  # fp16/bf16 也可以

        # 有效 token mask：True=有效
        if pad_id is not None:
            valid_mask = (toks != pad_id)  # [B, S]
        else:
            valid_mask = torch.ones_like(toks, dtype=torch.bool)

        # ===== 尾窗口掩码（向左 padding 友好）=====
        # 对每个样本 b，取其有效长度 len_b = valid_mask[b].sum()
        # 选择区间 [max(0, len_b - tail_window), len_b)
        B, S = toks.shape
        lengths = valid_mask.sum(dim=1)                                  # [B]
        tail_w = max(1, int(args.tail_window))
        # 构造位置索引 [0..S-1]
        idx = torch.arange(S, device=toks.device).unsqueeze(0).expand(B, S)  # [B, S]
        starts = (lengths - tail_w).clamp_min(0).unsqueeze(1)                # [B, 1]
        ends   = lengths.unsqueeze(1)                                        # [B, 1]
        tail_mask = (idx >= starts) & (idx < ends)                           # [B, S]
        # 与有效 mask 共同约束（避免 pad 被选到）
        tail_mask = tail_mask & valid_mask                                   # [B, S]

        # 扩展到与 resid_pre 对齐
        # resid_pre: [L, B, S, D]； 需要 [1,B,S,1]
        tail_mask4 = tail_mask.unsqueeze(0).unsqueeze(-1)                    # [1, B, S, 1]
        counts = tail_mask.sum(dim=1).clamp_min(1)                           # [B]
        # 对选中位置求和/均值
        sum_tail = (resid_pre * tail_mask4).sum(dim=2)                       # [L, B, D]
        mean_tail = sum_tail / counts.unsqueeze(0).unsqueeze(-1)             # [L, B, D]

        nl_list.append(mean_tail[:, 0, :].to("cpu"))  # [L, D]
        fol_list.append(mean_tail[:, 1, :].to("cpu"))  # [L, D]

        story_ids.append(rec.get("story_id"))
        labels.append(rec.get("label"))

        n_pairs += 1
        pbar.update(1)
        if args.max_pairs and n_pairs >= args.max_pairs:
            break

    pbar.close()

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
            "pooling": "tail_window",
            "tail_window": int(args.tail_window),
        },
        "note": "tensor[0,i,:,:] = NL of pair i; tensor[1,i,:,:] = FOL of pair i (tail-window pooled resid_pre)",
    }
    os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
    torch.save(out_obj, args.output_pt)
    print(f"Saved to {args.output_pt}")

if __name__ == "__main__":
    main()
