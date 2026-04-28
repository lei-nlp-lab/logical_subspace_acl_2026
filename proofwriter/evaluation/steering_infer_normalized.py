#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalized steering version of steering_infer.py
应用方案1：归一化 steering direction 到与原始激活相同的范数

主要修改：在 HFSteerer 的 hook 中添加归一化逻辑
"""

import sys
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM

# 导入原始的辅助函数
from steering_infer import (
    primary_device,
    find_decoder_layers,
    load_svcca,
    select_columns_by_corr
)


class HFSteererNormalized:
    """
    归一化版本的 HFSteerer

    Steering 公式从：
        H' = H + λ * (P @ H)

    改为：
        H' = H + λ * (P @ H / ||P @ H|| * ||H||)

    优点：
    - λ 有统一的语义：相对于原始激活范数的比例
    - 不同层、不同样本的 λ 可比性更好
    - 降低 λ 敏感度
    """

    def __init__(self,
                 model: AutoModelForCausalLM,
                 svcca_pt: str,
                 layers: List[int],
                 lambdas: Dict[int, float],
                 use_projectors: bool = False,
                 top_k: int = 0,
                 corr_min: float = 0.0,
                 anchor: str = "last",
                 window: int = 1):
        self.model = model
        self.device = primary_device(model)
        self.dtype = next(model.parameters()).dtype
        self.layers, self.family = find_decoder_layers(model)
        self.L = len(self.layers)
        self.target_layers = layers
        self.lambdas = lambdas
        self.use_projectors = use_projectors
        self.top_k = int(top_k) if top_k else 0
        self.corr_min = float(corr_min)
        self.anchor = anchor
        self.window = max(1, int(window))
        self.U: Dict[int, torch.Tensor] = {}
        self.P: Dict[int, torch.Tensor] = {}

        bases, projectors, corrs, _ = load_svcca(svcca_pt)
        for ell in self.target_layers:
            U = bases.get(int(ell))
            P = projectors.get(int(ell))
            cvec = corrs.get(int(ell))
            if (not self.use_projectors) and (U is not None):
                U = U.to(self.device, dtype=self.dtype)
                if (self.top_k or self.corr_min > 0.0) and (cvec is not None):
                    U = select_columns_by_corr(U, cvec.to(self.device), self.top_k, self.corr_min)
                self.U[ell] = U
            else:
                if P is None and U is not None:
                    U = U.to(self.device, dtype=self.dtype)
                    if (self.top_k or self.corr_min > 0.0) and (cvec is not None):
                        U = select_columns_by_corr(U, cvec.to(self.device), self.top_k, self.corr_min)
                    P = (U @ U.transpose(-1, -2)).to(self.device, dtype=self.dtype)
                elif P is not None:
                    P = P.to(self.device, dtype=self.dtype)
                else:
                    raise ValueError(f"Layer {ell}: neither U nor P found in svcca_pt.")
                self.P[ell] = P

        # hooks
        self.handles = []
        for ell in self.target_layers:
            layer_mod = self.layers[ell]
            h = layer_mod.register_forward_pre_hook(self._make_pre_hook(ell), with_kwargs=True)
            self.handles.append(h)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    # ---- hook with normalization ----
    def _make_pre_hook(self, ell: int):
        lam = float(self.lambdas.get(ell, 0.0))
        U = self.U.get(ell, None)
        P = self.P.get(ell, None)
        anchor = self.anchor
        win = self.window

        def hook(module, args, kwargs):
            if len(args) == 0:
                return (args, kwargs)
            H = args[0]  # [B,T,D]
            if not torch.is_tensor(H):
                return (args, kwargs)
            B, T, D = H.shape

            # 只在生成阶段（T==1）做 steering
            if T > 1:
                return (args, kwargs)

            # 位置掩码
            if anchor == "last":
                attn_mask = None
                if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                    attn_mask = kwargs["attention_mask"]
                if attn_mask is None:
                    attn_mask = H.new_ones((B, T), dtype=torch.long)
                lengths = attn_mask.to(torch.int64).sum(dim=1)
                mask = H.new_zeros((B, T), dtype=torch.bool)
                for b in range(B):
                    Lb = int(lengths[b].item())
                    if Lb <= 0:
                        continue
                    s = max(0, Lb - win)
                    e = Lb
                    mask[b, s:e] = True
            else:
                mask = H.new_ones((B, T), dtype=torch.bool)

            m = mask.unsqueeze(-1).to(H.dtype)  # [B,T,1]

            # ===== 核心修改：归一化 steering direction =====
            if P is not None:
                HP = torch.matmul(H, P)  # [B,T,D]

                # 计算原始激活和 steering direction 的范数
                H_norm = torch.norm(H, dim=-1, keepdim=True)  # [B,T,1]
                HP_norm = torch.norm(HP, dim=-1, keepdim=True) + 1e-8  # [B,T,1]，避免除零

                # 归一化 steering direction 到与 H 相同的范数
                HP_normalized = HP / HP_norm * H_norm  # [B,T,D]

                H_delta = lam * m * HP_normalized

            elif U is not None:
                Z = torch.matmul(H, U)  # [B,T,k]
                H_proj = torch.matmul(Z, U.transpose(-1, -2))  # [B,T,D]

                # 同样归一化
                H_norm = torch.norm(H, dim=-1, keepdim=True)  # [B,T,1]
                HP_norm = torch.norm(H_proj, dim=-1, keepdim=True) + 1e-8  # [B,T,1]
                H_proj_normalized = H_proj / HP_norm * H_norm  # [B,T,D]

                H_delta = lam * m * H_proj_normalized
            else:
                return (args, kwargs)

            H_out = H + H_delta
            new_args = (H_out,) + tuple(args[1:])
            return (new_args, kwargs)

        return hook
