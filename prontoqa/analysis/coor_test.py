#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, json, csv
from typing import Dict, Optional, List

import torch
from tqdm import tqdm

def load_svcca(svcca_pt: str):
    obj = torch.load(svcca_pt, map_location="cpu")
    bases = obj.get("bases") or {}
    projectors = obj.get("projectors") or {}
    corrs = obj.get("corrs") or {}
    cfg = obj.get("cfg", {})
    return bases, projectors, corrs, cfg

def select_columns_by_corr(U: torch.Tensor,
                           cvec: Optional[torch.Tensor],
                           top_k: int = 0,
                           corr_min: float = 0.0) -> torch.Tensor:
    if U is None: return U
    if cvec is None or cvec.numel() == 0:
        return U[:, :top_k] if (top_k and top_k < U.shape[1]) else U
    c = cvec.clone()
    keep = torch.ones_like(c, dtype=torch.bool)
    if corr_min > 0.0:
        keep = c >= corr_min
    U2 = U[:, keep]
    c2 = c[keep]
    if U2.shape[1] == 0:
        return U[:, :min(top_k or U.shape[1], U.shape[1])]
    if top_k and top_k < U2.shape[1]:
        idx = torch.argsort(c2, descending=True)[:top_k]
        return U2[:, idx]
    else:
        idx = torch.argsort(c2, descending=True)
        return U2[:, idx]

def parse_layers(s: str, L: int) -> List[int]:
    if not s:
        return list(range(L))
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b)+1))
        else:
            out.append(int(tok))
    out = sorted(set([x for x in out if 0 <= x < L]))
    if not out:
        raise ValueError("No valid layers parsed.")
    return out

def safe_corr_1d(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    # x,y: [N]
    x = x - x.mean()
    y = y - y.mean()
    sx = x.norm() + eps
    sy = y.norm() + eps
    return float((x @ y).item() / (sx.item() * sy.item()))

def corr_per_dim(Zx: torch.Tensor, Zy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Z*: [N, k]
    N, k = Zx.shape
    c = []
    for i in range(k):
        c.append(safe_corr_1d(Zx[:, i], Zy[:, i], eps))
    return torch.tensor(c, dtype=torch.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svcca_pt", required=True, help="训练集上算好的 SVCCA 结果（含 bases/projectors/corrs）")
    ap.add_argument("--val_resid_pt", required=True, help="验证集 resid（tensor[2,N,L,D] 的 .pt）")
    ap.add_argument("--out_csv", default="val_projection.csv", help="每层摘要输出")
    ap.add_argument("--out_corr_csv", default="val_corr_per_dim.csv", help="逐维相关输出")
    ap.add_argument("--layers", default="", help="评测层，如 '4,10' 或 '4-12'；空=两边都存在的层")
    ap.add_argument("--use_projectors", action="store_true", help="用 P=UU^T 计算 ρ；相关仍基于 U")
    ap.add_argument("--top_k", type=int, default=0, help="按训练 corr 从高到低裁列；0=不裁剪")
    ap.add_argument("--corr_min", type=float, default=0.0, help="仅保留 corr>=阈值 的列；0=不启用")
    ap.add_argument("--view_x_name", default="NL")
    ap.add_argument("--view_y_name", default="FOL")
    ap.add_argument("--center", action="store_true", help="在验证集上对 H 做样本维去均值后再投影")
    args = ap.parse_args()

    # ---- load svcca ----
    bases, projectors, corrs, svcfg = load_svcca(args.svcca_pt)

    # ---- load val resid ----
    val = torch.load(args.val_resid_pt, map_location="cpu")
    T = val["tensor"]  # [V, N, L, D]
    assert T.ndim == 4, f"expect [V,N,L,D], got {tuple(T.shape)}"
    V, N, L, D = T.shape
    views = val.get("views", [str(i) for i in range(V)])
    try:
        vx = views.index(args.view_x_name)
        vy = views.index(args.view_y_name) if args.view_y_name in views else None
    except ValueError:
        print(f"[error] view names not found in val_resid views={views}", file=sys.stderr)
        sys.exit(1)

    if not args.layers:
        sv_layers = sorted(set(int(k) for k in bases.keys()) | set(int(k) for k in projectors.keys()))
        layers = [l for l in range(L) if l in sv_layers]
    else:
        layers = parse_layers(args.layers, L)

    # ---- outputs ----
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_corr_csv) or ".", exist_ok=True)
    sum_writer = csv.writer(open(args.out_csv, "w", newline="", encoding="utf-8"))
    sum_writer.writerow([
        "layer","rho_x","rho_y","mean_corr_val","k_used",
        "mean_corr_train","top_k","corr_min","center","svcca_pt","val_resid_pt","view_x","view_y"
    ])
    corr_writer = csv.writer(open(args.out_corr_csv, "w", newline="", encoding="utf-8"))
    corr_writer.writerow([
        "layer","i","corr_val","k_used","svcca_pt","val_resid_pt"
    ])

    # ---- compute per layer ----
    pbar = tqdm(layers, desc="Validating U on val set")
    for ell in pbar:
        U = bases.get(int(ell))
        P = projectors.get(int(ell))
        cvec_train = corrs.get(int(ell))

        if U is None and P is None:
            continue

        # 列筛选
        U_sel = None
        if U is not None:
            U_sel = U.float()
            if (args.top_k or args.corr_min > 0.0) and (cvec_train is not None):
                U_sel = select_columns_by_corr(U_sel, cvec_train.float(), args.top_k, args.corr_min)

        if args.use_projectors:
            if P is None:
                if U_sel is None:
                    # 既没 P 也没 U
                    continue
                P = U_sel @ U_sel.t()
            else:
                P = P.float()
            # k 用列数表示（若只有 P 且来源不明，设为 -1）
            k_used = (U_sel.shape[1] if U_sel is not None else -1)
        else:
            if U_sel is None:
                continue
            k_used = U_sel.shape[1]

        # 取验证集 H（N×D）
        Hx = T[vx, :, ell, :].float()  # [N, D]
        Hy = T[vy, :, ell, :].float() if (vy is not None) else None
        if args.center:
            Hx = Hx - Hx.mean(dim=0, keepdim=True)
            if Hy is not None:
                Hy = Hy - Hy.mean(dim=0, keepdim=True)

        # ——— ρ: 可投影能量 ———
        def proj_frac(H: torch.Tensor) -> float:
            denom = torch.linalg.norm(H, ord="fro").pow(2).item() + 1e-12
            if args.use_projectors:
                HP = H @ P
                num = torch.linalg.norm(HP, ord="fro").pow(2).item()
            else:
                Z = H @ U_sel
                num = torch.linalg.norm(Z, ord="fro").pow(2).item()
            return float(num / denom)

        rho_x = proj_frac(Hx)
        rho_y = proj_frac(Hy) if Hy is not None else None

        # ——— 验证集上的逐维相关（在共享子空间 U 上）———
        # 用 U 做坐标：Zx = Hx U, Zy = Hy U；对每列做 Pearson corr
        mean_corr_val = ""
        if Hy is not None:
            Zx = Hx @ (U_sel if U_sel is not None else P)  # 若用 P，这里仍要 U_sel；否则没法逐维
            if U_sel is None:
                # 如果只提供了 P 而没 U，则无法逐维相关；跳过
                per_dim = None
            else:
                Zy = Hy @ U_sel
                per_dim = corr_per_dim(Zx, Zy)  # [k_used]
                mean_corr_val = float(per_dim.mean().item())
                # 写逐维表
                for i, v in enumerate(per_dim.tolist(), start=1):
                    corr_writer.writerow([ell, i, v, k_used, os.path.abspath(args.svcca_pt), os.path.abspath(args.val_resid_pt)])
        else:
            per_dim = None

        mean_corr_train = ""
        if cvec_train is not None and cvec_train.numel() > 0:
            mean_corr_train = f"{float(cvec_train.mean().item()):.6f}"

        sum_writer.writerow([
            ell,
            rho_x,
            (rho_y if rho_y is not None else ""),
            (f"{mean_corr_val:.6f}" if mean_corr_val != "" else ""),
            k_used,
            mean_corr_train,
            int(args.top_k),
            float(args.corr_min),
            bool(args.center),
            os.path.abspath(args.svcca_pt),
            os.path.abspath(args.val_resid_pt),
            args.view_x_name,
            args.view_y_name
        ])

    print(f"\nSaved summary CSV to {args.out_csv}")
    print(f"Saved per-dim corr CSV to {args.out_corr_csv}")

if __name__ == "__main__":
    main()
