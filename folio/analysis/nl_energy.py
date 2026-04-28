#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, math, json
import numpy as np
import torch
from tqdm.auto import tqdm

def frob2(t: torch.Tensor) -> torch.Tensor:
    # Frobenius norm squared
    return (t.float()**2).sum()

def main():
    ap = argparse.ArgumentParser(description="Compute per-layer projectable energy (rho) from saved SVCCA outputs")
    ap.add_argument("--svcca_pt", required=True, help="svcca 输出 .pt（包含 bases/projectors/corrs/cfg/from_resid）")
    ap.add_argument("--out_csv", default="", help="输出 CSV 路径（留空则用同前缀 _rho.csv）")
    ap.add_argument("--use_projectors", action="store_true", help="若存在，则直接用 P=UU^T 计算（默认用 U）")
    ap.add_argument("--view_x", type=int, default=0, help="X 侧的视角索引（默认 0=NL）")
    ap.add_argument("--view_y", type=int, default=1, help="Y 侧的视角索引（默认 1=FOL）")
    ap.add_argument("--center_auto", action="store_true",
                    help="沿用 svcca cfg.center；若没写进 cfg，则不去中心。与 --center/--no_center 互斥")
    ap.add_argument("--center", action="store_true", help="显式：计算 rho 前对样本维去中心（覆盖 center_auto）")
    ap.add_argument("--no_center", action="store_true", help="显式：不去中心（覆盖 center_auto）")
    ap.add_argument("--layers", type=str, default="",
                    help="逗号分隔的层索引子集（如 5,10,11）；为空则全层")
    args = ap.parse_args()

    # ---- load svcca outputs ----
    svcca = torch.load(args.svcca_pt, map_location="cpu")
    bases = svcca.get("bases") or {}
    projectors = svcca.get("projectors") or {}
    corrs = svcca.get("corrs") or {}
    cfg = svcca.get("cfg", {})
    cfg_per_layer = svcca.get("cfg_per_layer", {})
    from_resid = svcca.get("from_resid", None)
    views = svcca.get("views", ["NL","FOL"])

    if not bases and not projectors:
        raise SystemExit("没有找到 bases 或 projectors；无法计算 rho。")

    if from_resid is None or not os.path.exists(from_resid):
        raise SystemExit(f"from_resid 路径不可用：{from_resid}. 请确保 svcca_pt 里记录了原 resid 路径。")

    # center 逻辑优先级：--center / --no_center > --center_auto(cfg.center) > False
    if args.center and args.no_center:
        raise SystemExit("不能同时指定 --center 与 --no_center")
    if args.center:
        do_center = True
    elif args.no_center:
        do_center = False
    elif args.center_auto:
        do_center = bool(cfg.get("center", False))
    else:
        do_center = False

    # ---- load resid tensor ----
    resid_obj = torch.load(from_resid, map_location="cpu")
    T = resid_obj["tensor"].float()   # [V, N, L, D]
    assert T.ndim == 4, f"tensor ndim={T.ndim}, 期望4维 [V,N,L,D]"
    V, N, L, D = T.shape
    if args.view_x >= V or args.view_y >= V:
        raise SystemExit(f"view 索引越界：V={V}, 传入 x={args.view_x}, y={args.view_y}")

    # ---- layer subset ----
    if args.layers.strip():
        sel_layers = sorted(set(int(s) for s in args.layers.split(",") if s.strip()))
    else:
        sel_layers = list(range(L))

    # ---- prepare csv path ----
    if not args.out_csv:
        prefix = os.path.splitext(args.svcca_pt)[0]
        out_csv = prefix + "_rho.csv"
    else:
        out_csv = args.out_csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # ---- iterate layers ----
    rows = []
    pbar = tqdm(sel_layers, desc="Computing rho per layer", dynamic_ncols=True)
    for ell in pbar:
        if ell < 0 or ell >= L:
            continue

        # choose representation matrices
        # X: NL侧（默认 view 0），Y: FOL侧（默认 view 1）
        X = T[args.view_x, :, ell, :].clone()  # [N,D]
        Y = T[args.view_y, :, ell, :].clone()  # [N,D]

        if do_center:
            X -= X.mean(dim=0, keepdim=True)
            Y -= Y.mean(dim=0, keepdim=True)

        # get U or P for this layer
        U = bases.get(int(ell), None)
        P = projectors.get(int(ell), None)
        if U is not None:
            U = U.float()           # [D,k]
            k_used = U.shape[1]
        else:
            k_used = None
        if P is not None:
            P = P.float()           # [D,D]

        # compute rho
        eps = 1e-12
        if args.use_projectors and (P is not None):
            # 直接用 P：rho = ||X P||_F^2 / ||X||_F^2
            XP = X @ P              # [N,D]
            rho_x = float((frob2(XP) / (frob2(X) + eps)).item())
            YP = Y @ P
            rho_y = float((frob2(YP) / (frob2(Y) + eps)).item())
        elif U is not None:
            # 用 U：rho = ||X U||_F^2 / ||X||_F^2
            XU = X @ U              # [N,k]
            YU = Y @ U              # [N,k]
            rho_x = float((frob2(XU) / (frob2(X) + eps)).item())
            rho_y = float((frob2(YU) / (frob2(Y) + eps)).item())
        else:
            # 既没有 U 也没有 P
            rho_x = float("nan")
            rho_y = float("nan")

        # meta
        meta = cfg_per_layer.get(int(ell), {})
        dx = meta.get("dx", None)
        dy = meta.get("dy", None)
        if k_used is None:
            k_used = meta.get("k_used", None)

        cvec = corrs.get(int(ell), None)
        mean_corr = float(cvec.mean().item()) if cvec is not None and cvec.numel() > 0 else float("nan")

        pbar.set_postfix_str(f"layer={ell} rho_x={rho_x:.3f} rho_y={rho_y:.3f} mean={mean_corr:.3f}")

        rows.append({
            "layer": ell,
            "rho_x": rho_x,
            "rho_y": rho_y,
            "mean_corr": mean_corr,
            "k_used": int(k_used) if k_used is not None else None,
            "dx": int(dx) if dx is not None else None,
            "dy": int(dy) if dy is not None else None,
        })

    # ---- save CSV ----
    # 统一列顺序
    cols = ["layer", "rho_x", "rho_y", "mean_corr", "k_used", "dx", "dy",
            "from_resid", "svcca_pt", "view_x", "view_y", "center_used"]
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r_out = dict(r)
            r_out.update({
                "from_resid": from_resid,
                "svcca_pt": os.path.abspath(args.svcca_pt),
                "view_x": args.view_x,
                "view_y": args.view_y,
                "center_used": do_center,
            })
            w.writerow(r_out)

    print(f"Saved rho per layer to: {out_csv}")
    # 顺便打印 top-5 层（按 rho_x 排序）
    rows_sorted = sorted([r for r in rows if not math.isnan(r["rho_x"])], key=lambda z: z["rho_x"], reverse=True)
    print("Top-5 layers by rho_x:")
    for r in rows_sorted[:5]:
        print(f"  layer {r['layer']:>3}: rho_x={r['rho_x']:.4f}, mean_corr={r['mean_corr']:.4f}, k_used={r['k_used']}")

if __name__ == "__main__":
    main()
