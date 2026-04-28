#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, csv, sys
import numpy as np
import torch
from typing import Tuple
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

# =========================
# Utils
# =========================

def keep_by_var(pca: PCA, Z: np.ndarray, var_thresh: float) -> Tuple[np.ndarray, int]:
    """按累计方差阈值裁剪到 m 维。"""
    if hasattr(pca, "explained_variance_ratio_"):
        c = np.cumsum(pca.explained_variance_ratio_)
        m = int(np.searchsorted(c, var_thresh) + 1)
        m = max(1, min(m, Z.shape[1]))
        return Z[:, :m], m
    return Z, Z.shape[1]

def svcca_projector(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    pca_var: float,
    pca_cap: int,
    center: bool,
    shared: str = "avg",
    lib: str = "sklearn",
    cca_max_iter: int = 5000,
    seed: int = 0,
):
    """
    执行 SVCCA 并产出共享基 U、投影器 P 以及逐维 canonical 相关系数。
    返回: U[D,k_used], P[D,D], cvec[k_used], k_used, (dx,dy)
    """
    assert X.shape == Y.shape and X.ndim == 2
    N, D = X.shape

    # 可选：手动中心化（PCA 也会去中心；二者择一皆可，统一即可）
    if center:
        X = X - X.mean(0, keepdims=True)
        Y = Y - Y.mean(0, keepdims=True)

    # PCA：cap ≤ min(pca_cap, N-1, D)
    cap = max(1, min(pca_cap, N - 1, D))
    pca_x = PCA(n_components=cap, svd_solver="full", random_state=seed)
    pca_y = PCA(n_components=cap, svd_solver="full", random_state=seed)
    Xr = pca_x.fit_transform(X)      # [N, dx_cap]
    Yr = pca_y.fit_transform(Y)      # [N, dy_cap]

    # 按方差阈值截断
    Xr, dx = keep_by_var(pca_x, Xr, pca_var)
    Yr, dy = keep_by_var(pca_y, Yr, pca_var)

    k_used = max(1, min(int(k), dx, dy))

    # 线性 CCA（lib 可选 sklearn / cca-zoo）
    if lib == "ccazoo":
        try:
            from cca_zoo.models import CCA as CCA_ZOO
        except Exception as e:
            print("[warn] cca-zoo 未安装或导入失败，回退到 sklearn:", e, file=sys.stderr)
            lib = "sklearn"

    if lib == "ccazoo":
        cca = CCA_ZOO(latent_dimensions=k_used, random_state=seed, max_iter=cca_max_iter)
        cca.fit((Xr, Yr))
        Xc, Yc = cca.transform((Xr, Yr))       # [N,k]
        Ux, Uy = cca.weights                    # Ux:[dx,k], Uy:[dy,k]
    else:
        from sklearn.cross_decomposition import CCA as SKCCA
        cca = SKCCA(n_components=k_used, max_iter=cca_max_iter, scale=True)
        cca.fit(Xr, Yr)
        Xc, Yc = cca.transform(Xr, Yr)         # [N,k]
        Ux = cca.x_weights_[:, :k_used]        # [dx,k]
        Uy = cca.y_weights_[:, :k_used]        # [dy,k]

    # 将 CCA 权重从 PCA 子空间映回原空间
    Bx = pca_x.components_.T[:, :dx] @ Ux      # [D,k]
    By = pca_y.components_.T[:, :dy] @ Uy      # [D,k]

    if shared == "x":
        B = Bx
    elif shared == "y":
        B = By
    else:
        # avg：先做列符号对齐，避免相消
        signs = np.sign(np.sum(Bx * By, axis=0, keepdims=True))
        signs[signs == 0] = 1
        By = By * signs
        B  = 0.5 * (Bx + By)

    # QR 正交化得到共享基 U（reduced 模式）
    Q, _ = np.linalg.qr(B, mode="reduced")     # Q:[D,r]
    U = Q[:, :k_used]                          # [D,k]
    P = U @ U.T                                # [D,D]

    # 用 transform 出来的 canonical variates 计算逐维相关
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-8)
    Yc = (Yc - Yc.mean(0)) / (Yc.std(0) + 1e-8)
    cvec = (Xc * Yc).mean(0)                   # [k]

    return (
        U.astype(np.float32),
        P.astype(np.float32),
        cvec.astype(np.float32),
        int(k_used),
        (int(dx), int(dy)),
    )

# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resid_pt", required=True, help="包含 tensor[2,N,L,D] 的 .pt（fp32）")
    ap.add_argument("--out_path", required=True, help="输出 .pt（含 projectors/bases/corrs 等）")
    ap.add_argument("--pca_var", type=float, default=0.98, help="PCA 保留方差比例（0.97~0.995 常用）")
    ap.add_argument("--pca_cap", type=int, default=128, help="PCA 上限维数（如 128/256）")
    ap.add_argument("--k", type=int, default=32, help="canonical 方向数（32/64/128 网格）")
    ap.add_argument("--center", action="store_true", help="样本维去中心（X,Y 各自减均值）")
    ap.add_argument("--shared", choices=["avg", "x", "y"], default="x", help="共享基构造策略")
    ap.add_argument("--lib", choices=["sklearn", "ccazoo"], default="sklearn", help="CCA 实现选择")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--cca_max_iter", type=int, default=5000, help="CCA 最大迭代数")
    ap.add_argument("--no_save_P", action="store_true", help="只存 U 不存 P（节省体积）")
    ap.add_argument("--no_tqdm", action="store_true", help="关闭进度条输出（日志环境下更干净）")
    args = ap.parse_args()

    # -------- load --------
    obj = torch.load(args.resid_pt, map_location="cpu")
    T = obj["tensor"].cpu()
    assert T.dtype == torch.float32, f"Expected fp32 tensor, got {T.dtype}"
    views = obj.get("views", ["NL", "FOL"])
    story_ids = obj.get("story_ids", None)
    labels = obj.get("labels", None)
    V, N, L, D = T.shape
    assert V >= 2, f"expect at least two views, got {V}"
    print(f"Loaded resid tensor: shape={tuple(T.shape)} (views,N,L,D) dtype={T.dtype}")

    projectors = {}      # layer -> [D,D] (或不保存)
    bases = {}           # layer -> [D,k]
    corrs = {}           # layer -> [k]
    per_layer_cfg = {}   # layer -> dict(dx,dy,k_used)

    # -------- per-layer loop with tqdm --------
    use_tqdm = (not args.no_tqdm)
    iterator = tqdm(range(L), total=L, desc="SVCCA per-layer", dynamic_ncols=True) if use_tqdm else range(L)

    for ell in iterator:
        X = T[0, :, ell, :].numpy()  # view 0
        Y = T[1, :, ell, :].numpy()  # view 1

        U, P, cvec, k_used, (dx, dy) = svcca_projector(
            X=X, Y=Y,
            k=args.k,
            pca_var=args.pca_var,
            pca_cap=args.pca_cap,
            center=args.center,
            shared=args.shared,
            lib=args.lib,
            cca_max_iter=args.cca_max_iter,
            seed=args.seed,
        )

        bases[int(ell)] = torch.from_numpy(U)
        corrs[int(ell)] = torch.from_numpy(cvec)
        if not args.no_save_P:
            projectors[int(ell)] = torch.from_numpy(P)

        per_layer_cfg[int(ell)] = {"dx": dx, "dy": dy, "k_used": k_used}

        mean_corr = float(cvec.mean()) if len(cvec) else 0.0
        msg = f"[layer {ell:>3}] dx={dx:>4} dy={dy:>4} k_used={k_used:>4} mean_corr={mean_corr:.4f}"
        if use_tqdm:
            iterator.set_postfix_str(f"mean={mean_corr:.3f}, k={k_used}, dx={dx}, dy={dy}")
        else:
            print(msg)

    # -------- save pt --------
    out = {
        "projectors": projectors if not args.no_save_P else None,
        "bases": bases,
        "corrs": corrs,
        "views": views,
        "story_ids": story_ids,
        "labels": labels,
        "cfg": {
            "k": args.k,
            "pca_var": args.pca_var,
            "pca_cap": args.pca_cap,
            "center": args.center,
            "shared": args.shared,
            "cca_impl": args.lib,
            "seed": args.seed,
            "cca_max_iter": args.cca_max_iter,
            "save_P": (not args.no_save_P),
        },
        "cfg_per_layer": per_layer_cfg,
        "from_resid": os.path.abspath(args.resid_pt),
    }
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save(out, args.out_path)
    print(f"Saved outputs to {args.out_path}")

    # -------- save CSVs --------
    prefix = os.path.splitext(args.out_path)[0]

    csv_corr = prefix + "_corrs.csv"
    with open(csv_corr, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "i", "corr", "mean_corr_used_k", "k_used"])
        for ell in sorted(corrs.keys()):
            c = corrs[ell].numpy()
            m = float(c.mean()) if len(c) else 0.0
            for i, v in enumerate(c, start=1):
                w.writerow([ell, i, float(v), m, len(c)])
    print("Saved per-dimension correlations:", csv_corr)

    csv_mean = prefix + "_layer_mean_corr.csv"
    with open(csv_mean, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "mean_corr", "k_used", "dx", "dy"])
        for ell in sorted(corrs.keys()):
            c = corrs[ell].numpy()
            meta = per_layer_cfg[ell]
            w.writerow([ell, f"{float(c.mean()):.6f}", meta["k_used"], meta["dx"], meta["dy"]])
    print("Saved layer-mean correlations:", csv_mean)

if __name__ == "__main__":
    main()
