#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re, sys
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =========================
# Prompt & parsing
# =========================
PROMPT_TEMPLATE = """You are a careful logician. Use classical deductive reasoning.

Premises:
{prem_nl}

<CONCLUSION_START>
Hypothesis:
{hypo_nl}
<CONCLUSION_END>

Instructions:
- First, reason step by step.
- Then, on the last line, output exactly:
Truth value: <True|False|Uncertain>
"""

VAL_RE = re.compile(r"Truth\s*value\s*:\s*(True|False|Unknown|Uncertain)", re.IGNORECASE)

def norm_truth(x: str):
    if x is None: return None
    t = x.strip().lower()
    if t in {"true","t"}: return "True"
    if t in {"false","f"}: return "False"
    if t in {"unknown","uncertain","u"}: return "Unknown"
    m = VAL_RE.search(x)
    if m: return norm_truth(m.group(1))
    return None

def parse_truth(text: str):
    lines = [ln.strip() for ln in text.splitlines() if "truth value" in ln.lower()]
    if lines:
        m = VAL_RE.search(lines[-1])
        if m: return norm_truth(m.group(1))
    m = list(VAL_RE.finditer(text))
    return norm_truth(m[-1].group(1)) if m else None

def build_prompt(prem: str, concl: str) -> str:
    return PROMPT_TEMPLATE.format(prem_nl=prem.strip(), hypo_nl=concl.strip())

def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                parts = line.replace("}\n{","}|{").split("|")
                for p in parts:
                    p = p.strip()
                    if not p: continue
                    data.append(json.loads(p))
    return data

def batched(it, bs):
    for i in range(0, len(it), bs):
        yield it[i:i+bs]

# =========================
# HF steering helpers
# =========================
def primary_device(model):
    # 对 device_map=auto 的模型，取一个 cuda 设备；否则退回 model.device / cpu
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for dev in set(model.hf_device_map.values()):
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
    if hasattr(model, "device"):
        return model.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_decoder_layers(model) -> Tuple[List[torch.nn.Module], str]:
    """
    返回 [layer_modules], family_tag
    兼容 LLaMA/Mistral/Qwen2/GPT-NeoX/GPT2 等常见解码器容器。
    """
    # LLaMA / Mistral / Qwen2
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers), "meta-llama/mistral/qwen2"
    # GPT-NeoX / Dolly
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers), "gpt-neox"
    # GPT2 / GPT-J like
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h), "gpt2-family"
    # Falcon
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        return list(model.transformer.layers), "falcon"
    # Fallback: try model.layers
    if hasattr(model, "layers") and isinstance(model.layers, (list, tuple)):
        return list(model.layers), "generic"
    raise RuntimeError("Unrecognized decoder layer container in model; cannot place hooks.")

def parse_layers(s: str, L: int) -> List[int]:
    if not s.strip():
        raise ValueError("请用 --layers 指定至少一个层号（如 4,10 或 4-8）")
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        if "-" in tok:
            a, b = tok.split("-")
            a, b = int(a), int(b)
            out.extend(list(range(a, b+1)))
        else:
            out.append(int(tok))
    out = sorted(set([x for x in out if 0 <= x < L]))
    if not out:
        raise ValueError("层号解析为空或越界")
    return out

def parse_lambdas(s: Optional[str], default_lam: float, layers: List[int]) -> Dict[int, float]:
    if not s:
        return {ell: default_lam for ell in layers}
    d = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        k, v = tok.split(":")
        d[int(k)] = float(v)
    for ell in layers:
        d.setdefault(ell, default_lam)
    return d

def load_svcca(svcca_pt: str):
    obj = torch.load(svcca_pt, map_location="cpu")
    return obj.get("bases") or {}, obj.get("projectors") or {}, obj.get("corrs") or {}, obj.get("cfg", {})

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

class HFSteerer:
    """
    在 HF 模型的 decoder 层上，使用 forward-pre hook 在 resid_pre 注入：
        H' = H + λ * (H @ U @ U^T)   （或 H' = H + λ * (H @ P)）
    生效位置：anchor in {"last","all"}；"last" 使用 attention_mask 推出每条样本的最后有效 token，
    支持 window>1（最后 window 个 token 生效）。兼容左侧 padding。
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

    # ---- hook ----
    def _make_pre_hook(self, ell: int):
        lam = float(self.lambdas.get(ell, 0.0))
        U = self.U.get(ell, None)
        P = self.P.get(ell, None)
        anchor = self.anchor
        win = self.window

        def hook(module, args, kwargs):
            # 将 hidden_states 取出、修改后再放回
            # 兼容多种签名：大多数 decoder block 的第一个位置参数是 hidden_states
            if len(args) == 0:
                return (args, kwargs)  # 无法修改
            H = args[0]  # [B,T,D]
            if not torch.is_tensor(H):
                return (args, kwargs)
            B, T, D = H.shape

            # 只在生成阶段（T==1）做 steering，跳过 prompt 处理阶段（T>1）
            if T > 1:
                return (args, kwargs)

            # 位置掩码：从 attention_mask/causal mask 推最后有效 token（左 pad 支持）
            if anchor == "last":
                attn_mask = None
                # 优先从 kwargs 读
                if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                    attn_mask = kwargs["attention_mask"]  # [B,T]
                # 有些架构把 mask 存在 kwargs["encoder_attention_mask"]，但 decoder-only 通常没有
                if attn_mask is None:
                    # 退化：把全 True，当作“当前步在末尾”
                    attn_mask = H.new_ones((B, T), dtype=torch.long)
                # 最后有效位置 = 每行和-1
                lengths = attn_mask.to(torch.int64).sum(dim=1)  # [B]
                mask = H.new_zeros((B, T), dtype=torch.bool)
                # 对每个样本，最后 window 个有效 token 置 True
                for b in range(B):
                    Lb = int(lengths[b].item())
                    if Lb <= 0:
                        continue
                    s = max(0, Lb - win)
                    e = Lb
                    # 注意：如果是左 pad，末尾就是有效的最后 token，自然对齐
                    mask[b, s:e] = True
            else:
                # "all"
                mask = H.new_ones((B, T), dtype=torch.bool)

            m = mask.unsqueeze(-1).to(H.dtype)  # [B,T,1]

            if P is not None:
                HP = torch.matmul(H, P)              # [B,T,D]
                H_delta = lam * m * HP
            elif U is not None:
                Z = torch.matmul(H, U)               # [B,T,k]
                H_proj = torch.matmul(Z, U.transpose(-1, -2))  # [B,T,D]
                H_delta = lam * m * H_proj
            else:
                return (args, kwargs)

            H_out = H + H_delta
            # 把修改后的 H 写回 args[0]
            new_args = (H_out,) + tuple(args[1:])
            return (new_args, kwargs)

        return hook

# =========================
# Main eval (FOLIO)
# =========================
def main():
    ap = argparse.ArgumentParser()
    # 数据与模型
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--file", type=str, required=True,
                    help="Path to local FOLIO jsonl file (e.g., folio_v2_validation.jsonl)")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    ap.add_argument("--output", type=str, default="preds.jsonl")

    # ===== Steering 选项 =====
    ap.add_argument("--svcca_pt", type=str, default="", help="离线 SVCCA 结果（含 bases/projectors/corrs）")
    ap.add_argument("--layers", type=str, default="", help="逗号或区间，如 '4,10' 或 '4-6'")
    ap.add_argument("--lambda_", type=float, default=0.6, help="默认 λ（未单独指定的层用此值）")
    ap.add_argument("--lambdas", type=str, default="", help="逐层 λ，如 '4:0.6,10:0.4'")
    ap.add_argument("--use_projectors", action="store_true", help="使用 P=UU^T；默认用 U 做两次 matmul")
    ap.add_argument("--top_k", type=int, default=0, help="按 corr 排序保留前 top_k 列（0=不裁剪）")
    ap.add_argument("--corr_min", type=float, default=0.0, help="仅保留 corr>=corr_min 的列（0=不启用）")
    ap.add_argument("--anchor", choices=["last","all"], default="all", help="投影作用位置：last=最后 window 个有效 token；all=全序列")
    ap.add_argument("--window", type=int, default=1, help="anchor=last 时，最后多少个 token 生效")
    ap.add_argument("--steer_off", action="store_true", help="关闭 steering（做对照实验）")
    args = ap.parse_args()

    # 设置随机种子以确保可复现性
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


    # 读取数据
    rows = read_jsonl(args.file)
    if args.max_samples and args.max_samples < len(rows):
        rows = rows[:args.max_samples]

    def get_field(d, candidates, default=None):
        for k in candidates:
            if k in d: return d[k]
        return default

    samples = []
    for r in rows:
        prem = get_field(r, ["premises", "Premises - NL", "premises_nl", "premises-nl"])
        concl = get_field(r, ["conclusion", "Conclusions - NL", "conclusion_nl", "conclusion-nl"])
        lab  = get_field(r, ["label", "Truth Values", "truth", "gold_label"])
        if isinstance(prem, list): prem = " ".join(prem)
        if prem is None or concl is None or lab is None: continue
        gold = norm_truth(str(lab))
        if gold is None: continue
        samples.append({"prem": prem, "concl": concl, "gold": gold})
    if not samples:
        print("No valid samples parsed. Check your JSONL fields.", file=sys.stderr); sys.exit(1)

    # 模型与 tokenizer
    load_kwargs = {}
    if args.use_4bit: load_kwargs["load_in_4bit"] = True
    if args.use_8bit: load_kwargs["load_in_8bit"] = True

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=dtype,
            attn_implementation="flash_attention_2", low_cpu_mem_usage=True, **load_kwargs
        )
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=dtype,
            attn_implementation="sdpa", low_cpu_mem_usage=True, **load_kwargs
        )

    mdl.generation_config.pad_token_id = tok.pad_token_id
    mdl.generation_config.eos_token_id = tok.eos_token_id
    mdl.generation_config.use_cache = True

    pdev = primary_device(mdl)
    print("Primary device:", pdev)
    print("cuda_available:", torch.cuda.is_available(), "torch:", torch.__version__, "built_cuda:", torch.version.cuda)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ====== 注册 steering（resid_pre）======
    steerer = None
    if (not args.steer_off) and args.svcca_pt and args.layers:
        # 获取 decoder 层数用于解析层号
        layer_list, fam = find_decoder_layers(mdl)
        L_total = len(layer_list)
        sel_layers = parse_layers(args.layers, L_total)
        lam_map = parse_lambdas(args.lambdas, args.lambda_, sel_layers)
        print(f"[Steering ON] family={fam}  layers={sel_layers}  lambdas={lam_map}  "
              f"anchor={args.anchor} window={args.window}  useP={args.use_projectors}  "
              f"top_k={args.top_k} corr_min={args.corr_min}")
        steerer = HFSteerer(
            model=mdl,
            svcca_pt=args.svcca_pt,
            layers=sel_layers,
            lambdas=lam_map,
            use_projectors=args.use_projectors,
            top_k=args.top_k,
            corr_min=args.corr_min,
            anchor=args.anchor,
            window=args.window,
        )
    else:
        print("[Steering OFF] 运行基线推理。")

    # 推理评测
    results = []
    correct = 0
    parse_failed = 0
    per_class = defaultdict(lambda: {"tp":0,"total":0})
    bs = max(1, args.batch_size)

    def encode_batch(prompts: List[str]):
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
        # 重要：attention_mask 会被传给模型 → steerer 在 hook 内可见
        return enc

    def generate_texts(enc):
        enc = {k: v.to(pdev) for k, v in enc.items()}
        with torch.inference_mode():
            try:
                out = mdl.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                outs = []
                half = enc["input_ids"].shape[0]//2 or 1
                for part in (slice(0, half), slice(half, None)):
                    enc_sub = {k: v[part].to(pdev) for k, v in enc.items()}
                    outs.append(mdl.generate(**enc_sub, max_new_tokens=args.max_new_tokens, do_sample=False))
                out = torch.cat(outs, dim=0)
        texts = tok.batch_decode(out, skip_special_tokens=True)
        return texts

    samples_iter = list(batched(samples, bs))
    for batch in tqdm(samples_iter, ncols=100, desc="FOLIO eval"):
        prompts = []
        for s in batch:
            ptxt = build_prompt(s["prem"], s["concl"])
            if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
                msgs = [{"role":"system","content":"You are a helpful reasoning assistant."},
                        {"role":"user","content":ptxt}]
                chat_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                prompts.append(chat_text)
            else:
                prompts.append(ptxt)

        enc = encode_batch(prompts)
        texts = generate_texts(enc)

        for text, s in zip(texts, batch):
            pred = parse_truth(text)
            if pred is None:
                parse_failed += 1
                per_class[s["gold"]]["total"] += 1
                pred = "PARSE_FAILED"
            else:
                hit = int(pred == s["gold"])
                correct += hit
                per_class[s["gold"]]["total"] += 1
                if hit: per_class[s["gold"]]["tp"] += 1
            results.append({
                "gold": s["gold"], "pred": pred,
                "prem": s["prem"], "hypo": s["concl"],
                "gen": text
            })

    # 指标
    total = len(samples)
    parsed = total - parse_failed
    acc = correct / parsed if parsed else 0.0
    acc_with_failed = correct / total if total else 0.0
    print(f"\nTotal={total}  Parsed={parsed}  Parse_Failed={parse_failed} ({parse_failed/total*100:.1f}%)")
    print(f"Accuracy (on parsed samples)={acc*100:.2f}%")
    print(f"Accuracy (treat parse_failed as wrong): {acc_with_failed*100:.2f}%")
    for c in ["True","False","Unknown"]:
        if per_class[c]["total"]>0:
            pc = per_class[c]["tp"]/per_class[c]["total"]*100
            print(f"  {c:7s}: {pc:.2f}%  (n={per_class[c]['total']})")

    labels = ["True","False","Unknown"]
    all_preds = labels + ["PARSE_FAILED"]
    cm = {a:{b:0 for b in all_preds} for a in labels}
    for r in results: cm[r["gold"]][r["pred"]] += 1
    print("\nConfusion matrix (gold x pred):")
    print("         " + " ".join(f"{p:>12s}" for p in all_preds))
    for g in labels:
        print(f"  {g:7s}: " + " ".join(f"{cm[g][p]:>12d}" for p in all_preds))
    tp = {l:cm[l][l] for l in labels}
    fp = {l:sum(cm[g][l] for g in labels if g!=l) for l in labels}
    fn = {}
    for l in labels:
        fn[l] = sum(cm[l][p] for p in labels if p != l)
    prec = {l:(tp[l]/(tp[l]+(sum(cm[g][l] for g in labels if g!=l))) if (tp[l]+sum(cm[g][l] for g in labels if g!=l))>0 else 0.0) for l in labels}
    rec  = {l:(tp[l]/(tp[l]+fn[l]) if (tp[l]+fn[l])>0 else 0.0) for l in labels}
    f1   = {l:(2*prec[l]*rec[l]/(prec[l]+rec[l]) if (prec[l]+rec[l])>0 else 0.0) for l in labels}
    macro_f1 = sum(f1.values())/len(labels)
    print(f"Macro-F1 = {macro_f1*100:.2f}%")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved predictions to {args.output}")

    if steerer is not None:
        steerer.close()

if __name__ == "__main__":
    main()
