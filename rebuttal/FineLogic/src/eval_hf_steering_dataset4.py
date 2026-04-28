#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FineLogic-style evaluation on Dataset4-ProntoQA using HF generation + steering.

Key goals:
1) Keep prompt style aligned with FineLogic (direct/cot/fewshot).
2) Inject normalized steering from ProntoQA subspace.
3) Save FineLogic-compatible result/evaluation JSON files for each (layer, lambda, style).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Steering helpers from existing ProntoQA pipeline
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRONTOQA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "prontoqa"))
if PRONTOQA_DIR not in sys.path:
    sys.path.append(PRONTOQA_DIR)

from steering_infer_normalized import HFSteererNormalized as HFSteerer
from steering_infer import find_decoder_layers, primary_device


PARSER_FORMAT_RULES = (
    "Strict output format (mandatory):\n"
    "1) Start each reasoning line with 'Step k:' where k is 1,2,3...\n"
    "2) Each step must include an explicit conclusion id with colon in this pattern:\n"
    "   Step k: <refs joined by &> -> <conclusion_id>: <statement>\n"
    "3) Allowed conclusion ids are only: int1,int2,... or assump1,assump2,... or hypothesis.\n"
    "4) The final reasoning step must conclude with 'hypothesis:' (with colon), not plain 'hypothesis'.\n"
    "5) Do not write free-form paragraphs, bullet lists, or notes outside Step lines.\n"
    "6) After all Step lines, print exactly one final line: __PROVED__ or __DISPROVED__.\n"
    "Example:\n"
    "Step 1: fact18 -> int1: Max is a yumpus.\n"
    "Step 2: int1 & fact4 -> int2: Max is a dumpus.\n"
    "Step 3: int2 -> hypothesis: Max is sour.\n"
    "__PROVED__"
)


PROMPT_STYLES = {
    "direct": {
        "system": "Based on the provided facts, verify the hypothesis. Conclude with one of the markers: \"__PROVED__\" for true, or \"__DISPROVED__\" for false.",
        "suffix": "",
    },
    "cot": {
        "system": "Based on the provided facts, verify the hypothesis. Conclude with one of the markers: \"__PROVED__\" for proven, or \"__DISPROVED__\" for disproven.",
        "suffix": (
            "Let's think step by step.\n"
            + PARSER_FORMAT_RULES
        ),
    },
    "fewshot": {
        "system": "Based on the provided facts, verify the hypothesis. Conclude with one of the markers: \"__PROVED__\" for proven, or \"__DISPROVED__\" for disproven.",
        "suffix": "{fewshot_examples}",
    },
}

FEWSHOT_EXAMPLES = {
    "4": """Here are some examples of proofs for your reference:
[Start of example]
For example, for this question:
"facts": "fact1: This vascularization is unshaven. fact2: If this parapsychologist is not a past this vascularization shrinks Belarusian. fact3: Something is not choric and is not unshaven if the thing is not a Hugueninia. fact4: Something is not a Hugueninia if that thing shrinks Belarusian.",
"hypothesis": "This vascularization is a Hugueninia.",
We can conclude that
"fact3 -> int1: This vascularization neither is choric nor is unshaven if it is not a Hugueninia.; void -> assump1: Let's assume that this vascularization is not a kind of a Hugueninia.; int1 & assump1 -> int2: This vascularization is not choric and it is not unshaven.; int2 -> int3: This vascularization is not unshaven.; int3 & fact1 -> int4: This is contradiction.; [assump1] & int4 -> hypothesis;"
So the answer is
__PROVED__

There is also this question:
"facts": "fact1: Somebody is a Lophophorus if that it does trace excitableness or it does not Lot abridgement or both does not hold. fact2: There is nothing that traces excitableness and/or does not Lot abridgement. fact3: If the fact that something is vestiary and/or it is bumpy does not stand it inundates Vestris. fact4: Angeles is not non-Pasteurian. fact5: Angeles does Lot abridgement. fact6: If that somebody mists Ferber or this one does not disengage or both is wrong then that this one is a sabbatical is valid. fact7: If somebody is a Lophophorus that either it is a Gelechia or it does not chock impishness or both stands does not stand.",
"hypothesis": "Angeles is not a Lophophorus.",
We can prove it by
"fact2 -> int1: That Angeles traces excitableness or it does not Lot abridgement or both is invalid.; fact1 -> int2: Angeles is a Lophophorus if that Angeles does trace excitableness or it does not Lot abridgement or both does not stand.; int1 & int2 -> hypothesis;"
So the answer is
__DISPROVED__
[End of example]
You can refer to the proof method of the above question, think step by step, and give the result of this question.
"""
}

MARKERS = [
    "__PROVED__",
    "__DISPROVED__",
    "__UNKNOWN__",
    "__YES__",
    "__NO__",
    "DISPROVED",
    "PROVED",
    "UNKNOWN",
    "YES",
    "NO",
]


def model_result_filename(model_name: str, layer: int, lam: float, style: str, out_dir: str) -> str:
    safe = model_name.replace("/", "_")
    return os.path.join(out_dir, f"Dataset4-ProntoQA_results_{safe}_L{layer}_lam{lam:.3f}_{style}.json")


def model_eval_filename(model_name: str, layer: int, lam: float, style: str, out_dir: str) -> str:
    safe = model_name.replace("/", "_")
    return os.path.join(out_dir, f"Dataset4-ProntoQA_evaluation_{safe}_L{layer}_lam{lam:.3f}_{style}.json")


def evaluate_model_responses(results: List[Dict[str, Any]], model_name: str, style_name: str):
    evaluation = {"model": model_name, "styles": {style_name: {"total": 0, "correct": 0}}, "overall": {"total": 0, "correct": 0}}

    for pr in results:
        plabel = pr["problem"].get("proof_label", "")
        if plabel and not (plabel.startswith("__") and plabel.endswith("__")):
            expected = f"__{plabel}__"
        else:
            expected = plabel

        for resp in pr["responses"]:
            if not resp.get("success", False) or resp.get("model") != model_name:
                continue
            text = resp.get("response", "")
            found = "__UNKNOWN__"
            for mk in MARKERS:
                if mk in text:
                    found = mk if mk.startswith("__") else f"__{mk}__"
                    break

            evaluation["styles"][style_name]["total"] += 1
            evaluation["overall"]["total"] += 1

            if found == expected or (expected == "__YES__" and found == "__PROVED__") or (expected == "__NO__" and found == "__DISPROVED__"):
                evaluation["styles"][style_name]["correct"] += 1
                evaluation["overall"]["correct"] += 1

    st = evaluation["styles"][style_name]
    if st["total"]:
        st["accuracy"] = st["correct"] / st["total"]
    if evaluation["overall"]["total"]:
        evaluation["overall"]["accuracy"] = evaluation["overall"]["correct"] / evaluation["overall"]["total"]
    return evaluation


def process_problem_prompt(problem_data: Dict[str, Any], style_name: str) -> str:
    cfg = PROMPT_STYLES[style_name]
    in_data = problem_data.get("input", "")
    suffix = cfg["suffix"]
    if style_name == "fewshot":
        suffix = suffix.format(fewshot_examples=FEWSHOT_EXAMPLES["4"]) + "\n\n" + PARSER_FORMAT_RULES
    if in_data:
        return in_data + ("\n" + suffix if suffix else "")
    # Fallback path (Dataset4 should always have input)
    facts = problem_data.get("facts") or problem_data.get("original_data", {}).get("facts", "")
    hypo = problem_data.get("hypothesis") or problem_data.get("original_data", {}).get("hypothesis", "")
    return f"Facts: {facts}\nHypothesis: {hypo}\n{suffix}"


def normalize_problem_for_finelogic(problem_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure fields used by FineLogic eval_step exist:
      problem.input
      problem.proof_label
      problem.original_data.steps
    """
    p = dict(problem_data)
    if "input" not in p:
        p["input"] = str(problem_data.get("input", ""))
    if "proof_label" not in p:
        p["proof_label"] = str(problem_data.get("proof_label", ""))

    od = p.get("original_data")
    if not isinstance(od, dict):
        od = {}

    if "steps" not in od:
        expl = p.get("explanation", None)
        if isinstance(expl, list):
            od["steps"] = len(expl)
        elif isinstance(expl, str):
            lines = [ln for ln in expl.splitlines() if ln.strip()]
            od["steps"] = len(lines) if lines else -1
        else:
            od["steps"] = -1
    p["original_data"] = od
    return p


def apply_chat(tok, system_prompt: str, user_prompt: str, enable_thinking: bool) -> str:
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        chat_kwargs = dict(add_generation_prompt=True, tokenize=False)
        chat_kwargs["enable_thinking"] = bool(enable_thinking)
        try:
            return tok.apply_chat_template(msgs, **chat_kwargs)
        except TypeError:
            chat_kwargs.pop("enable_thinking", None)
            return tok.apply_chat_template(msgs, **chat_kwargs)
    return f"{system_prompt}\n\n{user_prompt}"


def batched(it, bs):
    for i in range(0, len(it), bs):
        yield it[i : i + bs]


def generate_with_style(
    mdl,
    tok,
    device,
    data: List[Dict[str, Any]],
    model_name: str,
    style_name: str,
    max_new_tokens: int,
    batch_size: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> List[Dict[str, Any]]:
    system_prompt = PROMPT_STYLES[style_name]["system"]
    structured = []
    for p in data:
        norm_p = normalize_problem_for_finelogic(p)
        structured.append({"problem": norm_p, "prompts": {style_name: process_problem_prompt(norm_p, style_name)}, "responses": []})

    batch_iter = batched(list(range(len(structured))), max(1, batch_size))
    total_batches = (len(structured) + max(1, batch_size) - 1) // max(1, batch_size)
    for batch_ids in tqdm(batch_iter, total=total_batches, desc=f"Samples ({style_name})", leave=False):
        chat_prompts = []
        for idx in batch_ids:
            up = structured[idx]["prompts"][style_name]
            chat_prompts.append(apply_chat(tok, system_prompt, up, enable_thinking))

        enc = tok(chat_prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        input_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.inference_mode():
            outputs = mdl.generate(**enc, **gen_kwargs)
        texts = []
        for row_i in range(outputs.shape[0]):
            start = int(input_lens[row_i])
            gen_ids = outputs[row_i, start:]
            texts.append(tok.decode(gen_ids, skip_special_tokens=True))

        for local_i, idx in enumerate(batch_ids):
            structured[idx]["responses"].append(
                {
                    "model": model_name,
                    "prompt_style": style_name,
                    "response": texts[local_i],
                    "success": True,
                    "idx": idx,
                }
            )
    return structured


def maybe_run_step_eval(input_json: str, out_detail: str, out_summary: str, concurrency: int):
    # Lazy import to avoid OPENAI_API_KEY requirement unless explicitly requested.
    from eval_step import run_pipeline
    import asyncio

    asyncio.run(run_pipeline(input_json, out_detail, out_summary, concurrency=concurrency))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model name label in outputs")
    ap.add_argument("--model_path", required=True, help="HF model path or id")
    ap.add_argument("--dataset4_path", default=os.path.join(SCRIPT_DIR, "..", "data", "eval", "Dataset4-ProntoQA.json"))
    ap.add_argument("--svcca_pt", required=True)
    ap.add_argument("--styles", default="cot", help="Comma-separated styles from {direct,cot,fewshot}")
    ap.add_argument("--layer_start", type=int, required=True)
    ap.add_argument("--layer_end", type=int, required=True)
    ap.add_argument("--lam_start", type=float, default=0.0)
    ap.add_argument("--lam_end", type=float, default=0.12)
    ap.add_argument("--lam_step", type=float, default=0.02)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    ap.add_argument("--use_projectors", action="store_true")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--corr_min", type=float, default=0.0)
    ap.add_argument("--anchor", choices=["last", "all"], default="all")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--out_dir", default=os.path.join(SCRIPT_DIR, "..", "results", "steering_dataset4"))
    ap.add_argument("--summary_csv", default=os.path.join(SCRIPT_DIR, "..", "results", "steering_dataset4", "Dataset4-ProntoQA_steering_summary.csv"))
    ap.add_argument("--run_step_eval", action="store_true")
    ap.add_argument("--step_concurrency", type=int, default=50)
    args = ap.parse_args()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    for s in styles:
        if s not in PROMPT_STYLES:
            raise ValueError(f"Unsupported style: {s}. choose from {list(PROMPT_STYLES.keys())}")

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open(args.dataset4_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.max_samples and args.max_samples < len(data):
        data = data[: args.max_samples]
    print(f"Loaded Dataset4 samples: {len(data)} from {args.dataset4_path}")

    load_kwargs = {}
    if args.use_4bit:
        load_kwargs["load_in_4bit"] = True
    if args.use_8bit:
        load_kwargs["load_in_8bit"] = True

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **load_kwargs,
        )
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **load_kwargs,
        )

    mdl.eval()
    for module in mdl.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    mdl.generation_config.pad_token_id = tok.pad_token_id
    mdl.generation_config.eos_token_id = tok.eos_token_id
    mdl.generation_config.use_cache = True

    device = primary_device(mdl)
    print("Primary device:", device)

    layer_list, _ = find_decoder_layers(mdl)
    total_layers = len(layer_list)
    if not (0 <= args.layer_start < total_layers and 0 <= args.layer_end < total_layers and args.layer_start <= args.layer_end):
        raise ValueError(f"Layer range invalid. model has {total_layers} layers")
    layers_to_scan = list(range(args.layer_start, args.layer_end + 1))

    lam_vals = []
    lam = args.lam_start
    while lam <= args.lam_end + 1e-9:
        lam_vals.append(round(lam, 6))
        lam += args.lam_step
    lam_vals = sorted(set(lam_vals))

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
    first_write = not os.path.exists(args.summary_csv)
    csvf = open(args.summary_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(csvf)
    if first_write:
        w.writerow(
            [
                "model",
                "model_path",
                "style",
                "layer",
                "lambda",
                "total",
                "correct",
                "accuracy",
                "dataset",
                "svcca_pt",
                "use_projectors",
                "top_k",
                "corr_min",
                "anchor",
                "window",
                "do_sample",
                "temperature",
                "top_p",
                "result_json",
                "eval_json",
            ]
        )
        csvf.flush()

    for layer in tqdm(layers_to_scan, desc="Layers"):
        for lam in tqdm(lam_vals, desc=f"Layer {layer} lambda", leave=False):
            steerer = None
            if lam != 0:
                steerer = HFSteerer(
                    model=mdl,
                    svcca_pt=args.svcca_pt,
                    layers=[layer],
                    lambdas={layer: float(lam)},
                    use_projectors=args.use_projectors,
                    top_k=args.top_k,
                    corr_min=args.corr_min,
                    anchor=args.anchor,
                    window=args.window,
                )
            try:
                for style in styles:
                    t0 = time.time()
                    results = generate_with_style(
                        mdl=mdl,
                        tok=tok,
                        device=device,
                        data=data,
                        model_name=args.model,
                        style_name=style,
                        max_new_tokens=args.max_new_tokens,
                        batch_size=args.batch_size,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        enable_thinking=args.enable_thinking,
                    )
                    eval_res = evaluate_model_responses(results, args.model, style)
                    elapsed = time.time() - t0

                    result_fp = model_result_filename(args.model, layer, lam, style, args.out_dir)
                    eval_fp = model_eval_filename(args.model, layer, lam, style, args.out_dir)
                    with open(result_fp, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    with open(eval_fp, "w", encoding="utf-8") as f:
                        json.dump(eval_res, f, ensure_ascii=False, indent=2)

                    total = eval_res["overall"]["total"]
                    correct = eval_res["overall"]["correct"]
                    acc = eval_res["overall"].get("accuracy", 0.0)
                    w.writerow(
                        [
                            args.model,
                            args.model_path,
                            style,
                            layer,
                            lam,
                            total,
                            correct,
                            acc,
                            os.path.abspath(args.dataset4_path),
                            os.path.abspath(args.svcca_pt),
                            int(args.use_projectors),
                            int(args.top_k),
                            float(args.corr_min),
                            args.anchor,
                            int(args.window),
                            int(args.do_sample),
                            float(args.temperature),
                            float(args.top_p),
                            result_fp,
                            eval_fp,
                        ]
                    )
                    csvf.flush()
                    print(f"[DONE] style={style} layer={layer} lam={lam:.3f} acc={acc:.4f} time={elapsed:.1f}s")

                    if args.run_step_eval and style in {"cot", "fewshot"}:
                        detail_fp = os.path.join(args.out_dir, f"Dataset4-ProntoQA_step_detail_{args.model.replace('/', '_')}_L{layer}_lam{lam:.3f}.json")
                        summary_fp = os.path.join(args.out_dir, f"Dataset4-ProntoQA_step_summary_{args.model.replace('/', '_')}_L{layer}_lam{lam:.3f}.json")
                        maybe_run_step_eval(result_fp, detail_fp, summary_fp, args.step_concurrency)
                        print(f"[STEP] wrote {summary_fp}")
            finally:
                if steerer is not None:
                    steerer.close()

    csvf.close()
    print(f"Summary CSV saved to: {args.summary_csv}")


if __name__ == "__main__":
    main()
