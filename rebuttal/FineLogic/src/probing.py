#!/usr/bin/env python
"""
Auto‑Probing Pipeline (vLLM 0.8.5, zero‑CLI, *embed* API)
=========================================================
One‑liner example
-----------------
```python
import probing_pipeline as pp
pp.run_probing(
    model_path="Qwen/Qwen1.5-1.8B",
    model_name="qwen1p5",
    tp         = 4           # tensor‑parallel shards (GPUs)
)
```
Outputs
-------
* **summary**   → `../results/probing_results/summary.json`
  ```json
  { "qwen1p5": {"A":0.81,"B":0.76,"C":0.69,"D":2.13}, ... }
  ```
* **task detail** → `../results/probing_results/probing_task_A_test.jsonl`, …
  Each line has `"results": {model_name: prediction, ...}` (merged across models).

Dependencies: `vllm==0.8.5`, `torch`, `numpy`, `scikit-learn`, `tqdm`
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from typing import List, Dict, Any, DefaultDict
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, mean_absolute_error

from vllm import LLM
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegressionCV, RidgeCV


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ---------------------------------------------------------------------
# Fixed paths / patterns
# ---------------------------------------------------------------------
DATA_DIR = Path("../data/probing")
EMB_DIR = Path("../results/emb_path")
DETAIL_DIR = Path("../results/probing_results")
SAMPLE_FP = DETAIL_DIR / "samplewise_details.json"
SUMMARY_FP = DETAIL_DIR / "summary.json"
for p in (EMB_DIR, DETAIL_DIR):
    p.mkdir(parents=True, exist_ok=True)

TASKS = ["A", "B", "C"]
TRAIN_FP = lambda t: DATA_DIR / f"probing_{t}_train.jsonl"
TEST_FP = lambda t: DATA_DIR / f"probing_{t}_test.jsonl"
EMB_FP = lambda m, t, s: EMB_DIR / f"{m}_task_{t}_{s}.npz"
DETAIL_FP = lambda t: DETAIL_DIR / f"probing_task_{t}_test.jsonl"


# ---------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------

def load_jsonl(p: Path):
    return [json.loads(l) for l in p.open()]


def dump_jsonl(rows, p: Path):
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Text building (chat template)
# ---------------------------------------------------------------------

def build_texts(rows, tok):
    texts = []
    for r in rows:
        chat = [
            {"role": "system", "content": r["sys_prompt"]},
            {"role": "user", "content": r["user_input"]},
            {"role": "assistant", "content": r["cut_output"]},
        ]
        texts.append(
            tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        )
    return texts


# ---------------------------------------------------------------------
# Embedding via vLLM.embed
# ---------------------------------------------------------------------

def embed_split(model_key: str, llm: LLM, tok, task: str, split: str):
    npz_path = EMB_FP(model_key, task, split)
    if npz_path.exists():
        print(f"[embed] cached {npz_path.name}")
        return

    rows = load_jsonl(TRAIN_FP(task)) if split == "train" else load_jsonl(TEST_FP(task))
    labels = np.asarray([r["label"] for r in rows], dtype=np.float32)
    texts = build_texts(rows, tok)

    outputs = llm.embed(texts, use_tqdm=True)  # Enable tqdm to show progress

    # Extract embedding vectors
    vecs = [o.outputs.embedding for o in outputs]

    np.savez(npz_path, X=np.asarray(vecs, dtype=np.float32), y=labels)
    print(f"[embed] saved {npz_path.relative_to(EMB_DIR.parent)}")


# ---------------------------------------------------------------------
# Train & evaluate probe
# ---------------------------------------------------------------------

def train_probe(train_npz: Path, test_npz: Path, task: str):
    data_tr, data_te = np.load(train_npz), np.load(test_npz)
    X_tr, y_tr = data_tr["X"], data_tr["y"]
    X_te, y_te = data_te["X"], data_te["y"]

    if task == 'clf':
        model = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(cv=5, max_iter=1000, n_jobs=-1, scoring='accuracy')
        )
    else:  # regression (Task D)
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-6, 3, 13)))

    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    metric = accuracy_score(y_te, preds.round()) if task == 'clf' else mean_absolute_error(y_te, preds)
    return preds.tolist(), float(metric)


# ---------------------------------------------------------------------
# Detail & summary updates
# ---------------------------------------------------------------------

def update_detail(task: str, preds, model_key: str):
    p = DETAIL_FP(task)
    rows = load_jsonl(p) if p.exists() else load_jsonl(TEST_FP(task))
    for r, pred in zip(rows, preds):
        r.setdefault("results", {})[model_key] = pred
    dump_jsonl(rows, p)
    print(f"[detail] updated {p.relative_to(DETAIL_DIR.parent)}")


def update_summary(model_key: str, summary_add: Dict[str, Any]):
    summary: Dict[str, Any] = {}
    if SUMMARY_FP.exists():
        summary = json.load(SUMMARY_FP.open())
    summary[model_key] = summary_add
    json.dump(summary, SUMMARY_FP.open("w"), indent=2)
    print("[summary] updated", SUMMARY_FP.relative_to(SUMMARY_FP.parent.parent))


# ---------------------------------------------------------------------
# Per‑problem diagnostics collector
# ---------------------------------------------------------------------

def collect_problem_stats(task: str, rows, preds, stats_dict):
    by_pid = defaultdict(list)
    for r, p in zip(rows, preds):
        by_pid[r["problem_id"]].append((r, p))

    for pid, lst in by_pid.items():
        if pid not in stats_dict:
            stats_dict[pid] = {}
        if task == 'A':
            lst.sort(key=lambda x: x[0]["cur_step"])
            n_steps = lst[0][0]["n_steps"]
            correct = [(pr == row["label"]) for row, pr in [(x[0], x[1]) for x in lst]]
            # suffix all true
            span = 0
            for i, (row_ok) in enumerate(correct):
                if all(correct[i:]):
                    span = n_steps - lst[i][0]["cur_step"]
                    break
            stats_dict[pid]["task A span"] = span
        elif task == 'B':
            stats_dict[pid]["task B all_correct"] = int(all(pr == row["label"] for row, pr in lst))
        elif task == 'C':
            stats_dict[pid]["task C all_correct"] = int(all(pr == row["label"] for row, pr in lst))
        elif task == 'D':
            lst.sort(key=lambda x: x[0]["cur_step"])
            n_steps = lst[0][0]["n_steps"]
            correct = [(round(pr) == row["label"]) for row, pr in [(x[0], x[1]) for x in lst]]
            span = 0
            for i in range(len(correct)):
                if all(correct[i:]):
                    span = n_steps - lst[i][0]["cur_step"]
                    break
            stats_dict[pid]["task D span"] = span


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run_probing(model_path: str, model_name: str = "model", tp: int = 4,
                dtype: str = "bfloat16", max_len: int = 4096):
    """Embed all tasks, train probes, compute global & per‑distance metrics."""

    model_key = model_name
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.padding_side, tok.truncation_side = "left", "right"

    llm = LLM(model=model_path,
              task="embed",
              tensor_parallel_size=tp,
              dtype=dtype,
              max_model_len=max_len,
              gpu_memory_utilization=0.97,
              max_num_seqs=1024,
              max_num_batched_tokens=1048576,
              enforce_eager=True)

    summary_entry = {}
    problem_stats: Dict[int, Dict[str, Any]] = {}

    for task in TASKS:
        print(f"=== Task {task} ({model_key}) ===")
        for split in ("train", "test"):
            embed_split(model_key, llm, tok, task, split)

        tr_npz = EMB_FP(model_key, task, "train")
        te_npz = EMB_FP(model_key, task, "test")
        task_type = 'reg' if task == 'D' else 'clf'

        preds, metric = train_probe(tr_npz, te_npz, task_type)
        summary_entry[task]=metric
        print(f"[task {task}] metric={metric:.4f}")
        update_detail(task, preds, model_key)

        rows = load_jsonl(TEST_FP(task))
        collect_problem_stats(task, rows, preds, problem_stats)

        # --- per‑distance stats for A & D ---
        if task in ("A", "D"):
            rows = load_jsonl(TEST_FP(task))
            bucket: DefaultDict[int, List[float]] = defaultdict(list)
            for r, pr in zip(rows, preds):
                dist = r["n_steps"] - r["cur_step"]
                if task == "A":
                    bucket[dist].append(1.0 if pr == r["label"] else 0.0)
                else:  # task D -> abs error
                    bucket[dist].append(abs(pr - r["label"]))
            agg = {str(k): (np.mean(v) if task == "A" else np.mean(v)) for k, v in bucket.items()}
            summary_entry[f"task {task}"] = agg
    # write samplewise details


    detail_rows = [{"problem_id": pid, **vals} for pid, vals in problem_stats.items()]
    existing = json.load(SAMPLE_FP.open()) if SAMPLE_FP.exists() else {}
    existing[model_key] = detail_rows
    json.dump(existing, SAMPLE_FP.open("w"), indent=2, ensure_ascii=False)

    # global averages of four extra metrics
    A_span = np.mean([v.get("task A span", 0) for v in problem_stats.values() if "task A span" in v])
    B_all = np.mean([v.get("task B all_correct", 0) for v in problem_stats.values() if "task B all_correct" in v])
    C_all = np.mean([v.get("task C all_correct", 0) for v in problem_stats.values() if "task C all_correct" in v])
    # D_span = np.mean([v.get("task D span", 0) for v in problem_stats.values() if "task D span" in v])
    summary_entry.update({"A_span": A_span, "B_all_correct": B_all, "C_all_correct": C_all})
    update_summary(model_key, summary_entry)
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

# ---------------------------------------------------------------------
# Optional CLI wrapper (for quick testing)
# ---------------------------------------------------------------------

def main():
    mp = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
    mn = sys.argv[2] if len(sys.argv) > 2 else "qwen7b"
    tp = int(os.environ.get("TP", "4"))
    run_probing(mp, model_name=mn, tp=tp)


if __name__ == "__main__":
    main()
