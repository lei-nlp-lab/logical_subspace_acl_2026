#!/usr/bin/env python
"""
step_validation_pipeline.py  (v1.6)
-----------------------------------
* Added a **progress bar** (tqdm) to show real‑time processing status for
  samples.  Installs with `pip install tqdm` if not already present.
"""

import os, re, json, asyncio
from collections import defaultdict, Counter
import aiohttp, backoff
from tqdm import tqdm  # progress bar

#########################
# Configuration         #
#########################
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.ai.it.ufl.edu").rstrip("/")
OPENAI_CHAT_COMPLETIONS_URL = f"{OPENAI_BASE_URL}/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
MODELS_ENV = os.getenv("OPENAI_EVAL_MODELS", "").strip()
MODELS = [m.strip() for m in MODELS_ENV.split(",") if m.strip()] if MODELS_ENV else ["gpt-4.1-mini", "gpt-4.1"]

STEP_RE = re.compile(r"^\s*Step\s*(\d+)\s*[:\.]", re.I | re.M)
REF_RE = re.compile(r"\b(?:fact\s*\d+|int\s*\d+|assump\s*\d+)\b", re.I)
# REFLINE_RE = re.compile(r"(fact\s*\d+|int\s*\d+|assump\s*\d+)\s*:\s*(.+)", re.I)
HYP_INLINE = re.compile(r"hypothesis[^:\n]*:\s*(.+)", re.I)
HYP_NEXT = re.compile(r"hypothesis[^:\n]*$", re.I)

FALL_FACT = re.compile(r"\b(fact\s*\d+|int\s*\d+|assump\s*\d+)\s*:\s*(.+)", re.I)
FALL_HYP = re.compile(r"\b(hypothesis)\s*:\s*(.+)", re.I)
# Reference "definition" fragments, possibly multiple per line separated by ';'
# Allows optional bullet prefix per fragment.
REFLINE_RE = re.compile(
    r"(?:^|;)\s*(?:[-*•]\s*)?(fact\s*\d+|int\s*\d+|assump\s*\d+)\b\s*:\s*(.+?)(?=$|;)",
    re.I,
)
STEP_PREFIX = re.compile(r"^\s*step\s+\d+\s*:\s*", re.I)
INLINE_REF_RE = re.compile(
    r"\b(fact\s*\d+|int\s*\d+|assump\s*\d+)\b\s*:\s*([^.;\n]+)",
    re.I
)

############################################
# GPT helper                               #
############################################

def _chat(session, msgs, model):
    payload = {"model": model, "messages": msgs, "temperature": 0.0}
    return session.post(OPENAI_CHAT_COMPLETIONS_URL, json=payload, headers=HEADERS)

@backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=5)
@backoff.on_exception(backoff.expo,
                            (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError),
                            max_tries=7,
                            factor=2)
async def ask_bool(session, msgs):
    """Return True/False parsed from GPT; raise RuntimeError if unable after retries.

    Enhancement:
    * Logs non‑200 responses & first 80 chars of error message.
    * Accept wider answer variants: "true.", "yes", "false," etc.
    * If OpenAI rate‑limits (429) schedules automatic retry via backoff.
    """
    for model in MODELS:
        try:
            async with _chat(session, msgs, model) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    print(f"[warn] {model} HTTP {resp.status}: {detail[:80]}")
                    raise RuntimeError("non‑200 response")
                data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip().lower()
                if content.startswith(("true", "yes", "1")):
                    return True
                if content.startswith(("false", "no", "0")):
                    return False
                # Unexpected body – treat as error to trigger retry/fallback
                print(f"[warn] {model} unexpected answer → '{content[:60]}'")
                raise RuntimeError("unparseable answer")
        except Exception as exc:
            print(f"[info] {model} failed ({exc}). trying next model…")
            continue
    return False

############################################
# Parsing helpers                          #
############################################

def _extract_hypothesis(text: str):
    if (m := HYP_INLINE.search(text)):
        return m.group(1).strip()
    lines = text.splitlines()
    for i, ln in enumerate(lines[:-1]):
        if HYP_NEXT.match(ln.strip()):
            nxt = lines[i + 1].strip()
            return nxt if nxt else None
    return None

def _parse_fallback(txt: str):
    txt = txt.strip()
    if txt.lower().startswith("facts:"):
        txt = txt[6:].lstrip()
    if txt.startswith(","):
        txt = txt[1:].lstrip()
    ref = {}
    for seg in re.split(r"[;\n\.]", txt):
        seg = seg.strip()
        if not seg:
            continue
        if (mf := FALL_FACT.match(seg)):
            key = mf.group(1).lower().replace(" ", "")
            ref[key] = mf.group(2).strip()
            continue
        if (mh := FALL_HYP.match(seg)):
            ref["hypothesis"] = mh.group(2).strip()
    return ref


def build_ref_dict(sample):
    txt = sample["responses"][0]["response"]
    ref = {}

    # Fast line-starting extraction (keep original logic)
    for ln in txt.splitlines():
        for m in REFLINE_RE.finditer(ln):
            key = m.group(1).lower().replace(" ", "")
            val = m.group(2).strip()
            if key not in ref and val:
                ref[key] = val

    # Extract hypothesis
    if (h := _extract_hypothesis(txt)):
        ref["hypothesis"] = h

    # Add facts if missing
    if not any(k.startswith("fact") for k in ref):
        ref.update(_parse_fallback(sample["problem"]["input"]))

    # If still missing int/assump, do global scan
    if not any(k.startswith(("int", "assump")) for k in ref):
        for m in INLINE_REF_RE.finditer(txt):
            key = m.group(1).lower().replace(" ", "")
            if key not in ref:          # Avoid overwriting line-start matches
                ref[key] = m.group(2).strip()

    return ref


INLINE_CONCL_RE = re.compile(r"\b(int\s*\d+|assump\s*\d+|hypothesis)\s*[:\.]", re.I)
SPLIT_DELIMS = [r";", r"\n", r"\."]      # semicolon → newline → period

def _split_by_delims(txt: str):
    """Try to split top-level segments using semicolon / newline / period in sequence."""
    for d in SPLIT_DELIMS:
        parts = [p.strip() for p in re.split(d, txt) if p.strip()]
        if len(parts) > 1:        # Only count as success if at least two segments
            return parts
    return [txt.strip()]

def split_steps(text: str):
    # First find formal Step markers
    idx = [m.start() for m in STEP_RE.finditer(text)]
    if idx:
        idx.append(len(text))
        steps = []
        for i, start in enumerate(idx[:-1]):
            block = text[start:idx[i + 1]].strip()
            n = int(STEP_RE.search(block).group(1))
            concl = None
            if (m := INLINE_CONCL_RE.search(block)):
                concl = m.group(1).lower().replace(" ", "")
            elif re.search(r"^\s*hypothesis\s*$", block, re.I | re.M):
                concl = "hypothesis"
            ante = {r.lower().replace(" ", "") for r in REF_RE.findall(block)}
            if concl:
                ante.discard(concl)
            steps.append({"n": n, "ante": ante, "concl": concl})
        return steps

    # No Step found → split sentence by sentence
    raw_parts = _split_by_delims(text)
    valid_parts = [p for p in raw_parts if INLINE_CONCL_RE.search(p)]

    if not valid_parts:
        # Fallback: treat whole text as one step
        ante = {r.lower().replace(" ", "") for r in REF_RE.findall(text)}
        return [{"n": 1, "ante": ante, "concl": None}]

    steps = []
    for i, block in enumerate(valid_parts, start=1):
        concl = INLINE_CONCL_RE.search(block).group(1).lower().replace(" ", "")
        ante = {r.lower().replace(" ", "") for r in REF_RE.findall(block)}
        ante.discard(concl)
        steps.append({"n": i, "ante": ante, "concl": concl})
    return steps


############################################
# Core evaluation                          #
############################################

def is_necessary(concl, later):
    if not concl:
        return True
    return concl in set().union(*(s["ante"] for s in later))

async def eval_step(session, step, ref, later):
    cid = step["concl"]
    if cid and cid.startswith("assump"):
        return {"skip": True}

    premises = [f"{r}: {ref.get(r, '???')}" for r in step["ante"]]
    concl_text = f"{cid}: {ref.get(cid, '???')}" if cid else "???"
    # print("premises:", premises)
    # print("conclusion:", concl_text)
    v_prompt = [{"role": "user", "content": "Premises:\n" + "\n".join(premises) + f"\nConclusion:\n{concl_text}\nDo the premises entail the conclusion? Answer true or false only."}]
    valid = await ask_bool(session, v_prompt)

    necessary = True if (cid and ref.get(cid, "").strip() == "⊥") else is_necessary(cid, later)

    atomic = False
    if valid:
        a_prompt = [{"role": "user", "content": "Premises:\n" + "\n".join(premises) + f"\nConclusion:\n{concl_text}\nIs this inference atomic? Answer true or false only."}]
        atomic = await ask_bool(session, a_prompt)

    return {"step": step["n"], "valid": valid, "necessary": necessary, "atomic": atomic, "skip": False}


async def analyse_sample(session, sample, sid):
    txt = sample["responses"][0]["response"]
    ground_truth_steps = sample["problem"]["original_data"]["steps"]
    ref = build_ref_dict(sample)
    steps = split_steps(txt)
    # print(ref)
    if not steps:
        return {"sample_id": sid,
                "error": "no_step_marker_found",
                "ground_truth_steps": ground_truth_steps,
                "raw_first_200": txt[:200]}
    results = {
        "sample_id": sid,
        "num_steps": len(steps),
        "ground_truth_steps": ground_truth_steps,
        "steps": []
    }
    for i, st in enumerate(steps):
        results["steps"].append(await eval_step(session, st, ref, steps[i + 1:]))
    return results

############################################
# Aggregation                              #
############################################

def aggregate(sample_results):
    """Return per‑num_steps averages, overall step‑weighted averages, **and**
    sample‑wise perfection stats (how many samples are 100% valid / necessary /
    atomic).
    """

    # -------- per‑num_steps bucket averages --------
    bucket = defaultdict(lambda: Counter(valid=0.0, necessary=0.0, atomic=0.0, samples=0))

    # -------- overall step‑weighted totals --------
    tot_true = Counter(valid=0, necessary=0, atomic=0, total=0)

    # -------- sample‑wise perfection counters --------
    sample_perfect = Counter(all_valid=0, all_necessary=0, all_atomic=0, all_three=0, total_samples=0)

    for samp in sample_results:
        if samp is None or samp.get("error"):
            continue
        non_skip = [st for st in samp["steps"] if not st.get("skip")]
        if not non_skip:
            continue
        sample_perfect["total_samples"] += 1

        # ----- sample‑wise perfection checks -----
        if all(st["valid"] for st in non_skip):
            sample_perfect["all_valid"] += 1
        if all(st["necessary"] for st in non_skip):
            sample_perfect["all_necessary"] += 1
        if all(st["atomic"] for st in non_skip):
            sample_perfect["all_atomic"] += 1
        if (all(st["valid"] for st in non_skip)
                and all(st["necessary"] for st in non_skip)
                and all(st["atomic"] for st in non_skip)):
            sample_perfect["all_three"] += 1

        # ----- per‑sample rates for bucket avg -----
        n_steps = len(non_skip)
        valid_rate = sum(st["valid"] for st in non_skip) / n_steps
        nec_rate = sum(st["necessary"] for st in non_skip) / n_steps
        atom_rate = sum(st["atomic"] for st in non_skip) / n_steps

        b = bucket[str(samp["ground_truth_steps"])]
        b["valid"] += valid_rate
        b["necessary"] += nec_rate
        b["atomic"] += atom_rate
        b["samples"] += 1

        # accumulate overall counts
        tot_true["valid"] += sum(st["valid"] for st in non_skip)
        tot_true["necessary"] += sum(st["necessary"] for st in non_skip)
        tot_true["atomic"] += sum(st["atomic"] for st in non_skip)
        tot_true["total"] += n_steps

    result = {
        str(k): {
            "valid": round(v["valid"] / v["samples"], 3),
            "necessary": round(v["necessary"] / v["samples"], 3),
            "atomic": round(v["atomic"] / v["samples"], 3)
        }
        for k, v in bucket.items() if v["samples"] > 0
    }

    if tot_true["total"]:
        result["overall_steps"] = {
            "valid": round(tot_true["valid"] / tot_true["total"], 3),
            "necessary": round(tot_true["necessary"] / tot_true["total"], 3),
            "atomic": round(tot_true["atomic"] / tot_true["total"], 3)
        }

    # sample perfection stats
    ts = sample_perfect["total_samples"] or 1  # avoid zero division
    result["overall_samples"] = {
        "total_samples": ts,
        "all_valid": {"count": sample_perfect["all_valid"], "ratio": round(sample_perfect["all_valid"] / ts, 3)},
        "all_necessary": {"count": sample_perfect["all_necessary"], "ratio": round(sample_perfect["all_necessary"] / ts, 3)},
        "all_atomic": {"count": sample_perfect["all_atomic"], "ratio": round(sample_perfect["all_atomic"] / ts, 3)},
        "all_three": {"count": sample_perfect["all_three"], "ratio": round(sample_perfect["all_three"] / ts, 3)}
    }

    return result

############################################
# Pipeline runner with progress bar        #
############################################

async def run_pipeline(inp, det, summ, concurrency=50):
    with open(inp, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    connector = aiohttp.TCPConnector(limit=concurrency)
    results = [None] * len(dataset)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def _run_one(sid, sample):
            try:
                return await analyse_sample(session, sample, sid)
            except Exception as exc:
                return {"sample_id": sid, "error": str(exc)[:120]}

        tasks = [_run_one(sid, sample) for sid, sample in enumerate(dataset)]
        pbar = tqdm(total=len(tasks), desc="Processing samples", unit="sample")
        for coro in asyncio.as_completed(tasks):
            res = await coro
            sid = res.get("sample_id")
            if isinstance(sid, int) and 0 <= sid < len(results):
                results[sid] = res
            else:
                results.append(res)
            pbar.update(1)
        pbar.close()

    with open(det, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(summ, "w", encoding="utf-8") as f:
        json.dump(aggregate(results), f, indent=2)

############################################
# CLI                                      #
############################################

if __name__ == "__main__":
    import argparse, pathlib, sys

    p = argparse.ArgumentParser()
    p.add_argument("--input", default="../results/Dataset1-FLD_results_qwen-7B_full_nl,4_nl_2epoch.json")
    p.add_argument("--output_detail", default="../results/detailed_FLD_mini.json")
    p.add_argument("--output_summary", default="../results/summary_FLD_mini.json")
    p.add_argument("--concurrency", type=int, default=50)
    a = p.parse_args()
    if not pathlib.Path(a.input).exists():
        sys.exit(f"File {a.input} not found")
    asyncio.run(run_pipeline(a.input, a.output_detail, a.output_summary, a.concurrency))
