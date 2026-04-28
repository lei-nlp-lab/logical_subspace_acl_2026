#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)

def norm_str(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return "\n".join(str(t) for t in x)
    return str(x)

def norm_label(x):
    # 统一到 True / False / Uncertain（把 Unknown 归到 Uncertain）
    if x is None:
        return None
    t = str(x).strip().lower()
    if t in {"true","t"}:   return "True"
    if t in {"false","f"}:  return "False"
    if t in {"uncertain","unknown","u"}: return "Uncertain"
    return str(x)

def make_text_nl(prem_nl, hyp_nl, label):
    return (
        "Premises:\n"
        f"{prem_nl}\n\n"
        "Hypothesis:\n"
        f"{hyp_nl}\n\n"
        f"The Hypothesis is {label}."
    )

def make_text_fol(prem_fol, hyp_fol, label):
    return (
        "Premises (FOL):\n"
        f"{prem_fol}\n\n"
        "Hypothesis (FOL):\n"
        f"{hyp_fol}\n\n"
        f"The Hypothesis is {label}."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="原始 JSONL 路径")
    ap.add_argument("--output", required=True, help="输出 JSONL 路径（每条一行，含 NL/FOL 的 pair）")
    args = ap.parse_args()

    wrote = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for rec in read_jsonl(args.input):
            story_id  = rec.get("story_id")
            label_raw = rec.get("label") or rec.get("Truth Values") or rec.get("truth") or rec.get("gold_label")
            label     = norm_label(label_raw)
            if label is None:
                continue

            prem_nl  = norm_str(rec.get("premises")      or rec.get("Premises - NL")      or rec.get("premises_nl"))
            hyp_nl   = norm_str(rec.get("conclusion")     or rec.get("Conclusions - NL")   or rec.get("conclusion_nl"))
            prem_fol = norm_str(rec.get("premises-FOL")   or rec.get("Premises - FOL")     or rec.get("premises_fol"))
            hyp_fol  = norm_str(rec.get("conclusion-FOL") or rec.get("Conclusions - FOL")  or rec.get("conclusion_fol"))

            # 两个视角都要有才打包成 pair
            if not (prem_nl and hyp_nl and prem_fol and hyp_fol):
                continue

            nl_text  = make_text_nl(prem_nl, hyp_nl, label)
            fol_text = make_text_fol(prem_fol, hyp_fol, label)

            out = {
                "story_id": story_id,
                "label": label,
                "pair": [
                    {"view": "NL",  "text": nl_text},
                    {"view": "FOL", "text": fol_text}
                ]
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"Done. Wrote {wrote} paired lines to {args.output}")

if __name__ == "__main__":
    main()
