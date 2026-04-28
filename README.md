# Logical Subspace Steering (LSS)

Code for the paper:

> **Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views**
> Feihao Fang, My T. Thai, Yuanyuan Lei. 2026.

## TL;DR

LSS aligns the residual-stream activations of an LLM on **natural-language (NL)** and **symbolic** views of the same logical problem with **SVCCA** to recover a shared *logical subspace*, then **steers** the model's activations along that subspace at inference time. No fine-tuning is required. The repository covers four datasets (ProntoQA, ProofWriter CWA/OWA, FOLIO, LogiQA), associated baselines, analysis tooling, and rebuttal experiments (HellaSwag, ReClor, FineLogic).

## Repository layout

```
.
├── LICENSE                          # MIT
├── README.md
├── requirements.txt                 # core deps
├── requirements-optional.txt        # rebuttal / vLLM / OpenAI-API deps
├── docs/
│   └── FILE_MANIFEST.md             # one-line description of every script
├── examples/slurm/                  # 7 sanitized SLURM templates
├── prontoqa/
│   ├── data/                        # paired-view ProntoQA splits
│   ├── data_processing/             # raw → paired views
│   ├── processing/                  # cca + residual extraction
│   ├── evaluation/                  # sweep + baselines
│   └── analysis/                    # paper analysis scripts
├── proofwriter/
│   ├── data/                        # CWA/OWA eval + paired SVCCA training set
│   ├── data_processing/             # NL → symbolic + paired views
│   ├── processing/                  # cca + residual extraction
│   └── evaluation/                  # CWA + OWA variants of every entry point
├── folio/
│   ├── data/                        # raw + paired FOLIO splits
│   ├── data_processing/             # process_json.py
│   ├── processing/                  # cca + residual extraction
│   ├── evaluation/                  # sweep + baselines
│   └── analysis/                    # FOLIO-specific analysis
├── generalization/
│   ├── data/                        # LogiQA / LogiQA-NLI eval samples + paired val
│   ├── evaluation/                  # apply ProntoQA SVCCA to LogiQA
│   ├── sample_logiqa.py
│   └── process_for_cca_proofs_only.py
└── rebuttal/                        # HellaSwag side-effect, ReClor transfer, FineLogic
    ├── FineLogic/                   # vendored — see its README/LICENSE
    ├── hellaswag/                   # vendored — see its README/LICENSE
    └── recolor/                     # ReClor preprocessing + sweep
```

## Installation

**Requirements**: Python ≥ 3.10 (tested on 3.11), a CUDA-capable GPU (≥ 24 GB recommended for the Llama-3.1-8B sweeps; 4-bit quantization via `--use_4bit` brings the floor to ~10 GB), and a working CUDA toolchain compatible with your installed `torch` build.

```bash
conda create -n logical-subspace python=3.11 -y
conda activate logical-subspace
pip install -r requirements.txt
```

Optional dependencies (rebuttal experiments, 4-bit / 8-bit quantization, vLLM, OpenAI-API data conversion):

```bash
pip install -r requirements-optional.txt
```

The optional file additionally pulls `bitsandbytes` (for `--use_4bit/--use_8bit` flags), `cca-zoo` (alternative CCA backend selectable via `--lib ccazoo`), `vllm` (for FineLogic / overall evaluation), and `openai` + `aiohttp` (for `proofwriter/data_processing/convert_nl_to_symbolic.py`).

## Models tested in the paper

The default scripts assume **`meta-llama/Llama-3.1-8B-Instruct`**. To switch model just pass a different `--model`. The paper additionally reports results on Llama-2-13B-Chat, Llama-3.2-3B-Instruct, Gemma-2-9B-IT, Phi-3-Mini-4K-Instruct, and Qwen3-4B; you may need to retune `--layer_start/--layer_end` and `--lam_*` ranges per architecture.

## End-to-end workflow

The four pipeline stages are: (1) prepare paired views → (2) extract residuals → (3) fit SVCCA → (4) evaluate steering. The bundled small data files let you skip stage 1 if you only want to reproduce evaluation.

### 1. Prepare paired (NL, symbolic) views

The repository already ships processed splits for every dataset (see `*/data/`), so this step is only needed if you want to regenerate them from upstream raw data.

ProntoQA — paired splits already provided in `prontoqa/data/` (1000 train / 500 val / 500 test, each item has 4 views: `NL_with_proof`, `NL_without_proof`, `FOL_with_proof`, `FOL_without_proof`):

```bash
# Optional — regenerate from a raw ProntoQA dump.
# process_for_cca.py: positional <input.json> <output.json>
python prontoqa/data_processing/process_for_cca.py raw_prontoqa.json \
    prontoqa/data/5hop_0shot_noadj_processed.json

# split_dataset.py: positional <input> <train_size> <val_size> <test_size> <output_prefix> [seed]
python prontoqa/processing/split_dataset.py \
    prontoqa/data/5hop_0shot_noadj_processed.json 1000 500 500 \
    prontoqa/data/5hop_0shot_noadj_processed
```

ProofWriter — paired splits already provided in `proofwriter/data/` (1500 train + 500 val for SVCCA, 501 CWA / 501 OWA depth-3 evaluation jsonls; each paired item has 4 views: `NL_with_proof`, `NL_without_proof`, `Symbolic_with_proof`, `Symbolic_without_proof`):

```bash
# Optional — regenerate from a raw ProofWriter dump.

# 1. extract (NL_proof, answer) pairs from raw ProofWriter jsonl
python proofwriter/data_processing/extract_proof_pairs.py \
    raw_proofwriter.jsonl proofwriter/data/proof_pairs.jsonl --min-depth 1 --max-depth 3

# 2. convert NL proofs to symbolic using an OpenAI-compatible API
export OPENAI_API_KEY=sk-...
python proofwriter/data_processing/convert_nl_to_symbolic.py \
    --input proofwriter/data/proof_pairs.jsonl \
    --output proofwriter/data/proof_pairs_symbolic.jsonl \
    --prompt-file proofwriter/data_processing/prompt_nl_to_symbolic.txt \
    --model gpt-4o --temperature 0.3

# 3. build the 4-view paired format. positional <input.jsonl> <output.json>
python proofwriter/data_processing/process_for_cca.py \
    proofwriter/data/proof_pairs_symbolic.jsonl proofwriter/data/train_train.json
```

FOLIO — paired splits already provided in `folio/data/` (1001 train / 203 val, each item has just 2 views: `NL` and `FOL`, since FOLIO doesn't ship with intermediate proofs the way ProntoQA / ProofWriter do). Raw FOLIO v2 splits are kept under the same directory for re-processing:

```bash
# Optional — regenerate paired views from raw FOLIO v2 jsonl
python folio/data_processing/process_json.py \
    --input folio/data/folio_v2_train.jsonl \
    --output folio/data/processed_train.jsonl
```

LogiQA — both the multiple-choice (`logiqa_500_sample.jsonl`) and NLI (`logiqa_nli_500_sample.jsonl`) 500-item evaluation subsets are already shipped under `generalization/data/`. The sister files `mcr_val.json` (MCR format) and `nli_val.json` (NLI format) are 500-item validation splits sampled from a different seed and are useful as held-out probes when reporting transfer numbers:

```bash
# Optional — regenerate a sample from upstream LogiQA 2.0
python generalization/sample_logiqa.py \
    --input path/to/raw_logiqa_test.txt \
    --output generalization/data/logiqa_500_sample.jsonl \
    --n_samples 500 --seed 42
```

### 2. Extract residual activations

For every dataset there is a thin wrapper around `transformer_lens` that emits a `[2, N, L, D]` tensor (NL view, symbolic view) on the **paired training set**:

```bash
# ProntoQA
python prontoqa/processing/get_residue_prontoqa.py \
    --input prontoqa/data/5hop_0shot_noadj_processed_train.json \
    --output_pt outputs/prontoqa_resid_llama8b.pt \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype fp32

# ProofWriter (--view2 selects which paired view is the second one)
python proofwriter/processing/get_residue_proofwriter.py \
    --input proofwriter/data/train_train.json \
    --output_pt outputs/proofwriter_resid_llama8b.pt \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --view2 Symbolic --dtype fp32

# FOLIO
python folio/processing/get_residue.py \
    --input folio/data/processed_train.jsonl \
    --output_pt outputs/folio_resid_llama8b.pt \
    --model meta-llama/Llama-3.1-8B-Instruct --dtype fp32
```

The `.pt` files are gitignored — they are large.

### 3. Fit the SVCCA subspace

The fitter is the same script across datasets, parameterised by the residual file:

```bash
python prontoqa/processing/cca.py \
    --resid_pt outputs/prontoqa_resid_llama8b.pt \
    --out_path outputs/prontoqa_cca_llama8b.pt \
    --pca_var 0.98 --pca_cap 128 --k 64 \
    --center --shared x --lib sklearn --no_save_P
```

Replace `prontoqa` with `proofwriter` / `folio` for the other two datasets.

### 4. Steering evaluation (layer × λ sweep)

The main entry points run a `(layer, λ)` grid and emit a CSV plus optional per-sample JSONL predictions.

ProntoQA:

```bash
python prontoqa/evaluation/infer_tuning_prontoqa_multilayer_normalized.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --file prontoqa/data/5hop_0shot_noadj_processed_test.json \
    --svcca_pt outputs/prontoqa_cca_llama8b.pt \
    --layer_start 10 --layer_end 20 \
    --view NL_without_proof \
    --lam_start 0.00 --lam_end 0.14 --lam_step 0.02 \
    --use_projectors --anchor all --window 1 \
    --out_csv outputs/prontoqa_sweep.csv --save_preds
```

ProofWriter, CWA / OWA — note the `_owa` suffix and the matching `--file`:

```bash
# CWA
python proofwriter/evaluation/infer_tuning_prontoqa_multilayer_normalized.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --file proofwriter/data/cwa_501_3hop.jsonl \
    --svcca_pt outputs/proofwriter_cca_llama8b.pt \
    --layer_start 10 --layer_end 20 --view NL_without_proof \
    --lam_start 0.00 --lam_end 0.14 --lam_step 0.02 \
    --use_projectors --out_csv outputs/proofwriter_cwa_sweep.csv --save_preds

# OWA
python proofwriter/evaluation/infer_tuning_prontoqa_multilayer_normalized_owa.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --file proofwriter/data/owa_501_3hop.jsonl \
    --svcca_pt outputs/proofwriter_cca_llama8b.pt \
    --layer_start 10 --layer_end 15 --view NL_without_proof \
    --lam_start 0.00 --lam_end 0.14 --lam_step 0.02 \
    --use_projectors --out_csv outputs/proofwriter_owa_sweep.csv --save_preds
```

FOLIO (no `--view` flag — the script always uses NL premises):

```bash
python folio/evaluation/infer_tuning_multilayer_normalized.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --file folio/data/folio_v2_validation.jsonl \
    --svcca_pt outputs/folio_cca_llama8b.pt \
    --layer_start 10 --layer_end 20 \
    --lam_start 0.00 --lam_end 0.14 --lam_step 0.02 \
    --use_projectors --out_csv outputs/folio_sweep.csv --save_preds
```

Common flags across all sweep scripts: `--top_k`, `--corr_min` (column pruning by SVCCA correlation), `--use_4bit/--use_8bit` (quantised loading), `--max_samples` (cap), `--anchor {last,all}` and `--window` (which token positions are steered).

### 5. Baselines

Each baseline reads the same paired-view JSON files. ProntoQA examples:

```bash
# Zero-shot CoT
python prontoqa/evaluation/prontoqa_zero_shot.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --file prontoqa/data/5hop_0shot_noadj_processed_test.json \
    --view NL_without_proof --output outputs/prontoqa_zs.jsonl

# 3-shot CoT
python prontoqa/evaluation/prontoqa_3shot_baseline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --file prontoqa/data/5hop_0shot_noadj_processed_test.json \
    --view NL_without_proof \
    --output outputs/prontoqa_3shot.jsonl --out_csv outputs/prontoqa_3shot.csv

# Self-consistency (5 paths)
python prontoqa/evaluation/infer_self_consistency_prontoqa.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --file prontoqa/data/5hop_0shot_noadj_processed_test.json \
    --view NL_without_proof --num_paths 5 --temperature 0.7

# Direct answer (no CoT)
python prontoqa/evaluation/infer_direct_answer.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --file prontoqa/data/5hop_0shot_noadj_processed_test.json \
    --view NL_without_proof --save_preds --out_csv outputs/prontoqa_direct.csv
```

ProofWriter has matching CWA / OWA scripts (`_owa` suffix). FOLIO has `infer_zero_shot.py`, `folio_3shot_baseline.py`, `infer_self_consistency.py`.

### 6. Cross-dataset transfer (LogiQA)

Fit SVCCA on ProntoQA (Step 3 above), then apply to LogiQA without retraining:

```bash
python generalization/evaluation/infer_tuning_logiqa_multilayer.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --file generalization/data/logiqa_500_sample.jsonl \
    --svcca_pt outputs/prontoqa_cca_llama8b.pt \
    --layer_start 10 --layer_end 20 \
    --lam_start 0.04 --lam_end 0.14 --lam_step 0.02 \
    --use_projectors --out_csv outputs/logiqa_transfer.csv --save_preds

# LogiQA-NLI variant
python generalization/evaluation/infer_tuning_logiqa_nli_multilayer.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --file generalization/data/logiqa_nli_500_sample.jsonl \
    --svcca_pt outputs/prontoqa_cca_llama8b.pt \
    --layer_start 10 --layer_end 20 \
    --out_csv outputs/logiqa_nli_transfer.csv --save_preds
```

### 7. Analysis

```bash
# Per-canonical-direction token analysis
python prontoqa/analysis/per_direction_analysis.py \
    --input prontoqa/data/5hop_0shot_noadj_processed_test.json \
    --svcca_pt outputs/prontoqa_cca_llama8b.pt --layer 16 \
    --model meta-llama/Llama-3.1-8B-Instruct --out_dir outputs/per_direction

# Token-level projection energy along CoT
python prontoqa/analysis/token_energy_cot.py \
    --input prontoqa/data/5hop_0shot_noadj_processed_test.json \
    --svcca_pt outputs/prontoqa_cca_llama8b.pt --layer 16 \
    --model meta-llama/Llama-3.1-8B-Instruct --out_dir outputs/token_energy

# ROC-AUC for "energy predicts correctness" across layers
python prontoqa/analysis/cot_energy_auc_multilayer.py \
    --preds_jsonl outputs/prontoqa_zs.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --cca_pt outputs/prontoqa_cca_llama8b.pt \
    --layer_start 0 --layer_end -1 --out_dir outputs/cot_auc

# McNemar test between two prediction files
python prontoqa/analysis/mcnemar_test.py \
    --baseline outputs/prontoqa_zs.jsonl \
    --treatment outputs/prontoqa_steered.jsonl \
    --baseline_name "zero-shot" --treatment_name "LSS-steered" \
    --output_json outputs/mcnemar.json

# SVCCA correlation generalization (train → val)
python prontoqa/analysis/coor_test.py \
    --svcca_pt outputs/prontoqa_cca_llama8b.pt \
    --val_resid_pt outputs/prontoqa_resid_val.pt \
    --out_csv outputs/val_projection.csv
```

### 8. Rebuttal experiments

HellaSwag side-effect probe (does steering hurt unrelated tasks?):

```bash
python rebuttal/preprocess_hellaswag.py \
    --input rebuttal/hellaswag/data/hellaswag_val.jsonl \
    --output rebuttal/hellaswag_val_processed.jsonl
python rebuttal/infer_tuning_hellaswag_multilayer_normalized_runtime_processed.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --file rebuttal/hellaswag_val_processed.jsonl \
    --svcca_pt outputs/prontoqa_cca_llama8b.pt \
    --layer_start 10 --layer_end 20 \
    --out_csv outputs/hellaswag_side_effect.csv --save_preds
```

ReClor cross-task transfer:

```bash
python rebuttal/recolor/preprocess_recolor.py \
    --input rebuttal/recolor/val.json \
    --output rebuttal/recolor/val_processed.jsonl
python rebuttal/recolor/infer_tuning_recolor_multilayer_normalized.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --file rebuttal/recolor/val_processed.jsonl \
    --svcca_pt outputs/prontoqa_cca_llama8b.pt \
    --layer_start 10 --layer_end 20 \
    --out_csv outputs/recolor_transfer.csv --save_preds
```

FineLogic step-level evaluation lives under `rebuttal/FineLogic/` and is a vendored fork of the upstream FineLogic harness. See `rebuttal/FineLogic/README.md` for its own setup; the integration script is `rebuttal/FineLogic/src/eval_hf_steering_dataset4.py`.

## Data sources

The repo bundles only small evaluation files. Larger corpora must be downloaded from upstream:

| Dataset | Upstream |
|---|---|
| ProntoQA | <https://github.com/asaparov/prontoqa> |
| ProofWriter v2020.12.3 | <https://allenai.org/data/proofwriter> |
| FOLIO v2 | <https://github.com/Yale-LILY/FOLIO> |
| LogiQA 2.0 | <https://github.com/csitfun/LogiQA2.0> |
| HellaSwag | <https://github.com/rowanz/hellaswag> |
| ReClor | <https://github.com/yuweihao/reclor> |
| FineLogic | <https://github.com/THU-KEG/FineLogic> (or upstream link in `rebuttal/FineLogic/README.md`) |

## SLURM templates

`examples/slurm/` contains seven sanitized templates covering one job per pipeline stage. Each script uses `${REPO_ROOT}` for the install root, `your-email@example.com` for SBATCH notifications, and `<your-env>` for the conda environment — replace these before submitting.

## Citation

```bibtex
@article{fang2026logical,
  title   = {Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views},
  author  = {Fang, Feihao and Thai, My T. and Lei, Yuanyuan},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). Vendored third-party code under `rebuttal/FineLogic/` and `rebuttal/hellaswag/` is governed by its own respective upstream license; see those subdirectories' LICENSE files.
