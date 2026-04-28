# File Manifest

Per-script reference for the repository.

## Per-dataset pipeline

Each of `prontoqa/`, `proofwriter/`, `folio/` follows the same three-stage layout:

| Stage | ProntoQA | ProofWriter | FOLIO |
|---|---|---|---|
| Build paired (NL, symbolic) views from raw data | `prontoqa/data_processing/process_for_cca.py` | `proofwriter/data_processing/{extract_proof_pairs,proof_to_text,convert_nl_to_symbolic,process_for_cca}.py` | `folio/data_processing/process_json.py` |
| Extract residual activations from a model | `prontoqa/processing/get_residue_prontoqa.py` | `proofwriter/processing/get_residue_proofwriter.py` | `folio/processing/get_residue.py` |
| Fit SVCCA subspace on the paired residuals | `prontoqa/processing/cca.py` | `proofwriter/processing/cca.py` | `folio/processing/cca.py` |

## Steering evaluation (layer × λ sweep)

| Dataset | Script |
|---|---|
| ProntoQA | `prontoqa/evaluation/infer_tuning_prontoqa_multilayer_normalized.py` |
| ProofWriter (CWA) | `proofwriter/evaluation/infer_tuning_prontoqa_multilayer_normalized.py` |
| ProofWriter (OWA) | `proofwriter/evaluation/infer_tuning_prontoqa_multilayer_normalized_owa.py` |
| FOLIO | `folio/evaluation/infer_tuning_multilayer_normalized.py` |
| LogiQA (transfer) | `generalization/evaluation/infer_tuning_logiqa_multilayer.py` |
| LogiQA-NLI (transfer) | `generalization/evaluation/infer_tuning_logiqa_nli_multilayer.py` |
| HellaSwag (side-effect probe) | `rebuttal/infer_tuning_hellaswag_multilayer_normalized_runtime.py` |
| HellaSwag (preprocessed) | `rebuttal/infer_tuning_hellaswag_multilayer_normalized_runtime_processed.py` |
| ReClor (transfer) | `rebuttal/recolor/infer_tuning_recolor_multilayer_normalized.py` |

## Baselines

| Method | ProntoQA | ProofWriter (CWA / OWA) | FOLIO |
|---|---|---|---|
| Zero-shot CoT | `prontoqa/evaluation/prontoqa_zero_shot.py` | — | `folio/evaluation/infer_zero_shot.py` |
| 3-shot CoT | `prontoqa/evaluation/prontoqa_3shot_baseline.py` | `proofwriter/evaluation/proofwriter_3shot_baseline.py` / `..._owa.py` | `folio/evaluation/folio_3shot_baseline.py` |
| Self-consistency | `prontoqa/evaluation/infer_self_consistency_prontoqa.py` | `proofwriter/evaluation/infer_self_consistency_prontoqa.py` / `..._owa.py` | `folio/evaluation/infer_self_consistency.py` |
| Direct answer | `prontoqa/evaluation/infer_direct_answer.py` | `proofwriter/evaluation/infer_direct_answer.py` / `..._owa.py` | — |
| 3-shot + steering | `prontoqa/evaluation/prontoqa_3shot_steering_multilayer_normalized.py` | — | — |

## Steering-hook utilities (imported by sweep scripts)

Each `evaluation/` directory ships its own copy so the per-dataset entry points are self-contained:

- `prontoqa/evaluation/steering_infer.py`, `steering_infer_normalized.py`
- `proofwriter/evaluation/steering_infer.py`, `steering_infer_normalized.py`
- `folio/evaluation/steering_infer.py`, `steering_infer_normalized.py`

`generalization/evaluation/infer_tuning_logiqa_*.py` reuses `proofwriter/evaluation/`'s helpers via a `sys.path.insert` relative to `__file__` (no extra setup needed).

## Analysis (mostly for ProntoQA)

| Script | Purpose |
|---|---|
| `prontoqa/analysis/per_direction_analysis.py` | Per-canonical-direction token category analysis |
| `prontoqa/analysis/token_energy_cot.py` | Token-level projection energy in CoT |
| `prontoqa/analysis/cot_energy_auc_multilayer.py` | ROC-AUC of correctness prediction via projection energy |
| `prontoqa/analysis/mcnemar_test.py` | McNemar significance test between baseline / treatment preds |
| `prontoqa/analysis/coor_test.py` | Per-dimension correlation between train and val SVCCA projections |
| `prontoqa/analysis/check_tokenization.py` | Sanity-check tokenizer alignment for residual extraction |
| `prontoqa/analysis/cot_keyword_error_analysis.py` | CoT failure-mode keyword analysis |
| `prontoqa/analysis/analyze_steps.py` | Per-step CoT correctness analysis |
| `folio/analysis/coor_test.py` | FOLIO version of `coor_test.py` |
| `folio/analysis/nl_energy.py` | NL-side projection energy on FOLIO |

## Generalization helpers

| Script | Purpose |
|---|---|
| `generalization/sample_logiqa.py` | Sample N items from LogiQA into the included `data/logiqa_500_sample.jsonl` |
| `generalization/process_for_cca_proofs_only.py` | Build a proof-only paired view from LogiQA-style data |

## Random-baseline / ablation utilities

| Script | Purpose |
|---|---|
| `prontoqa/evaluation/generate_random_projection.py` | Random orthonormal/uniform projection with the same shape as a real SVCCA file (for null-control) |
| `prontoqa/processing/split_dataset.py` | Split raw ProntoQA JSON into train/val/test |
| `proofwriter/processing/extract_random_testset.py`, `extract_balanced_testset.py` | Extract test subsets from ProofWriter |

## Rebuttal experiments

| Path | Purpose |
|---|---|
| `rebuttal/preprocess_hellaswag.py` | Convert raw HellaSwag jsonl into the format expected by the runtime sweep |
| `rebuttal/infer_tuning_hellaswag_multilayer_normalized_runtime.py` | Apply a logic-task SVCCA subspace to HellaSwag at runtime (side-effect probe) |
| `rebuttal/infer_tuning_hellaswag_multilayer_normalized_runtime_processed.py` | Same as above but on preprocessed input |
| `rebuttal/recolor/preprocess_recolor.py` | Convert raw ReClor split JSON into JSONL |
| `rebuttal/recolor/infer_tuning_recolor_multilayer_normalized.py` | Apply a logic-task SVCCA subspace to ReClor |
| `rebuttal/FineLogic/` | Vendored upstream FineLogic evaluation harness (third-party; see its README/LICENSE) |
| `rebuttal/hellaswag/` | Vendored upstream HellaSwag harness (third-party; see its README/LICENSE) |

## Example SLURM scripts

`examples/slurm/` contains seven sanitized SLURM templates covering one representative job per pipeline stage. Replace `${REPO_ROOT}`, `your-email@example.com`, and `<your-env>` with your own values before submitting.

## Bundled small data

| Path | Description |
|---|---|
| `prontoqa/data/5hop_0shot_noadj_processed_{train,val,test}.json` | Paired (NL, FOL) ProntoQA splits used in the paper |
| `proofwriter/data/{train_train,train_val}.json` | Paired ProofWriter SVCCA training/validation set |
| `proofwriter/data/{cwa,owa}_501_3hop.jsonl` | ProofWriter CWA / OWA depth-3 evaluation sets (501 each) |
| `folio/data/processed_{train,val}.jsonl` | Paired FOLIO splits (NL + FOL views) |
| `folio/data/folio_v2_{train,validation}.jsonl` | Raw FOLIO v2 splits (used by `process_json.py`) |
| `generalization/data/logiqa_{,nli_}500_sample.jsonl` | 500-item LogiQA / LogiQA-NLI eval samples |
| `generalization/data/{mcr,nli}_val.json` | Paired LogiQA validation views for SVCCA fitting |
| `rebuttal/recolor/{train,val,test}.json` | ReClor splits (vendored from upstream) |

Larger artifacts (residual `.pt` tensors, full LogiQA train, ProofWriter v2020.12.3 mirror, HellaSwag / FineLogic data dirs) are ignored via `.gitignore` and must be regenerated or downloaded from upstream — see the relevant section of `README.md`.
