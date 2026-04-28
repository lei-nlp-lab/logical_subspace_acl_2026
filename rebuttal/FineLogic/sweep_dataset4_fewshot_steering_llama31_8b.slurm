#!/bin/bash

#SBATCH --job-name=finelogic_d4_sweep
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=20gb
#SBATCH --time=1:30:00
#SBATCH --partition=hpg-b200
#SBATCH --output=finelogic_d4_sweep_%j.log

set -euo pipefail

module load conda
module load cuda/12.8.1
conda activate <your-env>  # adjust to your environment

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$ROOT/../.." && pwd)
SCRIPT=$ROOT/src/eval_hf_steering_dataset4.py
OUT_DIR=$ROOT/results/steering_dataset4
LOG_DIR=$ROOT/logs

mkdir -p "$OUT_DIR" "$LOG_DIR"

MODEL_NAME=llama31_8b
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
DATASET4_PATH=$ROOT/data/eval/Dataset4-ProntoQA.json
SVCCA_PT=$REPO_ROOT/rebuttal/llama_cca_64.pt

# Sweep range
LAYER_START=0
LAYER_END=0
LAM_START=0.00
LAM_END=0.00
LAM_STEP=0.02

# Few-shot prompt style from original FineLogic examples
STYLES=fewshot

MAX_SAMPLES=500
BATCH_SIZE=4
MAX_NEW_TOKENS=1024

SUMMARY_CSV=$OUT_DIR/Dataset4-ProntoQA_steering_summary_${MODEL_NAME}_${STYLES}_L${LAYER_START}-${LAYER_END}_job${SLURM_JOB_ID}.csv

echo "Running Dataset4 sweep"
echo "model=$MODEL_PATH styles=$STYLES layers=${LAYER_START}-${LAYER_END} lambda=${LAM_START}:${LAM_STEP}:${LAM_END}"
echo "summary_csv=$SUMMARY_CSV"

python3 "$SCRIPT" \
  --model "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --dataset4_path "$DATASET4_PATH" \
  --svcca_pt "$SVCCA_PT" \
  --styles "$STYLES" \
  --layer_start "$LAYER_START" \
  --layer_end "$LAYER_END" \
  --lam_start "$LAM_START" \
  --lam_end "$LAM_END" \
  --lam_step "$LAM_STEP" \
  --max_samples "$MAX_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --use_projectors \
  --out_dir "$OUT_DIR" \
  --summary_csv "$SUMMARY_CSV"

echo "Done. Results in $OUT_DIR"
