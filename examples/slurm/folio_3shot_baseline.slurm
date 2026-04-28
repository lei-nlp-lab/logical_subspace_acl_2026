#!/bin/bash

#SBATCH --job-name=folio_3shot_llama       # Job name
#SBATCH --mail-type=END,FAIL               # Mail events
#SBATCH --mail-user=your-email@example.com  # Where to send mail
#SBATCH --ntasks=1                         # Run on a single CPU
#SBATCH --mem=20gb                         # Job memory request
#SBATCH --time=2:00:00                     # Time limit hrs:min:sec
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --partition=hpg-b200
#SBATCH --output=${REPO_ROOT}/folio/logs/3shot_baseline_llama_%j.log

module load conda
module load cuda/12.8.1

conda activate <your-env>  # adjust to your environment

# ============================================================
# 3-Shot Baseline (No Steering) - FOLIO
# Model: Llama-3.1-8B-Instruct
# ============================================================

echo "Running 3-Shot Baseline with Llama-3.1-8B-Instruct on FOLIO"
python ${REPO_ROOT}/folio/evaluation/folio_3shot_baseline.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --file ${REPO_ROOT}/folio/data/raw/folio_v2_validation.jsonl \
    --max_new_tokens 1024 \
    --batch_size 4 \
    --output ${REPO_ROOT}/folio/outputs/3shot_baseline_llama_preds.jsonl \
    --out_csv ${REPO_ROOT}/folio/outputs/3shot_baseline_llama.csv
