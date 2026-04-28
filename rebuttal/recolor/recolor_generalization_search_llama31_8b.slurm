#!/bin/bash

#SBATCH --job-name=recolor_llama31_search
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=20gb
#SBATCH --time=16:00:00
#SBATCH --partition=hpg-b200
#SBATCH --output=recolor_llama31_search_%j.log

set -euo pipefail

module load conda
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate <your-env>  # adjust to your environment

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# ReClor generalization sweep:
# - model: Llama-3.1-8B-Instruct
# - subspace: learned from ProntoQA
# - data: processed ReClor val split
python3 "$SCRIPT_DIR/infer_tuning_recolor_multilayer_normalized.py" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --file "$SCRIPT_DIR/val_processed.jsonl" \
  --svcca_pt "$REPO_ROOT/rebuttal/llama_cca_64.pt" \
  --layer_start 7 --layer_end 14 \
  --lam_start 0.02 --lam_end 0.12 --lam_step 0.02 \
  --max_samples 500 \
  --batch_size 4 \
  --max_new_tokens 1024 \
  --out_csv "$SCRIPT_DIR/recolor_llama31_search_2.csv" \
  --save_preds
