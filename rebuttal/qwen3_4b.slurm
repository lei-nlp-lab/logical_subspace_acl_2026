#!/bin/bash

#SBATCH --job-name=logispace_rebuttal_4     # Job name
#SBATCH --mail-type=END,FAIL               # Mail events
#SBATCH --mail-user=YOUR_EMAIL             # Where to send mail
#SBATCH --ntasks=1                         # Run on a single CPU
#SBATCH --mem=20gb                         # Job memory request
#SBATCH --time=19:00:00                     # Time limit hrs:min:sec
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --partition=hpg-b200
#SBATCH --output=qwen3_4b_%j.log

set -euo pipefail

module load conda
module load cuda/12.8.1

conda activate <your-env>  # adjust to your environment

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

python "$REPO_ROOT/prontoqa/infer_tuning_prontoqa_multilayer_normalized.py" \
  --model Qwen/Qwen3-4B \
  --file "$REPO_ROOT/prontoqa/5hop_0shot_noadj_processed_test.json" \
  --svcca_pt "$SCRIPT_DIR/qwen3_4b_cca.pt" \
  --layer_start 0 --layer_end 7 \
  --view NL_with_proof \
  --lam_start 0.02 --lam_end 0.12 --lam_step 0.02 \
  --out_csv "$SCRIPT_DIR/qwen3_5.csv" \
  --save_preds
