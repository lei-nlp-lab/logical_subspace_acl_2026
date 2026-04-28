#!/bin/bash

#SBATCH --job-name=sc_pqa_llama           # Job name
#SBATCH --mail-type=END,FAIL              # Mail events
#SBATCH --mail-user=your-email@example.com # Where to send mail
#SBATCH --ntasks=1                        # Run on a single CPU
#SBATCH --mem=20gb                        # Job memory request
#SBATCH --time=6:00:00                    # Time limit hrs:min:sec
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --partition=hpg-b200
#SBATCH --output=${REPO_ROOT}/prontoqa/logs/sc_llama_%j.log

module load conda
module load cuda/12.8.1

conda activate <your-env>  # adjust to your environment

# ============================================================
# Self-Consistency Baseline - ProntoQA (Binary: True/False)
# Model: Llama-3.1-8B-Instruct
# ============================================================

echo "Running Self-Consistency with Llama-3.1-8B-Instruct on ProntoQA"
python ${REPO_ROOT}/prontoqa/evaluation/infer_self_consistency_prontoqa.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --file ${REPO_ROOT}/prontoqa/data/5hop_0shot_noadj_processed_test.json \
    --view NL_without_proof \
    --num_paths 3 \
    --temperature 0.7 \
    --top_p 1.0 \
    --min_valid_ratio 0.5 \
    --max_new_tokens 1024 \
    --out_csv ${REPO_ROOT}/prontoqa/outputs/self_consistency_llama.csv \
    --save_preds
