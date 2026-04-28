#!/bin/bash

#SBATCH --job-name=multi_view_steering    # Job name 
#SBATCH --mail-type=END,FAIL          # Mail events 
#SBATCH --mail-user=your-email@example.com  # Where to send mail 
#SBATCH --ntasks=1                    # Run on a single CPU 
#SBATCH --mem=20gb                     # Job memory request 
#SBATCH --time=12:00:00               # Time limit hrs:min:sec 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --partition=hpg-b200
#SBATCH --output=${REPO_ROOT}/prontoqa/logs/experiment_%j.log   # Standard output and error log 

module load conda
module load cuda/12.8.1

conda activate <your-env>  # adjust to your environment

python ${REPO_ROOT}/generalization/evaluation/infer_tuning_logiqa_multilayer.py --file ${REPO_ROOT}/generalization/data/logiqa_500_sample.jsonl --svcca_pt ${REPO_ROOT}/generalization/llama_cca.pt --layer_start 10 --layer_end 15 --use_projectors --out_csv ${REPO_ROOT}/generalization/logiqa_llama.csv --save_preds  --model meta-llama/Llama-3.1-8B-Instruct 