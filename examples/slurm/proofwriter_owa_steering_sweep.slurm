#!/bin/bash

#SBATCH --job-name=multi_view_steering    # Job name 
#SBATCH --mail-type=END,FAIL          # Mail events 
#SBATCH --mail-user=your-email@example.com  # Where to send mail 
#SBATCH --ntasks=1                    # Run on a single CPU 
#SBATCH --mem=60gb                     # Job memory request 
#SBATCH --time=21:00:00               # Time limit hrs:min:sec 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --partition=hpg-b200
#SBATCH --output=${REPO_ROOT}/prontoqa/logs/experiment_%j.log   # Standard output and error log 

module load conda
module load cuda/12.8.1

conda activate <your-env>  # adjust to your environment

python ${REPO_ROOT}/proofwriter/neweval/infer_tuning_prontoqa_multilayer_normalized_owa.py --model meta-llama/Llama-3.1-8B-Instruct --lam_start 0.02 --file ${REPO_ROOT}/proofwriter/data/owa_501_3hop.jsonl --svcca_pt ${REPO_ROOT}/proofwriter/outputs/residues/newllamacca.pt --layer_start 10 --layer_end 15 --view NL_without_proof --use_projectors --out_csv ${REPO_ROOT}/proofwriter/neweval/test_owa.csv --outdir_fig ${REPO_ROOT}/proofwriter/figures --save_preds

