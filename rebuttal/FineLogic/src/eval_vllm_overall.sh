#!/usr/bin/env bash
set -euo pipefail

MODEL_BASE=$1
DATASET=$2
DATASET_IDS="1 2 3 4"

for i in {1..3}; do
  model_name="llama3-8B_full_${DATASET:8}_${i}epoch"
  if [[ $DATASET == *"0_25"* ]]; then
    model_path="../../LLaMA-Factory/saves/llama3-8B/full/logical/${DATASET}/checkpoint-$((100*i))"
  else
    model_path="../../LLaMA-Factory/saves/llama3-8B/full/logical/${DATASET}/checkpoint-$((397*i))"
  fi
  
  echo "============================================================="
  echo "[ RUN ] model=${model_name} path=${model_path} datasets=${DATASET_IDS}"
  echo "============================================================="

  CUDA_VISIBLE_DEVICES=0,1,2,3 python -u eval_vllm_overall.py \
    --model "${model_name}" \
    --model_path "${model_path}" \
    --datasets ${DATASET_IDS}

  CUDA_VISIBLE_DEVICES=0,1,2,3 python -u probing.py "${model_path}" "${model_name}"
done