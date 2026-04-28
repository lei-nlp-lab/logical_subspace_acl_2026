#!/bin/bash

# Define datasets to loop through
datasets=(
########## Define the datasets to be used
)

cd ../../LLaMA-Factory/saves/llama3-8B/full/logical/
#chmod +x eval_vllm_overall.sh

# Loop through each dataset
for dataset in "${datasets[@]}"; do
  echo "Processing dataset: ${dataset}"
  
  # Return to root directory
  cd ../../../../
  
  # Initialize and activate conda environment
  eval "$(conda shell.bash hook)"
  conda activate llamafactory
  
  # Confirm logical_qwen.yaml file exists
  if [ ! -f "logical.yaml" ]; then
    echo "Error: logical.yaml file not found"
    echo "Please confirm file location and manually specify correct path"
    exit 1
  fi
  
  # Modify yaml file
  cat logical.yaml | sed "s/dataset: logical_ex/dataset: ${dataset}/" | sed "s|output_dir: saves/llama3-8B/full/logical/logical_ex|output_dir: saves/llama3-8B/full/logical/${dataset}|" > overall_temp.yaml
  
  # Train model
  CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train overall_temp.yaml
  
  # Switch to evaluation directory
  cd ../LogicalReasoning/src
  
  # Run evaluation script
  sh eval_vllm_overall.sh "llama3-8B_full" "${dataset}"

  # Delete dataset directory
  cd ../../LLaMA-Factory/saves/llama3-8B/full/logical/
  rm -rf "${dataset}"
  
  echo "Completed dataset: ${dataset}"
done
python process_result_by_step.py 1