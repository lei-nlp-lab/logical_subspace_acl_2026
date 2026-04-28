# FineLogic

This repository contains the code and evaluation scripts for our paper **"Dissecting Logical Reasoning in LLMs: A Fine-Grained Evaluation and Supervision Study"**, which performs a fine-grained analysis of reasoning capabilities and introduces supervision strategies to enhance logical performance in large language models.

## 🛠️ Installation

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
pip install vllm aiohttp backoff tqdm scikit-learn
cd ../
cd FineLogic
```

## 🚀 Usage

### 🔧 For Training and Evaluation

To run the complete training and evaluation pipeline:

```bash
# Move configuration file to LLaMA-Factory directory
mv logical.yaml ../LLaMA-Factory/

# Enter source directory and run training script
cd src
sh overall_llama.sh
```

### 📊 For Evaluation Only

If you only need to perform evaluation, you have two options:

**1: For the local model, use vLLM for evaluation**
```bash
sh src/eval_vllm_overall.sh
```
*Note: You need to change the model path in the script*

**2: For proprietary models, use API for evaluation**
```bash
python src/evaluate_api.py
```
