from __future__ import annotations

import asyncio
import os
import json
import time
import gc
from typing import List, Dict, Any
import torch.distributed as dist
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.distributed.parallel_state import destroy_model_parallel
import argparse, os, random
import re
from eval_step import run_pipeline
from probing import run_probing

# API model configuration
API_CONFIGS = {
    "gpt-4o": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY")
    },
    "deepseek-ai/DeepSeek-R1": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key": os.getenv("DEEPINFRA_API_KEY")
    }
}

# HuggingFace local model configuration
VLLM_CONFIGS: Dict[str, Dict[str, Any]] = {}

# Prompt style configuration
PROMPT_STYLES = {
    "direct": {
        "system": {
            "default": "Based on the provided facts, verify the hypothesis. Conclude with one of the markers: \"__PROVED__\" for proven, \"__DISPROVED__\" for disproven, or \"__UNKNOWN__\" if uncertain.",
            "3": "Based on the provided facts, verify the hypothesis. Conclude with one of the markers: \"__PROVED__\" for true, or \"__DISPROVED__\" for false.",
            "4": "Based on the provided facts, verify the hypothesis. Conclude with one of the markers: \"__PROVED__\" for true, or \"__DISPROVED__\" for false.",
        },
        "suffix": ""
    },
    "cot": {
        "system": {
            "default": "Based on the provided facts, verify the hypothesis. Conclude with one of the markers: \"__PROVED__\" for proven, \"__DISPROVED__\" for disproven, or \"__UNKNOWN__\" if uncertain.",
            "3": "Based on the provided facts, verify the hypothesis. Conclude with one of the markers: \"__PROVED__\" for proven, or \"__DISPROVED__\" for disproven.",
            "4": "Based on the provided facts, verify the hypothesis. Conclude with one of the markers: \"__PROVED__\" for proven, or \"__DISPROVED__\" for disproven.",
        },
        "suffix": "Let's think step by step, by considering whether each person is lying and if that leads to contradiction."
    },
    "fewshot": {
        "system": {
            "default": "Based on the provided facts, verify the hypothesis. Conclude with one of the markers: \"__PROVED__\" for proven, \"__DISPROVED__\" for disproven, or \"__UNKNOWN__\" if uncertain.",
            "3": "Based on the provided facts, verify the hypothesis. Conclude with one of the markers: \"__PROVED__\" for proven, or \"__DISPROVED__\" for disproven.",
            "4": "Based on the provided facts, verify the hypothesis. Conclude with one of the markers: \"__PROVED__\" for proven, or \"__DISPROVED__\" for disproven.",
        },
        "suffix": "{fewshot_examples}"
    }
}

# Few-shot examples
FEWSHOT_EXAMPLES = {
    "default": """Here are some examples of proofs for your reference:
[Start of example]
For example, for this question:
"facts": "fact1: This vascularization is unshaven. fact2: If this parapsychologist is not a past this vascularization shrinks Belarusian. fact3: Something is not choric and is not unshaven if the thing is not a Hugueninia. fact4: Something is not a Hugueninia if that thing shrinks Belarusian.",
"hypothesis": "This vascularization is a Hugueninia.",
We can conclude that
"fact3 -> int1: This vascularization neither is choric nor is unshaven if it is not a Hugueninia.; void -> assump1: Let's assume that this vascularization is not a kind of a Hugueninia.; int1 & assump1 -> int2: This vascularization is not choric and it is not unshaven.; int2 -> int3: This vascularization is not unshaven.; int3 & fact1 -> int4: This is contradiction.; [assump1] & int4 -> hypothesis;"
So the answer is
__PROVED__

There is also this question: 
"facts": "fact1: Somebody is a Lophophorus if that it does trace excitableness or it does not Lot abridgement or both does not hold. fact2: There is nothing that traces excitableness and/or does not Lot abridgement. fact3: If the fact that something is vestiary and/or it is bumpy does not stand it inundates Vestris. fact4: Angeles is not non-Pasteurian. fact5: Angeles does Lot abridgement. fact6: If that somebody mists Ferber or this one does not disengage or both is wrong then that this one is a sabbatical is valid. fact7: If somebody is a Lophophorus that either it is a Gelechia or it does not chock impishness or both stands does not stand.", 
"hypothesis": "Angeles is not a Lophophorus.",
We can prove it by 
"fact2 -> int1: That Angeles traces excitableness or it does not Lot abridgement or both is invalid.; fact1 -> int2: Angeles is a Lophophorus if that Angeles does trace excitableness or it does not Lot abridgement or both does not stand.; int1 & int2 -> hypothesis;"
So the answer is
__DISPROVED__

Or this question:
"facts": "fact1: Non-unperceivingness prevents non-lingualness. fact2: That decelerating Webster takes place and propping Lycian takes place is incorrect. fact3: That decelerating Webster does not occurs and propping Lycian takes place is wrong if anonymousness occurs. fact4: Decelerating Webster does not occurs and propping Lycian happens. fact5: Non-anonymousness prevents electrolytic. fact6: If phrasing Balthasar takes place that both bestowing and comp happen is wrong.",
"hypothesis": "Lingualness does not occur.",
We prove it in the following way:
"void -> assump1: Let's assume that that this charcuterie is not a Anomia and does dredge helpfulness is incorrect.; fact4 & assump1 -> int1: The cabman concludes Landowska.; fact2 & fact3 -> int2: That preschooler does not precede preciousness.; Without finding a way to either prove or disprove the hypothesis.\n\nFinal conclusion: __UNKNOWN__"
[End of example]
You can refer to the proof method of the above question, think step by step, and give the result of this question.
""",
    "3": """Here are some examples of proofs for your reference:
[Start of example]
For example, for this question:
"facts": "fact1: This vascularization is unshaven. fact2: If this parapsychologist is not a past this vascularization shrinks Belarusian. fact3: Something is not choric and is not unshaven if the thing is not a Hugueninia. fact4: Something is not a Hugueninia if that thing shrinks Belarusian.",
"hypothesis": "This vascularization is a Hugueninia.",
We can conclude that
"fact3 -> int1: This vascularization neither is choric nor is unshaven if it is not a Hugueninia.; void -> assump1: Let's assume that this vascularization is not a kind of a Hugueninia.; int1 & assump1 -> int2: This vascularization is not choric and it is not unshaven.; int2 -> int3: This vascularization is not unshaven.; int3 & fact1 -> int4: This is contradiction.; [assump1] & int4 -> hypothesis;"
So the answer is
__PROVED__

There is also this question: 
"facts": "fact1: Somebody is a Lophophorus if that it does trace excitableness or it does not Lot abridgement or both does not hold. fact2: There is nothing that traces excitableness and/or does not Lot abridgement. fact3: If the fact that something is vestiary and/or it is bumpy does not stand it inundates Vestris. fact4: Angeles is not non-Pasteurian. fact5: Angeles does Lot abridgement. fact6: If that somebody mists Ferber or this one does not disengage or both is wrong then that this one is a sabbatical is valid. fact7: If somebody is a Lophophorus that either it is a Gelechia or it does not chock impishness or both stands does not stand.", 
"hypothesis": "Angeles is not a Lophophorus.",
We can prove it by 
"fact2 -> int1: That Angeles traces excitableness or it does not Lot abridgement or both is invalid.; fact1 -> int2: Angeles is a Lophophorus if that Angeles does trace excitableness or it does not Lot abridgement or both does not stand.; int1 & int2 -> hypothesis;"
So the answer is
__DISPROVED__
[End of example]
You can refer to the proof method of the above question, think step by step, and give the result of this question.
""",
    "4": """Here are some examples of proofs for your reference:
[Start of example]
For example, for this question:
"facts": "fact1: This vascularization is unshaven. fact2: If this parapsychologist is not a past this vascularization shrinks Belarusian. fact3: Something is not choric and is not unshaven if the thing is not a Hugueninia. fact4: Something is not a Hugueninia if that thing shrinks Belarusian.",
"hypothesis": "This vascularization is a Hugueninia.",
We can conclude that
"fact3 -> int1: This vascularization neither is choric nor is unshaven if it is not a Hugueninia.; void -> assump1: Let's assume that this vascularization is not a kind of a Hugueninia.; int1 & assump1 -> int2: This vascularization is not choric and it is not unshaven.; int2 -> int3: This vascularization is not unshaven.; int3 & fact1 -> int4: This is contradiction.; [assump1] & int4 -> hypothesis;"
So the answer is
__PROVED__

There is also this question: 
"facts": "fact1: Somebody is a Lophophorus if that it does trace excitableness or it does not Lot abridgement or both does not hold. fact2: There is nothing that traces excitableness and/or does not Lot abridgement. fact3: If the fact that something is vestiary and/or it is bumpy does not stand it inundates Vestris. fact4: Angeles is not non-Pasteurian. fact5: Angeles does Lot abridgement. fact6: If that somebody mists Ferber or this one does not disengage or both is wrong then that this one is a sabbatical is valid. fact7: If somebody is a Lophophorus that either it is a Gelechia or it does not chock impishness or both stands does not stand.", 
"hypothesis": "Angeles is not a Lophophorus.",
We can prove it by 
"fact2 -> int1: That Angeles traces excitableness or it does not Lot abridgement or both is invalid.; fact1 -> int2: Angeles is a Lophophorus if that Angeles does trace excitableness or it does not Lot abridgement or both does not stand.; int1 & int2 -> hypothesis;"
So the answer is
__DISPROVED__
[End of example]
You can refer to the proof method of the above question, think step by step, and give the result of this question.
"""
}

# Utility functions

def model_result_filename(model_name, timestamp, filename):
    return f'{filename}_results_{model_name.replace("/", "_")}_{timestamp}.json'

def model_eval_filename(model_name, timestamp, filename):
    return f'{filename}_evaluation_{model_name.replace("/", "_")}_{timestamp}.json'

def combined_eval_filename(timestamp, filename):
    return f'{filename}_combined_evaluation.json'

def status_filename(filename):
    return f'{filename}_evaluation_status.json'

def load_status(filename):
    if os.path.exists(status_filename(filename)):
        with open(status_filename(filename), 'r') as f:
            return json.load(f)
    return {'completed_models': [], 'timestamp': int(time.time())}

def save_status(status, filename):
    with open(status_filename(filename), 'w') as f:
        json.dump(status, f, indent=2)

def load_combined_eval(timestamp, filename):
    combined_file = combined_eval_filename(timestamp, filename)
    if os.path.exists(combined_file):
        with open(combined_file, 'r') as f:
            return json.load(f)
    return {}

def save_combined_eval(combined_data, timestamp, filename):
    combined_file = combined_eval_filename(timestamp, filename)
    with open(combined_file, 'w') as f:
        json.dump(combined_data, f, indent=2)

# Model management class
class VLLMModelManager:
    def __init__(self, cfg_name: str, cfg: Dict[str, Any]):
        self.cfg_name = cfg_name
        self.cfg = cfg
        self.llm: LLM | None = None
        self.tokenizer = None

    def initialize(self):
        print(f"[vLLM] loading {self.cfg_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model_path"], trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.llm = LLM(
            model=self.cfg["model_path"],
            tensor_parallel_size=self.cfg.get("tensor_parallel", 4),
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
        )
        print(f"[vLLM] {self.cfg_name} ready ✔")

    def generate(self, prompts: List[str], system_prompt: str, max_new: int = 150) -> List[str]:
        if self.llm is None:
            self.initialize()
        chat_prompts = [
            self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p},
            ], tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        params = SamplingParams(
            max_tokens=max_new,
            temperature=0.7,
            top_p=0.9,
        )
        outputs = self.llm.generate(chat_prompts, params)
        return [o.outputs[0].text for o in outputs]

    def cleanup(self):
        if self.llm is not None:
            self.llm.shutdown()
        del self.tokenizer
        del self.llm
        gc.collect()

# Global cache
vllm_managers: Dict[str, VLLMModelManager] = {}

def get_manager(model_name: str) -> VLLMModelManager:
    if model_name not in vllm_managers:
        vllm_managers[model_name] = VLLMModelManager(model_name, VLLM_CONFIGS[model_name])
    return vllm_managers[model_name]

# API logic
def call_vllm_with_retry(model_name: str, batch_data: List[Dict[str, Any]], style_name: str, dataset_id: int = None, max_retries: int = 3):
    style_cfg = PROMPT_STYLES[style_name]
    mgr = get_manager(model_name)

    system_prompt = style_cfg["system"].get(str(dataset_id), style_cfg["system"]["default"])

    prompts = [item["input"] for item in batch_data]
    for attempt in range(max_retries):
        try:
            texts = mgr.generate(prompts, system_prompt=system_prompt, max_new=VLLM_CONFIGS[model_name].get("max_new_tokens", 3072))
            break
        except Exception as e:
            print(f"[vLLM] {model_name} generate failed: {e} (retry {attempt+1})")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)

    results = []
    for item, text in zip(batch_data, texts):
        results.append({
            "model": model_name,
            "prompt_style": style_name,
            "response": text,
            "success": True,
            "idx": item["idx"],
        })
    return results

# Evaluation logic
def evaluate_model_responses(results, model_name, dataset_id=None):
    evaluation = {"model": model_name, "styles": {}, "overall": {"total": 0, "correct": 0}}
    for s in PROMPT_STYLES:
        evaluation["styles"][s] = {"total": 0, "correct": 0}

    for pr in results:
        plabel = pr["problem"].get("proof_label", "")
        if plabel and not (plabel.startswith("__") and plabel.endswith("__")):
            expected = f"__{plabel}__"
        else:
            expected = plabel
        for resp in pr["responses"]:
            if not resp["success"] or resp["model"] != model_name:
                continue
            style = resp["prompt_style"]
            text  = resp.get("response", "")
            found = "__UNKNOWN__"
            for mk in ["__PROVED__", "__DISPROVED__", "__UNKNOWN__", "__YES__", "__NO__", "DISPROVED", "PROVED",
                        "UNKNOWN", "YES", "NO"]:
                if mk in text:
                    found = mk if mk.startswith("__") else f"__{mk}__"
                    break
            evaluation["styles"][style]["total"] += 1
            evaluation["overall"]["total"] += 1
            if found == expected or expected == "__YES__" and found == "__PROVED__" or expected == "__NO__" and found == "__DISPROVED__":
                evaluation["styles"][style]["correct"] += 1
                evaluation["overall"]["correct"] += 1

    # Calculate accuracy
    for s in PROMPT_STYLES:
        st = evaluation["styles"][s]
        if st["total"]:
            st["accuracy"] = st["correct"] / st["total"]
    if evaluation["overall"]["total"]:
        evaluation["overall"]["accuracy"] = evaluation["overall"]["correct"] / evaluation["overall"]["total"]
    return evaluation

# Batch processor
def process_problem_prompts(problem_data: Dict[str, Any], dataset_id: int = None):
    in_data  = problem_data.get("input")
    facts    = problem_data.get("facts") or problem_data.get("original_data", {}).get("facts", "")
    hypo     = problem_data.get("hypothesis") or problem_data.get("original_data", {}).get("hypothesis", "")
    prompts  = {}

    for s, cfg in PROMPT_STYLES.items():
        suffix = cfg["suffix"]
        if s == "fewshot":
            fewshot_key = str(dataset_id) if dataset_id in [3, 4] and str(dataset_id) in FEWSHOT_EXAMPLES else "default"
            suffix = suffix.format(fewshot_examples=FEWSHOT_EXAMPLES[fewshot_key])
        if in_data:
            prompts[s] = in_data + "\n" + suffix
        else:
            prompts[s] = f"Facts: {facts}\nHypothesis: {hypo}\n{suffix}"
    
    return prompts

def process_problems_for_model(data: List[Dict], model_name: str, dataset_id: int = None):
    structured = [{
        "problem": p,
        "prompts": process_problem_prompts(p, dataset_id),
        "responses": []
    } for p in data]

    for style in PROMPT_STYLES:
        batch = [{"input": item["prompts"][style], "idx": idx} for idx, item in enumerate(structured)]
        responses = call_vllm_with_retry(model_name, batch, style, dataset_id)
        for r in responses:
            structured[r["idx"]]["responses"].append(r)
    return structured

# Main function
def main(model_name: str, dataset_ids: list[int], model_path: str):
    # Set model path
    VLLM_CONFIGS[model_name] = {"model_path": model_path}
    
    Datasets = {
        1: "../data/Dataset1-FLD.json",
        2: "../data/DataSet2-FOLIO.json",
        3: "../data/Dataset3-Multi-LogiEval.json",
        4: "../data/Dataset4-ProntoQA.json",
    }
    dataset1_path = ""
    for i in dataset_ids:
        json_name = Datasets[i]
        status_base = json_name[:-5]
        resuts_dir = '../results/' + json_name.split("/")[-1].split(".")[0]
        status = load_status(status_base)
        ts = status["timestamp"]
        if i == 1:
            dataset1_path = model_result_filename(model_name, ts, resuts_dir)
            print(f"dataset1_path: {dataset1_path}")
        if model_name in status["completed_models"]:
            print(f"[skip] {model_name} already done for dataset {i}")
            continue
        with open(json_name) as f:
            data = json.load(f)
        print(f"[RUN] {model_name} for {json_name}")

        results = process_problems_for_model(data, model_name, i)
        json.dump(results, open(model_result_filename(model_name, ts, resuts_dir), "w"), indent=2)

        eval_res = evaluate_model_responses(results, model_name, i)
        json.dump(eval_res, open(model_eval_filename(model_name, ts, resuts_dir), "w"), indent=2)
        
        combined_data = load_combined_eval(ts, status_base)
        combined_data[model_name] = eval_res
        save_combined_eval(combined_data, ts, status_base)
        
        status["completed_models"].append(model_name)
        save_status(status, status_base)
        print(f"[DONE] {model_name} for {json_name}")
        print("Stepwise Analysis:")
    # if '../results/step/detail_' + model_name + '.json' not exists:
    if not os.path.exists('../results/step/summary_' + model_name + '.json'):
        asyncio.run(run_pipeline(dataset1_path,
                                 '../results/step/detail_' + model_name + '.json',
                                 '../results/step/summary_' + model_name + '.json'))
    else:
        print("Stepwise Analysis already done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model name")
    parser.add_argument("--model_path", required=True, help="model checkpoint path")
    parser.add_argument("--datasets", nargs="+", type=int,
                        default=[1, 2, 3, 4], 
                        help="dataset ids 1‑4, space‑separated")
    args = parser.parse_args()

    main(args.model, args.datasets, args.model_path)