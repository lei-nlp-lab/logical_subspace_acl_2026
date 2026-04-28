import asyncio
import aiohttp
import time
import os
import json
import random
import re
import numpy as np
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================== Configuration Area ================================

API_CONFIGS = {
    "gpt-4o": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "paste your api key"
    }
}

#, or \"__UNKNOWN__\" if uncertain.
PROMPT_STYLES = {
    "direct": {
        "system": "Based on the provided facts, answer the question. Conclude with one of the markers: \"__YES__\" for proven, \"__NO__\" for disproven",
        "suffix": ""
    },
    "cot": {
        "system": "Based on the provided facts, answer the question. Conclude with one of the markers: \"__YES__\" for proven, \"__NO__\" for disproven",
        "suffix": "Let's analyze this step by step."
    },
    "fewshot": {
        "system": "Based on the provided facts, answer the question. Conclude with one of the markers: \"__YES__\" for proven, \"__NO__\" for disproven",
        "suffix": "{fewshot_examples}"
    }
}

FEWSHOT_EXAMPLES = """Here are some examples of proofs for your reference:
[Start of example]
For example, for this question:
"PASTE AN EXAMPLE QUESTION AND SOLUTION HERE"
[End of example]
You can refer to the proof method of the above question, think step by step, and give the result of this question.
"""

# ======================== Utility Area ================================
def model_result_filename(model_name, timestamp,filename):
    return f'{filename}_results_{model_name.replace("/", "_")}_{timestamp}.json'

def model_eval_filename(model_name, timestamp,filename):
    return f'{filename}_evaluation_{model_name.replace("/", "_")}_{timestamp}.json'

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

# ======================== Core API Logic ============================

def call_model_api_with_retry(model_name, facts=None, hypothesis=None, prompt_style=None, max_retries=3, input_data=None):
    config = API_CONFIGS[model_name]
    style = PROMPT_STYLES[prompt_style] if prompt_style else PROMPT_STYLES["direct"]
    suffix = style["suffix"]
    if prompt_style == "fewshot":
        suffix = suffix.format(fewshot_examples=FEWSHOT_EXAMPLES)
    
    # Prioritize input_data, only use facts and hypothesis when input_data is not available
    if input_data:
        user_message = input_data+"\n"+suffix
    elif facts is not None and hypothesis is not None:
        user_message = f"Facts: {facts}\nHypothesis: {hypothesis}\n{suffix}"
    else:
        user_message = suffix  # If no valid input, only use suffix
        
    for attempt in range(max_retries):
        try:
            from openai import OpenAI
            if model_name in ["gpt-4o", "gpt-4.1"]:
                openai = OpenAI(
                    api_key=config['api_key']
                )
            else:
                openai = OpenAI(
                    api_key=config['api_key'],
                    base_url=config.get('base_url', 'api.openai.com')
                )
            response = openai.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": style["system"]},
                    {"role": "user", "content": user_message},
                ]
            )
            return {
                "model": model_name,
                "prompt_style": prompt_style,
                "response": response.choices[0].message.content,
                "success": True
            }
        except Exception as e:
            print(f"Exception for model {model_name}: {e}. Retrying...")
            time.sleep(1)
    return {
        "model": model_name,
        "prompt_style": prompt_style,
        "error": "Max retries exceeded",
        "success": False
    }

def process_problem(problem_data, model_name):
    # Check if direct input data exists (same level as original_data)
    input_data = problem_data.get("input")
    
    # Only try to extract facts and hypothesis when there's no input
    facts = None
    hypothesis = None
    proof_label = None
    
    if not input_data:
        # Extract facts and hypothesis
        if "original_data" in problem_data:
            facts = problem_data["original_data"].get("facts", "")
            hypothesis = problem_data["original_data"].get("hypothesis", "")
        else:
            facts = problem_data.get("facts", "")
            hypothesis = problem_data.get("hypothesis", "")
    
    # Try to get proof_label regardless of whether there's input or not
    proof_label = problem_data.get("proof_label")
    if proof_label is None and "original_data" in problem_data:
        proof_label = problem_data["original_data"].get("proof_label")
    
    styles = list(PROMPT_STYLES.keys())
    # Synchronous version, run each style one by one
    model_responses = []
    for style in styles:
        model_responses.append(
            call_model_api_with_retry(model_name, facts, hypothesis, style, input_data=input_data)
        )
    
    prompts = {}
    for style in PROMPT_STYLES:
        suffix = PROMPT_STYLES[style]["suffix"]
        if style == "fewshot":
            suffix = suffix.format(fewshot_examples=FEWSHOT_EXAMPLES)
        
        # If there's direct input data, use it; otherwise, construct standard prompt
        if input_data:
            prompts[style] = input_data+"\n"+suffix
        elif facts is not None and hypothesis is not None:
            prompts[style] = f"Facts: {facts}\nHypothesis: {hypothesis}\n{suffix}"
        else:
            prompts[style] = suffix
    
    return {
        "problem": {
            "facts": facts,
            "hypothesis": hypothesis,
            "proof_label": proof_label,
            "input": input_data  # Add input to the result to preserve original input
        },
        "prompts": prompts,
        "responses": model_responses
    }

def process_problems_for_model(data, model_name, max_workers=100):
    # Use concurrent processing for API models
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map problems to futures to maintain order
        future2problem = {executor.submit(process_problem, p, model_name): i for i, p in enumerate(data)}
        for future in tqdm(as_completed(future2problem), total=len(future2problem), desc=f'Processing({model_name})'):
            idx = future2problem[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"Problem {idx} generated an exception: {exc}")
                result = None
            results.append(result)
    # results are not in order originally, but it doesn't matter
    results = [r for r in results if r is not None]
    return results

def evaluate_model_responses(results, model_name):
    evaluation = {
        "model": model_name,
        "styles": {},
        "overall": {"total": 0, "correct": 0}
    }
    for style in PROMPT_STYLES.keys():
        evaluation["styles"][style] = {"total": 0, "correct": 0}
    for problem_result in results:
        proof_label = problem_result["problem"]["proof_label"]
        # If proof_label doesn't have underscores before and after, add them; otherwise, don't add
        if proof_label and not (proof_label.startswith("__") and proof_label.endswith("__")):
            expected_marker = f"__{proof_label}__"
        else:
            expected_marker = f"{proof_label}"
        for response_data in problem_result["responses"]:
            if not response_data["success"] or response_data["model"] != model_name:
                continue
            style = response_data["prompt_style"]
            response_text = response_data.get("response", "")
            found_marker = None
            for marker in ["__PROVED__", "__DISPROVED__", "__UNKNOWN__", "__YES__", "__NO__"]:
                if marker in response_text:
                    found_marker = marker
                    break
            # is_correct = found_marker == expected_marker
            if found_marker == expected_marker or expected_marker == "__YES__" and found_marker == "__PROVED__" or expected_marker == "__NO__" and found_marker == "__DISPROVED__":
                is_correct = True
            else:
                is_correct = False
            evaluation["styles"][style]["total"] += 1
            evaluation["overall"]["total"] += 1
            if is_correct:
                evaluation["styles"][style]["correct"] += 1
                evaluation["overall"]["correct"] += 1
    for style in PROMPT_STYLES.keys():
        if evaluation["styles"][style]["total"] > 0:
            evaluation["styles"][style]["accuracy"] = evaluation["styles"][style]["correct"] / evaluation["styles"][style]["total"]
    if evaluation["overall"]["total"] > 0:
        evaluation["overall"]["accuracy"] = evaluation["overall"]["correct"] / evaluation["overall"]["total"]
    return evaluation

# ======================= Main Process ======================
def main():
    json_name="Dataset1-FLD.json" # Set your json name here
    with open(json_name, 'r') as f:
        data = json.load(f)
    # Remove .json from json_name
    status = load_status(json_name[:-5])
    timestamp = status['timestamp']
    already = set(status['completed_models'])
    all_models = list(API_CONFIGS.keys())
    
    for model_name in all_models:
        if model_name in already:
            print(f"{model_name} already completed, skipping.")
            continue
        print(f"Starting to process model: {model_name}")
        
        # Use the processing function for API models
        results = process_problems_for_model(data, model_name, max_workers=100)
        
        # Save
        with open(model_result_filename(model_name, timestamp, json_name[:-5]), 'w') as f:
            json.dump(results, f, indent=2)
        evaluation = evaluate_model_responses(results, model_name)
        with open(model_eval_filename(model_name, timestamp, json_name[:-5]), 'w') as f:
            json.dump(evaluation, f, indent=2)
        # Record completed models
        status['completed_models'].append(model_name)
        save_status(status,json_name[:-5])
        print(f"{model_name} processing complete, saved.")
    
    print("All models processing complete.")

# If this script is run directly rather than as a module
if __name__ == "__main__":
    main()