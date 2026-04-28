import json
import os
from collections import defaultdict

def load_json_file(filename):
    """Helper function to load a JSON file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_proof_accuracy(results_file, dataset_file):
    """
    Analyzes proof accuracy from two JSON files.

    Args:
        results_file (str): Path to the JSON file containing model results
                            (e.g., Dataset1-FLD_results_Qwen_Qwen2.5-7B-Instruct_1746610537.json).
        dataset_file (str): Path to the JSON file containing original dataset information
                            (e.g., Dataset1-FLD.json).
                            
    Returns:
        dict: Dictionary containing the accuracy statistics
    """
    results_data = load_json_file(results_file)
    dataset_data = load_json_file(dataset_file)  # Debugging: print first 5 entries
    # Create a lookup for dataset_file based on 'input' for faster access
    dataset_lookup = {item['input']: item for item in dataset_data}
    # Structure to store counts:
    # stats[step_category][model_prompt_style]['correct']
    # stats[step_category][model_prompt_style]['total']
    stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

    # --- 1. Process data and categorize by steps ---
    for result_entry in results_data:
        problem_input = result_entry['problem']['input']
        ground_truth_label = result_entry['problem']['proof_label']

        # Find corresponding entry in dataset_file
        dataset_entry = dataset_lookup.get(problem_input)

        if not dataset_entry:
            print(f"Warning: Input not found in dataset_file: {problem_input[:50]}...")
            continue

        original_tree_steps = dataset_entry['original_data'].get('steps')
        original_proof_label = dataset_entry['original_data'].get('proof_label')

        step_category = None
        if original_proof_label in ["__UNKNOWN__", "UNKNOWN"]:
            step_category = "UNKNOWN"
        elif original_tree_steps is not None:
            try:
                step_category = int(original_tree_steps)
            except ValueError:
                print(f"Warning: Could not parse original_tree_steps '{original_tree_steps}' as int for input: {problem_input[:50]}...")
                step_category = "INVALID_STEP" # Or handle as UNKNOWN or skip
        else:
            print(f"Warning: original_tree_steps not found for input: {problem_input[:50]}...")
            step_category = "MISSING_STEP" # Or handle as UNKNOWN or skip

        if step_category is None:
            continue # Skip if we couldn't determine a category

        # --- 2. Match responses and check correctness ---
        for response_item in result_entry['responses']:
            model_name = response_item['model']
            prompt_style = response_item['prompt_style']
            model_response_text = response_item['response']

            model_prompt_key = f"{model_name}_{prompt_style}"

            # Check if the ground_truth_label is present in the model's response
            is_correct = ground_truth_label in model_response_text

            stats[step_category][model_prompt_key]['total'] += 1
            if is_correct:
                stats[step_category][model_prompt_key]['correct'] += 1
    
    # --- Grouped Step Accuracies ---
    grouped_step_results = {}
    
    step_groups = [
        ("0-3", range(0, 4)),
        ("4-7", range(4, 8)),
        ("8-11", range(8, 12)),
        ("12-15", range(12, 16)),
        ("16-19", range(16, 20))
    ]

    # Collect all unique model_prompt_keys for consistent reporting
    all_model_prompt_keys = set()
    for step_cat_data in stats.values():
        for mpk in step_cat_data.keys():
            all_model_prompt_keys.add(mpk)
    sorted_model_prompt_keys = sorted(list(all_model_prompt_keys))

    # Initialize the model-centric result structure
    model_results = {model_key: {} for model_key in sorted_model_prompt_keys}
    
    # Fill in the step group accuracies for each model
    for group_name, step_range in step_groups:
        for model_prompt_key in sorted_model_prompt_keys:
            group_correct = 0
            group_total = 0
            for step in step_range:
                if step in stats and model_prompt_key in stats[step]:
                    group_correct += stats[step][model_prompt_key]['correct']
                    group_total += stats[step][model_prompt_key]['total']

            accuracy = (group_correct / group_total) * 100 if group_total > 0 else 0
            model_results[model_prompt_key][group_name] = round(accuracy, 2)
    
    # UNKNOWN category
    if "UNKNOWN" in stats:
        for model_prompt_key in sorted_model_prompt_keys:
            if model_prompt_key in stats["UNKNOWN"]:
                data = stats["UNKNOWN"][model_prompt_key]
                total = data['total']
                correct = data['correct']
                accuracy = (correct / total) * 100 if total > 0 else 0
                model_results[model_prompt_key]["UNKNOWN"] = round(accuracy, 2)
            else:
                model_results[model_prompt_key]["UNKNOWN"] = 0.0
    
    return model_results

def print_results(results):
    """Print the results in a formatted way."""
    for model_name, step_accuracies in results.items():
        print(f"\nModel: {model_name}")
        for step_group, accuracy in step_accuracies.items():
            print(f"  {step_group}: {accuracy}%")

def process_all_results_files(comma_only=False):
    """
    Process all files in the ../results/ folder that start with 'Dataset1-FLD_results_'
    and store the results in a JSON file.
    
    Args:
        comma_only (bool): If True, only process files with comma in the filename.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    output_dir = os.path.join(results_dir, "step")
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_json_file = os.path.join(data_dir, 'Dataset1-FLD.json')
    
    # Check if the dataset file exists
    if not os.path.exists(dataset_json_file):
        print(f"Error: File not found - {dataset_json_file}")
        return
    
    # Find all results files
    all_files = [f for f in os.listdir(results_dir) if f.startswith('Dataset1-FLD_results_') and f.endswith('.json')]
    
    # Filter files based on comma_only parameter
    if comma_only:
        results_files = [f for f in all_files if ',' in f]
        print(f"Only processing files with comma in filename, found {len(results_files)} files")
    else:
        results_files = all_files
    
    if not results_files:
        print(f"Error: No qualifying result files found in {results_dir}")
        return
    
    all_results = {}
    
    for results_file in results_files:
        try:
            file_path = os.path.join(results_dir, results_file)
            print(f"Processing file: {results_file}")
            
            # Extract the model name from the filename
            # Format: Dataset1-FLD_results_MODEL_TIMESTAMP.json
            parts = results_file.split('_')
            if len(parts) >= 4:  # Make sure there are enough parts
                # The model name is everything between 'results_' and the timestamp
                model_key = '_'.join(parts[2:-1])
                
                # Process the file
                model_results = analyze_proof_accuracy(file_path, dataset_json_file)
                
                # Extract current model results from model_results
                current_model_results = {}
                for model_name in model_results:
                    if model_name.startswith(model_key):
                        current_model_results[model_name] = model_results[model_name]
                
                # Update all results
                all_results.update(model_results)
                
                # Print current file results
                print(f"\nResults - {model_key}:")
                print_results(current_model_results)
                print("\n" + "="*80 + "\n")
            else:
                print(f"Warning: Cannot extract model name from filename: {results_file}")
        except Exception as e:
            import traceback
            print(f"Error processing file {results_file}: {str(e)}")
            print(traceback.format_exc())
    
    # Set different output filename based on comma_only parameter
    output_filename = 'comma_model_accuracy_results.json' if comma_only else 'model_accuracy_results.json'
    output_file = os.path.join(output_dir, output_filename)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"All results saved to: {output_file}")

if __name__ == "__main__":
    import sys
    # Check command line arguments to decide whether to process only comma files
    comma_only = len(sys.argv) > 1 and sys.argv[1].lower() in ('comma', 'comma_only', 'true', '1')
    process_all_results_files(comma_only=comma_only)