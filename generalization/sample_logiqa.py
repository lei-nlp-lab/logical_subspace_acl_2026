"""
Script to randomly sample 500 examples from LogiQA test.txt
"""

import json
import random
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Sample examples from LogiQA test set")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to LogiQA test.txt (download from https://github.com/csitfun/LogiQA2.0)")
    parser.add_argument("--output", type=str,
                        default="generalization/data/logiqa_500_sample.jsonl",
                        help="Path to output sampled file")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of samples to draw")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Read all lines from input file
    with open(args.input, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Total examples in {args.input}: {len(lines)}")

    # Validate n_samples
    if args.n_samples > len(lines):
        print(f"Warning: requested {args.n_samples} samples but only {len(lines)} available. Using all.")
        args.n_samples = len(lines)

    # Randomly sample
    sampled_lines = random.sample(lines, args.n_samples)
    print(f"Sampled {len(sampled_lines)} examples")

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write sampled lines to output file
    with open(args.output, "w", encoding="utf-8") as f:
        for line in sampled_lines:
            f.write(line + "\n")

    print(f"Saved sampled examples to {args.output}")

    # Verify the saved file
    with open(args.output, "r", encoding="utf-8") as f:
        saved_count = sum(1 for _ in f)
    print(f"Verification: {saved_count} lines written")

if __name__ == "__main__":
    main()
