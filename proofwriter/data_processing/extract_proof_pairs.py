#!/usr/bin/env python3
"""
Extract (Natural Language, Proof Text) pairs from ProofWriter for CCA steering.

This script extracts parallel pairs of:
- Natural language questions/conclusions
- Corresponding natural language proof texts

These pairs can be used for CCA analysis similar to the (NL, FOL) pairs in ProntoQA.
"""

import json
import sys
from proof_to_text import ProofParser


class ConciseProofTextGenerator:
    """Generate concise natural language proof text (three-part format)."""

    def __init__(self, triples, rules, intermediates):
        """Initialize with context."""
        self.triples = triples
        self.rules = rules
        self.intermediates = intermediates

    def generate_proof_text(self, steps):
        """
        Generate concise proof text in three-part format:
        1. Premise1, premise2, and premise3. Rule text. Therefore conclusion.

        Args:
            steps: List of proof steps from parser

        Returns:
            Concise natural language proof text
        """
        lines = []
        derived = set()
        step_number = 1

        for step in steps:
            premises = step['premises']
            rule_id = step['rule']
            conclusion_id = step['conclusion']

            # Skip if already derived
            if conclusion_id in derived:
                continue

            # Get premise texts
            premise_texts = []
            for p in premises:
                if p.startswith('triple'):
                    text = self.triples[p]['text']
                    premise_texts.append(text)
                elif p.startswith('int'):
                    text = self.intermediates[p]['text']
                    premise_texts.append(text)
                else:
                    premise_texts.append(p)

            # Get rule text
            rule_text = self.rules[rule_id]['text'] if rule_id in self.rules else rule_id

            # Get conclusion text
            if conclusion_id.startswith('int'):
                conclusion_text = self.intermediates[conclusion_id]['text']
            else:
                conclusion_text = conclusion_id

            # Format in three-part structure
            # Part 1: Premises
            if len(premise_texts) == 1:
                premises_part = premise_texts[0].rstrip('.')
            elif len(premise_texts) == 2:
                premises_part = f"{premise_texts[0].rstrip('.')} and {premise_texts[1].rstrip('.')}"
            else:
                premises_part = ", ".join([p.rstrip('.') for p in premise_texts[:-1]]) + f", and {premise_texts[-1].rstrip('.')}"

            # Part 2: Rule
            rule_part = rule_text.rstrip('.')

            # Part 3: Conclusion
            conclusion_part = conclusion_text.rstrip('.')

            # Combine into single line
            line = f"{premises_part}. {rule_part}. Therefore {conclusion_part.lower()}."
            lines.append(line)

            derived.add(conclusion_id)
            step_number += 1

        return "\n".join(lines)


def extract_nl_proof_pairs(jsonl_file: str, output_file: str, min_depth: int = 1, max_depth: int = 3):
    """
    Extract (NL, Proof) pairs from ProofWriter dataset.

    Args:
        jsonl_file: Input JSONL file from ProofWriter
        output_file: Output file for (NL, Proof) pairs
        min_depth: Minimum proof depth to include
        max_depth: Maximum proof depth to include
    """
    pairs = []

    with open(jsonl_file, 'r') as f:
        for line in f:
            example = json.loads(line)

            for q_id, question in example['questions'].items():
                # Skip if not in desired depth range
                depth = question.get('QDep', 0)
                if depth < min_depth or depth > max_depth:
                    continue

                # Skip if no proof (allow both True and False answers)
                if 'answer' not in question or not question.get('proofsWithIntermediates'):
                    continue

                # Only process True/False, skip Unknown
                answer = question['answer']
                if answer not in [True, False]:
                    continue

                try:
                    # Get the natural language conclusion
                    nl_conclusion = question['question']

                    # Parse the proof
                    proof_data = question['proofsWithIntermediates'][0]
                    proof_repr = proof_data['representation']
                    intermediates = proof_data['intermediates']

                    parser = ProofParser(example['triples'], example['rules'], intermediates)
                    steps = parser.parse_proof(proof_repr)

                    generator = ConciseProofTextGenerator(example['triples'], example['rules'], intermediates)
                    proof_text = generator.generate_proof_text(steps)

                    # Extract all facts and rules from the theory
                    facts = [triple['text'] for triple in example['triples'].values()]
                    rules = [rule['text'] for rule in example['rules'].values()]

                    # Create pair with answer label
                    pair = {
                        'id': f"{example['id']}_{q_id}",
                        'nl_conclusion': nl_conclusion,
                        'proof_text': proof_text,
                        'answer': answer,  # True or False label
                        'depth': depth,
                        'theory_id': example['id'],
                        'facts': facts,
                        'rules': rules
                    }
                    pairs.append(pair)

                except Exception as e:
                    print(f"Error processing {example['id']}/{q_id}: {e}", file=sys.stderr)

    # Save pairs
    with open(output_file, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"Extracted {len(pairs)} (NL, Proof) pairs")
    print(f"Saved to {output_file}")

    # Print statistics
    depths = {}
    for pair in pairs:
        depth = pair['depth']
        depths[depth] = depths.get(depth, 0) + 1

    print("\nDistribution by proof depth:")
    for depth in sorted(depths.keys()):
        print(f"  Depth {depth}: {depths[depth]} pairs")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract (NL, Proof) pairs from ProofWriter for CCA analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:

  # Extract all proofs with depth 1-3
  python extract_proof_pairs.py meta-train.jsonl nl_proof_pairs.jsonl

  # Extract only depth 2-3 proofs
  python extract_proof_pairs.py meta-train.jsonl pairs.jsonl --min-depth 2 --max-depth 3

This creates a dataset similar to ProntoQA's (NL, FOL) pairs, but with:
- NL: Natural language conclusion
- Proof: Natural language proof text

These can be used for:
1. CCA analysis between NL representations and proof representations
2. Activation steering to guide proof generation
3. Analysis of how models represent multi-step reasoning
        '''
    )

    parser.add_argument('input_file', help='Input JSONL file from ProofWriter')
    parser.add_argument('output_file', help='Output JSONL file for (NL, Proof) pairs')
    parser.add_argument('--min-depth', type=int, default=1,
                       help='Minimum proof depth (default: 1)')
    parser.add_argument('--max-depth', type=int, default=3,
                       help='Maximum proof depth (default: 3)')

    args = parser.parse_args()

    extract_nl_proof_pairs(args.input_file, args.output_file,
                          args.min_depth, args.max_depth)


if __name__ == '__main__':
    main()
