#!/usr/bin/env python3
"""
ProofWriter Proof to Natural Language Text Generator

Parses ProofWriter's Polish notation proof format and generates natural language proof text.
Based on the official proof linearization syntax from the ProofWriter paper (Section 3.4):
- '%' connects rule→conclusion
- '&' expresses conjunctive premises
- '#' denotes inverse implication (←)
- Polish notation for tree structure
"""

import json
import re
from typing import Dict, List, Tuple, Any


class ProofParser:
    """Parser for ProofWriter proof representations."""

    def __init__(self, triples: Dict, rules: Dict, intermediates: Dict):
        """
        Initialize parser with context.

        Args:
            triples: Dictionary of fact IDs to fact objects
            rules: Dictionary of rule IDs to rule objects
            intermediates: Dictionary of intermediate conclusion IDs to objects
        """
        self.triples = triples
        self.rules = rules
        self.intermediates = intermediates

    def parse_proof(self, proof_str: str) -> List[Dict[str, Any]]:
        """
        Parse a proof string and return a list of proof steps in topological order.

        Args:
            proof_str: Polish notation proof string

        Returns:
            List of proof steps, each containing premises, rule, and conclusion
        """
        steps = []
        self._parse_expression(proof_str, steps)
        return steps

    def _parse_expression(self, expr: str, steps: List[Dict]) -> str:
        """
        Recursively parse an expression, adding steps as we go.

        Returns:
            The conclusion ID of this expression
        """
        expr = expr.strip()

        # Remove outer parentheses
        if expr.startswith('(') and expr.endswith(')'):
            # Check if these are the outermost parentheses
            depth = 0
            for i, char in enumerate(expr):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    # Not the outermost
                    break
            else:
                # These are the outermost
                expr = expr[1:-1]

        # Base case: single fact/triple/intermediate reference
        if not ' ' in expr and not '(' in expr and not '->' in expr:
            return expr

        # Find the main '->' at depth 0
        arrow_idx = self._find_main_arrow(expr)

        if arrow_idx == -1:
            # No arrow found, this might just be a fact reference
            return expr

        # Split into premises and rule/conclusion
        premises_part = expr[:arrow_idx].strip()
        rule_conc_part = expr[arrow_idx+2:].strip()

        # Remove outer parentheses from premises if present
        if premises_part.startswith('(') and premises_part.endswith(')'):
            depth = 0
            for i, char in enumerate(premises_part):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                if depth == 0 and i < len(premises_part) - 1:
                    # Not the outermost
                    break
            else:
                # These are the outermost
                premises_part = premises_part[1:-1]

        # Parse rule and conclusion from (rule % conclusion)
        if rule_conc_part.startswith('(') and rule_conc_part.endswith(')'):
            rule_conc_part = rule_conc_part[1:-1]

        parts = rule_conc_part.split('%')
        if len(parts) != 2:
            print(f"Warning: Could not parse rule/conclusion: {rule_conc_part}")
            return expr

        rule_id = parts[0].strip()
        conclusion_id = parts[1].strip()

        # Parse premises recursively
        premise_ids = []
        premise_tokens = self._tokenize_at_depth_0(premises_part)

        for token in premise_tokens:
            if '->' in token:
                # This is a nested rule application
                prem_id = self._parse_expression(token, steps)
                premise_ids.append(prem_id)
            else:
                premise_ids.append(token.strip())

        # Create proof step
        step = {
            'premises': premise_ids,
            'rule': rule_id,
            'conclusion': conclusion_id
        }
        steps.append(step)

        return conclusion_id

    def _find_main_arrow(self, s: str) -> int:
        """Find the index of the main '->' at depth 0."""
        depth = 0
        for i in range(len(s) - 1):
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
            elif depth == 0 and s[i:i+2] == '->':
                return i
        return -1

    def _tokenize_at_depth_0(self, s: str) -> List[str]:
        """Tokenize string by spaces at depth 0, respecting parentheses."""
        tokens = []
        current = ""
        depth = 0

        for char in s:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ' ' and depth == 0:
                if current.strip():
                    tokens.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            tokens.append(current.strip())

        return tokens


class ProofTextGenerator:
    """Generate natural language proof text from parsed proof steps."""

    def __init__(self, triples: Dict, rules: Dict, intermediates: Dict):
        """Initialize with context."""
        self.triples = triples
        self.rules = rules
        self.intermediates = intermediates

    def generate_proof_text(self, steps: List[Dict], verbose: bool = True) -> str:
        """
        Generate natural language proof text.

        Args:
            steps: List of proof steps from parser
            verbose: If True, show premises explicitly; if False, just show reasoning

        Returns:
            Natural language proof text
        """
        lines = []

        # Keep track of which intermediates we've already derived
        derived = set()
        step_number = 1

        for step in steps:
            premises = step['premises']
            rule_id = step['rule']
            conclusion_id = step['conclusion']

            # Skip if we've already derived this conclusion
            if conclusion_id in derived:
                continue

            # Get premise texts
            premise_texts = []
            for p in premises:
                if p.startswith('triple'):
                    text = self.triples[p]['text']
                    premise_texts.append(text)
                elif p.startswith('int'):
                    # Reference to a previously derived intermediate
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

            # Format the step - more concise version
            # Just show the inference with references to the premises
            if len(premise_texts) == 1:
                premises_str = premise_texts[0].rstrip('.')
            elif len(premise_texts) == 2:
                premises_str = f"{premise_texts[0].rstrip('.')} and {premise_texts[1].rstrip('.')}"
            else:
                premises_str = ", ".join([p.rstrip('.') for p in premise_texts[:-1]]) + f", and {premise_texts[-1].rstrip('.')}"

            line = f"{step_number}. Since {premises_str.lower()}, by the rule \"{rule_text}\", we can conclude that {conclusion_text.lower()}"
            lines.append(line)

            derived.add(conclusion_id)
            step_number += 1

        return "\n".join(lines)


def convert_proof_to_text(example: Dict, question_id: str) -> str:
    """
    Convert a ProofWriter question's proof to natural language text.

    Args:
        example: Full example from ProofWriter dataset
        question_id: Question ID (e.g., 'Q5')

    Returns:
        Natural language proof text
    """
    question = example['questions'][question_id]

    if not question['answer'] or question['QDep'] == 0:
        return f"Answer: {question['question']} - This is directly stated in the facts (depth 0)."

    proofs_with_intermediates = question['proofsWithIntermediates']
    if not proofs_with_intermediates:
        return f"Answer: {question['question']} - No proof available."

    # Use the first proof
    proof = proofs_with_intermediates[0]
    proof_repr = proof['representation']
    intermediates = proof['intermediates']

    # Parse and generate
    parser = ProofParser(example['triples'], example['rules'], intermediates)
    steps = parser.parse_proof(proof_repr)

    generator = ProofTextGenerator(example['triples'], example['rules'], intermediates)
    proof_text = generator.generate_proof_text(steps)

    return f"Question: {question['question']}\n\nProof:\n{proof_text}\n\nTherefore, {question['question']}"


def batch_convert_proofs(jsonl_file: str, output_file: str = None, max_examples: int = None):
    """
    Convert all proofs in a JSONL file to natural language text.

    Args:
        jsonl_file: Path to input JSONL file
        output_file: Path to output JSONL file (optional)
        max_examples: Maximum number of examples to process (optional)
    """
    results = []

    with open(jsonl_file, 'r') as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break

            example = json.loads(line)
            example_result = {
                'id': example.get('id', f'example_{i}'),
                'questions': {}
            }

            for q_id, question in example['questions'].items():
                if question.get('answer') and question.get('QDep', 0) > 0:
                    try:
                        proof_text = convert_proof_to_text(example, q_id)
                        example_result['questions'][q_id] = {
                            'question': question['question'],
                            'answer': question['answer'],
                            'depth': question.get('QDep', 0),
                            'proof_text': proof_text
                        }
                    except Exception as e:
                        print(f"Error processing {example_result['id']}/{q_id}: {e}")

            results.append(example_result)

    if output_file:
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"Wrote {len(results)} examples to {output_file}")

    return results


def main():
    """Example usage."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert ProofWriter proofs to natural language text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Convert a single question from a file
  python proof_to_text.py meta-train.jsonl -q Q5

  # Convert all proofs in a file (batch mode)
  python proof_to_text.py meta-train.jsonl --batch --max 10

  # Convert all proofs and save to output file
  python proof_to_text.py meta-train.jsonl --batch -o output.jsonl
        '''
    )

    parser.add_argument('input_file', help='Path to input JSONL file')
    parser.add_argument('-q', '--question', default='Q5',
                       help='Question ID to process (default: Q5)')
    parser.add_argument('--batch', action='store_true',
                       help='Process all examples in batch mode')
    parser.add_argument('-o', '--output',
                       help='Output file for batch mode')
    parser.add_argument('--max', type=int,
                       help='Maximum number of examples to process in batch mode')
    parser.add_argument('--show-original', action='store_true',
                       help='Show original proof representation')

    args = parser.parse_args()

    if args.batch:
        # Batch processing mode
        print(f"Processing {args.input_file} in batch mode...")
        results = batch_convert_proofs(args.input_file, args.output, args.max)

        # Print summary
        total_questions = sum(len(r['questions']) for r in results)
        print(f"\nProcessed {len(results)} examples with {total_questions} questions")

        # Show a few examples
        if not args.output:
            print("\nFirst few examples:")
            for i, result in enumerate(results[:3]):
                print(f"\n{'='*80}")
                print(f"Example {i+1}: {result['id']}")
                for q_id, q_data in list(result['questions'].items())[:2]:
                    print(f"\n{q_data['proof_text']}\n")
    else:
        # Single question mode
        with open(args.input_file, 'r') as f:
            example = json.loads(f.readline())

        # Generate proof text
        proof_text = convert_proof_to_text(example, args.question)
        print(proof_text)

        if args.show_original:
            print("\n" + "="*80 + "\n")
            question = example['questions'][args.question]
            if question['proofsWithIntermediates']:
                print("Original proof representation:")
                print(question['proofsWithIntermediates'][0]['representation'])


if __name__ == '__main__':
    main()
