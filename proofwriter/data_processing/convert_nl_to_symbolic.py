#!/usr/bin/env python3
"""
将 ProofWriter 的自然语言证明转换为符号化证明（TPTP-style）

使用 LiteLLM API 调用 LLM 进行转换
"""

import json
import argparse
import os
from openai import OpenAI
from tqdm import tqdm


def load_prompt_template(prompt_file):
    """加载 prompt 模板"""
    with open(prompt_file, 'r') as f:
        return f.read()


def convert_single_proof(client, nl_proof, prompt_template, model="gpt-4", temperature=0.3):
    """
    转换单个自然语言证明为符号化证明

    Args:
        client: OpenAI client
        nl_proof: 自然语言证明文本
        prompt_template: prompt 模板
        model: 使用的模型
        temperature: 采样温度 (0.0=确定性, 1.0=高随机性)

    Returns:
        symbolic_proof: 符号化证明列表
    """
    # 构造完整 prompt
    full_prompt = prompt_template.replace(
        "<<<INSERT_PROOFWRITER_PROOF_HERE>>>",
        nl_proof
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in formal logic and TPTP notation. Convert natural language proofs to symbolic form accurately."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            temperature=temperature,
            max_tokens=2000
        )

        # 解析响应
        response_text = response.choices[0].message.content.strip()

        # 尝试提取 JSON
        # 有时模型会在 JSON 前后加说明文字，需要提取
        if "```json" in response_text:
            # 提取 ```json ... ``` 中的内容
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "{" in response_text:
            # 提取第一个 { 到最后一个 }
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
        else:
            raise ValueError(f"No JSON found in response: {response_text}")

        result = json.loads(json_str)
        symbolic_proof = result.get("symbolic_proof", "")

        # 验证格式：应该是字符串，包含推理步骤
        if not isinstance(symbolic_proof, str):
            print(f"Warning: symbolic_proof is not a string: {type(symbolic_proof)}")
            return str(symbolic_proof) if symbolic_proof else ""

        return symbolic_proof

    except Exception as e:
        print(f"Error converting proof: {e}")
        print(f"Response: {response_text if 'response_text' in locals() else 'N/A'}")
        return None


def batch_convert_proofs(input_file, output_file, api_key, base_url, prompt_file, model="gpt-4", temperature=0.3):
    """
    批量转换证明

    Args:
        input_file: 输入 JSONL 文件（ProofWriter extract_proof_pairs.py 的输出）
        output_file: 输出 JSONL 文件（添加 symbolic_proof 字段）
        api_key: API key
        base_url: API base URL
        prompt_file: Prompt 模板文件
        model: 使用的模型
        temperature: 采样温度 (默认 0.3)
    """
    # 初始化 client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # 加载 prompt
    prompt_template = load_prompt_template(prompt_file)

    # 读取输入数据
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # 转换
    results = []
    success_count = 0
    fail_count = 0

    for item in tqdm(data, desc="Converting proofs"):
        nl_proof = item['proof_text']

        # 调用 API 转换
        symbolic_proof = convert_single_proof(
            client,
            nl_proof,
            prompt_template,
            model,
            temperature
        )

        if symbolic_proof is not None:
            item['symbolic_proof'] = symbolic_proof
            success_count += 1
        else:
            item['symbolic_proof'] = ""
            fail_count += 1

        results.append(item)

    # 保存结果
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✅ Conversion complete!")
    print(f"   Success: {success_count}")
    print(f"   Failed: {fail_count}")
    print(f"   Output: {output_file}")


def test_single_example(api_key, base_url, prompt_file, model="gpt-4", temperature=0.3):
    """测试单个例子"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt_template = load_prompt_template(prompt_file)

    # 测试例子
    test_proof = """Harry is green. If someone is green then they are young. Therefore harry is young.
Harry is blue, harry is green, and harry is young. If someone is blue and green and young then they are kind. Therefore harry is kind."""

    print("Testing with example proof:")
    print(test_proof)
    print(f"\nConverting... (temperature={temperature})")

    symbolic_proof = convert_single_proof(
        client,
        test_proof,
        prompt_template,
        model,
        temperature
    )

    print("\nResult:")
    print(json.dumps({"symbolic_proof": symbolic_proof}, indent=2, ensure_ascii=False))

    # 也打印格式化的版本（每行一个推理步骤）
    if symbolic_proof:
        print("\nFormatted (each line = one inference step):")
        for i, line in enumerate(symbolic_proof.split('\n'), 1):
            print(f"  Step {i}: {line}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert ProofWriter NL proofs to symbolic (TPTP-style)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:

  # Test with a single example
  python convert_nl_to_symbolic.py \\
    --test \\
    --api-key $API_KEY \\
    --base-url http://localhost:4000 \\
    --prompt-file prompt_nl_to_symbolic.txt

  # Batch convert (default temperature=0.3)
  python convert_nl_to_symbolic.py \\
    --input nl_proof_pairs.jsonl \\
    --output symbolic_proof_pairs.jsonl \\
    --api-key $API_KEY \\
    --base-url http://localhost:4000 \\
    --prompt-file prompt_nl_to_symbolic.txt \\
    --model gpt-4 \\
    --temperature 0.3

  # Use higher temperature for more diversity
  python convert_nl_to_symbolic.py \\
    --input nl_proof_pairs.jsonl \\
    --output symbolic_proof_pairs.jsonl \\
    --api-key $API_KEY \\
    --base-url http://localhost:4000 \\
    --prompt-file prompt_nl_to_symbolic.txt \\
    --model gpt-4 \\
    --temperature 0.7
        '''
    )

    parser.add_argument('--input', type=str,
                       help='Input JSONL file (from extract_proof_pairs.py)')
    parser.add_argument('--output', type=str,
                       help='Output JSONL file (with symbolic_proof field)')
    parser.add_argument('--api-key', type=str,
                       default=os.getenv('OPENAI_API_KEY', ''),
                       help='API key for LiteLLM (default: env OPENAI_API_KEY)')
    parser.add_argument('--base-url', type=str,
                       default=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                       help='Base URL for LiteLLM (default: env OPENAI_BASE_URL or https://api.openai.com/v1)')
    parser.add_argument('--prompt-file', type=str, required=True,
                       help='Prompt template file')
    parser.add_argument('--model', type=str, default='gpt-5',
                       help='Model to use (default: gpt-5)')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Sampling temperature (0.0=deterministic, 1.0=high randomness, default: 0.3)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: convert a single example')

    args = parser.parse_args()

    if args.test:
        # 测试模式
        test_single_example(
            args.api_key,
            args.base_url,
            args.prompt_file,
            args.model,
            args.temperature
        )
    else:
        # 批量转换模式
        if not args.input or not args.output:
            print("Error: --input and --output required for batch mode")
            return

        batch_convert_proofs(
            args.input,
            args.output,
            args.api_key,
            args.base_url,
            args.prompt_file,
            args.model,
            args.temperature
        )


if __name__ == '__main__':
    main()
