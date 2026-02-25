#!/usr/bin/env python3
"""
SwarmRouter-v1 Evaluation Script
Tests routing accuracy and JSON compliance.
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from unsloth import FastLanguageModel
from swarmrouter.schema import RouterOutput


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def generate_routing(model, tokenizer, user_message: str, max_new_tokens: int = 512) -> str:
    """Generate routing decision."""
    messages = [
        {"role": "system", "content": RouterOutput.system_prompt()},
        {"role": "user", "content": user_message}
    ]

    # Format with chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def extract_json(text: str) -> Dict:
    """Extract JSON from response text."""
    # Try to find JSON in markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end].strip()

    # Try to parse
    try:
        return json.loads(text)
    except:
        # Look for first { to last }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except:
                pass
        return None


def validate_routing(predicted: Dict, expected: Dict) -> Tuple[bool, List[str]]:
    """Validate routing decision against expected."""
    errors = []

    if predicted is None:
        return False, ["Failed to parse JSON"]

    # Check required fields
    required_fields = ["domain", "task_type", "complexity", "risk_level",
                      "latency_tier", "cost_sensitivity", "recommended_model",
                      "escalation_allowed", "requires_tools", "reasoning"]

    for field in required_fields:
        if field not in predicted:
            errors.append(f"Missing field: {field}")

    if errors:
        return False, errors

    # Validate with Pydantic
    try:
        RouterOutput(**predicted)
    except Exception as e:
        errors.append(f"Pydantic validation failed: {str(e)}")
        return False, errors

    # Check model routing correctness
    # Extract expected routing from assistant message
    expected_json = json.loads(expected["messages"][2]["content"])  # Assistant message

    if predicted["recommended_model"] != expected_json["recommended_model"]:
        errors.append(
            f"Model mismatch: predicted {predicted['recommended_model']}, "
            f"expected {expected_json['recommended_model']}"
        )

    if predicted["domain"] != expected_json["domain"]:
        errors.append(
            f"Domain mismatch: predicted {predicted['domain']}, "
            f"expected {expected_json['domain']}"
        )

    return len(errors) == 0, errors


def evaluate_model(model_path: Path, eval_dataset_path: Path, output_path: Path, n_samples: int = None):
    """Run full evaluation."""
    print("=" * 70)
    print("SwarmRouter-v1 Evaluation")
    print("=" * 70)

    # Load model
    print("\n📦 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=1024,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto"
    )
    FastLanguageModel.for_inference(model)  # Enable inference mode

    # Load eval dataset
    print("📂 Loading evaluation dataset...")
    eval_data = load_jsonl(eval_dataset_path)
    if n_samples:
        eval_data = eval_data[:n_samples]
    print(f"  Evaluating on {len(eval_data)} samples")

    # Run evaluation
    print("\n🔬 Running evaluation...")
    results = []
    correct = 0
    json_valid = 0
    model_correct = 0
    domain_correct = 0

    for i, example in enumerate(tqdm(eval_data)):
        user_message = example["messages"][1]["content"]

        # Generate routing
        response = generate_routing(model, tokenizer, user_message)

        # Parse JSON
        predicted = extract_json(response)

        # Validate
        is_valid, errors = validate_routing(predicted, example)

        if predicted is not None:
            json_valid += 1

        if is_valid:
            correct += 1

        # Check specific fields
        expected_json = json.loads(example["messages"][2]["content"])
        if predicted and predicted.get("recommended_model") == expected_json["recommended_model"]:
            model_correct += 1
        if predicted and predicted.get("domain") == expected_json["domain"]:
            domain_correct += 1

        result = {
            "idx": i,
            "user_message": user_message,
            "predicted": predicted,
            "expected": expected_json,
            "is_valid": is_valid,
            "errors": errors,
            "raw_response": response
        }
        results.append(result)

    # Calculate metrics
    total = len(results)
    accuracy = correct / total * 100
    json_accuracy = json_valid / total * 100
    model_accuracy = model_correct / total * 100
    domain_accuracy = domain_correct / total * 100

    # Print results
    print("\n" + "=" * 70)
    print("📊 Evaluation Results")
    print("=" * 70)
    print(f"Total samples:        {total}")
    print(f"JSON valid:           {json_valid} ({json_accuracy:.1f}%)")
    print(f"Fully correct:        {correct} ({accuracy:.1f}%)")
    print(f"Model routing:        {model_correct} ({model_accuracy:.1f}%)")
    print(f"Domain classification: {domain_correct} ({domain_accuracy:.1f}%)")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_path": str(model_path),
        "eval_dataset": str(eval_dataset_path),
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "total_samples": total,
            "json_valid": json_valid,
            "json_accuracy": json_accuracy,
            "fully_correct": correct,
            "full_accuracy": accuracy,
            "model_correct": model_correct,
            "model_accuracy": model_accuracy,
            "domain_correct": domain_correct,
            "domain_accuracy": domain_accuracy
        },
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")

    # Print error analysis
    if correct < total:
        print("\n❌ Error Analysis (first 5):")
        error_count = 0
        for result in results:
            if not result["is_valid"] and error_count < 5:
                print(f"\n  Sample {result['idx']}:")
                print(f"    User: {result['user_message'][:80]}...")
                print(f"    Errors: {', '.join(result['errors'])}")
                error_count += 1

    print("\n✅ Evaluation complete!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SwarmRouter model")
    parser.add_argument("--model-path", type=Path, default=Path("./models/swarmrouter-v1/final"),
                       help="Path to trained model")
    parser.add_argument("--eval-dataset", type=Path, default=Path("./data/swarmrouter_eval_400.jsonl"),
                       help="Path to evaluation dataset")
    parser.add_argument("--output", type=Path, default=Path("./outputs/eval_results.json"),
                       help="Output path for results")
    parser.add_argument("--n-samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all)")
    args = parser.parse_args()

    # Make paths absolute
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / args.model_path if not args.model_path.is_absolute() else args.model_path
    eval_dataset = base_dir / args.eval_dataset if not args.eval_dataset.is_absolute() else args.eval_dataset
    output = base_dir / args.output if not args.output.is_absolute() else args.output

    evaluate_model(model_path, eval_dataset, output, args.n_samples)


if __name__ == "__main__":
    main()
