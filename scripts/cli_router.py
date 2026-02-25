#!/usr/bin/env python3
"""
SwarmRouter-v1 CLI Tool
Interactive command-line interface for routing queries.
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))

from unsloth import FastLanguageModel
from swarmrouter.schema import RouterOutput


class RouterCLI:
    """CLI for SwarmRouter."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.history = []

    def load_model(self):
        """Load model."""
        print(f"Loading SwarmRouter from {self.model_path}...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.model_path),
            max_seq_length=1024,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map="auto"
        )
        FastLanguageModel.for_inference(self.model)

        print("✓ Model loaded\n")

    def generate_routing(self, user_message: str) -> dict:
        """Generate routing decision."""
        messages = [
            {"role": "system", "content": RouterOutput.system_prompt()},
            {"role": "user", "content": user_message}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)

        start_time = datetime.now()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip(), latency_ms

    def extract_json(self, text: str) -> dict:
        """Extract JSON from response."""
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        try:
            return json.loads(text)
        except:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except:
                    pass
            return None

    def print_routing(self, routing: dict, latency_ms: float):
        """Pretty print routing decision."""
        print("\n" + "=" * 70)
        print("🧭 Routing Decision")
        print("=" * 70)
        print(f"Domain:           {routing.get('domain', 'N/A')}")
        print(f"Task Type:        {routing.get('task_type', 'N/A')}")
        print(f"Complexity:       {routing.get('complexity', 'N/A')}")
        print(f"Risk Level:       {routing.get('risk_level', 'N/A')}")
        print(f"Recommended:      {routing.get('recommended_model', 'N/A')}")
        print(f"Escalation:       {'Yes' if routing.get('escalation_allowed') else 'No'}")
        print(f"Tools Required:   {', '.join(routing.get('requires_tools', [])) if routing.get('requires_tools') else 'None'}")
        print(f"Reasoning:        {routing.get('reasoning', 'N/A')}")
        print("-" * 70)
        print(f"Latency:          {latency_ms:.1f}ms")
        print("=" * 70 + "\n")

    def run_interactive(self):
        """Run interactive mode."""
        print("=" * 70)
        print("SwarmRouter CLI - Interactive Mode")
        print("=" * 70)
        print("\nCommands:")
        print("  /quit    - Exit")
        print("  /history - Show routing history")
        print("  /clear   - Clear history")
        print("\nEnter your queries below:\n")

        while True:
            try:
                user_input = input("Query> ").strip()

                if not user_input:
                    continue

                if user_input == "/quit":
                    print("\nGoodbye!")
                    break

                if user_input == "/history":
                    self.show_history()
                    continue

                if user_input == "/clear":
                    self.history = []
                    print("✓ History cleared\n")
                    continue

                # Generate routing
                response, latency_ms = self.generate_routing(user_input)
                routing = self.extract_json(response)

                if routing is None:
                    print("\n❌ Failed to parse routing decision")
                    print(f"Raw response: {response}\n")
                    continue

                # Validate
                try:
                    RouterOutput(**routing)
                    self.print_routing(routing, latency_ms)
                    self.history.append({
                        "query": user_input,
                        "routing": routing,
                        "latency_ms": latency_ms,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"\n❌ Validation error: {e}")
                    print(f"Raw response: {response}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

    def show_history(self):
        """Show routing history."""
        if not self.history:
            print("\nNo history yet\n")
            return

        print("\n" + "=" * 70)
        print("Routing History")
        print("=" * 70)

        for i, entry in enumerate(self.history, 1):
            print(f"\n{i}. Query: {entry['query'][:60]}...")
            print(f"   Model: {entry['routing']['recommended_model']}")
            print(f"   Domain: {entry['routing']['domain']}")
            print(f"   Latency: {entry['latency_ms']:.1f}ms")

        print("\n" + "=" * 70 + "\n")

    def run_single(self, query: str):
        """Run single query."""
        response, latency_ms = self.generate_routing(query)
        routing = self.extract_json(response)

        if routing is None:
            print("❌ Failed to parse routing decision")
            print(f"Raw response: {response}")
            sys.exit(1)

        try:
            RouterOutput(**routing)
            self.print_routing(routing, latency_ms)
        except Exception as e:
            print(f"❌ Validation error: {e}")
            print(f"Raw response: {response}")
            sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SwarmRouter CLI")
    parser.add_argument("--model-path", type=Path, default=Path("./models/swarmrouter-v1/final"),
                       help="Path to model")
    parser.add_argument("--query", type=str, default=None,
                       help="Single query mode")
    args = parser.parse_args()

    # Resolve model path
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / args.model_path if not args.model_path.is_absolute() else args.model_path

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    # Initialize CLI
    cli = RouterCLI(model_path)
    cli.load_model()

    # Run mode
    if args.query:
        cli.run_single(args.query)
    else:
        cli.run_interactive()


if __name__ == "__main__":
    main()
