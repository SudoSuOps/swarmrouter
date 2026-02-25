#!/usr/bin/env python3
"""
Synthetic data generator for SwarmRouter training dataset.
Generates diverse routing examples with proper labels.
"""

import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from swarmrouter.routing_rules import apply_routing_physics
from swarmrouter.schema import RouterOutput, format_system_prompt


class DatasetGenerator:
    """Generates synthetic routing examples."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.topics_dir = self.data_dir / "topics"
        self.templates_path = self.data_dir / "templates.json"

        # Load topics
        self.topics = self._load_topics()

        # Load templates
        with open(self.templates_path) as f:
            self.templates = json.load(f)

        # Domain distribution (sums to 100%)
        self.domain_distribution = {
            "medical": 0.20,
            "research": 0.20,
            "compute": 0.15,
            "cre": 0.15,
            "aviation": 0.10,
            "coding": 0.10,
            "operations": 0.05,
            "general": 0.05
        }

        # Complexity distribution
        self.complexity_distribution = {
            "low": 0.20,
            "medium": 0.50,
            "high": 0.30
        }

        # Task type distribution
        self.task_types = ["qa", "summarization", "reasoning", "generation", "planning", "triage"]

    def _load_topics(self) -> Dict[str, List[str]]:
        """Load all topic files."""
        topics = {}
        for topic_file in self.topics_dir.glob("*.txt"):
            domain = topic_file.stem
            with open(topic_file) as f:
                domain_topics = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Support both arrow-delimited and plain text formats
                    if "→" in line:
                        topic = line.split("→", 1)[1].strip()
                    else:
                        topic = line
                    if topic:
                        domain_topics.append(topic)
                topics[domain] = domain_topics
        return topics

    def _deterministic_sample(self, items: List, seed: str, k: int = 1):
        """Deterministic sampling using seed."""
        hash_obj = hashlib.md5(seed.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        rng = random.Random(hash_int)
        if k == 1:
            return rng.choice(items)
        return rng.sample(items, k)

    def _pick_domain(self, idx: int) -> str:
        """Pick domain based on distribution."""
        seed = f"domain_{idx}"
        hash_obj = hashlib.md5(seed.encode())
        hash_float = int(hash_obj.hexdigest()[:8], 16) / 0xffffffff

        cumulative = 0.0
        for domain, prob in self.domain_distribution.items():
            cumulative += prob
            if hash_float < cumulative:
                return domain
        return "general"

    def _pick_complexity(self, idx: int) -> str:
        """Pick complexity based on distribution."""
        seed = f"complexity_{idx}"
        hash_obj = hashlib.md5(seed.encode())
        hash_float = int(hash_obj.hexdigest()[:8], 16) / 0xffffffff

        cumulative = 0.0
        for complexity, prob in self.complexity_distribution.items():
            cumulative += prob
            if hash_float < cumulative:
                return complexity
        return "medium"

    def _pick_task_type(self, idx: int, domain: str) -> str:
        """Pick task type with domain-specific bias."""
        seed = f"task_{idx}_{domain}"

        # Domain-specific task type preferences
        if domain == "medical":
            task_pool = ["qa", "reasoning", "triage"] * 2 + ["summarization"]
        elif domain == "research":
            task_pool = ["summarization", "reasoning", "generation"] * 2 + ["qa"]
        elif domain in ["compute", "operations"]:
            task_pool = ["planning", "qa", "triage"] * 2 + ["reasoning"]
        elif domain in ["coding"]:
            task_pool = ["generation", "qa", "reasoning"] * 2 + ["planning"]
        elif domain == "general":
            task_pool = ["qa"] * 4 + ["summarization", "generation"]
        else:
            task_pool = self.task_types

        return self._deterministic_sample(task_pool, seed, k=1)

    def _augment_prompt(self, base_prompt: str, complexity: str, idx: int) -> str:
        """Add complexity-appropriate augmentation."""
        seed = f"aug_{idx}"

        if complexity == "low":
            return base_prompt
        elif complexity == "medium":
            modifiers = [
                "Provide specific examples and context for ",
                "Explain step-by-step: ",
                "Include practical considerations for ",
                "What are the key factors in "
            ]
            modifier = self._deterministic_sample(modifiers, seed, k=1)
            # Sometimes apply, sometimes don't
            hash_obj = hashlib.md5(seed.encode())
            if int(hash_obj.hexdigest()[:2], 16) % 100 < 60:
                return modifier + base_prompt.lower()
            return base_prompt
        else:  # high complexity
            modifiers = [
                "Provide a comprehensive analysis with trade-offs: ",
                "Synthesize multiple perspectives on ",
                "Critically evaluate and compare approaches to ",
                "Design an advanced solution for ",
                "Analyze the system-level implications of "
            ]
            modifier = self._deterministic_sample(modifiers, seed, k=1)
            return modifier + base_prompt.lower()

    def generate_example(self, idx: int) -> Dict:
        """Generate a single training example."""
        # Pick domain, complexity, task type
        domain = self._pick_domain(idx)
        complexity = self._pick_complexity(idx)
        task_type = self._pick_task_type(idx, domain)

        # Edge cases (5% of data)
        hash_obj = hashlib.md5(f"edge_{idx}".encode())
        is_edge_case = int(hash_obj.hexdigest()[:2], 16) % 100 < 5

        if is_edge_case:
            # Use edge case templates
            user_message = self._deterministic_sample(self.templates["edge_cases"], f"edge_{idx}", k=1)
            domain = "general"
            complexity = "low"
            task_type = "qa"
        else:
            # Pick topic and template
            topic = self._deterministic_sample(self.topics[domain], f"topic_{idx}_{domain}", k=1)
            template = self._deterministic_sample(self.templates[domain], f"template_{idx}_{domain}", k=1)

            # Generate base prompt
            user_message = template.format(topic=topic)

            # Augment based on complexity
            user_message = self._augment_prompt(user_message, complexity, idx)

        # Apply routing physics to get labels
        routing_decision = apply_routing_physics(
            domain=domain,
            user_message=user_message,
            task_type=task_type,
            complexity=complexity
        )

        # Build conversation format
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": format_system_prompt()
                },
                {
                    "role": "user",
                    "content": user_message
                },
                {
                    "role": "assistant",
                    "content": json.dumps(routing_decision, indent=2)
                }
            ],
            "metadata": {
                "domain": domain,
                "complexity": complexity,
                "task_type": task_type,
                "recommended_model": routing_decision["recommended_model"],
                "idx": idx
            }
        }

        return example

    def generate_dataset(self, n_samples: int, output_path: Path, shuffle_seed: int = 42):
        """Generate full dataset."""
        print(f"Generating {n_samples} training examples...")

        examples = []
        for idx in tqdm(range(n_samples)):
            example = self.generate_example(idx)
            examples.append(example)

        # Shuffle with seed for reproducibility
        rng = random.Random(shuffle_seed)
        rng.shuffle(examples)

        # Write to JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        print(f"✓ Wrote {len(examples)} examples to {output_path}")

        # Print distribution stats
        self._print_stats(examples)

    def _print_stats(self, examples: List[Dict]):
        """Print dataset statistics."""
        print("\n📊 Dataset Statistics:")

        # Domain distribution
        domain_counts = {}
        model_counts = {}
        complexity_counts = {}
        task_counts = {}

        for ex in examples:
            meta = ex["metadata"]
            domain_counts[meta["domain"]] = domain_counts.get(meta["domain"], 0) + 1
            model_counts[meta["recommended_model"]] = model_counts.get(meta["recommended_model"], 0) + 1
            complexity_counts[meta["complexity"]] = complexity_counts.get(meta["complexity"], 0) + 1
            task_counts[meta["task_type"]] = task_counts.get(meta["task_type"], 0) + 1

        print("\nDomain distribution:")
        for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            pct = count / len(examples) * 100
            print(f"  {domain:12s}: {count:5d} ({pct:5.1f}%)")

        print("\nModel routing distribution:")
        for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
            pct = count / len(examples) * 100
            print(f"  {model:15s}: {count:5d} ({pct:5.1f}%)")

        print("\nComplexity distribution:")
        for complexity, count in sorted(complexity_counts.items()):
            pct = count / len(examples) * 100
            print(f"  {complexity:8s}: {count:5d} ({pct:5.1f}%)")

        print("\nTask type distribution:")
        for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
            pct = count / len(examples) * 100
            print(f"  {task:15s}: {count:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    # Test generation
    data_dir = Path(__file__).parent
    generator = DatasetGenerator(data_dir)

    # Generate small test set
    test_output = data_dir / "test_samples.jsonl"
    generator.generate_dataset(n_samples=100, output_path=test_output)

    print(f"\n✓ Test generation complete. Review {test_output}")
