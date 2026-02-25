#!/usr/bin/env python3
"""
Main dataset creation script for SwarmRouter-v1.
Generates 60K training samples + 400 eval samples.
"""

import argparse
from pathlib import Path
from generator import DatasetGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate SwarmRouter training dataset")
    parser.add_argument("--train-size", type=int, default=60000, help="Number of training samples")
    parser.add_argument("--eval-size", type=int, default=400, help="Number of eval samples")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    # Initialize generator
    data_dir = Path(__file__).parent
    generator = DatasetGenerator(data_dir)

    # Output paths
    train_output = args.output_dir / "swarmrouter_train_60k.jsonl"
    eval_output = args.output_dir / "swarmrouter_eval_400.jsonl"

    print("=" * 60)
    print("SwarmRouter Dataset Generation")
    print("=" * 60)

    # Generate training set
    print(f"\n🔨 Generating training set ({args.train_size:,} samples)...")
    generator.generate_dataset(
        n_samples=args.train_size,
        output_path=train_output,
        shuffle_seed=args.seed
    )

    # Generate eval set with different seed
    print(f"\n🔬 Generating evaluation set ({args.eval_size:,} samples)...")
    generator.generate_dataset(
        n_samples=args.eval_size,
        output_path=eval_output,
        shuffle_seed=args.seed + 1000  # Different seed to avoid overlap
    )

    print("\n" + "=" * 60)
    print("✅ Dataset generation complete!")
    print("=" * 60)
    print(f"\nTraining set:   {train_output}")
    print(f"Evaluation set: {eval_output}")
    print("\nNext steps:")
    print("  1. Review sample outputs for quality")
    print("  2. Run training: python scripts/train.py")
    print("  3. Evaluate model: python scripts/eval.py")


if __name__ == "__main__":
    main()
