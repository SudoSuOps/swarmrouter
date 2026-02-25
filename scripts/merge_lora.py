#!/usr/bin/env python3
"""
Merge LoRA adapters into base model for faster inference.
Supports both SwarmRouter (dense) and SwarmCRE (MoE) models.
"""

import sys
import yaml
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from unsloth import FastLanguageModel


def merge_lora(model_path: Path, output_path: Path, quantization: str = "both",
               max_seq_length: int = 8192):
    """Merge LoRA adapters into base model."""
    print("=" * 70)
    print("SwarmCRE LoRA Merge")
    print("=" * 70)
    print(f"  Input:         {model_path}")
    print(f"  Output:        {output_path}")
    print(f"  Quantization:  {quantization}")
    print(f"  Max seq len:   {max_seq_length}")
    print("=" * 70)

    # Load model with LoRA
    print("\nLoading model with LoRA adapters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
    )

    print("Merging LoRA adapters into base model...")
    output_path.mkdir(parents=True, exist_ok=True)

    if quantization in ("none", "both"):
        print("  Saving BF16...")
        model.save_pretrained_merged(
            str(output_path / "bf16"),
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"  Saved to: {output_path / 'bf16'}")

    if quantization in ("4bit", "both"):
        print("  Saving 4-bit...")
        model.save_pretrained_merged(
            str(output_path / "4bit"),
            tokenizer,
            save_method="merged_4bit_forced",
        )
        print(f"  Saved to: {output_path / '4bit'}")

    print("\nMerge complete!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA adapters")
    parser.add_argument("--model-path", type=Path,
                        default=Path("./models/swarmcre-32b-v1/final"),
                        help="Path to LoRA model")
    parser.add_argument("--output", type=Path,
                        default=Path("./models/swarmcre-32b-v1-merged"),
                        help="Output path for merged model")
    parser.add_argument("--quantization", type=str, default="both",
                        choices=["none", "4bit", "both"])
    parser.add_argument("--max-seq-length", type=int, default=8192)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    model_path = base_dir / args.model_path if not args.model_path.is_absolute() else args.model_path
    output_path = base_dir / args.output if not args.output.is_absolute() else args.output

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    merge_lora(model_path, output_path, args.quantization, args.max_seq_length)


if __name__ == "__main__":
    main()
