#!/usr/bin/env python3
"""
Export SwarmRouter to GGUF format for llama.cpp and Ollama.
"""

import sys
import subprocess
from pathlib import Path


def export_gguf(model_path: Path, output_path: Path, quantization: str = "Q4_K_M"):
    """Export model to GGUF format."""
    print("=" * 70)
    print("SwarmRouter GGUF Export")
    print("=" * 70)
    print(f"Input:  {model_path}")
    print(f"Output: {output_path}")
    print(f"Quantization: {quantization}")
    print("=" * 70)

    # Check if llama.cpp is available
    print("\n🔍 Checking for llama.cpp...")

    llama_cpp_path = Path.home() / "llama.cpp"
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print("❌ llama.cpp not found. Please install:")
        print("  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        print("  cd ~/llama.cpp && make")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    print(f"✓ Found llama.cpp at {llama_cpp_path}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to GGUF (FP16 first)
    print("\n🔧 Converting to GGUF (FP16)...")
    fp16_path = output_path.parent / f"{output_path.stem}_fp16.gguf"

    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile", str(fp16_path),
        "--outtype", "f16"
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Conversion failed:\n{result.stderr}")
        sys.exit(1)

    print(f"✓ FP16 GGUF saved to: {fp16_path}")

    # Quantize if requested
    if quantization != "f16":
        print(f"\n🔧 Quantizing to {quantization}...")
        quantize_bin = llama_cpp_path / "llama-quantize"

        if not quantize_bin.exists():
            print("❌ llama-quantize not found. Run 'make' in llama.cpp directory.")
            sys.exit(1)

        quant_path = output_path.parent / f"{output_path.stem}_{quantization}.gguf"

        cmd = [
            str(quantize_bin),
            str(fp16_path),
            str(quant_path),
            quantization
        ]

        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Quantization failed:\n{result.stderr}")
            sys.exit(1)

        print(f"✓ Quantized GGUF saved to: {quant_path}")

        # Remove FP16 to save space (optional)
        print(f"\n🗑️  Removing intermediate FP16 file...")
        fp16_path.unlink()
        print("✓ Cleanup complete")

        final_path = quant_path
    else:
        final_path = fp16_path

    print("\n" + "=" * 70)
    print("✅ GGUF Export Complete!")
    print("=" * 70)
    print(f"\nOutput: {final_path}")
    print(f"Size: {final_path.stat().st_size / 1024 / 1024:.1f} MB")

    print("\n📦 Using with llama.cpp:")
    print(f"  ~/llama.cpp/llama-cli -m {final_path} -p 'Your prompt'")

    print("\n📦 Using with Ollama:")
    print("  1. Create Modelfile:")
    print(f"     FROM {final_path}")
    print("  2. Import:")
    print("     ollama create swarmrouter -f Modelfile")
    print("  3. Run:")
    print("     ollama run swarmrouter")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export model to GGUF")
    parser.add_argument("--model-path", type=Path, required=True,
                       help="Path to merged model (BF16 or FP16)")
    parser.add_argument("--output", type=Path, default=Path("./models/swarmrouter-v1.gguf"),
                       help="Output GGUF path")
    parser.add_argument("--quantization", type=str, default="Q4_K_M",
                       choices=["f16", "Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
                       help="GGUF quantization format")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    model_path = base_dir / args.model_path if not args.model_path.is_absolute() else args.model_path
    output_path = base_dir / args.output if not args.output.is_absolute() else args.output

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("\nMake sure to merge LoRA first:")
        print("  python scripts/merge_lora.py")
        sys.exit(1)

    export_gguf(model_path, output_path, args.quantization)


if __name__ == "__main__":
    main()
