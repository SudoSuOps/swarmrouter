"""
SwarmJudge-27B Training Script — Blackwell Edition
=====================================================

bf16 LoRA on Qwen3.5-27B (Mamba-Transformer hybrid, dense 27B) using the
RTX PRO 6000 Blackwell (96GB). Full precision, 8K sequence length,
proper foundation model training with packing.

Requires: causal-conv1d built with sm_120 support (CUDA 12.8+).

Base model:  Qwen/Qwen3.5-27B
Method:      bf16 LoRA r=64 alpha=32 (NOT QLoRA)
GPU:         Single RTX PRO 6000 Blackwell (96GB)
Data:        ~100K curated judge pairs (trajectory + agent trace + judge eval)
Output:      /data2/swarmjudge-27b/

Usage:
    python3 train_swarmjudge_27b.py [--dry-run] [--max-samples N] [--skip-merge] [--skip-gguf]

Launch on swarmrails:
    ssh swarmrails "cd /data2/swarmjudge_27b_data && \\
        nohup bash -c 'CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \\
        WANDB_MODE=disabled python3 train_swarmjudge_27b.py' \\
        > /data2/swarmjudge-27b/train.log 2>&1 &"
"""

import json
import os
import sys
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════
# Config — Blackwell 96GB, Mamba-Transformer hybrid
# ═══════════════════════════════════════════════════════════════════════

BASE_MODEL = "Qwen/Qwen3.5-27B"
MODEL_NAME = "SwarmJudge-27B-v1"
OUTPUT_DIR = Path("/data2/swarmjudge-27b")
DATA_DIR = Path("/data2/swarmjudge_27b_data")
TRAIN_FILE = DATA_DIR / "swarmjudge_27b_train.jsonl"
EVAL_FILE = DATA_DIR / "swarmjudge_27b_eval.jsonl"

# LoRA — bf16 (NOT QLoRA — we have 96GB, no quantization needed)
MAX_SEQ_LENGTH = 8192
LORA_R = 64
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training — Blackwell 96GB handles bf16 MoE comfortably
BATCH_SIZE = 2
GRAD_ACCUM = 16           # Effective batch = 32
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
NEFTUNE_ALPHA = 5.0
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 5
LR_SCHEDULER = "cosine"
SEED = 42

# GGUF export
LLAMA_CPP_DIR = Path("/home/swarm/llama.cpp")
GGUF_QUANT = "Q4_K_M"

# Force no wandb
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"


def format_chat(messages: list[dict]) -> str:
    """Format messages into Qwen3.5 ChatML template.

    Qwen3.5 uses the same <|im_start|>/<|im_end|> ChatML format as Qwen2.5.
    """
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file for provenance tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} Training — Blackwell Edition")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and model, estimate steps, then exit")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit training samples (0 = use all)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip LoRA merge step after training")
    parser.add_argument("--skip-gguf", action="store_true",
                        help="Skip GGUF quantization step after merge")
    args = parser.parse_args()

    print(f"{MODEL_NAME} Training — Blackwell Edition")
    print(f"{'='*60}")
    print(f"Base model:     {BASE_MODEL}")
    print(f"Architecture:   Mamba-Transformer Hybrid (27B dense)")
    print(f"LoRA:           r={LORA_R}, alpha={LORA_ALPHA}, bf16 (NOT QLoRA)")
    print(f"Seq length:     {MAX_SEQ_LENGTH}")
    print(f"Batch:          {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective")
    print(f"Epochs:         {NUM_EPOCHS}")
    print(f"LR:             {LEARNING_RATE}")
    print(f"Scheduler:      {LR_SCHEDULER}")
    print(f"Packing:        True")
    print(f"NEFTune:        alpha={NEFTUNE_ALPHA}")
    print(f"Train data:     {TRAIN_FILE}")
    print(f"Eval data:      {EVAL_FILE}")
    print(f"Output:         {OUTPUT_DIR}")
    print()

    # ── Validate data exists ─────────────────────────────────────────
    if not TRAIN_FILE.exists():
        print(f"ERROR: Training data not found at {TRAIN_FILE}")
        print(f"Run the curator first: python3 -m data.judge.assemble_swarmjudge_27b --stage")
        sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────
    print("Loading training data...")
    train_texts = []
    with open(TRAIN_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = format_chat(r["messages"])
            train_texts.append({"text": text})

    if args.max_samples > 0:
        train_texts = train_texts[:args.max_samples]

    print(f"  Train samples: {len(train_texts):,}")

    eval_texts = []
    if EVAL_FILE.exists():
        with open(EVAL_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                text = format_chat(r["messages"])
                eval_texts.append({"text": text})
        print(f"  Eval samples:  {len(eval_texts):,}")
    else:
        print(f"  WARNING: No eval file at {EVAL_FILE} — training without eval")

    # ── Load model — bf16 LoRA, NOT 4-bit ────────────────────────────
    print(f"\nLoading {BASE_MODEL} (bf16, full precision)...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,           # Auto-detect bf16 on Blackwell
        load_in_4bit=False,   # Full precision — we have 96GB
    )

    print("Applying LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimized checkpointing
        random_state=SEED,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    import torch
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    vram_used = torch.cuda.memory_allocated() / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU:  {gpu_name}")
    print(f"  VRAM: {vram_used:.1f} / {vram_gb:.1f} GB ({vram_used/vram_gb*100:.1f}% after model load)")

    # ── Estimate steps ───────────────────────────────────────────────
    effective_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = len(train_texts) // effective_batch
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    save_count = total_steps // SAVE_STEPS

    print(f"\n  Steps/epoch:   {steps_per_epoch:,}")
    print(f"  Total steps:   {total_steps:,}")
    print(f"  Warmup steps:  {warmup_steps:,}")
    print(f"  Save points:   {save_count}")

    if args.dry_run:
        # Mamba-Transformer hybrid 27B — estimate ~15-20s/step on Blackwell with causal-conv1d
        est_sec_per_step = 18
        est_hours = total_steps * est_sec_per_step / 3600
        print(f"\n  DRY RUN — would train {total_steps:,} steps")
        print(f"  Estimated time: ~{est_hours:.1f}h @ ~{est_sec_per_step}s/step (27B Mamba hybrid)")
        print(f"  Estimated VRAM peak: ~55-65 GB (bf16 27B + LoRA + optimizer + KV cache)")
        return

    # ── Dataset ──────────────────────────────────────────────────────
    from datasets import Dataset

    train_ds = Dataset.from_list(train_texts)
    eval_ds = Dataset.from_list(eval_texts) if eval_texts else None

    # ── Train ────────────────────────────────────────────────────────
    from trl import SFTTrainer, SFTConfig

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type=LR_SCHEDULER,
        optim="adamw_torch",
        fp16=False,
        bf16=True,              # Blackwell native bf16
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=EVAL_STEPS if eval_ds else None,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,           # Packing — efficient with 96GB
        neftune_noise_alpha=NEFTUNE_ALPHA,
        dataset_text_field="text",
        report_to="none",       # No wandb
        seed=SEED,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    print(f"\n{'='*60}")
    print(f"Starting {MODEL_NAME} training at {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    train_start = datetime.now()
    result = trainer.train()
    train_end = datetime.now()
    train_duration = train_end - train_start

    # ── Save adapter ─────────────────────────────────────────────────
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nAdapter saved to {final_dir}")

    # ── Training metadata ────────────────────────────────────────────
    meta = {
        "model_name": MODEL_NAME,
        "base_model": BASE_MODEL,
        "architecture": "Mamba-Transformer Hybrid (27B dense)",
        "precision": "bf16 LoRA (NOT QLoRA)",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_samples": len(train_texts),
        "eval_samples": len(eval_texts),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": effective_batch,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler": LR_SCHEDULER,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "neftune_alpha": NEFTUNE_ALPHA,
        "packing": True,
        "gradient_checkpointing": True,
        "total_steps": result.global_step,
        "final_train_loss": result.training_loss,
        "train_runtime_sec": result.metrics.get("train_runtime", 0),
        "train_duration": str(train_duration),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(trainable / total * 100, 4),
        "gpu": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "started_at": train_start.isoformat(),
        "completed_at": train_end.isoformat(),
    }

    meta_path = OUTPUT_DIR / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    print(f"\nTraining complete.")
    print(f"  Steps:    {result.global_step:,}")
    print(f"  Loss:     {result.training_loss:.4f}")
    print(f"  Duration: {train_duration}")

    # ── Merge LoRA → full model ──────────────────────────────────────
    if not args.skip_merge:
        print(f"\n{'='*60}")
        print("Merging LoRA adapter into full model (16-bit)...")
        print(f"{'='*60}")

        merged_dir = OUTPUT_DIR / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"Merged model saved to {merged_dir}")

        # Compute SHA-256 of merged safetensors for provenance
        print("Computing checksums...")
        checksums = {}
        for sf in sorted(merged_dir.glob("*.safetensors")):
            checksums[sf.name] = sha256_file(sf)
            print(f"  {sf.name}: {checksums[sf.name][:16]}...")

        checksum_path = merged_dir / "SHA256SUMS.txt"
        with open(checksum_path, "w") as f:
            for name, digest in sorted(checksums.items()):
                f.write(f"{digest}  {name}\n")
        print(f"Checksums saved to {checksum_path}")

        # Update metadata with merge info
        meta["merged_dir"] = str(merged_dir)
        meta["merged_checksums"] = checksums
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # ── GGUF export ──────────────────────────────────────────────
        if not args.skip_gguf:
            print(f"\n{'='*60}")
            print(f"Exporting GGUF ({GGUF_QUANT}) via llama.cpp...")
            print(f"{'='*60}")

            gguf_dir = OUTPUT_DIR / "gguf"
            gguf_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Convert merged model to GGUF F16
            gguf_f16 = gguf_dir / f"swarmjudge-27b-v1-f16.gguf"
            convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

            if not convert_script.exists():
                print(f"  WARNING: {convert_script} not found — skipping GGUF export")
                print(f"  To export manually:")
                print(f"    python3 {convert_script} {merged_dir} --outfile {gguf_f16} --outtype f16")
                print(f"    {LLAMA_CPP_DIR}/llama-quantize {gguf_f16} {gguf_dir}/swarmjudge-27b-v1-{GGUF_QUANT.lower()}.gguf {GGUF_QUANT}")
            else:
                print(f"  Converting to GGUF F16...")
                convert_result = subprocess.run(
                    [
                        sys.executable, str(convert_script),
                        str(merged_dir),
                        "--outfile", str(gguf_f16),
                        "--outtype", "f16",
                    ],
                    capture_output=True, text=True, timeout=1800,
                )
                if convert_result.returncode != 0:
                    print(f"  ERROR in GGUF conversion: {convert_result.stderr[-500:]}")
                    print(f"  Skipping quantization.")
                else:
                    f16_size_gb = gguf_f16.stat().st_size / 1e9
                    print(f"  F16 GGUF: {gguf_f16} ({f16_size_gb:.1f} GB)")

                    # Step 2: Quantize to Q4_K_M
                    gguf_quant_path = gguf_dir / f"swarmjudge-27b-v1-{GGUF_QUANT.lower()}.gguf"
                    quantize_bin = LLAMA_CPP_DIR / "llama-quantize"

                    if not quantize_bin.exists():
                        print(f"  WARNING: {quantize_bin} not found — skipping quantization")
                        print(f"  To quantize manually:")
                        print(f"    {quantize_bin} {gguf_f16} {gguf_quant_path} {GGUF_QUANT}")
                    else:
                        print(f"  Quantizing to {GGUF_QUANT}...")
                        quant_result = subprocess.run(
                            [
                                str(quantize_bin),
                                str(gguf_f16),
                                str(gguf_quant_path),
                                GGUF_QUANT,
                            ],
                            capture_output=True, text=True, timeout=3600,
                        )
                        if quant_result.returncode != 0:
                            print(f"  ERROR in quantization: {quant_result.stderr[-500:]}")
                        else:
                            quant_size_gb = gguf_quant_path.stat().st_size / 1e9
                            gguf_sha = sha256_file(gguf_quant_path)
                            print(f"  {GGUF_QUANT} GGUF: {gguf_quant_path} ({quant_size_gb:.1f} GB)")
                            print(f"  SHA-256: {gguf_sha}")

                            # Update metadata
                            meta["gguf_path"] = str(gguf_quant_path)
                            meta["gguf_quant"] = GGUF_QUANT
                            meta["gguf_size_gb"] = round(quant_size_gb, 2)
                            meta["gguf_sha256"] = gguf_sha
                            with open(meta_path, "w") as f:
                                json.dump(meta, f, indent=2)

                            # Write GGUF checksum file
                            gguf_sha_path = gguf_dir / "GGUF_SHA256SUM.txt"
                            with open(gguf_sha_path, "w") as f:
                                f.write(f"{gguf_sha}  {gguf_quant_path.name}\n")
                            print(f"  Checksum saved to {gguf_sha_path}")

                    # Clean up F16 intermediate if quantization succeeded
                    if gguf_quant_path.exists() and gguf_f16.exists():
                        print(f"  Removing intermediate F16 GGUF ({f16_size_gb:.1f} GB)...")
                        gguf_f16.unlink()

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{MODEL_NAME} — TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Base:       {BASE_MODEL}")
    print(f"  Steps:      {result.global_step:,}")
    print(f"  Loss:       {result.training_loss:.4f}")
    print(f"  Duration:   {train_duration}")
    print(f"  Adapter:    {final_dir}")
    if not args.skip_merge:
        print(f"  Merged:     {OUTPUT_DIR / 'merged'}")
    if not args.skip_merge and not args.skip_gguf:
        gguf_out = OUTPUT_DIR / "gguf" / f"swarmjudge-27b-v1-{GGUF_QUANT.lower()}.gguf"
        if gguf_out.exists():
            print(f"  GGUF:       {gguf_out}")
    print(f"  Metadata:   {meta_path}")
    print(f"\nDone at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
