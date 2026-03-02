"""
SwarmRouter-4B-0 Block Zero — Whale Training Script
=====================================================

bf16 LoRA on Qwen3.5-4B-Base using Unsloth.
4B in bf16 = ~8GB weights — fits on whale 3090 (24GB) comfortably.

Base model:  Qwen/Qwen3.5-4B-Base
Method:      bf16 LoRA r=64 alpha=32
GPU:         RTX 3090 (24GB) — whale
Data:        68,179 sealed router pairs (swarmrouter-4b0-v1 build)
Output:      /home/swarm/swarmrouter-4b0/

Block-0 cycle: Train → Evaluate (judge) → v1

Usage:
    python3 train_router_4b0.py [--dry-run] [--max-samples N] [--skip-merge]

Launch on whale:
    source /home/swarm/router-v3-env/bin/activate
    cd /home/swarm/swarmrouter-4b0
    nohup python3 train_router_4b0.py > train.log 2>&1 &
"""

import json
import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════
# Config — whale RTX 3090 (24GB), 4B-Base bf16 LoRA
# ═══════════════════════════════════════════════════════════════════════

BASE_MODEL  = "Qwen/Qwen3.5-4B-Base"
MODEL_NAME  = "SwarmRouter-4B-0"
OUTPUT_DIR  = Path("/home/swarm/swarmrouter-4b0")
DATA_DIR    = Path("/home/swarm/swarmrouter-4b0/data")

# Dataset — sealed build artifact (symlinked into DATA_DIR on staging)
TRAIN_FILE  = DATA_DIR / "train.jsonl"
EVAL_FILE   = DATA_DIR / "eval.jsonl"

# Build provenance
BUILD_ID    = "swarmrouter-4b0-v1-20260302_180342"
SHA256_TRAIN = "d60772d5db2408823c638e69de928b4266a18ae0f5989ff52647f16063f8770c"
SHA256_EVAL  = "aa87ef6bd3d4cedc63ae3243806a09a42e33ae9de8c35cdd7c94a131fa86a820"

# LoRA — bf16 (4B fits in 24GB without quantization)
MAX_SEQ_LENGTH = 2048
LORA_R         = 64
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training — whale 3090 (24GB), 4B model
BATCH_SIZE   = 4
GRAD_ACCUM   = 8         # Effective batch = 32
NUM_EPOCHS   = 2
LEARNING_RATE = 2.0e-4  # Higher LR for smaller model
WARMUP_RATIO  = 0.03
WEIGHT_DECAY  = 0.01
MAX_GRAD_NORM = 1.0
NEFTUNE_ALPHA = 5.0
SAVE_STEPS    = 500
EVAL_STEPS    = 500
LOGGING_STEPS = 25
SAVE_TOTAL_LIMIT = 3
LR_SCHEDULER  = "cosine"
SEED          = 42

os.environ["WANDB_MODE"]     = "disabled"
os.environ["WANDB_DISABLED"] = "true"


def format_chat(messages: list[dict]) -> str:
    """Qwen3.5 ChatML format."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    return "\n".join(parts) + "\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_data(path: Path, expected_sha: str, label: str) -> bool:
    print(f"  Verifying {label}...")
    actual = sha256_file(path)
    if actual != expected_sha:
        print(f"  ERROR: SHA256 mismatch on {label}")
        print(f"    Expected: {expected_sha}")
        print(f"    Got:      {actual}")
        return False
    print(f"  ✓ {label} SHA256 verified")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} Block Zero — Whale Edition")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data + model, estimate steps, exit")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit training samples for testing (0 = all)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip LoRA merge after training (GGUF done separately on swarmrails)")
    args = parser.parse_args()

    effective_batch = BATCH_SIZE * GRAD_ACCUM

    print(f"{MODEL_NAME} Block Zero — Whale Edition")
    print(f"{'='*60}")
    print(f"Base model:     {BASE_MODEL}")
    print(f"Architecture:   Qwen3.5-4B-Base (DeltaNet hybrid)")
    print(f"LoRA:           r={LORA_R}, alpha={LORA_ALPHA}, bf16 LoRA")
    print(f"Seq length:     {MAX_SEQ_LENGTH}")
    print(f"Batch:          {BATCH_SIZE} x {GRAD_ACCUM} = {effective_batch} effective")
    print(f"Epochs:         {NUM_EPOCHS}")
    print(f"LR:             {LEARNING_RATE} ({LR_SCHEDULER})")
    print(f"Packing:        True")
    print(f"NEFTune:        alpha={NEFTUNE_ALPHA}")
    print(f"Build ID:       {BUILD_ID}")
    print(f"Train data:     {TRAIN_FILE}")
    print(f"Eval data:      {EVAL_FILE}")
    print(f"Output:         {OUTPUT_DIR}")
    print()

    # ── Verify sealed data ───────────────────────────────────────────
    if not TRAIN_FILE.exists():
        print(f"ERROR: Training data not found at {TRAIN_FILE}")
        print(f"Stage with: python3 stage_whale.py")
        sys.exit(1)
    if not EVAL_FILE.exists():
        print(f"ERROR: Eval data not found at {EVAL_FILE}")
        sys.exit(1)

    print("Verifying sealed build data...")
    if not verify_data(TRAIN_FILE, SHA256_TRAIN, "train"):
        sys.exit(1)
    if not verify_data(EVAL_FILE, SHA256_EVAL, "eval"):
        sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────
    print("\nLoading training data...")
    train_texts = []
    with open(TRAIN_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            train_texts.append({"text": format_chat(r["messages"])})

    if args.max_samples > 0:
        train_texts = train_texts[:args.max_samples]
    print(f"  Train samples: {len(train_texts):,}")

    eval_texts = []
    with open(EVAL_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            eval_texts.append({"text": format_chat(r["messages"])})
    print(f"  Eval samples:  {len(eval_texts):,}")

    # ── Load model ───────────────────────────────────────────────────
    print(f"\nLoading {BASE_MODEL} (bf16, full precision on 24GB)...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,           # Auto-detect bf16
        load_in_4bit=False,   # 4B model fits in 24GB at bf16 — no quantization
    )

    print("Applying LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    import torch
    vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
    vram_used = torch.cuda.memory_allocated() / 1e9
    gpu_name  = torch.cuda.get_device_name(0)
    print(f"  GPU:  {gpu_name}")
    print(f"  VRAM: {vram_used:.1f} / {vram_gb:.1f} GB ({vram_used/vram_gb*100:.1f}% after model load)")

    # ── Estimate steps ───────────────────────────────────────────────
    steps_per_epoch = len(train_texts) // effective_batch
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = int(total_steps * WARMUP_RATIO)
    save_count      = total_steps // SAVE_STEPS

    print(f"\n  Steps/epoch:   {steps_per_epoch:,}")
    print(f"  Total steps:   {total_steps:,}")
    print(f"  Warmup steps:  {warmup_steps:,}")
    print(f"  Save points:   {save_count}")

    if args.dry_run:
        est_sec_per_step = 8   # 4B on 3090 — ~6-10s/step estimate
        est_hours = total_steps * est_sec_per_step / 3600
        print(f"\n  DRY RUN — would train {total_steps:,} steps")
        print(f"  Estimated time: ~{est_hours:.1f}h @ ~{est_sec_per_step}s/step")
        print(f"  Estimated VRAM peak: ~14-18 GB (bf16 4B + LoRA + grad checkpointing)")
        return

    # ── Dataset ──────────────────────────────────────────────────────
    from datasets import Dataset

    train_ds = Dataset.from_list(train_texts)
    eval_ds  = Dataset.from_list(eval_texts)

    # ── Train ────────────────────────────────────────────────────────
    from trl import SFTTrainer, SFTConfig

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type=LR_SCHEDULER,
        optim="adamw_torch",
        fp16=False,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
        neftune_noise_alpha=NEFTUNE_ALPHA,
        dataset_text_field="text",
        report_to="none",
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
    print(f"Starting {MODEL_NAME} Block Zero at {datetime.now().isoformat()}")
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
        "model_name":      MODEL_NAME,
        "block":           "0",
        "build_id":        BUILD_ID,
        "base_model":      BASE_MODEL,
        "architecture":    "Qwen3.5-4B-Base (DeltaNet hybrid)",
        "precision":       "bf16 LoRA",
        "lora_r":          LORA_R,
        "lora_alpha":      LORA_ALPHA,
        "lora_dropout":    LORA_DROPOUT,
        "target_modules":  TARGET_MODULES,
        "max_seq_length":  MAX_SEQ_LENGTH,
        "train_samples":   len(train_texts),
        "eval_samples":    len(eval_texts),
        "epochs":          NUM_EPOCHS,
        "batch_size":      BATCH_SIZE,
        "grad_accum":      GRAD_ACCUM,
        "effective_batch": effective_batch,
        "learning_rate":   LEARNING_RATE,
        "lr_scheduler":    LR_SCHEDULER,
        "warmup_ratio":    WARMUP_RATIO,
        "neftune_alpha":   NEFTUNE_ALPHA,
        "packing":         True,
        "sha256_train":    SHA256_TRAIN,
        "sha256_eval":     SHA256_EVAL,
        "total_steps":     result.global_step,
        "final_train_loss":result.training_loss,
        "train_runtime_sec": result.metrics.get("train_runtime", 0),
        "train_duration":  str(train_duration),
        "trainable_params":trainable,
        "total_params":    total,
        "gpu":             gpu_name,
        "vram_gb":         round(vram_gb, 1),
        "started_at":      train_start.isoformat(),
        "completed_at":    train_end.isoformat(),
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
        print("Merging LoRA adapter into full model (bf16)...")
        print(f"{'='*60}")

        merged_dir = OUTPUT_DIR / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"Merged model saved to {merged_dir}")

        print("Computing SHA256 checksums...")
        checksums = {}
        for sf in sorted(merged_dir.glob("*.safetensors")):
            checksums[sf.name] = sha256_file(sf)
            print(f"  {sf.name}: {checksums[sf.name][:16]}...")

        sha_path = merged_dir / "SHA256SUMS.txt"
        with open(sha_path, "w") as f:
            for name, digest in sorted(checksums.items()):
                f.write(f"{digest}  {name}\n")
        print(f"Checksums → {sha_path}")

        meta["merged_dir"]       = str(merged_dir)
        meta["merged_checksums"] = checksums
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{MODEL_NAME} — BLOCK ZERO COMPLETE")
    print(f"{'='*60}")
    print(f"  Base:       {BASE_MODEL}")
    print(f"  Build ID:   {BUILD_ID}")
    print(f"  Steps:      {result.global_step:,}")
    print(f"  Loss:       {result.training_loss:.4f}")
    print(f"  Duration:   {train_duration}")
    print(f"  Adapter:    {final_dir}")
    if not args.skip_merge:
        print(f"  Merged:     {OUTPUT_DIR / 'merged'}")
    print(f"\nNext (Block-0 → Evaluate cycle):")
    print(f"  scp merged model to swarmrails for GGUF quantization")
    print(f"  Run eval suite: judge scores Block-0 outputs")
    print(f"  PASS → seal as v1 | FAIL → targeted fine-tune → v1")
    print(f"\nDone at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
