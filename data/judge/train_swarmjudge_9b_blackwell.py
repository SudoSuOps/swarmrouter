#!/usr/bin/env python3
"""
SwarmJudge-9B Training Script — Blackwell Edition
===================================================

bf16 full LoRA on RTX PRO 6000 Blackwell (96GB).
Same 23K Phase 1 PASS pairs as 27B Block-0.

Base model:  Qwen/Qwen3.5-9B
Method:      bf16 LoRA r=64 (full precision — 96GB VRAM)
GPU:         RTX PRO 6000 Blackwell 96GB (CUDA_VISIBLE_DEVICES=1)
Data:        23,000 PASS pairs (Llama 4 Maverick judge)
Output:      /data2/swarmjudge-9b-b0/

Run:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
    WANDB_MODE=disabled python3 train_swarmjudge_9b_blackwell.py
"""

import json
import os
import sys
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

os.environ["WANDB_MODE"]     = "disabled"
os.environ["WANDB_DISABLED"] = "true"

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

BASE_MODEL  = "Qwen/Qwen3.5-9B"
MODEL_NAME  = "SwarmJudge-9B-B0-Phase1"
OUTPUT_DIR  = Path("/data2/swarmjudge-9b-b0")
DATA_DIR    = Path("/data2/swarmjudge_27b_data")
TRAIN_FILE  = DATA_DIR / "phase1_b0" / "phase1_train.jsonl"
EVAL_FILE   = DATA_DIR / "phase1" / "phase1_eval.jsonl"

# bf16 full LoRA — 96GB Blackwell
MAX_SEQ_LENGTH = 4096           # Judge pairs avg ~1500-2000 tok; packing fills the window
LORA_R         = 64
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training — Blackwell tuned
BATCH_SIZE    = 4
GRAD_ACCUM    = 8               # Effective batch = 32
NUM_EPOCHS    = 1
LEARNING_RATE = 5e-5            # Standard bf16 LoRA LR
WARMUP_RATIO  = 0.05
WEIGHT_DECAY  = 0.01
MAX_GRAD_NORM = 1.0
NEFTUNE_ALPHA = 5.0
SAVE_STEPS    = 100
EVAL_STEPS    = 100
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 3
LR_SCHEDULER  = "cosine"
SEED          = 42


def format_chat(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    return "\n".join(parts) + "\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} Training — Blackwell Edition")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-merge",  action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"{MODEL_NAME} Training — Blackwell Edition")
    print(f"{'='*60}")
    print(f"Base:           {BASE_MODEL}")
    print(f"Method:         bf16 LoRA r={LORA_R} (full precision)")
    print(f"Epochs:         {NUM_EPOCHS}")
    print(f"Batch (eff):    {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"Max seq:        {MAX_SEQ_LENGTH}")
    print(f"Train data:     {TRAIN_FILE}")
    print(f"Eval data:      {EVAL_FILE}")
    print(f"Output:         {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    if not TRAIN_FILE.exists():
        print(f"ERROR: Training data not found at {TRAIN_FILE}")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────
    print("Loading training data...")
    with open(TRAIN_FILE) as f:
        train_raw = [json.loads(l) for l in f if l.strip()]
    if args.max_samples:
        train_raw = train_raw[:args.max_samples]
    if args.dry_run:
        train_raw = train_raw[:50]
        print("DRY RUN — 50 samples")

    eval_raw = []
    if EVAL_FILE.exists():
        with open(EVAL_FILE) as f:
            eval_raw = [json.loads(l) for l in f if l.strip()]
        if args.dry_run:
            eval_raw = eval_raw[:20]

    print(f"  Train samples: {len(train_raw):,}")
    print(f"  Eval samples:  {len(eval_raw):,}")

    train_texts = [format_chat(p["messages"]) for p in train_raw]
    eval_texts  = [format_chat(p["messages"]) for p in eval_raw]

    train_sha = sha256_file(TRAIN_FILE)
    print(f"  Train SHA256:  {train_sha[:16]}...")

    # ── Load model (bf16 LoRA) ─────────────────────────────────────
    print(f"\nLoading {BASE_MODEL} (bf16 LoRA — full precision)...")
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    import torch

    model, _proc = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,     # bf16 full precision — 96GB Blackwell
    )

    # Load text-only tokenizer directly to bypass unsloth_zoo's VL dispatch.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )

    model.print_trainable_parameters()

    # ── Dataset (text-field, packing handles tokenization) ─────────
    from datasets import Dataset

    print("\nBuilding datasets...")
    train_ds = Dataset.from_dict({"text": train_texts})
    eval_ds  = Dataset.from_dict({"text": eval_texts}) if eval_texts else None

    # ── Training ───────────────────────────────────────────────────
    from trl import SFTTrainer, SFTConfig

    # packing=True: bins short sequences into 4096-token windows → no padding waste
    # With 23K pairs avg ~1500 tok, packing roughly doubles throughput
    steps_per_epoch = len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    print(f"\n  Steps/epoch:   {steps_per_epoch:,}")
    print(f"  Total steps:   {total_steps:,}")
    print(f"  Warmup steps:  {warmup_steps:,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=EVAL_STEPS if eval_ds else None,
        bf16=True,
        fp16=False,
        seed=SEED,
        dataloader_num_workers=4,
        remove_unused_columns=True,
        report_to="none",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=True,
        neftune_noise_alpha=NEFTUNE_ALPHA,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print(f"\nStarting {MODEL_NAME} training at {datetime.now().isoformat()}")
    trainer.train()
    print(f"\nTraining complete at {datetime.now().isoformat()}")

    # ── Save ───────────────────────────────────────────────────────
    final_dir = OUTPUT_DIR / "final"
    print(f"\nSaving adapter → {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # ── Provenance ─────────────────────────────────────────────────
    prov = {
        "model_name":    MODEL_NAME,
        "base_model":    BASE_MODEL,
        "method":        "bf16 LoRA r=64 (full precision)",
        "gpu":           "RTX PRO 6000 Blackwell 96GB",
        "lora_r":        LORA_R,
        "lora_alpha":    LORA_ALPHA,
        "max_seq":       MAX_SEQ_LENGTH,
        "learning_rate": LEARNING_RATE,
        "epochs":        NUM_EPOCHS,
        "train_samples": len(train_ds),
        "train_sha256":  train_sha,
        "judge_model":   "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "verdict_filter": "PASS_ONLY",
        "completed_utc": datetime.utcnow().isoformat(),
    }
    with open(OUTPUT_DIR / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(f"Provenance → {OUTPUT_DIR / 'provenance.json'}")

    if not args.skip_merge:
        print("\nMerging LoRA adapter into base model...")
        merged_dir = OUTPUT_DIR / "merged"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        print(f"Merged model → {merged_dir}")

    print(f"\n{'='*60}")
    print(f"{MODEL_NAME} — DONE")
    print(f"  Adapter: {final_dir}")
    print(f"  GPU:     RTX PRO 6000 Blackwell 96GB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
