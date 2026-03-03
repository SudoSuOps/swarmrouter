"""
SwarmJudge-14B Training Script — Blackwell Edition
=====================================================

bf16 LoRA on Qwen2.5-14B-Instruct using the RTX PRO 6000 Blackwell (96GB).
Full precision, full sequence length, proper foundation model training.

Usage:
    python3 train_swarmjudge_14b.py [--dry-run] [--max-samples N]
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════
# Config — Blackwell 96GB
# ═══════════════════════════════════════════════════════════════════════

BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
OUTPUT_DIR = Path("/data2/swarmjudge-14b")
DATA_DIR = Path("/data2/swarmjudge_data")
TRAIN_FILE = DATA_DIR / "swarmjudge_7b_train.jsonl"
EVAL_FILE = DATA_DIR / "swarmjudge_7b_eval.jsonl"

# LoRA — bf16 (not QLoRA)
MAX_SEQ_LENGTH = 4096
LORA_R = 64
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training — Blackwell can handle bigger batches
BATCH_SIZE = 2
GRAD_ACCUM = 16       # Effective batch = 32
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
NEFTUNE_ALPHA = 5.0
SAVE_STEPS = 500
LOGGING_STEPS = 10
LR_SCHEDULER = "cosine"


def format_chat(messages: list[dict]) -> str:
    """Format messages into Qwen2.5 chat template."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    print(f"SwarmJudge-14B Training — Blackwell Edition")
    print(f"{'='*60}")
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, bf16 (NOT QLoRA)")
    print(f"Data: {TRAIN_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # ── Load data ───────────────────────────────────────────────────
    print("Loading training data...")
    train_texts = []
    with open(TRAIN_FILE) as f:
        for line in f:
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
                r = json.loads(line)
                text = format_chat(r["messages"])
                eval_texts.append({"text": text})
        print(f"  Eval samples: {len(eval_texts):,}")

    # ── Model — bf16 LoRA, NOT 4-bit ────────────────────────────────
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,       # Auto-detect bf16
        load_in_4bit=False,  # Full precision — we have 96GB
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    import torch
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram_used:.1f} / {vram_gb:.1f} GB")

    # ── Estimate steps ──────────────────────────────────────────────
    steps_per_epoch = len(train_texts) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"  Steps/epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")

    if args.dry_run:
        print(f"\n  DRY RUN — would train {total_steps} steps")
        print(f"  Estimated time: ~{total_steps * 19 / 3600:.1f}h @ ~19s/step")
        return

    # ── Dataset ─────────────────────────────────────────────────────
    from datasets import Dataset

    train_ds = Dataset.from_list(train_texts)
    eval_ds = Dataset.from_list(eval_texts) if eval_texts else None

    # ── Train ───────────────────────────────────────────────────────
    from trl import SFTTrainer, SFTConfig

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        fp16=False,
        bf16=True,  # Blackwell native bf16
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,   # Packing works with 96GB
        neftune_noise_alpha=NEFTUNE_ALPHA,
        dataset_text_field="text",
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=SAVE_STEPS if eval_ds else None,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    print(f"\n{'='*60}")
    print(f"Starting SwarmJudge-14B training at {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    result = trainer.train()

    # ── Save ────────────────────────────────────────────────────────
    final_dir = OUTPUT_DIR / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nAdapter saved to {final_dir}")

    # ── Merge to full model ─────────────────────────────────────────
    print("Merging adapter to full model...")
    merged_dir = OUTPUT_DIR / "merged"
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to {merged_dir}")

    # ── Metadata ────────────────────────────────────────────────────
    meta = {
        "model_name": "SwarmJudge-14B-v1",
        "base_model": BASE_MODEL,
        "precision": "bf16 LoRA (NOT QLoRA)",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_samples": len(train_texts),
        "eval_samples": len(eval_texts),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "packing": True,
        "total_steps": result.global_step,
        "final_train_loss": result.training_loss,
        "train_runtime_sec": result.metrics.get("train_runtime", 0),
        "trainable_params": trainable,
        "gpu": "RTX PRO 6000 Blackwell (96GB)",
        "completed_at": datetime.now().isoformat(),
    }

    meta_path = OUTPUT_DIR / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")
    print(f"\nTraining complete. Loss: {result.training_loss:.4f}")


if __name__ == "__main__":
    main()
