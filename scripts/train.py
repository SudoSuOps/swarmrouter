#!/usr/bin/env python3
"""
SwarmCRE Training Script
QLoRA fine-tuning on Qwen3.5-35B-A3B (MoE) using Unsloth.
Target: RTX 6000 Blackwell (48GB)

Model family:
  SwarmCRE-32B  → Qwen3.5-35B-A3B  (this script)
  SwarmCRE-14B  → Qwen3.5-27B      (same script, different config)
  SwarmCRE-8B   → distilled from 32B (see scripts/distill.py)
  SwarmCRE-4B   → distilled from 8B
"""

import os
import sys
import json
import yaml
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig


def load_config(config_path: Path) -> Dict:
    """Load training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonl_dataset(jsonl_path: Path, max_samples: int = 0) -> List[Dict]:
    """Load JSONL dataset with optional sample limit."""
    data = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            data.append(record)
            if max_samples and len(data) >= max_samples:
                break
    return data


def formatting_prompts_func(example):
    """Format example(s) using Qwen3.5 chat template.

    Always returns a list of strings (required by Unsloth SFTTrainer).
    Uses instruct mode (no thinking blocks) — CRE answers are deterministic math.
    """
    messages = example["messages"]
    # Single example: messages is a list of dicts [{"role": ..., "content": ...}, ...]
    if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
        text = "".join(
            f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            for msg in messages
        )
        return [text]
    # Batch: messages is a list of lists
    return [
        "".join(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n" for msg in msgs)
        for msgs in messages
    ]


def print_model_info(model, config: Dict):
    """Print model architecture and training info."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = trainable_params / total_params * 100 if total_params > 0 else 0

    print(f"  Total parameters:     {total_params:>14,}")
    print(f"  Trainable parameters: {trainable_params:>14,} ({pct:.2f}%)")
    print(f"  LoRA rank:            {config['lora']['r']}")
    print(f"  LoRA alpha:           {config['lora']['lora_alpha']}")
    print(f"  Max sequence length:  {config['training']['max_seq_length']:,}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SwarmCRE Training")
    parser.add_argument("--config", type=Path, default=None,
                        help="Config file (default: configs/train.yaml)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit training samples (0 = all, useful for smoke tests)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model and data but don't train")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from checkpoint directory")
    args = parser.parse_args()

    # Load config
    base_dir = Path(__file__).parent.parent
    config_path = args.config or (base_dir / "configs" / "train.yaml")
    config = load_config(config_path)

    model_name = config["model"]["base_model"]
    output_dir = base_dir / config["model"]["output_dir"]

    print("=" * 70)
    print("SwarmCRE Training")
    print("=" * 70)
    print(f"  Base model:    {model_name}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Epochs:        {config['training']['num_train_epochs']}")
    print(f"  Sequence len:  {config['training']['max_seq_length']}")
    print(f"  Batch size:    {config['training']['per_device_train_batch_size']} x "
          f"{config['training']['gradient_accumulation_steps']} = "
          f"{config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    if args.max_samples:
        print(f"  Sample limit:  {args.max_samples:,}")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────
    print("\nLoading model and tokenizer...")

    # Build max_memory dict if specified
    max_memory = None
    hw_config = config.get("hardware", {})
    if hw_config.get("max_memory"):
        max_memory = hw_config["max_memory"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config["training"]["max_seq_length"],
        dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float16,
        load_in_4bit=config["model"]["load_in_4bit"],
        device_map=hw_config.get("device_map", "auto"),
    )

    # ── Add LoRA adapters ─────────────────────────────────────
    print("Adding LoRA adapters...")

    lora_config = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=config["training"]["seed"],
        modules_to_save=lora_config.get("modules_to_save"),
    )

    print_model_info(model, config)

    # ── Load datasets ─────────────────────────────────────────
    print("\nLoading datasets...")

    train_path = base_dir / config["training"]["train_dataset"]
    eval_path = base_dir / config["training"]["eval_dataset"]

    train_data = load_jsonl_dataset(train_path, max_samples=args.max_samples)
    eval_data = load_jsonl_dataset(eval_path)

    # Pre-format into text strings to avoid Unsloth's 64-worker tokenization OOM
    print("  Formatting chat templates...")
    for record in train_data:
        record["text"] = "".join(
            f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            for msg in record["messages"]
        )
    for record in eval_data:
        record["text"] = "".join(
            f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            for msg in record["messages"]
        )

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Eval samples:     {len(eval_dataset):,}")

    if args.dry_run:
        print("\n-- DRY RUN: skipping training --")
        print("Model loaded successfully. Config validated.")
        return

    # ── Training arguments ────────────────────────────────────
    print("\nConfiguring training...")
    output_dir.mkdir(parents=True, exist_ok=True)

    tc = config["training"]

    # Build SFTConfig
    sft_kwargs = dict(
        output_dir=str(output_dir),

        # Dataset
        max_seq_length=tc["max_seq_length"],
        dataset_text_field="text",
        dataset_num_proc=8,  # Limit tokenization workers to avoid OOM

        # Training
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=tc["per_device_eval_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],

        # Optimization
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        warmup_ratio=tc["warmup_ratio"],
        optim=tc["optim"],
        weight_decay=tc["weight_decay"],
        max_grad_norm=tc["max_grad_norm"],

        # Logging
        logging_steps=tc["logging_steps"],
        eval_strategy=tc["eval_strategy"],
        eval_steps=tc["eval_steps"],
        save_strategy=tc["save_strategy"],
        save_steps=tc["save_steps"],
        save_total_limit=tc["save_total_limit"],

        # Precision
        fp16=tc["fp16"],
        bf16=tc["bf16"],

        # Efficiency
        dataloader_num_workers=tc["dataloader_num_workers"],
        dataloader_pin_memory=tc["dataloader_pin_memory"],

        # Misc
        seed=tc["seed"],
    )

    # Optional: packing and NEFTune
    if tc.get("packing"):
        sft_kwargs["packing"] = True
    if tc.get("neftune_noise_alpha"):
        sft_kwargs["neftune_noise_alpha"] = tc["neftune_noise_alpha"]
    if tc.get("warmup_steps"):
        sft_kwargs["warmup_steps"] = tc["warmup_steps"]

    # W&B
    wandb_config = config.get("wandb", {})
    if wandb_config.get("enabled"):
        sft_kwargs["report_to"] = "wandb"
        os.environ["WANDB_PROJECT"] = wandb_config.get("project", "swarmcre")
        run_name = wandb_config.get("name", f"swarmcre-{datetime.now().strftime('%Y%m%d-%H%M')}")
        sft_kwargs["run_name"] = run_name
        if wandb_config.get("entity"):
            os.environ["WANDB_ENTITY"] = wandb_config["entity"]
    else:
        sft_kwargs["report_to"] = "none"

    training_args = SFTConfig(**sft_kwargs)

    # ── Create trainer ────────────────────────────────────────
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Stats
    effective_batch = tc["per_device_train_batch_size"] * tc["gradient_accumulation_steps"]
    total_steps = len(train_dataset) * tc["num_train_epochs"] // effective_batch
    print(f"  Total training steps: ~{total_steps:,}")
    print(f"  Effective batch size: {effective_batch}")

    # ── Train ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    start_time = datetime.now()

    if args.resume:
        print(f"  Resuming from: {args.resume}")
        trainer.train(resume_from_checkpoint=str(args.resume))
    else:
        trainer.train()

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    print("\n" + "=" * 70)
    print(f"Training complete! Time: {training_time:.0f}s ({training_time/3600:.1f}h)")
    print("=" * 70)

    # ── Save ──────────────────────────────────────────────────
    print("\nSaving final model...")
    final_output = output_dir / "final"
    final_output.mkdir(exist_ok=True)

    model.save_pretrained(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    print(f"  Model saved to: {final_output}")

    # Save training metadata
    metadata = {
        "model_family": "SwarmCRE",
        "model_variant": "SwarmCRE-32B",
        "base_model": model_name,
        "architecture": "MoE (35B total / 3B active)",
        "lora_config": config["lora"],
        "training_config": config["training"],
        "hardware": config.get("hardware", {}),
        "training_time_seconds": training_time,
        "training_time_hours": round(training_time / 3600, 2),
        "total_steps": total_steps,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "dataset": {
            "train": str(train_path),
            "eval": str(eval_path),
            "factory": "SwarmCRE Dataset Factory v1",
            "task_types": [
                "underwriting_calc", "ic_memo", "lease_reasoning",
                "market_comp_narrative", "rent_roll_extraction",
                "t12_normalization", "lease_abstract_extraction",
                "risk_triage", "exchange_1031", "tax_analysis",
                "loi_deliverable", "structured_agent_output",
            ],
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nTraining metadata saved.")
    print("\nNext steps:")
    print("  1. Evaluate: python scripts/eval.py --model models/swarmcre-32b-v1/final")
    print("  2. Merge LoRA: python scripts/merge_lora.py --model-path models/swarmcre-32b-v1/final")
    print("  3. Export GGUF: python scripts/export_gguf.py")
    print("  4. Serve API: python scripts/serve_api.py")


if __name__ == "__main__":
    main()
