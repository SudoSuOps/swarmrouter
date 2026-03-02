#!/usr/bin/env python3
"""
SwarmRouter v3 — Training Script
QLoRA fine-tuning of Qwen3.5-4B-Base using HuggingFace PEFT + TRL.
No Unsloth dependency — works with transformers 5.x (qwen3_5 arch).
Target: whale — 1x NVIDIA GeForce RTX 3090 (24GB)
"""

import os
import sys
import json
import yaml
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


def load_config(config_path: Path) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonl_dataset(jsonl_path: Path, max_samples: int = 0) -> List[Dict]:
    data = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if max_samples and len(data) >= max_samples:
                break
    return data


def print_model_info(model, config: Dict):
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

    parser = argparse.ArgumentParser(description="SwarmRouter v3 Training")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    config_path = args.config or (base_dir / "configs" / "train_whale_v3.yaml")
    config = load_config(config_path)

    model_name = config["model"]["base_model"]
    output_dir = base_dir / config["model"]["output_dir"]

    print("=" * 70)
    print("SwarmRouter v3 Training (PEFT/TRL — no Unsloth)")
    print("=" * 70)
    print(f"  Base model:    {model_name}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Epochs:        {config['training']['num_train_epochs']}")
    print(f"  Sequence len:  {config['training']['max_seq_length']}")
    tc = config["training"]
    print(f"  Batch size:    {tc['per_device_train_batch_size']} x "
          f"{tc['gradient_accumulation_steps']} = "
          f"{tc['per_device_train_batch_size'] * tc['gradient_accumulation_steps']}")
    if args.max_samples:
        print(f"  Sample limit:  {args.max_samples:,}")
    print("=" * 70)

    # ── BitsAndBytes NF4 config ───────────────────────────────
    print("\nLoading model and tokenizer...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["load_in_4bit"],
        bnb_4bit_quant_type=config["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=config["model"]["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Base model — set chat template explicitly for SFTTrainer packing
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% endfor %}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # Memory-efficient; flash_attention_2 unsupported by DeltaNet hybrid
    )

    # Prepare for k-bit training (gradient checkpointing + cast norms)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # ── LoRA adapters ─────────────────────────────────────────
    print("Adding LoRA adapters...")
    lora_config = config["lora"]
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    print_model_info(model, config)

    # ── Load and format datasets ───────────────────────────────
    print("\nLoading datasets...")
    train_path = base_dir / config["training"]["train_dataset"]
    eval_path = base_dir / config["training"]["eval_dataset"]

    train_data = load_jsonl_dataset(train_path, max_samples=args.max_samples)
    eval_data = load_jsonl_dataset(eval_path)

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
        # Quick tokenization test
        sample = train_data[0]["text"]
        tokens = tokenizer(sample, return_tensors="pt")
        print(f"  Sample token count: {tokens['input_ids'].shape[1]}")
        print(f"  GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        return

    # ── Training config ────────────────────────────────────────
    print("\nConfiguring training...")
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_kwargs = dict(
        output_dir=str(output_dir),
        max_length=tc["max_seq_length"],
        dataset_text_field="text",
        dataset_num_proc=1,
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=tc["per_device_eval_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        gradient_checkpointing=tc["gradient_checkpointing"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        warmup_ratio=tc["warmup_ratio"],
        optim=tc["optim"],
        weight_decay=tc["weight_decay"],
        max_grad_norm=tc["max_grad_norm"],
        logging_steps=tc["logging_steps"],
        eval_strategy=tc["eval_strategy"],
        eval_steps=tc["eval_steps"],
        save_strategy=tc["save_strategy"],
        save_steps=tc["save_steps"],
        save_total_limit=tc["save_total_limit"],
        fp16=tc["fp16"],
        bf16=tc["bf16"],
        dataloader_num_workers=tc["dataloader_num_workers"],
        dataloader_pin_memory=tc["dataloader_pin_memory"],
        seed=tc["seed"],
        report_to="none",
    )

    if tc.get("packing"):
        sft_kwargs["packing"] = True
    if tc.get("neftune_noise_alpha"):
        sft_kwargs["neftune_noise_alpha"] = tc["neftune_noise_alpha"]
    if tc.get("warmup_steps"):
        sft_kwargs["warmup_steps"] = tc["warmup_steps"]

    training_args = SFTConfig(**sft_kwargs)

    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    effective_batch = tc["per_device_train_batch_size"] * tc["gradient_accumulation_steps"]
    total_steps = len(train_dataset) * tc["num_train_epochs"] // effective_batch
    print(f"  Total training steps: ~{total_steps:,}")
    print(f"  Effective batch size: {effective_batch}")

    # ── Train ──────────────────────────────────────────────────
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

    # ── Save ───────────────────────────────────────────────────
    print("\nSaving final model...")
    final_output = output_dir / "final"
    final_output.mkdir(exist_ok=True)

    model.save_pretrained(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    print(f"  Model saved to: {final_output}")

    metadata = {
        "model_family": "SwarmRouter",
        "model_variant": "SwarmRouter-v3",
        "base_model": model_name,
        "architecture": "Qwen3.5-4B-Base (DeltaNet hybrid)",
        "training_method": "QLoRA NF4 via HuggingFace PEFT",
        "lora_config": config["lora"],
        "training_config": config["training"],
        "training_time_seconds": training_time,
        "training_time_hours": round(training_time / 3600, 2),
        "total_steps": total_steps,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nTraining metadata saved.")
    print("\nNext steps:")
    print("  1. Merge LoRA: python scripts/merge_lora.py --model-path models/swarmrouter-v3/final")
    print("  2. Eval:       python scripts/eval.py --model models/swarmrouter-v3/final")
    print("  3. Export GGUF via llama.cpp on swarmrails")


if __name__ == "__main__":
    main()
