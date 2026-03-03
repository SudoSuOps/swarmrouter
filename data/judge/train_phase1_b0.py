#!/usr/bin/env python3
"""
SwarmJudge-27B Block-0 Phase 1 Training
========================================
23,000 PASS pairs — Llama 4 Maverick judge — 1 epoch — identity lock

Data:   /data2/swarmjudge_27b_data/phase1_b0/phase1_train.jsonl
Eval:   /data2/swarmjudge_27b_data/phase1/phase1_eval.jsonl
Output: /data2/swarmjudge-27b-b0/
Steps:  ~719  (23K / eff_batch_32 × 1 epoch)
ETA:    ~12h on Blackwell
"""
import sys
import importlib
from pathlib import Path

# ── Patch constants before importing main module ────────────────────
DATA_DIR = Path("/data2/swarmjudge_27b_data")

# Load the base training module
spec = importlib.util.spec_from_file_location(
    "train_base", DATA_DIR / "train_swarmjudge_27b.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["train_base"] = mod

# Override Phase 1 config BEFORE exec
mod.MODEL_NAME  = "SwarmJudge-27B-B0-Phase1"
mod.OUTPUT_DIR  = Path("/data2/swarmjudge-27b-b0")
mod.TRAIN_FILE  = Path("/data2/swarmjudge_27b_data/phase1_b0/phase1_train.jsonl")
mod.EVAL_FILE   = Path("/data2/swarmjudge_27b_data/phase1/phase1_eval.jsonl")
mod.NUM_EPOCHS  = 1
mod.SAVE_STEPS  = 100
mod.EVAL_STEPS  = 100

spec.loader.exec_module(mod)

# Re-apply overrides (exec_module resets them from source)
mod.MODEL_NAME  = "SwarmJudge-27B-B0-Phase1"
mod.OUTPUT_DIR  = Path("/data2/swarmjudge-27b-b0")
mod.TRAIN_FILE  = Path("/data2/swarmjudge_27b_data/phase1_b0/phase1_train.jsonl")
mod.EVAL_FILE   = Path("/data2/swarmjudge_27b_data/phase1/phase1_eval.jsonl")
mod.NUM_EPOCHS  = 1
mod.SAVE_STEPS  = 100
mod.EVAL_STEPS  = 100

if __name__ == "__main__":
    mod.main()
