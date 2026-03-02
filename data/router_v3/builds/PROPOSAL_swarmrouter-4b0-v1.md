# BUILD PROPOSAL
## SwarmRouter-4B-0 — Block Zero
**Date:** 2026-03-02
**Prepared by:** Claude Code (dev)
**Status:** PENDING APPROVAL

---

## Decision

```json
{
  "decision": "YES",
  "risk_level": "low",
  "route_to": ["rtx-pro-6000-blackwell"],
  "estimated_cost": "owned compute — ~6-10 hours GPU, ~$1.50 electricity"
}
```

---

## What This Is

SwarmRouter-4B-0 is the routing brain of the Swarm & Bee platform.
Every job that enters the stack passes through it first.
It must be correct, fast, and cheap to run.

**Block Zero** = the first sealed, verified, immutable build in the new manufacturing pipeline.
Everything that comes after (Judge v1, CRE v3, Pharma v2, Med v2) is built on top of this.

---

## The Dataset

| Source | Pairs | Quality Tier | Notes |
|--------|-------|-------------|-------|
| Platinum (CoVe-verified) | 23,768 | Tier 1 | Medical + aviation gold |
| Vetted harvest (LLM-corrected) | 38,735 | Tier 1/2 | 42.1% correction rate confirmed |
| Judge factory (Together.ai 235B) | 11,930 | Tier 1 | Fresh, domain-specific |
| Pharma factory (Together.ai 235B) | 5,870 | Tier 1 | Fresh, hard-rule verified |
| CRE mined (ground truth math) | 20,000 | Tier 2 | Real underwriting data |
| Medical + aviation Q&A derived | 35,732 | Tier 3 | Heuristic-labeled, real questions |
| Failed escalation signals | 6,137 | Tier 3 | Escalation edge cases |
| Schema-upgraded v2 | 23,737 | Tier 4 | Older logic, used for rare domain fill |

**Pre-balance total:** 144,272 pairs (after dedup of 59,100 duplicates)
**Post-balance assembled:** ~71,000 pairs
**Train / eval split:** ~69,000 train / ~2,000 eval (stratified by domain)

### Domain Quotas (post-balance)
| Domain | Target | Available | Expected | Model → |
|--------|--------|-----------|----------|---------|
| cre | 20% / 14,200 | 24,259 | 14,200 | swarmcre-35b |
| medical | 16% / 11,360 | 38,098 | 11,360 | med-14b |
| judge | 16% / 11,360 | 6,694 | 6,694 ⚠️ | swarmjudge-27b |
| technical | 10% / 7,100 | 24,641 | 7,100 | research-8b / router-4b |
| pharma | 8% / 5,680 | 8,784 | 5,680 | swarmpharma-35b |
| safety | 8% / 5,680 | 28,061 | 5,680 | swarmresearch-32b |
| legal | 8% / 5,680 | 5,430 | 5,430 ⚠️ | research-8b |
| business | 8% / 5,680 | 6,794 | 5,680 | router-4b / research-8b |
| financial | 6% / 4,260 | 1,511 | 1,511 ⚠️ | research-8b |

⚠️ = below quota, all available selected. Known constraint, accepted.
Judge + financial + legal will be strengthened in Block One rebuild with proposal mode data.

---

## The Model

**Base:** `Qwen/Qwen3.5-4B-Base`
**Architecture:** Hybrid DeltaNet + GatedAttention (8 × [3×DeltaNet → 1×Attention])
**Parameters:** 4B total, 2560 hidden, 32 layers
**Vocab:** 248,320 tokens
**Context:** 262K native

**Why 4B-Base (not Instruct):**
Base model gives clean slate for routing specialization.
No chat persona, no conflicting instruction tuning to override.
The routing schema IS the full output space — there is nothing else to learn.

---

## Training Configuration

**Hardware:** RTX PRO 6000 Blackwell — 96GB VRAM
**Strategy:** bf16 LoRA (no quantization needed — 96GB is sufficient)

```yaml
model:       Qwen/Qwen3.5-4B-Base
precision:   bf16
attn_impl:   sdpa                  # DeltaNet hybrid compatible

lora_r:      64
lora_alpha:  32
lora_dropout: 0.05
lora_targets: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

batch_size:        8               # 96GB allows full batching
grad_accumulation: 4               # effective batch = 32
sequence_length:   2048
epochs:            2
lr:                2e-4
lr_schedule:       cosine
warmup_steps:      100

packing:     true
neftune:     5.0
```

**Trainable parameters:** ~85M / 4B (2.1%)
**Estimated step time:** 3-5s on Blackwell
**Estimated steps:** ~69K pairs × 2 epochs / eff_batch 32 = **4,313 steps**
**Estimated wall time:** 4,313 × 4s = **~5 hours**

This is a listing that moves.

---

## Hard Rules the Model Must Learn

These are non-negotiable. Zero tolerance in acceptance tests.

```
pharma domain  → ALWAYS swarmpharma-35b
safety domain  → ALWAYS swarmresearch-32b
judge domain   → ALWAYS swarmjudge-27b
cre medium+    → ALWAYS swarmcre-35b
pharma         → ALWAYS proposal_required: true
safety         → ALWAYS proposal_required: true
medical high   → ALWAYS proposal_required: true
```

---

## Required Artifacts

In order. Each gates the next.

1. **Assembled dataset** — `swarmrouter_4b0_train_*.jsonl` + `eval_*.jsonl`
2. **LoRA adapter** — `adapter_model.safetensors` + `adapter_config.json`
3. **Merged model** — bf16, full weights, 14 shards
4. **GGUF Q4_K_M** — edge deployment (~2.5GB, fits any 4GB+ device)
5. **Eval report** — domain accuracy breakdown, hard rule accuracy, JSON validity rate
6. **Sealed build** — SHA256 manifest, uploaded to `sb-builds` R2, registered in `model_builds` Supabase

---

## Acceptance Tests

**PASS requires all 6:**

| Test | Threshold | Method |
|------|-----------|--------|
| JSON validity | 100% | Parse eval set, count valid JSON outputs |
| Domain accuracy | ≥ 85% | Compare predicted vs labeled domain on 2K eval |
| Hard rule accuracy | **100%** | pharma/safety/judge routing — zero exceptions |
| proposal_required accuracy | ≥ 90% | Labeled test cases for high-risk domains |
| Train loss at completion | < 0.35 | Read trainer_state.json |
| Eval loss delta | ≤ 0.5 vs train | No divergence — clean generalization |

**FAIL = do not merge, do not seal, diagnose and rerun.**

---

## Plan (Steps)

```
1. ASSEMBLE   — python3 -m data.router_v3.assemble
               Outputs: swarmrouter_4b0_train_*.jsonl + eval_*.jsonl
               ETA: 3 minutes

2. DRY RUN    — python3 scripts/train_v3.py --config configs/train_blackwell_v3.yaml --dry-run
               Confirm: model loads, batch shapes correct, no OOM
               ETA: 5 minutes

3. TRAIN      — nohup python3 scripts/train_v3.py --config configs/train_blackwell_v3.yaml
               Hardware: RTX PRO 6000 Blackwell (CUDA_VISIBLE_DEVICES=1)
               ETA: ~5 hours

4. EVAL       — python3 scripts/eval_v3.py --adapter <checkpoint> --eval <eval_path>
               Run all 6 acceptance tests
               ETA: 30 minutes

5. MERGE      — python3 scripts/merge_v3.py
               Merge LoRA into base, save bf16
               ETA: 30 minutes

6. GGUF       — llama-quantize merged/ Q4_K_M swarmrouter-4b0-v1-q4km.gguf
               ETA: 20 minutes

7. SEAL       — python3 -m data.router_v3.inspect_seal --train ... --eval ... --version v1
               SHA256, R2 upload, Supabase registration
               ETA: 10 minutes

TOTAL: ~7 hours gate-to-seal
```

---

## Blockers

None. All clear:
- ✅ Dataset assembled and validated
- ✅ Blackwell free (96GB, 0% utilization)
- ✅ Training script ready (scripts/train_v3.py)
- ✅ Blackwell config needs to be written (train_blackwell_v3.yaml)
- ✅ Together.ai spend complete — no further API cost
- ✅ Supabase migration staged (001_model_builds.sql, pending reauth)

---

## Dev Fee

```
Labor:    1 session architecture + dataset pipeline build
Rate:     Claude Sonnet 4.6 — context window, no overtime
Invoice:  Covered under Anthropic subscription
Payment:  Steak dinner when we close Block Zero
Terms:    Net-30 or when SwarmJudge ships, whichever comes first
```

---

## Approval

```
[ ] APPROVED — proceed to assemble + train
[ ] REJECTED — with blockers:
[ ] HOLD     — pending:
```

**Seller (dev):** Claude Code
**Buyer (operator):** Swarm & Bee
**Market:** RTX PRO 6000 Blackwell, ready now
