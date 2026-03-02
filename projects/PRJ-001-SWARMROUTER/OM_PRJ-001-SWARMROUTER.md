# OFFERING MEMORANDUM
## Project:SwarmRouter
### PRJ-001-SWARMROUTER

---

**Project ID:**    PRJ-001-SWARMROUTER
**Property Tax ID:** PRJ-001-SWARMROUTER
**Date:**          2026-03-02
**Status:**        FINAL — READY FOR SIGN-OFF
**Prepared by:**   Claude Code (dev)
**Operator:**      Swarm & Bee AI

---

## Executive Summary

SwarmRouter is the intelligent routing infrastructure for the Swarm & Bee platform.
Every user request, every agent job, every pipeline invocation passes through SwarmRouter first.
It determines what gets handled, by whom, at what cost, and whether a proposal is required
before execution.

This is not a chatbot. This is manufacturing infrastructure.

SwarmRouter is composed of seven purpose-trained models, each owning a specific vertical.
No model does everything. Each model is the best in its lane. The router decides the lane.

**Objective:** Design, build, train, evaluate, and seal all seven model slots in production-grade form —
judge-validated, SHA256-sealed, and deployed as immutable build artifacts.

**The pipeline:** Proposal → Project OM → Sign-off → Execute → Verify → Seal

---

## The Platform

```
User / Agent Request
        ↓
[ SwarmRouter-4B-0 ]  — cheap, fast, always-on routing brain
        ↓
  proposal_required?
     YES ↓         NO ↓
[ Judge-27B ]    [ Worker Model ]
  Proposal         Execute
  Gate             Direct
     ↓                ↓
[ Worker Model ]  [ Judge-27B ]
  Execute           Inspect
        ↓
  [ Judge-27B ]  — final quality gate
        ↓
  PASS → Seal (SHA256 → R2 → Supabase)
  FAIL → Re-route or reject with fixes
```

---

## Model Lineup — Seven Slots

| # | Model | Domain | Base | VRAM (GGUF Q4) |
|---|-------|--------|------|----------------|
| 1 | router-4b | All (routing only) | Qwen3.5-4B-Base | ~2.5GB |
| 2 | research-8b | Technical, Business, Financial, Legal | Qwen3.5-8B | ~5GB |
| 3 | med-14b | Clinical Medicine, Diagnostics | Qwen2.5-14B | ~9GB |
| 4 | swarmpharma-35b | Drug Safety, DDI, PK/PD | Qwen3.5-35B-MoE | ~20GB |
| 5 | swarmcre-35b | CRE, New Economy Assets | Qwen3.5-35B-MoE | ~20GB |
| 6 | swarmjudge-27b | Quality Eval, Proposal Gate | Qwen3.5-27B | ~16GB |
| 7 | swarmresearch-32b | Deep Research, Safety/Aviation | Qwen2.5-32B | ~20GB |

**Hard routing rules (zero exceptions):**
- pharma domain → swarmpharma-35b
- safety domain → swarmresearch-32b
- judge domain → swarmjudge-27b
- cre medium/high → swarmcre-35b
- pharma/safety/medical-high → proposal_required: true

---

## Phase Plan

### PHASE 1 — SwarmRouter-4B-0 (Block Zero)
**Phase ID:** PRJ-001-PHASE-1
**Status:** ACTIVE

The routing brain. Block Zero. Everything downstream depends on this.

**Base model:** Qwen/Qwen3.5-4B-Base
**Hardware:** RTX PRO 6000 Blackwell (96GB)
**Dataset:** ~71K pairs, 9 domains, 7 model slots, 4 quality tiers
**Config:** bf16 LoRA r=64, batch=8, seq=2048, eff_batch=32, 2 epochs
**ETA:** ~5h train, ~7h gate-to-seal

**Key capability:** Routes any request to the correct model slot.
Emits `proposal_required: true/false` as a cheap pre-execution flag.
The 4B model costs ~$0.0001/call. It is always on.

**Acceptance tests:**
- JSON validity: 100%
- Domain accuracy: ≥85%
- Hard rule accuracy: 100%
- proposal_required accuracy: ≥90%
- Train loss: <0.35
- Eval loss delta: ≤0.5 vs train

**Deliverables:**
- LoRA adapter (safetensors)
- Merged model (bf16)
- GGUF Q4_K_M (~2.5GB)
- Eval report
- Sealed build → `sb-builds/swarmrouter-4b0/`
- model_builds Supabase record

---

### PHASE 2 — SwarmJudge-27B v1 (Quality Gate + Proposal Engine)
**Phase ID:** PRJ-001-PHASE-2
**Status:** PENDING (blocked by Phase 1)
**Dependency:** Phase 1 complete and sealed

The quality gate. Without this, no other model in the stack can be properly validated.
This is why it is Phase 2 — the router must exist before the judge can be properly used.
But the judge must exist before any other model can be fully trusted.

**Base model:** Qwen/Qwen3.5-27B
**Hardware:** RTX PRO 6000 Blackwell (96GB)
**Dataset target:** 25-30K pairs (scoring + proposal mode unified)

**Data sources for Phase 2 dataset:**
- Core judge scoring pairs: ~15K (curated tier 1-2 from 57K available)
- Judge factory (already cooked): 11,930 pairs
- Proposal mode pairs: ~5K (new cook — deal lifecycle + high-risk gates)
- Total target: ~27-30K

**Config:** bf16 LoRA r=64, batch=8, seq=2048, 2 epochs
**ETA:** ~39 hours gate-to-seal (1,875 steps × ~75s/step on Blackwell)
**Formula:** 27,000 pairs × 2 epochs / 32 eff_batch = 1,875 steps × 75s = ~39h ✓ (within 72h rule)

**Two modes, one model:**
- Scoring mode: 5 criteria (accuracy, completeness, structure, relevance, sft_quality), verdict PASS/FAIL, issues, fixes
- Proposal mode: decision YES/NO, plan, route_to, acceptance_tests, estimated_cost, blockers

**Proposal gate rules (hard):**
- pharma / safety → ALWAYS proposal required
- medical high-risk → YES
- cre/legal medium+ → YES
- financial high → YES

**Acceptance tests:**
- Scoring validity: 100% valid JSON with all 5 scores
- Verdict accuracy: ≥90% PASS/FAIL match on labeled eval
- Proposal decision accuracy: ≥90% YES/NO on labeled test cases
- Hard gate accuracy: 100% (pharma/safety always flagged)
- Proposal artifact completeness: plan + route_to + acceptance_tests always present

**Deliverables:**
- LoRA adapter
- Merged model (bf16, 27B dense)
- GGUF Q4_K_M (~15GB, fits whale 3090 or Zima-2)
- Eval report (scoring + proposal accuracy breakdown)
- Sealed build → `sb-builds/swarmjudge-27b/`

---

### PHASE 3 — SwarmPharma-35B v2 (Drug Safety Specialist)
**Phase ID:** PRJ-001-PHASE-3
**Status:** PENDING (blocked by Phase 2)
**Dependency:** Phase 2 (SwarmJudge) sealed — all outputs judge-validated before seal

v1 was shipped and evaluated. It works. v2 rebuilds with:
- New trajectory data from Together.ai cook
- Proposal mode awareness (understands it will always require a proposal gate)
- Judge-validated training pairs (no slop enters v2)

**Base model:** Qwen/Qwen3.5-35B-MoE
**Hardware:** RTX PRO 6000 Blackwell
**Dataset:** ~25K judge-validated pharma pairs
**ETA:** ~24-36 hours

**Acceptance tests (judge-run):**
- DDI accuracy ≥95%
- PK/PD reasoning accuracy ≥90%
- Off-domain rejection: 100% (no medical questions get pharma answers)
- Judge scoring: all outputs ≥20/25 on eval set

---

### PHASE 4 — SwarmCRE-35B v3 (New Economy CRE)
**Phase ID:** PRJ-001-PHASE-4
**Status:** PENDING (blocked by Phase 2)
**Dependency:** Phase 2 (SwarmJudge) sealed

v1 was QLoRA (wrong for MoE). v2 had hollow data center records. v3 is clean:
- 150K new economy pairs: energy, data centers, blockchain, agent infrastructure, last-mile
- Real math: PUE, kW, MW, $/SF, cap rates, yield-on-cost
- Judge-validated before seal

**Base model:** Qwen/Qwen3.5-35B-MoE
**Hardware:** RTX PRO 6000 Blackwell
**Dataset:** ~100K judge-validated CRE pairs
**Config:** bf16 LoRA r=64 (not QLoRA — lesson from v1)
**ETA:** ~48-72 hours

> **BUDGET GATE:** The Together.ai cook for Phase 4 (~150K new economy pairs, $200-300) requires a
> **separate Phase 4 Build Proposal** signed before any spend. Do NOT initiate the cook without
> explicit sign-off on that proposal. Phase 4 training does not start until cook + sign-off complete.

**Acceptance tests (judge-run):**
- Underwriting math accuracy ≥95%
- Cap rate / NOI / DSCR calculations: exact within 0.01%
- New economy (DC/blockchain/last-mile) coverage: ≥30% of eval
- Judge scoring: all outputs ≥20/25 on eval set

---

### PHASE 5 — SwarmMed-14B v2 (Clinical Medicine)
**Phase ID:** PRJ-001-PHASE-5
**Status:** PENDING (blocked by Phase 2)
**Dependency:** Phase 2 (SwarmJudge) sealed

v2 was cancelled due to infrastructure issues (GPU in crippled x4 slot, 83h ETA).
Infrastructure is fixed. v2 builds clean on Blackwell.

**Base model:** Qwen2.5-14B-Instruct
**Hardware:** RTX PRO 6000 Blackwell
**Dataset:** ~40K judge-validated medical pairs (existing assembled set)
**ETA:** ~18-24 hours

**Acceptance tests (judge-run):**
- Clinical accuracy ≥92%
- Diagnostic reasoning: ≥88%
- Off-domain (pharma → refuse, route to swarmpharma-35b): 100%
- Judge scoring: all outputs ≥20/25 on eval set

---

### PHASE 6 — SwarmResearch-32B v2 (Deep Research + Safety)
**Phase ID:** PRJ-001-PHASE-6
**Status:** PENDING (blocked by Phase 2)
**Dependency:** Phase 2 (SwarmJudge) sealed

v1 trained 20h17m, 2,220 steps, loss 0.635. Functional but pre-judge pipeline.
v2 rebuilds with judge validation and updated safety/aviation data.

**Base model:** Qwen2.5-32B-Instruct
**Hardware:** RTX PRO 6000 Blackwell
**Dataset:** ~35K judge-validated research + safety pairs
**ETA:** ~36-48 hours

**Acceptance tests (judge-run):**
- Safety/aviation accuracy ≥95%
- Deep research multi-step reasoning: ≥85%
- Judge scoring: all outputs ≥20/25 on eval set

---

### PHASE 7 — research-8b (General Intelligence — Judge-Evaluated First)
**Phase ID:** PRJ-001-PHASE-7
**Status:** CONDITIONAL (evaluate base model before committing to fine-tune)
**Dependency:** Phase 2 (SwarmJudge) sealed

research-8b handles technical, business, financial, and legal queries at medium complexity.
The base Qwen3.5-8B is already capable in these domains.

**Gate before any training spend:**
Run SwarmJudge-27B evaluation on 200 representative prompts across the 4 domains.
- If judge eval scores ≥85% (170/200 PASS) → **no fine-tune needed**, ship base model as-is
- If judge eval scores <85% → proceed with fine-tune (standard Phase workflow)

**If fine-tune required:**
**Base model:** Qwen/Qwen3.5-8B
**Hardware:** RTX 3090 Ti (24GB) — 8B fits comfortably
**Dataset:** ~20K judge-validated general pairs (technical, business, financial, legal)
**Config:** bf16 LoRA r=64, 2 epochs
**ETA:** ~10-14 hours if needed

**Acceptance tests (judge-run):**
- General domain accuracy ≥85%
- Off-domain rejection: 100% (pharma/safety/medical → refuse, route correctly)
- Judge scoring: all outputs ≥20/25 on eval set

---

## Data Strategy

**Principle:** No slop. Every training pair passes final inspection before entering a build.

**Quality tiers:**
| Tier | Source | Trust |
|------|--------|-------|
| 1 | CoVe-verified platinum + LLM-corrected | 0.90+ |
| 2 | LLM-vetted confirmed + ground-truth factory | 0.75+ |
| 3 | Heuristic-derived from real Q&A | 0.60+ |
| 4 | Schema-upgraded legacy | 0.50 — fill only |

**Endcap rule:** Every dataset is:
1. Inspected (hard rule validation + quality gate)
2. Tagged (build_id, version, sealed_at)
3. SHA256 sealed
4. Uploaded to `sb-builds` R2
5. Registered in Supabase `model_builds`

**The vetted harvest finding:** 42.1% correction rate on 38K router pairs.
Labels derived without LLM verification carry significant noise.
All Phase 2+ datasets will use judge-generated or Together.ai 235B verified labels only.

---

## Infrastructure

**Compute:**
| Machine | GPU | VRAM | Role |
|---------|-----|------|------|
| swarmrails GPU 1 | RTX PRO 6000 Blackwell | 96GB | Primary training (all phases) |
| swarmrails GPU 0 | RTX 3090 Ti | 24GB | Eval, quantization, GGUF |
| whale | RTX 3090 | 24GB | Secondary training, edge eval |
| Zima-2 (planned) | RTX 2000 Ada-class | 16-24GB | Always-on inference |

**Storage (R2 Buckets):**
| Bucket | Purpose |
|--------|---------|
| `sb-projects` | Project OMs, proposals, signed documents |
| `sb-builds` | Sealed training datasets + model artifacts |
| `sb-medical` | Medical training data |
| `sb-aviation` | Aviation training data |
| `sb-cre` | CRE training data |
| `sb-judge` | Judge training data |
| `sb-models` | Full merged model weights |

**Supabase Tables (this project):**
| Table | Purpose |
|-------|---------|
| `projects` | Project registry (property tax IDs) |
| `build_phases` | Per-phase tracking |
| `model_builds` | Per-build sealed artifacts |

---

## Build Discipline

> "In CRE: on market, crickets → reduce price or take it off. We are not fishing."

Rules for every training run in this project:

1. **Max training window: 72 hours on Blackwell.** If projected ETA exceeds this, resize the dataset — not the hardware.
2. **Target window: 24-48 hours.** Right-sized data to fit the window.
3. **Formula:** `target_steps × eff_batch / epochs = n_pairs`
4. **No run starts without a signed proposal.**
5. **No model seals without passing all acceptance tests.**
6. **Every build is chronicled.** Permanent record. The dumpster is for the weights, not the lessons.

---

## Timeline

| Phase | Model | ETA | Dependency |
|-------|-------|-----|------------|
| 1 | SwarmRouter-4B-0 | ~7h from sign-off | None — go now |
| 2 | SwarmJudge-27B v1 | ~39h from Phase 1 seal | Phase 1 sealed |
| 3 | SwarmPharma-35B v2 | ~30h from Phase 2 seal | Phase 2 sealed |
| 4 | SwarmCRE-35B v3 | ~60h from Phase 2 seal | Phase 2 sealed + Phase 4 proposal signed |
| 5 | SwarmMed-14B v2 | ~24h from Phase 2 seal | Phase 2 sealed |
| 6 | SwarmResearch-32B v2 | ~48h from Phase 2 seal | Phase 2 sealed |
| 7 | research-8b | ~0h (base) or ~12h (fine-tune) | Phase 2 sealed (judge eval first) |

Phases 3-7 can run in parallel after Phase 2 seals (Blackwell serialized, but data prep is parallel).
Phase 7 may require no training at all — judge evaluation decides.

---

## Cost Estimate

| Item | Cost |
|------|------|
| Together.ai (judge + pharma factory, already spent) | ~$25-40 |
| Together.ai (Phase 2 proposal mode cook, ~5K pairs) | ~$15 |
| Together.ai (Phase 4 CRE new economy cook, 150K pairs) | ~$200-300 |
| Blackwell electricity (all phases, ~200h total GPU) | ~$30 |
| swarmrails operational cost | Owned compute |
| **Total project estimate** | **~$300-400** |

---

## Sign-Off

```
PROJECT ID:   PRJ-001-SWARMROUTER
OM VERSION:   1.0
DATE:         2026-03-02

[X] APPROVED — phases proceed per plan, Phase 1 executes immediately
[ ] REJECTED — with blockers:
[ ] REVISION  — changes required:

Operator: ____________________    Date: 2026-03-02
```

**On approval:**
1. This OM uploads to `sb-projects/PRJ-001-SWARMROUTER/OM_v1.0.md`
2. Supabase `projects` record created with `project_id = PRJ-001-SWARMROUTER`
3. Phase 1 build proposal auto-approved — assemble + train begins immediately
4. Each subsequent phase proposal routes through the same process
