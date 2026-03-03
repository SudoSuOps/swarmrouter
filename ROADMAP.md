# SwarmJudge-9B-CRE — Block-0 Roadmap

> Commercial Real Estate AI franchise. Investment-grade. Field-ready. Agent-native.

---

## Vision

A vertical specialist AI stack purpose-built for commercial real estate — from the broker on the street to the investment committee at HQ. The model is not a chatbot. It is a **quality engine** that evaluates, scores, flags, and fixes CRE analysis in real time.

**The broker never leaves the field.** Site visits, dials, texts, doc reviews, LOI negotiations — the model meets them there. On the phone, on the tablet, on the desktop. Instant, accurate, investment-grade.

---

## Block-0: SwarmJudge-9B-CRE

### Architecture

- **Base model**: Qwen/Qwen3.5-9B (Mamba-Transformer hybrid)
- **Method**: bf16 LoRA r=64, full precision on RTX PRO 6000 Blackwell (96GB)
- **Judge**: Llama 4 Maverick 17Bx128E (5-criterion scoring)
- **Scoring**: accuracy · completeness · structure · relevance · sft_quality (1–5 each, 25 total)
- **Verdict**: PASS (≥20, all≥3, accuracy≥4) or FAIL (with issues + fixes)

---

### Phase 1 — Identity Lock ✅ COMPLETE

**Goal**: Lock the model's identity as a CRE evaluator. PASS-only pairs. No failure signal yet — just teach the model what correct CRE analysis looks like at every level of complexity.

| Metric | Value |
|--------|-------|
| Train pairs | 23,000 PASS |
| Cook runs | r1–r6 (seeds 300–800) |
| Judge model | Llama 4 Maverick 17Bx128E |
| Train loss | 0.2217 |
| Runtime | 4h 50m (Blackwell) |
| Adapter | `/data2/swarmjudge-9b-b0/final/` |
| Merged | `/data2/swarmjudge-9b-b0/merged/` (18GB bf16) |

**Task coverage**:
- 42% underwriting_calc — cap rates, NOI, debt service, LTV
- 26% ic_memo — investment committee analysis
- 10% t12_normalization — trailing 12, rent roll
- 10% lease_reasoning — NNN lease clauses, CAM, TI
- 10% risk_triage — deal risk identification
- 2%  exchange_1031 — tax-deferred exchange

---

### Phase 2 — Failure Recognition + Trajectory ⏳ TRAINING NOW

**Goal**: Teach the model what bad CRE analysis looks like and how to fix it. Mixed PASS/FAIL. Every FAIL pair includes explicit `issues[]` and `fixes[]` — the model learns to prescribe corrections, not just flag failures.

| Metric | Value |
|--------|-------|
| Train pairs | 23,000 (70% PASS / 30% FAIL) |
| Cook runs | r7–r11 (seeds 900–1300) |
| FAIL source | r1–r6 fill-in (6,900 banked free) |
| PASS source | r7–r11 new trajectory cooks (16,100) |
| Base | Phase 1 merged → new LoRA @ LR=2e-5 |
| Output | `/data2/swarmjudge-9b-cre-p2/` |
| ETA | ~5h on Blackwell |

**What Phase 2 teaches**:
- Hollow data detection (numbers without substance)
- Math verification failure (cap rate errors, DSCR miscalc)
- Incomplete analysis (missing risk factors, no market context)
- Lease precision gaps (wrong CAM structure, missing clauses)
- How to prescribe specific, actionable fixes

---

### Phase 3 — Best and Final 📋 PLANNED

**Goal**: Fill the investment-grade CRE gaps. Best-and-final model before Block-1.

**Target gaps**:
- PSA (Purchase and Sale Agreement) — key terms, due diligence, earnest money, contingencies
- LOI (Letter of Intent) — price, exclusivity, deposit, closing conditions
- STNL / NNN deep — credit tenant analysis, corporate guarantee, dark value, absolute vs modified NNN
- SNDA / Estoppel — subordination, non-disturbance, tenant confirmation letters
- Lease abstract — full abstraction workflow for investment-grade properties
- Investment-grade underwriting — S&P rated tenants, cap rate spreads by credit tier
- New era CRE — RWA tokenization, stablecoin settlement, on-chain title, DeFi collateral

**New era specifics**:
- ERC-1400 compliant token structure for fractional NNN ownership
- USDC closing on PSA vs traditional title
- Smart contract lease terms + automated rent distribution
- SEC Reg D / Reg A+ offering structure for tokenized CRE
- Blockchain escrow mechanics
- Stablecoin yield vs traditional 1031 analysis

---

## Block-1: Genesis

**Triggers**: Phase 3 eval passes quality bar → Block-0 final shipped → Block-1 begins.

**What changes**:
- New base (Qwen3.5-9B next checkpoint or equivalent)
- 20%+ new-era CRE in training mix
- Live production data from SwarmSkills agent economy
- Phase 1 identity lock from Block-0 Phase 3 as starting point
- Full GGUF pipeline: Q8_0 (desktop) · Q4_K_M (edge) · Q3_K_M (mobile)

---

## SwarmSkills Agent Economy

SwarmSkills are the live commercial intelligence layer (19 skills, live at router.swarmandbee.com). SwarmJudge-9B-CRE is the **quality gate** that makes the agent economy self-improving.

**19 live skills**:
`broker_senior` · `broker_junior` · `intelligence_query` · `bookmaker` · `deal_tracker` · `developer` · `signal_scraper` · `investor` · `exchange_1031` · `market_report` · `lead_scorer` · `email_composer` · `comp_analyzer` · `rent_roll_analyzer` · `debt_analyzer` · `tax_assessor` · `site_selector` · `portfolio_optimizer` · `news_digest`

**The flywheel**:

```
Broker query
    ↓
Router-2B → SwarmSkill runs
    ↓
SwarmJudge-9B-CRE evaluates output
    ↓
PASS → deliver to broker (< 2s)
FAIL → retry with 122B CEO or flag
    ↓
Every judged output = new training pair
    ↓
Block-1 trains on real production deals
```

Block-1 Genesis trains on real broker queries, real NNN comps, real PSAs, real lease abstracts. No lab data. The moat is production traffic.

---

## SwarmHQ Compute Fleet

**3 cluster rigs × 4× RTX PRO 6000 Blackwell (96GB) = 12 GPUs · 1,152GB VRAM**

```
Rig 1:  [9B-CRE] [9B-CRE] [9B-CRE] [Router-2B]
Rig 2:  [9B-CRE] [9B-CRE] [9B-CRE] [Router-2B]
Rig 3:  [122B CEO fp8] ──────── [122B CEO fp8]  +  2× overflow
```

**CEO models**:
- Qwen3.5-122B fp8 → 2× RTX 6000 (192GB) — complex IC memos, deal structuring
- Qwen3.5-397B MoE Q4_K_M → 3× RTX 6000 (22B active/token) — portfolio-level, board memos

**Routing logic**:
- Router-2B dispatches 80% of broker queries to 9B-CRE (fast, < 2s)
- Complex deals escalate to 122B CEO
- Board-level / portfolio analysis to 397B

---

## Distribution Ladder

```
SwarmHQ  (1,152GB fleet)
  └── 9B-CRE bf16 · 122B CEO · 397B MoE
         ↓  private network / API
Mac Mini · Desktop Appliance  (32–64GB)
  └── 9B-CRE Q8_0  (~9GB, near-lossless)
         ↓
Jetson Orin · Zima-S  (8–16GB)
  └── 9B-CRE Q4_K_M  (~5GB)
  └── 4B-CRE Q4_K_M  (~2.5GB)
         ↓
BeeMini · Mobile · Field  (4–8GB)
  └── Router-2B Q4_K_M  (~1.2GB)  ← instant dispatch
  └── 4B-CRE Q4_K_M  (~2.5GB)    ← local fast answers
  └── escalate → Cloudflare Worker (Qwen3-30B edge)
         ↓
Broker  ·  site visit  ·  dials  ·  texts  ·  doc review
```

Every device runs models trained from the same Block-0 lineage. The field broker gets investment-grade analysis on their phone. No friction. No waiting. No login walls.

---

## Franchise Stack

| Model | Size | Format | Target device | Role |
|-------|------|--------|---------------|------|
| SwarmJudge-9B-CRE | 18GB bf16 / 5GB Q4 | safetensors / GGUF | Rig GPU / Zima-S | Main CRE judge |
| SwarmJudge-4B-CRE | ~8GB bf16 / 2.5GB Q4 | GGUF | BeeMini / Jetson | Edge judge |
| SwarmRouter-2B | ~4GB bf16 / 1.2GB Q4 | GGUF | Mobile / BeeMini | Field router |
| 122B CEO | ~122GB fp8 | vLLM | 2× Rig GPU | Complex analysis |
| 397B MoE CEO | ~220GB Q4 | llama.cpp | 3× Rig GPU | Portfolio / board |

---

## Milestones

- [x] Block-0 Phase 1 — Identity lock (23K PASS, loss 0.2217)
- [ ] Block-0 Phase 2 — Failure recognition (23K PASS+FAIL, ~5h ETA)
- [ ] Block-0 Phase 3 — Best and final (PSA/LOI/STNL/RWA, ~25K pairs)
- [ ] Block-0 GGUF — Q8_0 + Q4_K_M quantization and edge deploy
- [ ] Block-1 Genesis — Live production training loop, new-era CRE at 20%+
