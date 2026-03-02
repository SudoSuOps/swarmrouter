#!/usr/bin/env python3
"""
SwarmRouter-4B-0 Dataset Factory
==================================
Async Together.ai generation across 9 domain streams.
Model: Qwen/Qwen3-235B-A22B-Instruct-2507-tput
Primary verticals: SwarmMed | SwarmCRE | SwarmJudge

Stream targets (total ~70K factory pairs):
  medical:   12K  — SwarmMed PRIMARY
  pharma:     6K  — SwarmPharma specialist
  cre:       18K  — SwarmCRE PRIMARY (largest stream)
  safety:     4K  — Aviation + industrial safety
  technical:  6K  — Software, engineering, ML infra
  business:   6K  — Strategy, ops, management
  financial:  5K  — Financial modeling, valuations
  legal:      5K  — Contracts, compliance, research
  judge:     12K  — SwarmJudge PRIMARY (quality/eval/policy)
  edge:       4K  — Adversarial, multi-domain, ambiguous
  ─────────────────
  Total:     78K

Usage:
    python3 -m data.router_v3.factory --stream all
    python3 -m data.router_v3.factory --stream cre --count 18000
    python3 -m data.router_v3.factory --stream judge --dry-run
"""

import asyncio
import json
import os
import random
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

from openai import AsyncOpenAI

from data.router_v3.schema import (
    DOMAINS, TASK_TYPES, COMPLEXITY, RISK_LEVELS,
    LATENCY_TIERS, COST_SENSITIVITY, MODELS, TOOLS,
    SYSTEM_PROMPT, get_recommended_model, needs_proposal,
)

API_KEY = os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = "https://api.together.xyz/v1"
GEN_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
FAST_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# ── Stream definitions ─────────────────────────────────────────────
STREAMS = {
    "medical": {
        "target": 12000,
        "domain": "medical",
        "description": "Clinical medicine — SwarmMed PRIMARY",
        "seed_topics": [
            "acute MI management in elderly patients",
            "sepsis protocol and antibiotic stewardship",
            "heart failure with reduced ejection fraction",
            "diabetic ketoacidosis emergency management",
            "stroke thrombolysis eligibility criteria",
            "post-surgical complication assessment",
            "oncology treatment toxicity management",
            "ICU ventilator weaning protocols",
            "pediatric fever workup and triage",
            "psychiatric medication interactions",
            "chronic kidney disease staging and management",
            "autoimmune disease differential diagnosis",
            "trauma scoring and resuscitation",
            "infectious disease outbreak protocols",
            "pre-operative cardiac risk assessment",
        ],
        "risk_weights": {"low": 0.15, "medium": 0.40, "high": 0.35, "critical": 0.10},
        "complexity_weights": {"low": 0.20, "medium": 0.45, "high": 0.35},
    },
    "pharma": {
        "target": 6000,
        "domain": "pharma",
        "description": "Drug safety, DDI, PK/PD — SwarmPharma",
        "seed_topics": [
            "drug-drug interaction between warfarin and NSAIDs",
            "renal dose adjustment for antibiotics",
            "QT prolongation risk with antipsychotics",
            "CYP450 enzyme inhibition and drug metabolism",
            "therapeutic drug monitoring for vancomycin",
            "pediatric dosing weight-based calculations",
            "anticoagulation bridging perioperative",
            "chemotherapy dose modification toxicity",
            "hepatic impairment pharmacokinetics",
            "pregnancy category drug safety",
            "insulin regimen optimization T1DM",
            "immunosuppressant trough levels transplant",
            "opioid equianalgesic conversion",
            "narrow therapeutic index monitoring",
            "biosimilar substitution protocols",
        ],
        "risk_weights": {"low": 0.10, "medium": 0.35, "high": 0.40, "critical": 0.15},
        "complexity_weights": {"low": 0.15, "medium": 0.40, "high": 0.45},
    },
    "cre": {
        "target": 18000,
        "domain": "cre",
        "description": "Commercial real estate + new economy — SwarmCRE PRIMARY",
        "seed_topics": [
            "data center underwriting PUE and critical power costs",
            "industrial warehouse cap rate analysis Alliance TX",
            "cold storage last-mile logistics lease valuation",
            "bitcoin mining facility and blockchain hosting economics",
            "hyperscaler data center NNN lease triple net terms",
            "AI training cluster colocation pricing GPU per kW",
            "industrial outdoor storage IOS market comp analysis",
            "tokenized real estate ERC-1400 digital security structure",
            "energy infrastructure power purchase agreement MW pricing",
            "flex industrial R&D space conversion cap rate analysis",
            "cross-dock logistics T12 normalization and NOI",
            "micro-fulfillment center urban last-mile proforma ROI",
            "GPU cluster colocation REIT yield analysis",
            "CRE DSCR calculation for new economy assets underwriting",
            "1031 exchange qualified intermediary DST timeline",
            "industrial land entitlement risk and absorption forecast",
            "tech campus flex space vacancy and absorption rates",
            "REIT 10-K NAV calculation and implied cap rate",
            "cap rate compression primary vs secondary industrial markets",
            "rent roll normalization infill warehouse lease audit",
            "small bay industrial lease up assumptions and absorption",
            "cold storage tenant credit analysis and lease structure",
            "last-mile logistics site selection criteria demographics",
            "data center liquidity discount DCF sensitivity analysis",
            "CRE broker opinion of value BOV methodology",
        ],
        "risk_weights": {"low": 0.10, "medium": 0.45, "high": 0.35, "critical": 0.10},
        "complexity_weights": {"low": 0.15, "medium": 0.40, "high": 0.45},
    },
    "safety": {
        "target": 4000,
        "domain": "safety",
        "description": "Aviation ops, industrial safety, critical risk",
        "seed_topics": [
            "MEL deferral procedures for avionics failure",
            "TCAS RA response procedures and pilot compliance",
            "engine failure V1 cut takeoff decision making",
            "rejected takeoff abort criteria and speeds",
            "fuel endurance calculation divert alternate selection",
            "pressurization failure emergency descent procedure",
            "windshear encounter recovery procedures energy management",
            "runway contamination braking action reports and limits",
            "cargo dangerous goods Class 3 flammable handling",
            "industrial confined space entry gas monitoring protocols",
        ],
        "risk_weights": {"low": 0.05, "medium": 0.25, "high": 0.45, "critical": 0.25},
        "complexity_weights": {"low": 0.10, "medium": 0.40, "high": 0.50},
    },
    "technical": {
        "target": 6000,
        "domain": "technical",
        "description": "Software, engineering, ML infra",
        "seed_topics": [
            "async/await Python concurrency and deadlock prevention",
            "React state management architecture comparison Redux vs Zustand",
            "database index optimization strategies for high write workloads",
            "Kubernetes pod scheduling resource limits and QoS",
            "vLLM throughput optimization for 70B models",
            "CUDA memory management OOM debugging strategies",
            "distributed training gradient accumulation and ZeRO",
            "quantization GGUF Q4_K_M vs Q8_0 quality tradeoffs",
            "REST API versioning and backwards compatibility",
            "microservices data consistency saga vs 2PC patterns",
        ],
        "risk_weights": {"low": 0.30, "medium": 0.45, "high": 0.20, "critical": 0.05},
        "complexity_weights": {"low": 0.25, "medium": 0.45, "high": 0.30},
    },
    "business": {
        "target": 6000,
        "domain": "business",
        "description": "Strategy, operations, management",
        "seed_topics": [
            "go-to-market strategy for enterprise SaaS product",
            "OKR framework implementation and quarterly review",
            "vendor contract negotiation key terms and leverage",
            "M&A due diligence checklist and red flags",
            "supply chain resilience and single-source risk",
            "customer churn analysis and retention playbook",
            "board presentation structure for Series B raise",
            "competitive moat analysis and defensibility",
            "team structure and org design for scale",
            "product-market fit signals and validation",
        ],
        "risk_weights": {"low": 0.30, "medium": 0.45, "high": 0.20, "critical": 0.05},
        "complexity_weights": {"low": 0.25, "medium": 0.45, "high": 0.30},
    },
    "financial": {
        "target": 5000,
        "domain": "financial",
        "description": "Financial modeling, valuation, analysis",
        "seed_topics": [
            "DCF valuation model assumptions and sensitivity",
            "LBO model leverage and returns analysis",
            "comparable company analysis EV/EBITDA multiples",
            "working capital optimization and cash conversion cycle",
            "options pricing Black-Scholes implied volatility",
            "credit risk assessment for leveraged buyout",
            "treasury yield curve inversion and recession signal",
            "portfolio VaR calculation and stress testing",
            "private equity IRR waterfall distribution",
            "revenue recognition ASC 606 SaaS contracts",
        ],
        "risk_weights": {"low": 0.20, "medium": 0.45, "high": 0.30, "critical": 0.05},
        "complexity_weights": {"low": 0.20, "medium": 0.45, "high": 0.35},
    },
    "legal": {
        "target": 5000,
        "domain": "legal",
        "description": "Contracts, compliance, regulations",
        "seed_topics": [
            "NDA mutual vs one-way confidentiality provisions",
            "GDPR data processing agreement requirements",
            "SaaS MSA limitation of liability clause negotiation",
            "IP assignment vs license in employment agreements",
            "securities law Reg D exemption requirements",
            "commercial lease CAM reconciliation dispute",
            "force majeure clause applicability and drafting",
            "antitrust merger notification HSR thresholds",
            "HIPAA business associate agreement requirements",
            "construction lien waiver and mechanics lien rights",
        ],
        "risk_weights": {"low": 0.20, "medium": 0.45, "high": 0.30, "critical": 0.05},
        "complexity_weights": {"low": 0.20, "medium": 0.45, "high": 0.35},
    },
    "judge": {
        "target": 12000,
        "domain": "judge",
        "description": "Quality evaluation, scoring, drift, policy — SwarmJudge PRIMARY",
        "seed_topics": [
            "evaluate clinical accuracy of this medical diagnosis response",
            "score this CRE underwriting output for math accuracy and completeness",
            "detect quality drift in agent responses over last 7 days",
            "assess aviation safety briefing for operational correctness",
            "baseline quality score for new pharma agent before deployment",
            "regression test this research summary against gold standard",
            "identify silent failures in production routing decisions",
            "score agent output on 5 criteria: accuracy completeness structure relevance sft_quality",
            "quality gate check before deploying new model version to production",
            "compare two agent responses and recommend the better one with reasoning",
            "detect hallucinations in this medical diagnosis output",
            "flag this CRE underwriting for reviewer — potential cap rate math errors",
            "evaluate chain-of-thought reasoning quality and logical consistency",
            "measure response consistency across 50 similar queries for drift",
            "assess if this answer meets SFT training quality threshold",
            "policy check: does this response comply with medical disclaimer requirements",
            "evaluate this pharma DDI answer for patient safety completeness",
            "score output quality for RLHF training signal generation",
            "detect confident hallucinations in this legal compliance answer",
            "compare model outputs before and after fine-tuning for regression",
        ],
        "risk_weights": {"low": 0.15, "medium": 0.40, "high": 0.30, "critical": 0.15},
        "complexity_weights": {"low": 0.15, "medium": 0.45, "high": 0.40},
    },
    "edge": {
        "target": 8000,
        "domain": None,
        "description": "Adversarial, ambiguous, multi-domain edge cases",
        "seed_topics": [
            "query spanning medical and pharma domains simultaneously",
            "CRE underwriting request that also needs legal review",
            "ambiguous request — could be financial or business domain",
            "aviation safety query disguised as general operations",
            "pharma question with embedded quality evaluation request",
            "complex multi-step request requiring escalation decision",
            "query with conflicting domain signals — needs careful triage",
            "CRE new economy query with compute infrastructure components",
            "medical research query that also needs judge quality gate",
            "legal compliance + financial modeling combined scenario",
        ],
        "risk_weights": {"low": 0.10, "medium": 0.35, "high": 0.40, "critical": 0.15},
        "complexity_weights": {"low": 0.10, "medium": 0.40, "high": 0.50},
    },
}

TOTAL_TARGET = sum(s["target"] for s in STREAMS.values())


def build_gen_prompt(stream_name: str, stream_cfg: dict, batch_size: int = 10) -> str:
    domain = stream_cfg.get("domain") or random.choice(DOMAINS)
    topics = stream_cfg["seed_topics"]
    topic_sample = random.sample(topics, min(3, len(topics)))

    complexity = random.choices(
        list(stream_cfg["complexity_weights"].keys()),
        weights=list(stream_cfg["complexity_weights"].values()),
    )[0]
    risk = random.choices(
        list(stream_cfg["risk_weights"].keys()),
        weights=list(stream_cfg["risk_weights"].values()),
    )[0]

    eff_domain = domain if domain else "business"
    recommended_model = get_recommended_model(eff_domain, complexity, risk)

    edge_instruction = ""
    if stream_name == "edge":
        edge_instruction = """
IMPORTANT: These are EDGE CASES. Generate queries that are:
- Ambiguous in domain classification (real signal confusion)
- Spanning multiple verticals naturally
- Containing misleading surface features
- Requiring careful analysis to route correctly
The routing decision should reflect the PRIMARY domain after careful reasoning."""

    return f"""You are generating training data for SwarmRouter-4B-0, an AI routing model for the Swarm & Bee platform.

Generate {batch_size} diverse, realistic user queries for the '{stream_name}' stream.
Domain: {domain if domain else '<determine from query>'}
Complexity target: {complexity}
Risk level target: {risk}
{edge_instruction}

Topic hints (use as inspiration, vary freely): {', '.join(topic_sample)}

For each query, output a JSON object with these exact keys:
- "query": the user's request (natural language, realistic, 20-200 words, specific details)
- "domain": one of {DOMAINS}
- "task_type": one of {TASK_TYPES}
- "complexity": "{complexity}"
- "risk_level": "{risk}"
- "latency_tier": appropriate latency expectation
- "cost_sensitivity": budget sensitivity of the requester
- "recommended_model": "{recommended_model}"
- "escalation_allowed": true/false
- "requires_tools": array of needed tools (["none"] if none)
- "reasoning": max 120 chars explaining why this model was chosen

Query quality requirements:
- REALISTIC — things real professionals actually ask
- SPECIFIC — include numbers, names, scenarios, context
- VARIED — different phrasings, lengths, technical depths
- For medical: real drug names, lab values, clinical presentations
- For CRE: real asset classes, market names, financial metrics, new economy
- For judge: clear quality evaluation or scoring intent
- NO generic placeholder queries like "explain X" or "help me with Y"

Output ONLY a JSON array of {batch_size} objects. No markdown, no explanation, no preamble."""


async def generate_batch(
    client: AsyncOpenAI,
    stream_name: str,
    stream_cfg: dict,
    batch_size: int = 10,
    model: str = GEN_MODEL,
    semaphore: asyncio.Semaphore = None,
) -> list[dict]:
    sem = semaphore or asyncio.Semaphore(1)
    prompt = build_gen_prompt(stream_name, stream_cfg, batch_size)

    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.85,
                timeout=60,
            )
            raw = resp.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            pairs = json.loads(raw)
            if not isinstance(pairs, list):
                return []

            results = []
            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                query = pair.get("query", "").strip()
                if not query or len(query) < 15:
                    continue

                domain = pair.get("domain", "business")
                task_type = pair.get("task_type", "qa")
                complexity = pair.get("complexity", "medium")
                risk_level = pair.get("risk_level", "medium")
                latency_tier = pair.get("latency_tier", "interactive")
                cost_sensitivity = pair.get("cost_sensitivity", "medium")
                recommended_model = pair.get("recommended_model", "research-8b")
                escalation_allowed = pair.get("escalation_allowed", True)
                requires_tools = pair.get("requires_tools", ["none"])
                reasoning = pair.get("reasoning", "")[:120]

                if domain not in DOMAINS:
                    domain = "business"
                if task_type not in TASK_TYPES:
                    task_type = "qa"
                if complexity not in COMPLEXITY:
                    complexity = "medium"
                if risk_level not in RISK_LEVELS:
                    risk_level = "medium"
                if latency_tier not in LATENCY_TIERS:
                    latency_tier = "interactive"
                if cost_sensitivity not in COST_SENSITIVITY:
                    cost_sensitivity = "medium"
                if recommended_model not in MODELS:
                    recommended_model = get_recommended_model(domain, complexity, risk_level)
                if not isinstance(requires_tools, list):
                    requires_tools = ["none"]
                requires_tools = [t for t in requires_tools if t in TOOLS] or ["none"]

                # proposal_required: trust model if bool, else derive
                proposal_req = pair.get("proposal_required")
                if not isinstance(proposal_req, bool):
                    proposal_req = needs_proposal(domain, complexity, risk_level)

                routing_answer = {
                    "domain": domain,
                    "task_type": task_type,
                    "complexity": complexity,
                    "risk_level": risk_level,
                    "latency_tier": latency_tier,
                    "cost_sensitivity": cost_sensitivity,
                    "recommended_model": recommended_model,
                    "escalation_allowed": bool(escalation_allowed),
                    "proposal_required": proposal_req,
                    "requires_tools": requires_tools,
                    "reasoning": reasoning,
                }

                results.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": json.dumps(routing_answer, indent=2)},
                    ],
                    "metadata": {
                        "domain": domain,
                        "task_type": task_type,
                        "complexity": complexity,
                        "risk_level": risk_level,
                        "recommended_model": recommended_model,
                        "stream": stream_name,
                        "model": model,
                        "generated_at": datetime.now(datetime.UTC).isoformat()
                            if hasattr(datetime, 'UTC')
                            else datetime.utcnow().isoformat(),
                    },
                })

            return results

        except json.JSONDecodeError:
            return []
        except Exception as e:
            print(f"  [warn] batch error ({stream_name}): {type(e).__name__}: {e}")
            return []


async def run_stream(
    client: AsyncOpenAI,
    stream_name: str,
    stream_cfg: dict,
    target: int,
    output_path: Path,
    model: str,
    concurrency: int,
    batch_size: int,
) -> int:
    sem = asyncio.Semaphore(concurrency)
    batches_needed = (target + batch_size - 1) // batch_size
    total_generated = 0
    failed = 0

    print(f"\n[{stream_name}] target={target:,} batches={batches_needed} model={model.split('/')[-1]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a") as f:
        tasks = [
            generate_batch(client, stream_name, stream_cfg, batch_size, model, sem)
            for _ in range(batches_needed)
        ]

        for i, coro in enumerate(asyncio.as_completed(tasks)):
            results = await coro
            if results:
                for record in results:
                    f.write(json.dumps(record) + "\n")
                total_generated += len(results)
            else:
                failed += 1

            if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                pct = total_generated / target * 100
                print(f"  [{stream_name}] {total_generated:,}/{target:,} ({pct:.0f}%) failed_batches={failed}")

            if total_generated >= target:
                break

    print(f"  [{stream_name}] DONE — {total_generated:,} pairs")
    return total_generated


async def main_async(args):
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    model = FAST_MODEL if args.fast else GEN_MODEL
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    if args.dry_run:
        print("=== DRY RUN ===")
        stream_name = args.stream if args.stream != "all" else "cre"
        result = await generate_batch(client, stream_name, STREAMS[stream_name], 3, model)
        print(f"Generated {len(result)} pairs")
        for r in result:
            print(f"  domain={r['metadata']['domain']} model={r['metadata']['recommended_model']}")
            print(f"  Q: {r['messages'][1]['content'][:120]}")
            print(f"  A (model): {json.loads(r['messages'][2]['content'])['recommended_model']}")
            print()
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if args.stream == "all":
        streams_to_run = list(STREAMS.items())
    elif args.stream in STREAMS:
        streams_to_run = [(args.stream, STREAMS[args.stream])]
    else:
        print(f"Unknown stream: {args.stream}. Options: all | {' | '.join(STREAMS.keys())}")
        return

    total_target = sum(
        args.count if args.count else s["target"]
        for _, s in streams_to_run
    )

    print("=" * 70)
    print("SwarmRouter-4B-0 Dataset Factory")
    print("Primary verticals: SwarmMed | SwarmCRE | SwarmJudge")
    print("=" * 70)
    print(f"  Model:        {model.split('/')[-1]}")
    print(f"  Streams:      {[s[0] for s in streams_to_run]}")
    print(f"  Concurrency:  {args.concurrency}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Total target: {total_target:,}")
    print("=" * 70)

    grand_total = 0
    start = time.time()

    for stream_name, stream_cfg in streams_to_run:
        target = args.count if args.count else stream_cfg["target"]
        output_path = output_dir / f"router_v3_{stream_name}_{timestamp}.jsonl"
        count = await run_stream(
            client, stream_name, stream_cfg, target,
            output_path, model, args.concurrency, args.batch_size,
        )
        grand_total += count

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print(f"COMPLETE — {grand_total:,} pairs in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Output dir: {output_dir}")
    print("=" * 70)
    print("\nNext: python3 -m data.router_v3.assemble")


def main():
    parser = argparse.ArgumentParser(description="SwarmRouter-4B-0 Factory")
    parser.add_argument("--stream", default="all")
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
