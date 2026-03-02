#!/usr/bin/env python3
"""
SwarmRouter-4B-0 — Harvest Miner
==================================
Converts all existing data sources to v3 routing pair format.

Sources:
  1. harvest-feb23 router pairs (38,798) — old schema conversion
  2. harvest-feb23 medical Q&A (15,720) — derive routing labels
  3. harvest-feb23 aviation Q&A (20,012) — derive routing labels
  4. swarmcre_dataset (881K) — mine 20K CRE routing labels
  5. existing 60K router pairs — update to v3 schema
  6. medical + aviation failed (6,137) — escalation signals

Usage:
    python3 -m data.router_v3.mine_harvest
    python3 -m data.router_v3.mine_harvest --source cre --limit 20000
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

from data.router_v3.schema import SYSTEM_PROMPT, DOMAINS, MODELS, get_recommended_model, needs_proposal

OUTPUT_DIR = Path(__file__).parent / "output"

# ── Source paths ───────────────────────────────────────────────────
HARVEST_ROUTER_DIR  = Path("/home/swarm/Desktop/harvest-feb23/deduped/router")
HARVEST_MEDICAL_DIR = Path("/home/swarm/Desktop/harvest-feb23/deduped/medical")
HARVEST_AVIATION_DIR= Path("/home/swarm/Desktop/harvest-feb23/deduped/aviation")
SWARMCRE_TRAIN      = Path("/home/swarm/Desktop/swarmrouter/data/swarmcre_dataset/output/swarmcre_train.jsonl")
OLD_ROUTER_60K      = Path("/home/swarm/Desktop/swarmrouter/data/swarmrouter_train_60k.jsonl")
MEDICAL_FAILED      = Path("/home/swarm/Desktop/gold-for-cove/promoted/failed.jsonl")
AVIATION_FAILED     = Path("/home/swarm/Desktop/swarmrouter/aviation_promoted/failed.jsonl")

# ── Old → new domain mapping ───────────────────────────────────────
OLD_DOMAIN_MAP = {
    "medicine": "medical",
    "medical":  "medical",
    "pharma":   "pharma",
    "pharmacology": "pharma",
    "aviation": "safety",
    "legal":    "legal",
    "coding":   "technical",
    "research": "technical",
    "general_chat": "business",
    "general":  "business",
    "compute":  "technical",
    "operations": "business",
    "cre":      "cre",
    "finance":  "financial",
    "financial": "financial",
}

OLD_MODEL_MAP = {
    "medical-14b":  "med-14b",
    "med-14b":      "med-14b",
    "research-8b":  "research-8b",
    "research-32b": "swarmresearch-32b",
    "router-3b":    "router-4b",
    "legal-14b":    "research-8b",   # no legal specialist, use research-8b
    "aviation-32b": "swarmresearch-32b",
}

# ── CRE task_type → routing complexity/risk ───────────────────────
CRE_TASK_COMPLEXITY = {
    "underwriting_calc":        ("high",   "high"),
    "ic_memo":                  ("high",   "medium"),
    "lease_reasoning":          ("medium", "medium"),
    "market_comp_narrative":    ("medium", "low"),
    "t12_normalization":        ("medium", "medium"),
    "rent_roll_extraction":     ("medium", "low"),
    "lease_abstract_extraction":("low",    "low"),
    "risk_triage":              ("high",   "high"),
    "exchange_1031":            ("high",   "high"),
    "tax_analysis":             ("medium", "medium"),
    "structured_agent_output":  ("high",   "medium"),
    "loi_deliverable":          ("medium", "medium"),
}


def make_record(question: str, routing: dict, stream: str, source: str = "derived") -> dict:
    """Build a v3 training record."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": json.dumps(routing, indent=2)},
        ],
        "metadata": {
            "domain": routing["domain"],
            "task_type": routing["task_type"],
            "complexity": routing["complexity"],
            "risk_level": routing["risk_level"],
            "recommended_model": routing["recommended_model"],
            "stream": stream,
            "source": source,
            "generated_at": datetime.utcnow().isoformat(),
        },
    }


def base_routing(domain, complexity, risk_level, task_type="qa",
                 latency="interactive", cost="medium") -> dict:
    return {
        "domain": domain,
        "task_type": task_type,
        "complexity": complexity,
        "risk_level": risk_level,
        "latency_tier": latency,
        "cost_sensitivity": cost,
        "recommended_model": get_recommended_model(domain, complexity, risk_level),
        "escalation_allowed": risk_level not in ("critical",),
        "proposal_required": needs_proposal(domain, complexity, risk_level),
        "requires_tools": ["none"],
        "reasoning": f"{domain} {complexity} {risk_level} risk",
    }


# ═══════════════════════════════════════════════════════════════════
# 1. Harvest Router (38K) — old schema conversion
# ═══════════════════════════════════════════════════════════════════
def mine_harvest_router(limit: int = 0) -> list[dict]:
    results = []
    files = list(HARVEST_ROUTER_DIR.glob("*.jsonl"))

    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    question = record.get("question", "").strip()
                    if not question or len(question) < 15:
                        continue

                    # Parse old answer JSON
                    answer_raw = record.get("answer", "{}")
                    try:
                        ans = json.loads(answer_raw)
                    except json.JSONDecodeError:
                        continue

                    old_domain = ans.get("domain", "general")
                    domain = OLD_DOMAIN_MAP.get(old_domain, "business")
                    old_model = ans.get("route_to_model", "research-8b")

                    difficulty = ans.get("difficulty", "medium")
                    complexity = difficulty if difficulty in ("low", "medium", "high") else "medium"

                    # Infer risk from domain + complexity
                    if domain == "safety":
                        risk_level = "high"
                    elif domain in ("medical", "pharma") and complexity == "high":
                        risk_level = "high"
                    elif domain in ("medical", "pharma"):
                        risk_level = "medium"
                    else:
                        risk_level = "low" if complexity == "low" else "medium"

                    task_raw = ans.get("task", "reasoning")
                    task_type = task_raw if task_raw in (
                        "qa", "summarization", "reasoning", "generation",
                        "planning", "triage", "evaluation", "extraction"
                    ) else "reasoning"

                    routing = base_routing(domain, complexity, risk_level, task_type)
                    # Respect original model if it maps cleanly
                    mapped_model = OLD_MODEL_MAP.get(old_model)
                    if mapped_model and mapped_model in MODELS:
                        routing["recommended_model"] = mapped_model
                    routing["reasoning"] = f"converted from {old_domain} harvest → {domain}"[:120]

                    results.append(make_record(question, routing, "harvest_router", "harvest_converted"))

                except Exception:
                    continue

                if limit and len(results) >= limit:
                    break
        if limit and len(results) >= limit:
            break

    print(f"  Harvest router: {len(results):,} pairs")
    return results


# ═══════════════════════════════════════════════════════════════════
# 2. Harvest Medical Q&A (15,720) — derive routing
# ═══════════════════════════════════════════════════════════════════
def mine_harvest_medical(limit: int = 0) -> list[dict]:
    results = []
    files = list(HARVEST_MEDICAL_DIR.glob("*.jsonl"))

    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Could be Q&A or messages format
                    question = record.get("question", "")
                    if not question:
                        msgs = record.get("messages", [])
                        question = next((m["content"] for m in msgs if m["role"] == "user"), "")
                    if not question or len(question) < 15:
                        continue

                    specialty = record.get("specialty", "general_medicine").lower()
                    # Infer complexity from question length
                    words = len(question.split())
                    complexity = "high" if words > 80 else "medium" if words > 30 else "low"
                    risk_level = "high" if any(w in question.lower() for w in
                        ["emergency", "acute", "critical", "urgent", "stat", "icu"]) else "medium"

                    domain = "pharma" if any(w in specialty for w in
                        ["pharm", "drug", "toxicol"]) else "medical"

                    routing = base_routing(domain, complexity, risk_level, "reasoning",
                                         "interactive", "low")
                    routing["reasoning"] = f"{specialty} {complexity} → {routing['recommended_model']}"[:120]

                    results.append(make_record(question, routing, "harvest_medical", "harvest_derived"))

                except Exception:
                    continue

                if limit and len(results) >= limit:
                    break
        if limit and len(results) >= limit:
            break

    print(f"  Harvest medical: {len(results):,} pairs")
    return results


# ═══════════════════════════════════════════════════════════════════
# 3. Harvest Aviation Q&A (20,012) — derive routing
# ═══════════════════════════════════════════════════════════════════
def mine_harvest_aviation(limit: int = 0) -> list[dict]:
    results = []
    files = list(HARVEST_AVIATION_DIR.glob("*.jsonl"))

    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    question = record.get("question", "")
                    if not question:
                        msgs = record.get("messages", [])
                        question = next((m["content"] for m in msgs if m["role"] == "user"), "")
                    if not question or len(question) < 15:
                        continue

                    words = len(question.split())
                    q_lower = question.lower()
                    complexity = "high" if words > 80 else "medium" if words > 30 else "medium"
                    is_critical = any(w in q_lower for w in
                        ["emergency", "failure", "abort", "tcas", "mayday", "fire",
                         "stall", "upset", "evacuate", "windshear", "crash"])
                    risk_level = "critical" if is_critical else "high"

                    routing = base_routing("safety", complexity, risk_level, "reasoning",
                                          "interactive", "low")
                    routing["escalation_allowed"] = False  # safety: never downgrade
                    routing["reasoning"] = f"aviation {complexity} {risk_level} → swarmresearch-32b"[:120]

                    results.append(make_record(question, routing, "harvest_aviation", "harvest_derived"))

                except Exception:
                    continue

                if limit and len(results) >= limit:
                    break
        if limit and len(results) >= limit:
            break

    print(f"  Harvest aviation: {len(results):,} pairs")
    return results


# ═══════════════════════════════════════════════════════════════════
# 4. SwarmCRE Dataset (881K) — mine 20K routing labels
# ═══════════════════════════════════════════════════════════════════
def mine_swarmcre(limit: int = 20000) -> list[dict]:
    results = []
    if not SWARMCRE_TRAIN.exists():
        print(f"  [warn] CRE dataset not found: {SWARMCRE_TRAIN}")
        return results

    # Sample evenly across task types for diversity
    by_task = {}
    with open(SWARMCRE_TRAIN) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                task = record.get("task_type", "unknown")
                by_task.setdefault(task, []).append(record)
            except Exception:
                continue

    # Per-task quota
    tasks = list(by_task.keys())
    per_task = max(1, limit // len(tasks))

    for task, records in by_task.items():
        complexity, risk_level = CRE_TASK_COMPLEXITY.get(task, ("medium", "medium"))
        sample = random.sample(records, min(per_task, len(records)))

        for record in sample:
            msgs = record.get("messages", [])
            question = next((m["content"] for m in msgs if m["role"] == "user"), "")
            if not question:
                question = record.get("question", "")
            if not question or len(question) < 15:
                continue

            task_type_map = {
                "underwriting_calc": "reasoning",
                "ic_memo": "generation",
                "lease_reasoning": "reasoning",
                "market_comp_narrative": "generation",
                "t12_normalization": "extraction",
                "rent_roll_extraction": "extraction",
                "lease_abstract_extraction": "extraction",
                "risk_triage": "triage",
                "exchange_1031": "planning",
                "tax_analysis": "reasoning",
                "structured_agent_output": "generation",
            }
            task_type = task_type_map.get(task, "reasoning")
            asset = record.get("metadata", {}).get("asset_type", "")

            routing = base_routing("cre", complexity, risk_level, task_type, "batch", "medium")
            routing["reasoning"] = f"CRE {task} {asset} → swarmcre-35b"[:120]
            if task in ("underwriting_calc", "risk_triage", "exchange_1031"):
                routing["requires_tools"] = ["calculator"]

            results.append(make_record(question, routing, "swarmcre_dataset", "cre_factory"))

            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    print(f"  SwarmCRE: {len(results):,} pairs ({len(tasks)} task types)")
    return results


# ═══════════════════════════════════════════════════════════════════
# 5. Existing 60K Router — update to v3 schema
# ═══════════════════════════════════════════════════════════════════
def mine_old_60k(limit: int = 0) -> list[dict]:
    results = []
    if not OLD_ROUTER_60K.exists():
        print(f"  [warn] 60K router not found")
        return results

    # Domain remap from old 60K schema to v3
    domain_remap = {
        "medical":    "medical",
        "cre":        "cre",
        "aviation":   "safety",
        "research":   "technical",
        "coding":     "technical",
        "operations": "business",
        "compute":    "technical",
        "general":    "business",
    }
    model_remap = {
        "router-3b":    "router-4b",
        "research-8b":  "research-8b",
        "med-14b":      "med-14b",
        "research-32b": "swarmresearch-32b",
    }

    with open(OLD_ROUTER_60K) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                msgs = record.get("messages", [])
                meta = record.get("metadata", {})

                question = next((m["content"] for m in msgs if m["role"] == "user"), "")
                if not question or len(question) < 15:
                    continue

                old_domain = meta.get("domain", "general")
                domain = domain_remap.get(old_domain, "business")
                complexity = meta.get("complexity", "medium")
                old_model = meta.get("recommended_model", "research-8b")
                task_type = meta.get("task_type", "qa")
                if task_type not in ("qa","summarization","reasoning","generation",
                                     "planning","triage","evaluation","extraction"):
                    task_type = "qa"

                # Infer risk
                if domain == "safety":
                    risk_level = "high"
                elif domain in ("medical", "pharma") and complexity == "high":
                    risk_level = "high"
                else:
                    risk_level = "medium" if complexity != "low" else "low"

                routing = base_routing(domain, complexity, risk_level, task_type)
                mapped = model_remap.get(old_model)
                if mapped:
                    routing["recommended_model"] = mapped
                routing["reasoning"] = f"v2→v3 {old_domain}→{domain} {complexity}"[:120]

                results.append(make_record(question, routing, "v2_router_60k", "schema_upgraded"))

            except Exception:
                continue

            if limit and len(results) >= limit:
                break

    print(f"  Old 60K router: {len(results):,} pairs (schema upgraded v2→v3)")
    return results


# ═══════════════════════════════════════════════════════════════════
# 6. Failed pairs — escalation signals
# ═══════════════════════════════════════════════════════════════════
def mine_failed(limit: int = 0) -> list[dict]:
    results = []

    sources = [
        (MEDICAL_FAILED, "medical", "med-14b"),
        (AVIATION_FAILED, "safety", "swarmresearch-32b"),
    ]

    for path, domain, model in sources:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    question = record.get("question", "").strip()
                    if not question or len(question) < 15:
                        continue

                    specialty = record.get("specialty", "")
                    cove_score = record.get("cove_score", 0)

                    # Failed = low quality → escalate signal
                    complexity = "high"  # If it failed, it was hard
                    risk_level = "high"

                    routing = base_routing(domain, complexity, risk_level, "triage")
                    routing["recommended_model"] = model
                    routing["escalation_allowed"] = True
                    routing["reasoning"] = f"failed {specialty} → escalate to {model}"[:120]

                    results.append(make_record(question, routing, "failed_escalation", "cove_failed"))

                except Exception:
                    continue

                if limit and len(results) >= limit:
                    break

    print(f"  Failed pairs (escalation signals): {len(results):,} pairs")
    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SwarmRouter-4B-0 Harvest Miner")
    parser.add_argument("--source", default="all",
                        help="all | router | medical | aviation | cre | v2 | failed")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit per source (0 = all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    print("=" * 60)
    print("SwarmRouter-4B-0 — Harvest Miner")
    print("=" * 60)

    sources_map = {
        "router":   mine_harvest_router,
        "medical":  mine_harvest_medical,
        "aviation": mine_harvest_aviation,
        "cre":      mine_swarmcre,
        "v2":       mine_old_60k,
        "failed":   mine_failed,
    }

    if args.source == "all":
        to_run = list(sources_map.keys())
    elif args.source in sources_map:
        to_run = [args.source]
    else:
        print(f"Unknown source: {args.source}. Options: all | {' | '.join(sources_map)}")
        return

    for source_name in to_run:
        fn = sources_map[source_name]
        pairs = fn(args.limit) if args.limit else fn()
        all_results.extend(pairs)

    random.shuffle(all_results)

    source_tag = args.source if args.source != "all" else "all"
    output_path = OUTPUT_DIR / f"router_v3_harvest_{source_tag}_{timestamp}.jsonl"
    with open(output_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nTotal mined: {len(all_results):,} pairs → {output_path}")

    # Distribution
    domains = Counter(r["metadata"]["domain"] for r in all_results)
    models = Counter(r["metadata"]["recommended_model"] for r in all_results)
    streams = Counter(r["metadata"]["stream"] for r in all_results)
    print("\nDomains:", dict(sorted(domains.items(), key=lambda x: -x[1])))
    print("Models:", dict(sorted(models.items(), key=lambda x: -x[1])))
    print("Streams:", dict(sorted(streams.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
