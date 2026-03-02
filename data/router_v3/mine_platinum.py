#!/usr/bin/env python3
"""
Mine existing platinum Q&A pairs → routing pairs.
Converts cove-verified medical + aviation platinum into SwarmRouter-4B-0 format.

Sources:
  - Medical platinum: 8,532 pairs (cove_score 20+, accuracy>=4)
  - Aviation platinum: 15,236 pairs (tier verified)

Usage:
    python3 -m data.router_v3.mine_platinum
"""

import json
import random
from pathlib import Path
from datetime import datetime

from data.router_v3.schema import SYSTEM_PROMPT, get_recommended_model, needs_proposal

# ── Source paths ───────────────────────────────────────────────────
MEDICAL_PLATINUM = Path("/home/swarm/Desktop/gold-for-cove/promoted/platinum_promoted.jsonl")
AVIATION_PLATINUM = Path("/home/swarm/Desktop/swarmrouter/aviation_promoted/platinum_promoted.jsonl")
OUTPUT_DIR = Path(__file__).parent / "output"

# ── Specialty → complexity mapping ────────────────────────────────
HIGH_COMPLEXITY_SPECIALTIES = {
    "cardiology", "oncology", "neurology", "critical_care", "nephrology",
    "hematology", "gastroenterology", "pulmonology", "endocrinology",
    "rheumatology", "infectious_disease", "transplant", "trauma",
    "interventional_radiology", "neuroradiology", "nuclear_medicine",
}

HIGH_RISK_SPECIALTIES = {
    "emergency_medicine", "critical_care", "trauma", "cardiology",
    "oncology", "neurology", "nephrology", "transplant", "neonatology",
    "pediatric_critical_care", "anesthesiology",
}

PHARMA_SPECIALTIES = {
    "pharmacology", "clinical_pharmacology", "toxicology", "pharmacy",
    "drug_safety", "pharmacokinetics",
}

# ── Aviation risk + complexity always elevated ────────────────────
AVIATION_CRITICAL_TOPICS = {
    "emergency", "failure", "abort", "TCAS", "RA", "windshear",
    "fire", "pressurization", "stall", "upset", "evacuation", "mayday",
}


def get_medical_routing(specialty: str, cove_score: int, question: str) -> dict:
    """Derive routing from medical specialty + cove_score."""
    specialty_lower = specialty.lower().replace(" ", "_").replace("-", "_")

    # Pharma redirect
    if specialty_lower in PHARMA_SPECIALTIES:
        domain = "pharma"
        complexity = "high"
        risk_level = "high"
        task_type = "qa"
        recommended_model = "swarmpharma-35b"
    else:
        domain = "medical"
        # Complexity from cove_score + specialty
        if cove_score >= 23 or specialty_lower in HIGH_COMPLEXITY_SPECIALTIES:
            complexity = "high"
        elif cove_score >= 20:
            complexity = "medium"
        else:
            complexity = "low"

        # Risk from specialty
        if specialty_lower in HIGH_RISK_SPECIALTIES:
            risk_level = "high"
        else:
            risk_level = "medium"

        # Task type from question keywords
        q_lower = question.lower()
        if any(w in q_lower for w in ["manage", "treat", "protocol", "plan"]):
            task_type = "planning"
        elif any(w in q_lower for w in ["diagnos", "differentiat", "what is"]):
            task_type = "reasoning"
        elif any(w in q_lower for w in ["calcul", "dose", "mg", "mg/kg"]):
            task_type = "qa"
        else:
            task_type = "qa"

        recommended_model = get_recommended_model(domain, complexity, risk_level)

    # Latency: batch for non-urgent, interactive for triage
    latency_tier = "batch"
    cost_sensitivity = "low"

    requires_tools = ["none"]
    if "calcul" in question.lower() or "dose" in question.lower():
        requires_tools = ["calculator"]

    reasoning = f"{specialty} {complexity} complexity {risk_level} risk → {recommended_model}"[:120]

    return {
        "domain": domain,
        "task_type": task_type,
        "complexity": complexity,
        "risk_level": risk_level,
        "latency_tier": latency_tier,
        "cost_sensitivity": cost_sensitivity,
        "recommended_model": recommended_model,
        "escalation_allowed": True,
        "proposal_required": needs_proposal(domain, complexity, risk_level),
        "requires_tools": requires_tools,
        "reasoning": reasoning,
    }


def get_aviation_routing(specialty: str, question: str) -> dict:
    """Derive routing from aviation question — always elevated risk."""
    q_upper = question.upper()
    domain = "safety"  # v3 schema: "aviation" → "safety"

    # Check for critical/emergency content
    is_critical = any(kw.upper() in q_upper for kw in AVIATION_CRITICAL_TOPICS)
    risk_level = "critical" if is_critical else "high"

    # Complexity from question length + technical depth
    word_count = len(question.split())
    if word_count > 80:
        complexity = "high"
    elif word_count > 40:
        complexity = "medium"
    else:
        complexity = "medium"

    # Task type
    q_lower = question.lower()
    if any(w in q_lower for w in ["procedure", "checklist", "protocol"]):
        task_type = "planning"
    elif any(w in q_lower for w in ["calcul", "fuel", "weight", "performance"]):
        task_type = "qa"
    elif any(w in q_lower for w in ["assess", "determin", "evaluat"]):
        task_type = "triage"
    else:
        task_type = "reasoning"

    recommended_model = "swarmresearch-32b"  # Aviation always 32B
    requires_tools = ["none"]

    reasoning = f"safety/aviation {complexity} {risk_level} risk → swarmresearch-32b"[:120]

    return {
        "domain": domain,
        "task_type": task_type,
        "complexity": complexity,
        "risk_level": risk_level,
        "latency_tier": "interactive",
        "cost_sensitivity": "low",
        "recommended_model": recommended_model,
        "escalation_allowed": False,  # Aviation/safety: never escalate to lower model
        "proposal_required": True,    # Safety domain: always requires proposal gate
        "requires_tools": requires_tools,
        "reasoning": reasoning,
    }


def mine_medical(max_pairs: int = 8532) -> list[dict]:
    """Mine medical platinum → routing pairs."""
    results = []
    with open(MEDICAL_PLATINUM) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = record.get("question", "").strip()
            specialty = record.get("specialty", "general_medicine")
            cove_score = record.get("cove_score", 20)
            cove_verdict = record.get("cove_verdict", {})

            if not question or cove_score < 20:
                continue

            routing = get_medical_routing(specialty, cove_score, question)
            routing_answer = json.dumps(routing, indent=2)

            results.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": routing_answer},
                ],
                "metadata": {
                    "domain": routing["domain"],
                    "task_type": routing["task_type"],
                    "complexity": routing["complexity"],
                    "risk_level": routing["risk_level"],
                    "recommended_model": routing["recommended_model"],
                    "stream": "platinum_medical",
                    "source": "cove_verified",
                    "cove_score": cove_score,
                    "specialty": specialty,
                    "generated_at": datetime.utcnow().isoformat(),
                },
            })

            if len(results) >= max_pairs:
                break

    print(f"  Mined {len(results):,} medical platinum pairs")
    return results


def mine_aviation(max_pairs: int = 15236) -> list[dict]:
    """Mine aviation platinum → routing pairs."""
    results = []
    with open(AVIATION_PLATINUM) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = record.get("question", "").strip()
            specialty = record.get("specialty", "aviation_operations")

            if not question:
                continue

            routing = get_aviation_routing(specialty, question)
            routing_answer = json.dumps(routing, indent=2)

            results.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": routing_answer},
                ],
                "metadata": {
                    "domain": routing["domain"],
                    "task_type": routing["task_type"],
                    "complexity": routing["complexity"],
                    "risk_level": routing["risk_level"],
                    "recommended_model": routing["recommended_model"],
                    "stream": "platinum_aviation",
                    "source": "cove_verified",
                    "specialty": specialty,
                    "generated_at": datetime.utcnow().isoformat(),
                },
            })

            if len(results) >= max_pairs:
                break

    print(f"  Mined {len(results):,} aviation platinum pairs")
    return results


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("SwarmRouter-4B-0 — Platinum Miner")
    print("=" * 60)

    print("\nMining medical platinum...")
    med_pairs = mine_medical()

    print("Mining aviation platinum...")
    ava_pairs = mine_aviation()

    all_pairs = med_pairs + ava_pairs
    random.shuffle(all_pairs)

    output_path = OUTPUT_DIR / f"router_v3_platinum_{timestamp}.jsonl"
    with open(output_path, "w") as f:
        for record in all_pairs:
            f.write(json.dumps(record) + "\n")

    print(f"\nTotal mined: {len(all_pairs):,} pairs → {output_path}")

    # Distribution report
    from collections import Counter
    domains = Counter(r["metadata"]["domain"] for r in all_pairs)
    models = Counter(r["metadata"]["recommended_model"] for r in all_pairs)
    print("\nDomain distribution:", dict(domains))
    print("Model distribution:", dict(models))


if __name__ == "__main__":
    main()
