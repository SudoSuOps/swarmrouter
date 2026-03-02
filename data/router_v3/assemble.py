#!/usr/bin/env python3
"""
SwarmRouter-4B-0 Dataset Assembler
====================================
Combines all factory + platinum sources, deduplicates,
validates schema, balances distribution, splits train/eval.

Target: ~80K train + 2K eval

Usage:
    python3 -m data.router_v3.assemble
    python3 -m data.router_v3.assemble --report-only
"""

import json
import random
import argparse
import hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime

from data.router_v3.schema import DOMAINS, MODELS, TASK_TYPES, COMPLEXITY, RISK_LEVELS

OUTPUT_DIR = Path(__file__).parent / "output"
FINAL_DIR = Path(__file__).parent.parent  # swarmrouter/data/

TRAIN_TARGET = 80000
EVAL_TARGET = 2000
SEED = 42


def fingerprint(text: str) -> str:
    """Short fingerprint for dedup."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def validate_record(record: dict) -> bool:
    """Validate a training record against v3 schema."""
    try:
        msgs = record.get("messages", [])
        if len(msgs) != 3:
            return False
        if msgs[0]["role"] != "system":
            return False
        if msgs[1]["role"] != "user":
            return False
        if msgs[2]["role"] != "assistant":
            return False

        query = msgs[1]["content"].strip()
        if len(query) < 15:
            return False

        answer = json.loads(msgs[2]["content"])

        # Validate required keys
        required = ["domain", "task_type", "complexity", "risk_level",
                    "latency_tier", "cost_sensitivity", "recommended_model",
                    "escalation_allowed", "requires_tools", "reasoning"]
        for key in required:
            if key not in answer:
                return False

        # Validate enums
        if answer["domain"] not in DOMAINS:
            return False
        if answer["task_type"] not in TASK_TYPES:
            return False
        if answer["complexity"] not in COMPLEXITY:
            return False
        if answer["risk_level"] not in RISK_LEVELS:
            return False
        if answer["recommended_model"] not in MODELS:
            return False
        if not isinstance(answer["requires_tools"], list):
            return False

        # Reasoning length
        if len(answer.get("reasoning", "")) > 120:
            return False

        return True

    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def load_all_sources() -> list[dict]:
    """Load all generated and mined pairs from output directory.

    If both raw harvest and vetted harvest exist, skip the raw harvest —
    the vetted version has been quality-verified and routing-corrected.
    """
    all_records = []
    files = sorted(OUTPUT_DIR.glob("router_v3_*.jsonl"))

    # If vetted file exists, skip raw harvest to avoid mixing unverified pairs
    has_vetted = any("vetted" in f.name for f in files)
    raw_harvest_prefix = "router_v3_harvest_20260302_1649"  # the raw 38K router file

    print(f"Loading from {len(files)} source files...")
    skipped = []
    for path in files:
        # Skip raw harvest router pairs if vetted version exists
        if has_vetted and path.name.startswith(raw_harvest_prefix):
            skipped.append(path.name)
            continue

        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    all_records.append(record)
                    count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  {path.name}: {count:,}")

    if skipped:
        print(f"  [skipped raw harvest — using vetted version]: {skipped}")

    print(f"  Total loaded: {len(all_records):,}")
    return all_records


def dedup(records: list[dict]) -> list[dict]:
    """Deduplicate by query fingerprint."""
    seen = set()
    unique = []
    dupes = 0
    for record in records:
        query = record["messages"][1]["content"]
        fp = fingerprint(query)
        if fp in seen:
            dupes += 1
            continue
        seen.add(fp)
        unique.append(record)
    print(f"  Dedup: {len(records):,} → {len(unique):,} ({dupes:,} removed)")
    return unique


def quality_gate(records: list[dict]) -> list[dict]:
    """Filter invalid records."""
    valid = [r for r in records if validate_record(r)]
    invalid = len(records) - len(valid)
    print(f"  Quality gate: {len(records):,} → {len(valid):,} ({invalid:,} rejected)")
    return valid


# Source quality tiers — higher = more trusted supervision signal
# Tier 1: CoVe-verified platinum + vetted-and-corrected pairs
# Tier 2: LLM-vetted (unchanged) + CRE math ground truth
# Tier 3: Derived from real data (keyword heuristics)
# Tier 4: Schema-upgraded v2 pairs (stale model slots, older logic)
SOURCE_TIERS = {
    "cove_verified":      1,   # Platinum — human+LLM verified
    "vetted_corrected":   1,   # Vetted + label was wrong → corrected
    "vetted_confirmed":   2,   # Vetted + label was already correct
    "cre_factory":        2,   # Ground truth CRE math
    "harvest_derived":    3,   # Heuristic-derived from real Q&A
    "cove_failed":        3,   # Escalation signals
    "schema_upgraded":    4,   # v2 schema, older logic
}


def assign_tier(record: dict) -> int:
    """Assign quality tier to a record based on source + correction flag."""
    meta = record.get("metadata", {})
    source = meta.get("source", "unknown")
    stream = meta.get("stream", "")

    if source == "cove_verified":
        return 1
    if stream == "vetted_harvest":
        if meta.get("corrected", False):
            return 1   # Label was wrong, vetted model fixed it → high trust
        return 2       # Label was already right, confirmed
    if source == "cre_factory":
        return 2
    if source in ("harvest_derived",):
        return 3
    if source == "cove_failed":
        return 3
    if source == "schema_upgraded":
        return 4
    return 3  # Default: derived


def confidence_filter(records: list[dict], min_tier: int = 4) -> list[dict]:
    """Drop records below minimum quality tier.

    Tier mapping to user confidence thresholds:
      Tier 1 = confidence 0.90+ (platinum/vetted-corrected) → ALWAYS keep
      Tier 2 = confidence 0.75+ (vetted-confirmed/CRE)      → keep
      Tier 3 = confidence 0.60+ (derived)                   → keep, lower priority
      Tier 4 = confidence ~0.50 (schema_upgraded v2)        → keep if rare domain
    """
    tier_counts = Counter(assign_tier(r) for r in records)
    print(f"  Quality tiers before filter: { {f'tier{k}': v for k, v in sorted(tier_counts.items())} }")

    # Hard drop: tier > min_tier UNLESS it's a rare domain (pharma/judge/financial)
    rare_domains = {"pharma", "judge", "financial"}
    kept, dropped = [], []
    for r in records:
        tier = assign_tier(r)
        domain = r.get("metadata", {}).get("domain", "")
        if tier <= min_tier:
            kept.append(r)
        elif domain in rare_domains:
            kept.append(r)   # Keep rare domains regardless of tier
        else:
            dropped.append(r)

    print(f"  Confidence filter (min tier {min_tier}): {len(records):,} → {len(kept):,} ({len(dropped):,} dropped)")
    return kept


def print_distribution(records: list[dict], label: str = ""):
    """Print domain + model distribution."""
    domains = Counter(r["metadata"]["domain"] for r in records)
    models = Counter(r["metadata"]["recommended_model"] for r in records)
    streams = Counter(r["metadata"].get("stream", "?") for r in records)
    complexity = Counter(r["metadata"].get("complexity", "?") for r in records)

    print(f"\n{'─'*50}")
    print(f"Distribution: {label} ({len(records):,} pairs)")
    print(f"{'─'*50}")
    print("Domains:")
    for d, c in sorted(domains.items(), key=lambda x: -x[1]):
        bar = "█" * int(c / len(records) * 40)
        pct = c / len(records) * 100
        print(f"  {d:<20} {c:>6,} ({pct:4.1f}%) {bar}")
    print("\nModels:")
    for m, c in sorted(models.items(), key=lambda x: -x[1]):
        bar = "█" * int(c / len(records) * 40)
        pct = c / len(records) * 100
        print(f"  {m:<22} {c:>6,} ({pct:4.1f}%) {bar}")
    print("\nComplexity:", dict(sorted(complexity.items())))
    print("Streams (top 10):", dict(streams.most_common(10)))


def balance_domains(records: list[dict], target: int) -> list[dict]:
    """Soft-balance domain distribution, preserving high-quality sources."""
    by_domain = {}
    for r in records:
        d = r["metadata"]["domain"]
        by_domain.setdefault(d, []).append(r)

    # Target per domain (soft floor/ceil)
    n_domains = len(by_domain)
    base_per_domain = target // n_domains

    # Domain quotas — some domains are more important
    # Three primary verticals: SwarmMed | SwarmCRE | SwarmJudge
    quotas = {
        "medical":    int(target * 0.16),   # 16% — SwarmMed PRIMARY
        "pharma":     int(target * 0.08),   #  8% — SwarmPharma specialist
        "cre":        int(target * 0.20),   # 20% — SwarmCRE PRIMARY (largest)
        "safety":     int(target * 0.08),   #  8% — Aviation + industrial safety
        "technical":  int(target * 0.10),   # 10% — Software, ML, engineering
        "business":   int(target * 0.08),   #  8% — Strategy, ops, management
        "financial":  int(target * 0.06),   #  6% — Financial modeling
        "legal":      int(target * 0.08),   #  8% — Contracts, compliance
        "judge":      int(target * 0.16),   # 16% — SwarmJudge PRIMARY
    }

    balanced = []
    for domain, quota in quotas.items():
        available = by_domain.get(domain, [])
        if not available:
            print(f"  [warn] no pairs for domain '{domain}'")
            continue
        # Sort by quality tier — tier 1 first, tier 4 last
        available.sort(key=assign_tier)
        selected = available[:quota]

        balanced.extend(selected)
        print(f"  {domain:<20} quota={quota:,} available={len(available):,} selected={len(selected):,}")

    return balanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true",
                        help="Just report distribution, don't write output")
    args = parser.parse_args()

    random.seed(SEED)

    print("=" * 60)
    print("SwarmRouter-4B-0 Dataset Assembler")
    print("=" * 60)

    # ── Load all sources ───────────────────────────────────────
    all_records = load_all_sources()
    if not all_records:
        print("No source files found. Run factory.py and mine_platinum.py first.")
        return

    # ── Deduplicate ────────────────────────────────────────────
    print("\nDeduplicating...")
    records = dedup(all_records)

    # ── Quality gate ───────────────────────────────────────────
    print("Quality gate...")
    records = quality_gate(records)

    # ── Confidence filter ──────────────────────────────────────
    # Tier 1-3 = keep always; Tier 4 (schema_upgraded v2) = keep only for rare domains
    # For 4B: clean supervision matters more than volume
    print("Confidence filter...")
    records = confidence_filter(records, min_tier=4)

    print_distribution(records, "pre-balance")

    if args.report_only:
        return

    if len(records) < TRAIN_TARGET + EVAL_TARGET:
        print(f"\n[warn] Only {len(records):,} valid pairs. Need {TRAIN_TARGET + EVAL_TARGET:,}.")
        print("Run more factory streams first.")

    # ── Balance + sample ───────────────────────────────────────
    print(f"\nBalancing to {TRAIN_TARGET + EVAL_TARGET:,} pairs...")
    total_needed = min(TRAIN_TARGET + EVAL_TARGET, len(records))
    balanced = balance_domains(records, total_needed)
    random.shuffle(balanced)

    # ── Eval split — stratified by domain ─────────────────────
    by_domain = {}
    for r in balanced:
        d = r["metadata"]["domain"]
        by_domain.setdefault(d, []).append(r)

    eval_set = []
    eval_per_domain = EVAL_TARGET // len(by_domain)
    for domain_records in by_domain.values():
        eval_set.extend(domain_records[:eval_per_domain])

    eval_fps = {fingerprint(r["messages"][1]["content"]) for r in eval_set}
    train_set = [r for r in balanced if fingerprint(r["messages"][1]["content"]) not in eval_fps]

    # Trim to targets
    train_set = train_set[:TRAIN_TARGET]
    eval_set = eval_set[:EVAL_TARGET]

    random.shuffle(train_set)

    print_distribution(train_set, "train")
    print_distribution(eval_set, "eval")

    # ── Write output ───────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    train_path = FINAL_DIR / f"swarmrouter_4b0_train_{timestamp}.jsonl"
    eval_path = FINAL_DIR / f"swarmrouter_4b0_eval_{timestamp}.jsonl"

    with open(train_path, "w") as f:
        for r in train_set:
            f.write(json.dumps(r) + "\n")

    with open(eval_path, "w") as f:
        for r in eval_set:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Train: {len(train_set):,} → {train_path}")
    print(f"  Eval:  {len(eval_set):,} → {eval_path}")
    print(f"{'='*60}")
    print("\nNext: update configs/train_whale_v3.yaml with new dataset paths")
    print("Then: python3 scripts/train_v3.py --config configs/train_whale_v3.yaml")


if __name__ == "__main__":
    main()
