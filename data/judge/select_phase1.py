#!/usr/bin/env python3
"""
Phase 1 Selector — SwarmJudge-27B Block-0
==========================================

Selects the cleanest pairs from the 103K staging set for Phase 1 training.

Rules:
  - Schema-correct ONLY: 5-criterion (sft_quality), 25-point max
  - 60% PASS / 40% FAIL target ratio
  - Priority: named source > unknown, clearest verdict signal first
  - Outputs: phase1_train.jsonl + phase1_report.json

Usage:
  python3 select_phase1.py \
      --input  /data2/swarmjudge_27b_data/swarmjudge_27b_train.jsonl \
      --output /data2/swarmjudge_27b_data/phase1/ \
      --target 25000
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
# Schema check
# ═══════════════════════════════════════════════════════════════════════

def is_correct_schema(pair: dict) -> bool:
    """True if pair uses the 5-criterion SwarmJudge schema."""
    asst = next((m["content"] for m in pair.get("messages", []) if m["role"] == "assistant"), "")
    has_sft     = "sft_quality" in asst
    has_wrong   = "grounding" in asst or "coherence" in asst
    return has_sft and not has_wrong


def parse_assistant(pair: dict) -> dict | None:
    """Parse assistant message JSON. Returns None on failure."""
    asst = next((m["content"] for m in pair.get("messages", []) if m["role"] == "assistant"), "")
    try:
        return json.loads(asst)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Scoring — how clean is the signal?
# ═══════════════════════════════════════════════════════════════════════

def selection_score(pair: dict, parsed: dict) -> float:
    """
    Higher = cleaner signal = higher priority for Phase 1.
    Phase 1 wants unambiguous examples, not borderline ones.
    """
    meta    = pair.get("metadata", {})
    verdict = parsed.get("verdict", "").upper()
    total   = parsed.get("total", 0) or 0
    scores  = parsed.get("scores", {})
    source  = meta.get("source", "") or ""

    score = 0.0

    # Source bonus — named, curated sources are higher signal
    if source.startswith("trajectory_cook_"):
        score += 3.0

    if verdict == "PASS":
        # Clearest PASS = 25/25 with all 5s
        if total == 25:
            score += 5.0
        elif total >= 23:
            score += 3.0
        elif total >= 20:
            score += 1.0
        else:
            score -= 2.0  # PASS with low total = suspicious, borderline

        # Penalize if any criterion is low (borderline PASS)
        min_criterion = min(scores.values()) if scores else 0
        if min_criterion >= 4:
            score += 1.0
        elif min_criterion == 3:
            score += 0.0
        else:
            score -= 1.0  # PASS with a criterion < 3 violates the threshold

    elif verdict == "FAIL":
        # Clearest FAIL = low score with specific named issues
        issues = parsed.get("issues", [])
        fixes  = parsed.get("fixes", [])

        if total <= 5:
            score += 5.0
        elif total <= 10:
            score += 3.0
        elif total <= 15:
            score += 1.0
        else:
            score -= 1.0  # High-scoring FAIL is borderline — Phase 2 material

        # Bonus for well-articulated failures (has both issues and fixes)
        if len(issues) >= 2 and len(fixes) >= 2:
            score += 2.0
        elif len(issues) >= 1:
            score += 1.0

    return score


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Selector — SwarmJudge-27B Block-0")
    parser.add_argument("--input",  default="/data2/swarmjudge_27b_data/swarmjudge_27b_train.jsonl")
    parser.add_argument("--output", default="/data2/swarmjudge_27b_data/phase1/")
    parser.add_argument("--target", type=int, default=25000)
    parser.add_argument("--pass-ratio", type=float, default=0.60,
                        help="Target fraction of PASS pairs (default: 0.60)")
    parser.add_argument("--exclude-domains", nargs="+", default=[],
                        help="Domains to exclude (e.g. --exclude-domains aviation)")
    parser.add_argument("--take-all", action="store_true",
                        help="Take every schema-correct pair, ignore --target and ratio")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_pass = int(args.target * args.pass_ratio)
    target_fail = args.target - target_pass
    print(f"Target: {args.target:,} pairs  ({target_pass:,} PASS + {target_fail:,} FAIL)")

    # ── Load ──
    print(f"Loading {args.input} ...")
    with open(args.input) as f:
        all_pairs = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(all_pairs):,} pairs")

    # ── Filter to correct schema + excluded domains ──
    correct = []
    wrong_schema = 0
    parse_errors = 0
    domain_excluded = 0
    exclude = [d.lower() for d in args.exclude_domains]

    for pair in all_pairs:
        if not is_correct_schema(pair):
            wrong_schema += 1
            continue
        parsed = parse_assistant(pair)
        if parsed is None:
            parse_errors += 1
            continue
        if exclude:
            domain = (pair.get("metadata", {}).get("domain", "") or "").lower()
            source = (pair.get("metadata", {}).get("source", "") or "").lower()
            if any(ex in domain or ex in source for ex in exclude):
                domain_excluded += 1
                continue
        correct.append((pair, parsed))

    print(f"Schema-correct: {len(correct):,}  |  wrong-schema: {wrong_schema:,}  |  parse-err: {parse_errors:,}  |  domain-excluded: {domain_excluded:,}")

    # ── Split by verdict ──
    pass_pairs = [(p, d) for p, d in correct if d.get("verdict", "").upper() == "PASS"]
    fail_pairs = [(p, d) for p, d in correct if d.get("verdict", "").upper() == "FAIL"]
    print(f"Available: {len(pass_pairs):,} PASS  |  {len(fail_pairs):,} FAIL")

    # ── Score and sort ──
    pass_pairs.sort(key=lambda x: selection_score(x[0], x[1]), reverse=True)
    fail_pairs.sort(key=lambda x: selection_score(x[0], x[1]), reverse=True)

    # ── Select ──
    if args.take_all:
        n_pass = len(pass_pairs)
        n_fail = len(fail_pairs)
        print(f"--take-all: using all {n_pass:,} PASS + {n_fail:,} FAIL")
    else:
        n_pass = min(target_pass, len(pass_pairs))
        n_fail = min(target_fail, len(fail_pairs))
        if n_pass < target_pass:
            print(f"⚠ Only {n_pass:,} PASS available (wanted {target_pass:,})")
        if n_fail < target_fail:
            print(f"⚠ Only {n_fail:,} FAIL available (wanted {target_fail:,})")

    selected_pass = [p for p, d in pass_pairs[:n_pass]]
    selected_fail = [p for p, d in fail_pairs[:n_fail]]
    selected = selected_pass + selected_fail

    # Shuffle (don't feed all PASS then all FAIL to the trainer)
    random.shuffle(selected)

    total_selected = len(selected)
    print(f"Selected: {total_selected:,}  ({n_pass:,} PASS + {n_fail:,} FAIL)")

    # ── Write ──
    out_path = output_dir / "phase1_train.jsonl"
    with open(out_path, "w") as f:
        for pair in selected:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Written → {out_path}")

    # ── Distribution report ──
    source_dist   = Counter()
    domain_dist   = Counter()
    verdict_dist  = Counter()
    score_dist    = Counter()

    for pair in selected:
        m = pair.get("metadata", {})
        parsed = parse_assistant(pair)
        source_dist[m.get("source", "") or "unknown"] += 1
        domain_dist[m.get("domain", "") or "unknown"] += 1
        v = (parsed or {}).get("verdict", "?").upper()
        t = (parsed or {}).get("total", 0) or 0
        verdict_dist[v] += 1
        score_dist["25" if t == 25 else "23-24" if t >= 23 else "20-22" if t >= 20 else "15-19" if t >= 15 else "<=14"] += 1

    report = {
        "generated_utc":   datetime.now(timezone.utc).isoformat(),
        "input_file":      args.input,
        "output_file":     str(out_path),
        "total_selected":  total_selected,
        "target":          args.target,
        "pass_count":      n_pass,
        "fail_count":      n_fail,
        "pass_ratio":      round(n_pass / total_selected, 3) if total_selected else 0,
        "schema_wrong_excluded": wrong_schema,
        "parse_errors_excluded": parse_errors,
        "source_distribution": dict(source_dist.most_common()),
        "domain_distribution": dict(domain_dist.most_common()),
        "verdict_distribution": dict(verdict_dist),
        "score_band_distribution": dict(score_dist),
    }

    report_path = output_dir / "phase1_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Print summary ──
    print()
    print("═" * 55)
    print(f"Phase 1 Selection Complete")
    print("═" * 55)
    print(f"  Total selected:    {total_selected:,} / {args.target:,} target")
    print(f"  PASS:              {n_pass:,}  ({n_pass/total_selected*100:.1f}%)")
    print(f"  FAIL:              {n_fail:,}  ({n_fail/total_selected*100:.1f}%)")
    print(f"  Schema excluded:   {wrong_schema:,} (8-crit wrong schema)")
    print()
    print("  Sources:")
    for s, c in source_dist.most_common():
        print(f"    {c:>6,}  {s}")
    print()
    print("  Domains:")
    for d, c in domain_dist.most_common():
        print(f"    {c:>6,}  {d}")
    print()
    print("  Score bands:")
    for b in ["25", "23-24", "20-22", "15-19", "<=14"]:
        print(f"    {score_dist.get(b, 0):>6,}  {b}")
    print()
    print(f"  Output:  {out_path}")
    print(f"  Report:  {report_path}")
    print("═" * 55)

    if total_selected < args.target * 0.8:
        print(f"\n⚠ WARNING: Selected {total_selected:,} pairs, below 80% of target {args.target:,}.")
        print("  Consider reducing --target or cooking additional schema-correct pairs.")


if __name__ == "__main__":
    main()
