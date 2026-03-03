#!/usr/bin/env python3
"""
Phase 2 Assembler — SwarmJudge-9B Block-0
==========================================

Mixes FAIL fill-in pairs + new PASS trajectory pairs.
Phase 1 was PASS-only (identity lock).
Phase 2 teaches the model what failure looks like + how to fix it.

Target: 23,000 pairs (70% PASS / 30% FAIL)
  FAIL:  ~6,900  (from r1-r6 existing cooks — free)
  PASS:  ~16,100 (from r7-r9 new cooks + Phase 1 overflow)

Cook runs consumed:
  Phase 1 (PASS only):  r1-r6  seeds 300-800  (23K selected)
  Phase 2 new PASS:     r7-r9  seeds 900-1100

Output: /data2/swarmjudge_27b_data/phase2_9b/phase2_train.jsonl
R2:     9b-block-0-phase2

Usage:
  python3 assemble_phase2.py
  python3 assemble_phase2.py --status
  python3 assemble_phase2.py --target 23000 --fail-ratio 0.30
"""

import json
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

random.seed(42)

BASE = Path("/data2/swarmjudge_27b_data")

# Phase 1 runs (used for FAIL pairs only — PASS already in Phase 1 train set)
PHASE1_DIRS = [
    BASE / "cre_judge_cook_llama4",       # r1 seed=300
    BASE / "cre_judge_cook_llama4_r2",    # r2 seed=400
    BASE / "cre_judge_cook_llama4_r3",    # r3 seed=500
    BASE / "cre_judge_cook_llama4_r4",    # r4 seed=600
    BASE / "cre_judge_cook_llama4_r5",    # r5 seed=700
    BASE / "cre_judge_cook_llama4_r6",    # r6 seed=800
]

# Phase 2 new cook runs (fresh PASS + FAIL, seeds not used in Phase 1)
PHASE2_DIRS = [
    BASE / "cre_judge_cook_llama4_r7",    # r7 seed=900
    BASE / "cre_judge_cook_llama4_r8",    # r8 seed=1000
    BASE / "cre_judge_cook_llama4_r9",    # r9 seed=1100
    BASE / "cre_judge_cook_llama4_r10",   # r10 seed=1200
    BASE / "cre_judge_cook_llama4_r11",   # r11 seed=1300 top-up
]

# Phase 1 train fingerprints (to prevent contamination)
PHASE1_TRAIN = BASE / "phase1_b0" / "phase1_train.jsonl"

R2_BUCKET = "9b-block-0-phase2"


def get_verdict(pair: dict) -> str:
    asst = next((m["content"] for m in pair.get("messages", []) if m["role"] == "assistant"), "")
    try:
        return json.loads(asst).get("verdict", "").upper()
    except Exception:
        return ""


def is_clean(pair: dict) -> bool:
    """Verify pair uses 5-criterion schema."""
    asst = next((m["content"] for m in pair.get("messages", []) if m["role"] == "assistant"), "")
    try:
        ev = json.loads(asst)
        scores = ev.get("scores", {})
        required = {"accuracy", "completeness", "structure", "relevance", "sft_quality"}
        bad = {"grounding", "coherence", "efficiency", "format", "safety"}
        return required.issubset(scores.keys()) and not any(b in scores for b in bad)
    except Exception:
        return False


def has_substance(pair: dict) -> bool:
    """FAIL pairs must have at least one issue and at least one fix."""
    asst = next((m["content"] for m in pair.get("messages", []) if m["role"] == "assistant"), "")
    try:
        ev = json.loads(asst)
        return (
            get_verdict(pair) == "PASS" or
            (len(ev.get("issues", [])) >= 1 and len(ev.get("fixes", [])) >= 1)
        )
    except Exception:
        return False


def load_phase1_fingerprints() -> set:
    fps = set()
    if PHASE1_TRAIN.exists():
        with open(PHASE1_TRAIN) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                    fp = p.get("metadata", {}).get("fingerprint", "")
                    if fp:
                        fps.add(fp)
                except Exception:
                    pass
    return fps


def load_pairs(dirs: list, target_verdict: str = None) -> tuple[list, dict]:
    """Load clean pairs from cook dirs, optionally filtered by verdict."""
    pairs = []
    seen_fps = set()
    stats = {"total": 0, "pass": 0, "fail": 0, "dirty": 0, "dup": 0}

    for d in dirs:
        f = d / "cre_judge_train.jsonl"
        if not f.exists():
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                stats["total"] += 1
                pair = json.loads(line)
                v = get_verdict(pair)

                if target_verdict and v != target_verdict:
                    continue
                if not is_clean(pair):
                    stats["dirty"] += 1
                    continue
                if not has_substance(pair):
                    stats["dirty"] += 1
                    continue

                fp = pair.get("metadata", {}).get("fingerprint", "")
                if fp and fp in seen_fps:
                    stats["dup"] += 1
                    continue
                if fp:
                    seen_fps.add(fp)

                if v == "PASS":
                    stats["pass"] += 1
                elif v == "FAIL":
                    stats["fail"] += 1

                pairs.append(pair)

    return pairs, stats, seen_fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",      type=int,   default=23000)
    parser.add_argument("--fail-ratio",  type=float, default=0.30)
    parser.add_argument("--output",      default=str(BASE / "phase2_9b"))
    parser.add_argument("--push-r2",     action="store_true")
    parser.add_argument("--status",      action="store_true")
    args = parser.parse_args()

    fail_target = int(args.target * args.fail_ratio)
    pass_target = args.target - fail_target

    print(f"\n{'═'*60}")
    print(f"Phase 2 Assembler — SwarmJudge-9B Block-0")
    print(f"{'═'*60}")
    print(f"  Target:     {args.target:,} pairs")
    print(f"  PASS quota: {pass_target:,} ({100-int(args.fail_ratio*100)}%)")
    print(f"  FAIL quota: {fail_target:,} ({int(args.fail_ratio*100)}%)")
    print()

    # ── Status mode ─────────────────────────────────────────────────
    if args.status:
        print("Phase 1 dirs (FAIL pairs available):")
        total_fail = 0
        for d in PHASE1_DIRS:
            f = d / "cre_judge_train.jsonl"
            if not f.exists():
                print(f"  {d.name:<35} MISSING")
                continue
            pairs = [json.loads(l) for l in open(f) if l.strip()]
            fail_n = sum(1 for p in pairs if get_verdict(p) == "FAIL" and is_clean(p))
            total_fail += fail_n
            print(f"  {d.name:<35} {fail_n:>5,} FAIL")

        print(f"\nPhase 2 dirs (new cook):")
        total_new = 0
        for d in PHASE2_DIRS:
            f = d / "cre_judge_train.jsonl"
            if not f.exists():
                print(f"  {d.name:<35} NOT STARTED")
                continue
            pairs = [json.loads(l) for l in open(f) if l.strip()]
            pass_n = sum(1 for p in pairs if get_verdict(p) == "PASS" and is_clean(p))
            fail_n = sum(1 for p in pairs if get_verdict(p) == "FAIL" and is_clean(p))
            total_new += pass_n + fail_n
            print(f"  {d.name:<35} {pass_n:>5,} PASS  {fail_n:>5,} FAIL")

        print(f"\n  Available FAIL (Phase 1): {total_fail:,} / need {fail_target:,}")
        print(f"  Available new pairs:      {total_new:,} / need {pass_target:,} PASS")
        remaining_pass = max(0, pass_target - total_new)
        if remaining_pass:
            print(f"  NEED: cook ~{int(remaining_pass / 0.61):,} more pairs for Phase 2 PASS")
        else:
            print(f"  READY: Enough Phase 2 PASS pairs available ✓")
        return

    # ── Load Phase 1 FAIL pairs ──────────────────────────────────────
    print("Loading Phase 1 FAIL pairs (fill-in)...")
    phase1_fps = load_phase1_fingerprints()
    print(f"  Phase 1 train fingerprints loaded: {len(phase1_fps):,} (contamination guard)")

    fail_pairs, fail_stats, fail_seen = load_pairs(PHASE1_DIRS, target_verdict="FAIL")
    # Exclude any FAIL pairs whose fingerprint appears in Phase 1 train (shouldn't happen but be safe)
    fail_pairs = [p for p in fail_pairs if p.get("metadata", {}).get("fingerprint", "") not in phase1_fps]
    print(f"  Clean FAIL pairs from r1-r6: {len(fail_pairs):,}")

    # ── Load Phase 2 new pairs ───────────────────────────────────────
    print("\nLoading Phase 2 new cook pairs (trajectory)...")
    all_seen = fail_seen | phase1_fps
    new_pass = []
    new_fail = []

    for d in PHASE2_DIRS:
        f = d / "cre_judge_train.jsonl"
        if not f.exists():
            print(f"  {d.name:<35} MISSING — skipping")
            continue
        run_p = run_f = 0
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                pair = json.loads(line)
                if not is_clean(pair) or not has_substance(pair):
                    continue
                fp = pair.get("metadata", {}).get("fingerprint", "")
                if fp and fp in all_seen:
                    continue
                if fp:
                    all_seen.add(fp)
                v = get_verdict(pair)
                if v == "PASS":
                    new_pass.append(pair)
                    run_p += 1
                elif v == "FAIL":
                    new_fail.append(pair)
                    run_f += 1
        print(f"  {d.name:<35} {run_p:>5,} PASS  {run_f:>5,} FAIL")

    print(f"\n  Total FAIL available:   {len(fail_pairs) + len(new_fail):,}")
    print(f"  Total new PASS:         {len(new_pass):,}")

    # ── Check readiness ─────────────────────────────────────────────
    total_fail_avail = len(fail_pairs) + len(new_fail)
    total_pass_avail = len(new_pass)

    if total_fail_avail < fail_target:
        print(f"\n  INSUFFICIENT FAIL: {total_fail_avail:,} < {fail_target:,} needed")
        return
    if total_pass_avail < pass_target:
        print(f"\n  INSUFFICIENT PASS: {total_pass_avail:,} < {pass_target:,} needed")
        print(f"  Cook ~{int((pass_target - total_pass_avail) / 0.61):,} more pairs (r8+) and re-run")
        return

    # ── Select and balance ───────────────────────────────────────────
    random.shuffle(fail_pairs)
    random.shuffle(new_fail)
    random.shuffle(new_pass)

    # Fill FAIL quota from Phase 1 FAIL first, supplement with new FAIL
    all_fail = fail_pairs + new_fail
    random.shuffle(all_fail)
    selected_fail = all_fail[:fail_target]
    selected_pass = new_pass[:pass_target]
    selected = selected_fail + selected_pass
    random.shuffle(selected)

    print(f"\n  Selected: {len(selected_fail):,} FAIL + {len(selected_pass):,} PASS = {len(selected):,} total")

    # ── Write ────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "phase2_train.jsonl"

    with open(out_path, "w") as f:
        for p in selected:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ── Distribution report ──────────────────────────────────────────
    task_dist  = Counter()
    score_dist = Counter()
    verdict_dist = Counter()
    for p in selected:
        m = p.get("metadata", {})
        task_dist[m.get("task_type", "unknown")] += 1
        verdict_dist[get_verdict(p)] += 1
        asst = next((msg["content"] for msg in p["messages"] if msg["role"] == "assistant"), "")
        try:
            total = json.loads(asst).get("total", 0)
            score_dist["25" if total == 25 else "23-24" if total >= 23 else "20-22" if total >= 20 else "15-19" if total >= 15 else "<15"] += 1
        except Exception:
            pass

    report = {
        "generated_utc":     datetime.now(timezone.utc).isoformat(),
        "phase":             "phase2",
        "target":            args.target,
        "total_selected":    len(selected),
        "pass_count":        verdict_dist.get("PASS", 0),
        "fail_count":        verdict_dist.get("FAIL", 0),
        "judge_model":       "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "r2_bucket":         R2_BUCKET,
        "task_distribution": dict(task_dist.most_common()),
        "score_distribution": dict(score_dist),
        "verdict_split":     dict(verdict_dist),
    }
    report_path = output_dir / "phase2_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'═'*60}")
    print(f"Phase 2 Assembly — SwarmJudge-9B Block-0")
    print(f"{'═'*60}")
    print(f"  Total:    {len(selected):,}  ({verdict_dist.get('PASS',0):,} PASS / {verdict_dist.get('FAIL',0):,} FAIL)")
    print(f"  Judge:    Llama 4 Maverick 17Bx128E")
    print(f"  Output:   {out_path}")
    print(f"\n  Tasks:")
    for t, c in task_dist.most_common():
        print(f"    {c:>6,}  {t}")
    print(f"\n  Score bands:")
    for b in ["25", "23-24", "20-22", "15-19", "<15"]:
        print(f"    {score_dist.get(b,0):>6,}  {b}")
    print(f"{'═'*60}")

    # ── Push to R2 ───────────────────────────────────────────────────
    if args.push_r2:
        print(f"\nPushing to R2 bucket: {R2_BUCKET}...")
        r2_key = "phase2/phase2_train.jsonl"
        cmd = ["aws", "s3", "cp", str(out_path),
               f"s3://{R2_BUCKET}/{r2_key}",
               "--endpoint-url", "https://6abec5e82728df0610a98be9364918e4.r2.cloudflarestorage.com"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Pushed → r2://{R2_BUCKET}/{r2_key}")
        else:
            print(f"  R2 push failed: {result.stderr}")


if __name__ == "__main__":
    main()
