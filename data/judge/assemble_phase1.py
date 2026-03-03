#!/usr/bin/env python3
"""
Phase 1 Assembler — SwarmJudge-27B Block-0
==========================================

Collects PASS pairs from all Llama 4 Maverick cook runs.
NO contamination. NO existing data. Clean Llama 4 PASS pairs only.

Target: 23,000 PASS pairs
Output: /data2/swarmjudge_27b_data/phase1_b0/phase1_train.jsonl
R2:     27b-block-0-pass

Cook runs consumed:
  r1  seed=300  5K  processed
  r2  seed=400  10K processed
  r3  seed=500  8K  processed
  r4  seed=600  8K  processed
  r5  seed=700  8K  processed

Usage:
  python3 assemble_phase1.py
  python3 assemble_phase1.py --target 23000 --push-r2
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

COOK_DIRS = [
    BASE / "cre_judge_cook_llama4",       # r1 seed=300
    BASE / "cre_judge_cook_llama4_r2",    # r2 seed=400
    BASE / "cre_judge_cook_llama4_r3",    # r3 seed=500
    BASE / "cre_judge_cook_llama4_r4",    # r4 seed=600
    BASE / "cre_judge_cook_llama4_r5",    # r5 seed=700
    BASE / "cre_judge_cook_llama4_r6",    # r6 seed=800
]

R2_BUCKET = "27b-block-0-pass"


def get_verdict(pair: dict) -> str:
    asst = next((m["content"] for m in pair.get("messages", []) if m["role"] == "assistant"), "")
    try:
        return json.loads(asst).get("verdict", "").upper()
    except Exception:
        return ""


def is_clean(pair: dict) -> bool:
    """Verify pair uses 5-criterion schema — no contamination."""
    asst = next((m["content"] for m in pair.get("messages", []) if m["role"] == "assistant"), "")
    try:
        ev = json.loads(asst)
        scores = ev.get("scores", {})
        required = {"accuracy", "completeness", "structure", "relevance", "sft_quality"}
        bad = {"grounding", "coherence", "efficiency", "format", "safety"}
        return required.issubset(scores.keys()) and not any(b in scores for b in bad)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",   type=int, default=23000)
    parser.add_argument("--output",   default=str(BASE / "phase1_b0"))
    parser.add_argument("--push-r2",  action="store_true", help="Push to R2 bucket after assembly")
    parser.add_argument("--status",   action="store_true", help="Show cook progress and exit")
    args = parser.parse_args()

    # ── Status mode ─────────────────────────────────────────────────────
    if args.status:
        total_pass = 0
        total_all  = 0
        for d in COOK_DIRS:
            f = d / "cre_judge_train.jsonl"
            if not f.exists():
                print(f"  {d.name:<35} NOT STARTED")
                continue
            pairs = [json.loads(l) for l in open(f) if l.strip()]
            pass_n = sum(1 for p in pairs if get_verdict(p) == "PASS")
            total_all  += len(pairs)
            total_pass += pass_n
            print(f"  {d.name:<35} {len(pairs):>6,} cooked  |  {pass_n:>6,} PASS ({pass_n/max(len(pairs),1)*100:.0f}%)")
        print(f"\n  TOTAL: {total_all:,} cooked  |  {total_pass:,} PASS  |  need {args.target:,}")
        remaining = max(0, args.target - total_pass)
        if remaining:
            print(f"  NEED:  {remaining:,} more PASS (cook ~{int(remaining/0.61):,} more pairs)")
        else:
            print(f"  READY: {total_pass:,} PASS >= {args.target:,} target ✓")
        return

    # ── Load all PASS pairs from all cook runs ───────────────────────────
    print("Loading Llama 4 cook runs — PASS only, no contamination...\n")
    all_pass = []
    seen_fps = set()

    for d in COOK_DIRS:
        f = d / "cre_judge_train.jsonl"
        if not f.exists():
            print(f"  {d.name:<35} MISSING — skipping")
            continue
        run_pass = 0
        run_dirty = 0
        run_dup = 0
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                pair = json.loads(line)
                if get_verdict(pair) != "PASS":
                    continue
                if not is_clean(pair):
                    run_dirty += 1
                    continue
                fp = pair.get("metadata", {}).get("fingerprint", "")
                if fp and fp in seen_fps:
                    run_dup += 1
                    continue
                if fp:
                    seen_fps.add(fp)
                all_pass.append(pair)
                run_pass += 1
        print(f"  {d.name:<35} {run_pass:>6,} PASS  (dirty={run_dirty}, dup={run_dup})")

    print(f"\n  Total clean PASS: {len(all_pass):,}")

    if len(all_pass) < args.target:
        print(f"\n  INSUFFICIENT: {len(all_pass):,} < {args.target:,} target")
        print(f"  Cook ~{int((args.target - len(all_pass)) / 0.61):,} more pairs and re-run")
        return

    # ── Select 23K ──────────────────────────────────────────────────────
    random.shuffle(all_pass)
    selected = all_pass[:args.target]
    print(f"  Selected: {len(selected):,} / {len(all_pass):,} available")

    # ── Write ───────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "phase1_train.jsonl"

    with open(out_path, "w") as f:
        for p in selected:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ── Domain / task distribution ──────────────────────────────────────
    task_dist   = Counter()
    score_dist  = Counter()
    for p in selected:
        m = p.get("metadata", {})
        task_dist[m.get("task_type", "unknown")] += 1
        asst = next((msg["content"] for msg in p["messages"] if msg["role"] == "assistant"), "")
        try:
            total = json.loads(asst).get("total", 0)
            score_dist["25" if total == 25 else "23-24" if total >= 23 else "20-22" if total >= 20 else "<20"] += 1
        except Exception:
            pass

    report = {
        "generated_utc":   datetime.now(timezone.utc).isoformat(),
        "target":          args.target,
        "total_selected":  len(selected),
        "total_available": len(all_pass),
        "judge_model":     "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "verdict":         "PASS_ONLY",
        "contamination":   "NONE",
        "r2_bucket":       R2_BUCKET,
        "task_distribution":  dict(task_dist.most_common()),
        "score_distribution": dict(score_dist),
    }
    report_path = output_dir / "phase1_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'═'*55}")
    print(f"Phase 1 Assembly — SwarmJudge-27B Block-0")
    print(f"{'═'*55}")
    print(f"  Pairs:    {len(selected):,}  (PASS only, no contamination)")
    print(f"  Judge:    Llama 4 Maverick 17Bx128E")
    print(f"  Output:   {out_path}")
    print(f"\n  Tasks:")
    for t, c in task_dist.most_common():
        print(f"    {c:>6,}  {t}")
    print(f"\n  Score bands:")
    for b in ["25", "23-24", "20-22", "<20"]:
        print(f"    {score_dist.get(b,0):>6,}  {b}")
    print(f"{'═'*55}")

    # ── Push to R2 ──────────────────────────────────────────────────────
    if args.push_r2:
        print(f"\nPushing to R2 bucket: {R2_BUCKET}...")
        r2_key = f"phase1/phase1_train.jsonl"
        cmd = ["aws", "s3", "cp", str(out_path),
               f"s3://{R2_BUCKET}/{r2_key}",
               "--endpoint-url", "https://6abec5e82728df0610a98be9364918e4.r2.cloudflarestorage.com"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Pushed → r2://{R2_BUCKET}/{r2_key}")
        else:
            print(f"  R2 push failed: {result.stderr}")
            print(f"  Run manually: aws s3 cp {out_path} s3://{R2_BUCKET}/{r2_key} --endpoint-url ...")


if __name__ == "__main__":
    main()
