"""
SwarmJudge-27B Dataset Curator
================================

Quality-curates ~100K training pairs from three cook sources:
  1. Trajectory cook   (42,739 records — 5-criteria CoVe verdicts)
  2. Agent trace cook   (~40,000 records — agent archetypes + failure modes)
  3. Judge eval cook    (~80,000 records — RAG/multi-agent/code/safety)

Curation strategy: take high-confidence pairs from both extremes.
Skip the ambiguous middle to give the model a clean decision boundary.

Pulls data directly from swarmrails + whale via SSH/SCP, runs locally.

Usage:
    python3 -m data.judge.assemble_swarmjudge_27b [--target 100000] [--eval-pct 5]
    python3 data/judge/assemble_swarmjudge_27b.py --dry-run
    python3 data/judge/assemble_swarmjudge_27b.py --target 100000 --stage
"""

import json
import random
import hashlib
import subprocess
import shutil
import sys
import os
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Generator

# ═══════════════════════════════════════════════════════════════════════
# Remote paths
# ═══════════════════════════════════════════════════════════════════════

TRAJECTORY_REMOTE = "swarmrails:/data2/swarmjudge_data/trajectory_cooked/swarmjudge_trajectory_train.jsonl"
AGENT_TRACE_REMOTE = "swarmrails:/data2/swarmjudge_agent_traces/agent_judge_train.jsonl"
JUDGE_EVAL_WHALE_REMOTE = "whale:~/cre_neweconomy/judge_eval_train.jsonl"
JUDGE_EVAL_RAILS_REMOTE = "swarmrails:/data2/judge_eval_cook/judge_eval_train.jsonl"

LOCAL_CACHE = Path("/tmp/swarmjudge_27b_assembly")
OUTPUT_DIR = Path("/tmp/swarmjudge_27b")

# ═══════════════════════════════════════════════════════════════════════
# Quality Thresholds
# ═══════════════════════════════════════════════════════════════════════

# Trajectory cook: 5-criteria scores (25 total)
TRAJ_PASS_FLOOR = 23      # Only take strong PASS (23-25 out of 25)
TRAJ_FAIL_CEILING = 17    # Only take clear FAIL (<= 17 out of 25)
# Skip the mushy 18-22 middle — ambiguous signal

# Agent trace cook: health_score (0-100)
AGENT_PASS_FLOOR = 85     # Strong PASS
AGENT_FAIL_CEILING = 65   # Clear FAIL
# Skip 66-84 middle

# Judge eval cook: verdict-based (already two-tier validated)
# All records passed 235B judge — trust the verdict, filter only format issues
JUDGE_MIN_MESSAGES = 3    # Must have system + user + assistant


# ═══════════════════════════════════════════════════════════════════════
# Data Pull
# ═══════════════════════════════════════════════════════════════════════

def pull_remote(remote_path: str, local_name: str) -> Path:
    """SCP a file from remote machine to local cache."""
    local_path = LOCAL_CACHE / local_name
    if local_path.exists():
        age_min = (datetime.now().timestamp() - local_path.stat().st_mtime) / 60
        if age_min < 30:
            print(f"  CACHE HIT: {local_name} ({age_min:.0f}min old)")
            return local_path

    print(f"  Pulling {remote_path} ...")
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["scp", remote_path, str(local_path)],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"  ERROR pulling {remote_path}: {result.stderr}")
        return local_path  # May not exist
    size_mb = local_path.stat().st_size / 1e6
    print(f"  OK: {local_name} ({size_mb:.1f} MB)")
    return local_path


def iter_jsonl(path: Path) -> Generator[dict, None, None]:
    """Stream JSONL records."""
    if not path.exists():
        print(f"  MISSING: {path}")
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ═══════════════════════════════════════════════════════════════════════
# Source 1: Trajectory Cook Curator
# ═══════════════════════════════════════════════════════════════════════

def curate_trajectory(path: Path) -> tuple[list[dict], dict]:
    """Curate trajectory cook — take extremes, skip middle.

    These records have:
      messages: [{role, content}, ...]
      metadata: {domain, specialty, verdict, total, source, ...}
    """
    stats = {
        "raw": 0, "accepted": 0, "rejected_middle": 0,
        "rejected_format": 0, "pass": 0, "fail": 0,
        "by_domain": Counter(), "by_score": Counter(),
    }
    accepted = []

    for rec in iter_jsonl(path):
        stats["raw"] += 1
        meta = rec.get("metadata", {})
        messages = rec.get("messages", [])
        total = int(meta.get("total", 0))
        verdict = meta.get("verdict", "UNKNOWN")

        # Format check
        if len(messages) < 3:
            stats["rejected_format"] += 1
            continue

        # Quality gate: take extremes only
        if verdict == "PASS" and total >= TRAJ_PASS_FLOOR:
            stats["pass"] += 1
            accepted.append(rec)
        elif verdict == "FAIL" and total <= TRAJ_FAIL_CEILING:
            stats["fail"] += 1
            accepted.append(rec)
        else:
            stats["rejected_middle"] += 1
            continue

        stats["accepted"] += 1
        stats["by_domain"][meta.get("domain", "unknown")] += 1
        stats["by_score"][total] += 1

    return accepted, stats


# ═══════════════════════════════════════════════════════════════════════
# Source 2: Agent Trace Cook Curator
# ═══════════════════════════════════════════════════════════════════════

def curate_agent_traces(path: Path) -> tuple[list[dict], dict]:
    """Curate agent traces — take high/low health, skip middle.

    These records have:
      messages: [{role, content}, ...]
      metadata: {agent_type, quality_mode, verdict, health_score, failure_mode, ...}
    """
    stats = {
        "raw": 0, "accepted": 0, "rejected_middle": 0,
        "rejected_format": 0, "pass": 0, "fail": 0,
        "by_archetype": Counter(), "by_quality_mode": Counter(),
    }
    accepted = []

    for rec in iter_jsonl(path):
        stats["raw"] += 1
        meta = rec.get("metadata", {})
        messages = rec.get("messages", [])
        health = int(meta.get("health_score", 0))
        verdict = meta.get("verdict", "UNKNOWN")

        if len(messages) < 3:
            stats["rejected_format"] += 1
            continue

        # Quality gate: take extremes
        if verdict == "PASS" and health >= AGENT_PASS_FLOOR:
            stats["pass"] += 1
            accepted.append(rec)
        elif verdict == "FAIL" and health <= AGENT_FAIL_CEILING:
            stats["fail"] += 1
            accepted.append(rec)
        else:
            stats["rejected_middle"] += 1
            continue

        stats["accepted"] += 1
        stats["by_archetype"][meta.get("agent_type", "unknown")] += 1
        stats["by_quality_mode"][meta.get("quality_mode", "unknown")] += 1

    return accepted, stats


# ═══════════════════════════════════════════════════════════════════════
# Source 3: Judge Eval Cook Curator
# ═══════════════════════════════════════════════════════════════════════

def curate_judge_eval(path: Path) -> tuple[list[dict], dict]:
    """Curate judge eval — trust 235B verdicts, filter format issues.

    These records have:
      id, stream, category, quality_mode, messages, metadata
    """
    stats = {
        "raw": 0, "accepted": 0, "rejected_format": 0,
        "pass": 0, "fail": 0,
        "by_stream": Counter(), "by_quality_mode": Counter(),
        "by_category": Counter(),
    }
    accepted = []

    for rec in iter_jsonl(path):
        stats["raw"] += 1
        meta = rec.get("metadata", {})
        messages = rec.get("messages", [])
        verdict = meta.get("verdict", "UNKNOWN")

        if len(messages) < JUDGE_MIN_MESSAGES:
            stats["rejected_format"] += 1
            continue

        # Verify assistant response is parseable JSON
        assistant_msg = messages[-1].get("content", "") if messages else ""
        try:
            parsed = json.loads(assistant_msg)
            if "verdict" not in parsed:
                stats["rejected_format"] += 1
                continue
        except (json.JSONDecodeError, TypeError):
            # Try extracting JSON from think-mode output
            if '{"verdict"' in assistant_msg:
                try:
                    start = assistant_msg.index('{"verdict"')
                    # Find matching closing brace
                    depth = 0
                    for i in range(start, len(assistant_msg)):
                        if assistant_msg[i] == '{':
                            depth += 1
                        elif assistant_msg[i] == '}':
                            depth -= 1
                            if depth == 0:
                                json.loads(assistant_msg[start:i+1])
                                break
                    else:
                        stats["rejected_format"] += 1
                        continue
                except (json.JSONDecodeError, ValueError):
                    stats["rejected_format"] += 1
                    continue
            else:
                stats["rejected_format"] += 1
                continue

        if verdict == "PASS":
            stats["pass"] += 1
        else:
            stats["fail"] += 1

        stats["accepted"] += 1
        stats["by_stream"][meta.get("stream", rec.get("stream", "unknown"))] += 1
        stats["by_quality_mode"][meta.get("quality_mode", rec.get("quality_mode", "unknown"))] += 1
        stats["by_category"][meta.get("category", rec.get("category", "unknown"))] += 1
        accepted.append(rec)

    return accepted, stats


# ═══════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════

def fingerprint(rec: dict) -> str:
    """Generate fingerprint from user message content."""
    messages = rec.get("messages", [])
    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    content = "".join(user_msgs)[:500]
    return hashlib.md5(content.encode()).hexdigest()[:12]


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove exact and near-duplicates."""
    seen = set()
    unique = []
    for rec in records:
        fp = fingerprint(rec)
        if fp in seen:
            continue
        seen.add(fp)
        unique.append(rec)
    return unique


# ═══════════════════════════════════════════════════════════════════════
# Balancer
# ═══════════════════════════════════════════════════════════════════════

def balance_and_select(
    trajectory: list[dict],
    agent_traces: list[dict],
    judge_eval: list[dict],
    target: int,
) -> list[dict]:
    """Balance across sources with target PASS/FAIL ratio ~55/45.

    Allocation strategy:
      - Trajectory:   20% of target (proven data, strong signal)
      - Agent traces:  25% of target (diverse archetypes)
      - Judge eval:   55% of target (largest source, 4 new streams)
    """
    traj_target = int(target * 0.20)
    agent_target = int(target * 0.25)
    judge_target = target - traj_target - agent_target

    print(f"\nAllocation targets:")
    print(f"  Trajectory:   {traj_target:,} (from {len(trajectory):,} available)")
    print(f"  Agent traces: {agent_target:,} (from {len(agent_traces):,} available)")
    print(f"  Judge eval:   {judge_target:,} (from {len(judge_eval):,} available)")

    selected = []

    # Select from each source with PASS/FAIL balancing
    for source, source_target, source_name in [
        (trajectory, traj_target, "trajectory"),
        (agent_traces, agent_target, "agent_traces"),
        (judge_eval, judge_target, "judge_eval"),
    ]:
        random.shuffle(source)

        # Split into PASS and FAIL
        passes = [r for r in source if _get_verdict(r) == "PASS"]
        fails = [r for r in source if _get_verdict(r) == "FAIL"]

        # Target 55% PASS, 45% FAIL per source
        pass_target = int(source_target * 0.55)
        fail_target = source_target - pass_target

        # If one side is short, redistribute
        if len(passes) < pass_target:
            pass_take = len(passes)
            fail_target = min(source_target - pass_take, len(fails))
        elif len(fails) < fail_target:
            fail_take = len(fails)
            pass_target = min(source_target - fail_take, len(passes))
            fail_target = fail_take

        pass_take = min(pass_target, len(passes))
        fail_take = min(fail_target, len(fails))

        taken_passes = passes[:pass_take]
        taken_fails = fails[:fail_take]

        # Tag source
        for r in taken_passes + taken_fails:
            meta = r.get("metadata", {})
            meta["curator_source"] = source_name
            r["metadata"] = meta

        selected.extend(taken_passes)
        selected.extend(taken_fails)

        total_taken = pass_take + fail_take
        print(f"  {source_name}: took {total_taken:,} ({pass_take:,} PASS + {fail_take:,} FAIL)")

    random.shuffle(selected)
    return selected


def _get_verdict(rec: dict) -> str:
    """Extract verdict from record (handles both metadata formats)."""
    meta = rec.get("metadata", {})
    return meta.get("verdict", "UNKNOWN")


# ═══════════════════════════════════════════════════════════════════════
# Stats Reporter
# ═══════════════════════════════════════════════════════════════════════

def print_stats(name: str, stats: dict):
    """Pretty print curation stats."""
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Raw:          {stats['raw']:,}")
    print(f"  Accepted:     {stats['accepted']:,} ({stats['accepted']/max(stats['raw'],1)*100:.1f}%)")
    if "rejected_middle" in stats:
        print(f"  Skip middle:  {stats['rejected_middle']:,}")
    print(f"  Bad format:   {stats['rejected_format']:,}")
    print(f"  PASS:         {stats['pass']:,}")
    print(f"  FAIL:         {stats['fail']:,}")

    for key in sorted(stats.keys()):
        if key.startswith("by_"):
            label = key[3:].replace("_", " ").title()
            print(f"\n  By {label}:")
            for k, v in stats[key].most_common(15):
                print(f"    {k}: {v:,}")


# ═══════════════════════════════════════════════════════════════════════
# Staging (SCP to swarmrails)
# ═══════════════════════════════════════════════════════════════════════

def stage_to_swarmrails(output_dir: Path):
    """Push assembled data to swarmrails for training."""
    remote_dir = "/data2/swarmjudge_27b_data"
    print(f"\nStaging to swarmrails:{remote_dir} ...")

    # Create remote dir
    subprocess.run(
        ["ssh", "swarmrails", f"mkdir -p {remote_dir}"],
        check=True, timeout=30
    )

    # SCP all output files
    for f in output_dir.iterdir():
        print(f"  Pushing {f.name} ...")
        subprocess.run(
            ["scp", str(f), f"swarmrails:{remote_dir}/{f.name}"],
            check=True, timeout=300
        )

    print(f"  STAGED: swarmrails:{remote_dir}/")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Curate SwarmJudge-27B training data")
    parser.add_argument("--target", type=int, default=100000,
                        help="Target curated record count (default: 100K)")
    parser.add_argument("--eval-pct", type=int, default=5,
                        help="Percentage held out for eval (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Pull data and show stats but don't write output")
    parser.add_argument("--no-pull", action="store_true",
                        help="Use cached data only (skip SCP)")
    parser.add_argument("--stage", action="store_true",
                        help="SCP output to swarmrails after assembly")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("SwarmJudge-27B Dataset Curator")
    print(f"Target: {args.target:,} curated pairs")
    print("=" * 60)

    # ── Pull data ────────────────────────────────────────────────
    if not args.no_pull:
        print("\n[1/6] Pulling data from remote machines...")
        traj_path = pull_remote(TRAJECTORY_REMOTE, "trajectory.jsonl")
        agent_path = pull_remote(AGENT_TRACE_REMOTE, "agent_traces.jsonl")
        judge_whale_path = pull_remote(JUDGE_EVAL_WHALE_REMOTE, "judge_eval_whale.jsonl")
        judge_rails_path = pull_remote(JUDGE_EVAL_RAILS_REMOTE, "judge_eval_rails.jsonl")
    else:
        print("\n[1/6] Using cached data...")
        traj_path = LOCAL_CACHE / "trajectory.jsonl"
        agent_path = LOCAL_CACHE / "agent_traces.jsonl"
        judge_whale_path = LOCAL_CACHE / "judge_eval_whale.jsonl"
        judge_rails_path = LOCAL_CACHE / "judge_eval_rails.jsonl"

    # Merge judge eval files into one
    judge_path = LOCAL_CACHE / "judge_eval_merged.jsonl"
    print("  Merging judge eval files...")
    with open(judge_path, "w") as out:
        for src in [judge_whale_path, judge_rails_path]:
            if src.exists():
                with open(src) as f:
                    for line in f:
                        out.write(line)
    merged_lines = sum(1 for _ in open(judge_path))
    print(f"  Merged: {merged_lines:,} records")

    # ── Curate each source ───────────────────────────────────────
    print("\n[2/6] Curating trajectory cook (extremes only)...")
    print(f"  PASS floor: {TRAJ_PASS_FLOOR}/25  |  FAIL ceiling: {TRAJ_FAIL_CEILING}/25")
    traj_curated, traj_stats = curate_trajectory(traj_path)
    print_stats("Trajectory Cook", traj_stats)

    print(f"\n[3/6] Curating agent trace cook (extremes only)...")
    print(f"  PASS floor: {AGENT_PASS_FLOOR}/100  |  FAIL ceiling: {AGENT_FAIL_CEILING}/100")
    agent_curated, agent_stats = curate_agent_traces(agent_path)
    print_stats("Agent Trace Cook", agent_stats)

    print(f"\n[4/6] Curating judge eval cook (format-verified)...")
    judge_curated, judge_stats = curate_judge_eval(judge_path)
    print_stats("Judge Eval Cook", judge_stats)

    # ── Deduplicate ──────────────────────────────────────────────
    print(f"\nDeduplicating...")
    traj_dedup = deduplicate(traj_curated)
    agent_dedup = deduplicate(agent_curated)
    judge_dedup = deduplicate(judge_curated)
    print(f"  Trajectory:   {len(traj_curated):,} → {len(traj_dedup):,} ({len(traj_curated)-len(traj_dedup)} dups)")
    print(f"  Agent traces: {len(agent_curated):,} → {len(agent_dedup):,} ({len(agent_curated)-len(agent_dedup)} dups)")
    print(f"  Judge eval:   {len(judge_curated):,} → {len(judge_dedup):,} ({len(judge_curated)-len(judge_dedup)} dups)")

    total_available = len(traj_dedup) + len(agent_dedup) + len(judge_dedup)
    print(f"\n  Total available after curation: {total_available:,}")

    if args.dry_run:
        print(f"\n  DRY RUN — would select {min(args.target, total_available):,} from {total_available:,}")
        print(f"\n  Curated pool composition:")
        all_pass = sum(1 for r in traj_dedup + agent_dedup + judge_dedup if _get_verdict(r) == "PASS")
        all_fail = sum(1 for r in traj_dedup + agent_dedup + judge_dedup if _get_verdict(r) == "FAIL")
        print(f"    PASS: {all_pass:,} ({all_pass/max(total_available,1)*100:.1f}%)")
        print(f"    FAIL: {all_fail:,} ({all_fail/max(total_available,1)*100:.1f}%)")
        return

    # ── Balance and select ───────────────────────────────────────
    print(f"\n[5/6] Balancing and selecting {args.target:,} pairs...")
    selected = balance_and_select(traj_dedup, agent_dedup, judge_dedup, args.target)

    # ── Final stats ──────────────────────────────────────────────
    final_pass = sum(1 for r in selected if _get_verdict(r) == "PASS")
    final_fail = len(selected) - final_pass
    source_counts = Counter()
    for r in selected:
        source_counts[r.get("metadata", {}).get("curator_source", "unknown")] += 1

    print(f"\n{'='*60}")
    print(f"FINAL CURATED DATASET: {len(selected):,} pairs")
    print(f"  PASS: {final_pass:,} ({final_pass/len(selected)*100:.1f}%)")
    print(f"  FAIL: {final_fail:,} ({final_fail/len(selected)*100:.1f}%)")
    print(f"\n  By source:")
    for src, cnt in source_counts.most_common():
        print(f"    {src}: {cnt:,} ({cnt/len(selected)*100:.1f}%)")

    # ── Train/eval split ─────────────────────────────────────────
    eval_count = max(1, int(len(selected) * args.eval_pct / 100))
    random.shuffle(selected)
    eval_set = selected[:eval_count]
    train_set = selected[eval_count:]

    # ── Write output ─────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "swarmjudge_27b_train.jsonl"
    eval_path = OUTPUT_DIR / "swarmjudge_27b_eval.jsonl"

    train_bytes = 0
    with open(train_path, "w") as f:
        for r in train_set:
            line = json.dumps(r, ensure_ascii=False)
            f.write(line + "\n")
            train_bytes += len(line)

    eval_bytes = 0
    with open(eval_path, "w") as f:
        for r in eval_set:
            line = json.dumps(r, ensure_ascii=False)
            f.write(line + "\n")
            eval_bytes += len(line)

    # ── Assembly metadata ────────────────────────────────────────
    meta = {
        "model": "SwarmJudge-27B",
        "base": "Qwen3.5-27B-A3B",
        "assembled_at": datetime.now().isoformat(),
        "target": args.target,
        "total_curated": len(selected),
        "train_count": len(train_set),
        "eval_count": len(eval_set),
        "pass_count": final_pass,
        "fail_count": final_fail,
        "pass_pct": round(final_pass / len(selected) * 100, 1),
        "sources": dict(source_counts),
        "curation_thresholds": {
            "trajectory_pass_floor": TRAJ_PASS_FLOOR,
            "trajectory_fail_ceiling": TRAJ_FAIL_CEILING,
            "agent_pass_floor": AGENT_PASS_FLOOR,
            "agent_fail_ceiling": AGENT_FAIL_CEILING,
        },
        "source_stats": {
            "trajectory": {
                "raw": traj_stats["raw"],
                "curated": len(traj_dedup),
                "acceptance_rate": round(len(traj_dedup) / max(traj_stats["raw"], 1) * 100, 1),
            },
            "agent_traces": {
                "raw": agent_stats["raw"],
                "curated": len(agent_dedup),
                "acceptance_rate": round(len(agent_dedup) / max(agent_stats["raw"], 1) * 100, 1),
            },
            "judge_eval": {
                "raw": judge_stats["raw"],
                "curated": len(judge_dedup),
                "acceptance_rate": round(len(judge_dedup) / max(judge_stats["raw"], 1) * 100, 1),
            },
        },
    }
    meta_path = OUTPUT_DIR / "assembly_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"OUTPUT:")
    print(f"  Train: {train_path}")
    print(f"         {len(train_set):,} records ({train_bytes/1e6:.1f} MB)")
    print(f"  Eval:  {eval_path}")
    print(f"         {len(eval_set):,} records ({eval_bytes/1e6:.1f} MB)")
    print(f"  Meta:  {meta_path}")

    # ── Stage to swarmrails ──────────────────────────────────────
    if args.stage:
        stage_to_swarmrails(OUTPUT_DIR)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
