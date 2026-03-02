#!/usr/bin/env python3
"""
SwarmRouter-4B-0 — Inspect + Seal (Endcap Build)
==================================================
Final step in the dataset pipeline. Takes the assembled train/eval pair,
runs hard-rule validation, tags every pair with build metadata,
computes SHA256 manifest, uploads to sb-builds R2 bucket,
and registers the build in Supabase model_builds table.

Each build is an isolated, versioned, immutable deliverable.
NOT mixed with other data in the stack.

Usage:
    python3 -m data.router_v3.inspect_seal --train <path> --eval <path>
    python3 -m data.router_v3.inspect_seal --train <path> --eval <path> --dry-run
    python3 -m data.router_v3.inspect_seal --train <path> --eval <path> --version v1
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from data.router_v3.schema import (
    DOMAINS, MODELS, TASK_TYPES, COMPLEXITY, RISK_LEVELS,
    LATENCY_TIERS, COST_SENSITIVITY, TOOLS,
)

OUTPUT_DIR = Path(__file__).parent / "output"
BUILDS_DIR = Path(__file__).parent / "builds"

# Hard routing rules — violations are REJECTED not corrected
HARD_RULES = {
    "pharma":  "swarmpharma-35b",
    "safety":  "swarmresearch-32b",
    "judge":   "swarmjudge-27b",
}

MODEL_NAME  = "swarmrouter-4b0"
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


# ── Validation ──────────────────────────────────────────────────────

def inspect_record(record: dict, idx: int) -> tuple[bool, str]:
    """Return (ok, reason). Strict — hard rules fail the pair."""
    try:
        msgs = record.get("messages", [])
        if len(msgs) != 3:
            return False, "wrong message count"

        query = msgs[1]["content"].strip()
        if len(query) < 15:
            return False, "query too short"

        answer = json.loads(msgs[2]["content"])

        required = [
            "domain", "task_type", "complexity", "risk_level",
            "latency_tier", "cost_sensitivity", "recommended_model",
            "escalation_allowed", "requires_tools", "reasoning",
        ]
        for key in required:
            if key not in answer:
                return False, f"missing key: {key}"

        domain = answer["domain"]
        model  = answer["recommended_model"]

        if domain not in DOMAINS:
            return False, f"invalid domain: {domain}"
        if answer["task_type"] not in TASK_TYPES:
            return False, f"invalid task_type: {answer['task_type']}"
        if answer["complexity"] not in COMPLEXITY:
            return False, f"invalid complexity: {answer['complexity']}"
        if answer["risk_level"] not in RISK_LEVELS:
            return False, f"invalid risk_level: {answer['risk_level']}"
        if model not in MODELS:
            return False, f"invalid model: {model}"
        if not isinstance(answer["requires_tools"], list):
            return False, "requires_tools not list"
        if len(answer.get("reasoning", "")) > 120:
            return False, "reasoning > 120 chars"

        # Hard rule enforcement — zero tolerance
        if domain in HARD_RULES and model != HARD_RULES[domain]:
            return False, f"hard rule violation: {domain} → {model} (must be {HARD_RULES[domain]})"

        return True, "ok"

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return False, f"parse error: {e}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Stats ────────────────────────────────────────────────────────────

def compute_stats(records: list[dict]) -> dict:
    domains   = Counter(r["metadata"]["domain"] for r in records)
    models    = Counter(r["metadata"]["recommended_model"] for r in records)
    streams   = Counter(r["metadata"].get("stream", "?") for r in records)
    complexity= Counter(r["metadata"].get("complexity", "?") for r in records)
    sources   = Counter(r["metadata"].get("source", "?") for r in records)
    return {
        "total":      len(records),
        "domains":    dict(domains),
        "models":     dict(models),
        "streams":    dict(streams),
        "complexity": dict(complexity),
        "sources":    dict(sources),
    }


def print_stats(stats: dict, label: str):
    print(f"\n{'─'*54}")
    print(f"  {label}  ({stats['total']:,} pairs)")
    print(f"{'─'*54}")
    print("  Domains:")
    for d, c in sorted(stats["domains"].items(), key=lambda x: -x[1]):
        pct = c / stats["total"] * 100
        bar = "█" * int(pct / 2.5)
        print(f"    {d:<22} {c:>6,} ({pct:4.1f}%) {bar}")
    print("\n  Models:")
    for m, c in sorted(stats["models"].items(), key=lambda x: -x[1]):
        pct = c / stats["total"] * 100
        print(f"    {m:<24} {c:>6,} ({pct:4.1f}%)")
    print(f"\n  Complexity: {stats['complexity']}")
    print(f"  Streams (top 5): {dict(Counter(stats['streams']).most_common(5))}")


# ── Tag + Seal ───────────────────────────────────────────────────────

def tag_record(record: dict, build_id: str, version: str, sealed_at: str) -> dict:
    """Stamp every pair with build provenance. Immutable after this."""
    r = dict(record)
    r["metadata"] = dict(record.get("metadata", {}))
    r["metadata"]["build_id"]  = build_id
    r["metadata"]["version"]   = version
    r["metadata"]["sealed_at"] = sealed_at
    return r


# ── R2 Upload ────────────────────────────────────────────────────────

def upload_to_r2(local_path: Path, r2_key: str, bucket: str = "sb-builds") -> bool:
    """Upload file to R2 via wrangler CLI."""
    cmd = [
        "npx", "wrangler", "r2", "object", "put",
        f"{bucket}/{r2_key}",
        "--file", str(local_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [warn] R2 upload failed: {result.stderr[:200]}")
        return False
    return True


# ── Supabase Registration ────────────────────────────────────────────

def register_build(build_record: dict) -> bool:
    """POST build record to Supabase model_builds table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("  [warn] SUPABASE_URL/KEY not set — skipping Supabase registration")
        return False

    import urllib.request
    import urllib.error

    data = json.dumps(build_record).encode()
    req = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/model_builds",
        data=data,
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status in (200, 201)
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        print(f"  [warn] Supabase registration failed: {e.code} {body}")
        return False
    except Exception as e:
        print(f"  [warn] Supabase error: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SwarmRouter-4B-0 Inspect + Seal")
    parser.add_argument("--train", required=True, help="Assembled train JSONL path")
    parser.add_argument("--eval",  required=True, help="Assembled eval JSONL path")
    parser.add_argument("--version", default="v1", help="Build version tag (default: v1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate + report only, no R2/Supabase writes")
    args = parser.parse_args()

    train_path = Path(args.train)
    eval_path  = Path(args.eval)

    if not train_path.exists():
        print(f"ERROR: train file not found: {train_path}")
        sys.exit(1)
    if not eval_path.exists():
        print(f"ERROR: eval file not found: {eval_path}")
        sys.exit(1)

    # Build ID: deterministic from version + date
    now_str    = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    build_id   = f"{MODEL_NAME}-{args.version}-{now_str}"
    sealed_at  = datetime.utcnow().isoformat() + "Z"

    print("=" * 54)
    print(f"  SwarmRouter-4B-0 — Inspect + Seal")
    print(f"  Build ID : {build_id}")
    print(f"  Version  : {args.version}")
    print(f"  Mode     : {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 54)

    # ── Load ────────────────────────────────────────────────────────
    def load_jsonl(path: Path) -> list[dict]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    print(f"\nLoading train: {train_path.name}")
    train_raw = load_jsonl(train_path)
    print(f"  Loaded {len(train_raw):,} records")

    print(f"Loading eval : {eval_path.name}")
    eval_raw  = load_jsonl(eval_path)
    print(f"  Loaded {len(eval_raw):,} records")

    # ── Inspect ─────────────────────────────────────────────────────
    print("\nRunning final inspection...")

    hard_rule_violations = Counter()
    rejection_reasons    = Counter()

    def inspect_all(records, label):
        passed, rejected = [], []
        for i, r in enumerate(records):
            ok, reason = inspect_record(r, i)
            if ok:
                passed.append(r)
            else:
                rejected.append((i, reason))
                rejection_reasons[reason.split(":")[0].strip()] += 1
                if "hard rule" in reason:
                    hard_rule_violations[reason] += 1
        print(f"  {label}: {len(records):,} in → {len(passed):,} passed, {len(rejected):,} rejected")
        if rejected[:5]:
            for idx, reason in rejected[:5]:
                print(f"    [#{idx}] {reason}")
        return passed

    train_passed = inspect_all(train_raw, "train")
    eval_passed  = inspect_all(eval_raw, "eval")

    if hard_rule_violations:
        print(f"\n  Hard rule violations: {dict(hard_rule_violations)}")

    # ── Stats ────────────────────────────────────────────────────────
    train_stats = compute_stats(train_passed)
    eval_stats  = compute_stats(eval_passed)
    print_stats(train_stats, "TRAIN (post-inspection)")
    print_stats(eval_stats,  "EVAL  (post-inspection)")

    if args.dry_run:
        print(f"\n{'='*54}")
        print("  DRY RUN complete — no files written")
        print(f"{'='*54}")
        return

    # ── Tag ──────────────────────────────────────────────────────────
    print(f"\nTagging {len(train_passed)+len(eval_passed):,} pairs with build_id={build_id}...")
    train_sealed = [tag_record(r, build_id, args.version, sealed_at) for r in train_passed]
    eval_sealed  = [tag_record(r, build_id, args.version, sealed_at) for r in eval_passed]

    # ── Write sealed files ───────────────────────────────────────────
    BUILDS_DIR.mkdir(parents=True, exist_ok=True)
    build_dir = BUILDS_DIR / build_id
    build_dir.mkdir(parents=True, exist_ok=True)

    train_out = build_dir / f"{build_id}_train.jsonl"
    eval_out  = build_dir / f"{build_id}_eval.jsonl"

    print(f"Writing sealed train → {train_out.name}")
    with open(train_out, "w") as f:
        for r in train_sealed:
            f.write(json.dumps(r) + "\n")

    print(f"Writing sealed eval  → {eval_out.name}")
    with open(eval_out, "w") as f:
        for r in eval_sealed:
            f.write(json.dumps(r) + "\n")

    # ── SHA256 ───────────────────────────────────────────────────────
    print("Computing SHA256...")
    sha_train = sha256_file(train_out)
    sha_eval  = sha256_file(eval_out)
    print(f"  train: {sha_train}")
    print(f"  eval : {sha_eval}")

    # ── Manifest ─────────────────────────────────────────────────────
    manifest = {
        "build_id":      build_id,
        "model_name":    MODEL_NAME,
        "version":       args.version,
        "sealed_at":     sealed_at,
        "train": {
            "file":     train_out.name,
            "pairs":    len(train_sealed),
            "sha256":   sha_train,
            "r2_key":   f"{MODEL_NAME}/{build_id}/train.jsonl",
        },
        "eval": {
            "file":     eval_out.name,
            "pairs":    len(eval_sealed),
            "sha256":   sha_eval,
            "r2_key":   f"{MODEL_NAME}/{build_id}/eval.jsonl",
        },
        "domain_dist":  train_stats["domains"],
        "model_dist":   train_stats["models"],
        "stream_dist":  train_stats["streams"],
        "complexity":   train_stats["complexity"],
        "hard_rule_violations": dict(hard_rule_violations),
        "rejection_reasons":    dict(rejection_reasons),
    }

    manifest_path = build_dir / f"{build_id}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # SHA256SUM file
    sha_path = build_dir / "SHA256SUMS.txt"
    with open(sha_path, "w") as f:
        f.write(f"{sha_train}  {train_out.name}\n")
        f.write(f"{sha_eval}  {eval_out.name}\n")
        f.write(f"{sha256_file(manifest_path)}  {manifest_path.name}\n")

    print(f"\nManifest  → {manifest_path.name}")
    print(f"SHA256SUMs → {sha_path.name}")

    # ── R2 Upload ────────────────────────────────────────────────────
    print(f"\nUploading to R2 bucket 'sb-builds'...")
    r2_prefix = f"{MODEL_NAME}/{build_id}"

    files_to_upload = [
        (train_out,    f"{r2_prefix}/train.jsonl"),
        (eval_out,     f"{r2_prefix}/eval.jsonl"),
        (manifest_path,f"{r2_prefix}/manifest.json"),
        (sha_path,     f"{r2_prefix}/SHA256SUMS.txt"),
    ]

    for local, key in files_to_upload:
        ok = upload_to_r2(local, key)
        status = "✓" if ok else "✗"
        print(f"  {status} {key}")

    # ── Supabase ─────────────────────────────────────────────────────
    build_record = {
        "build_id":     build_id,
        "model_name":   MODEL_NAME,
        "version":      args.version,
        "sealed_at":    sealed_at,
        "train_pairs":  len(train_sealed),
        "eval_pairs":   len(eval_sealed),
        "sha256_train": sha_train,
        "sha256_eval":  sha_eval,
        "manifest":     manifest,
        "domain_dist":  train_stats["domains"],
        "model_dist":   train_stats["models"],
        "status":       "sealed",
        "r2_bucket":    "sb-builds",
        "r2_prefix":    r2_prefix,
    }

    print(f"\nRegistering in Supabase model_builds...")
    ok = register_build(build_record)
    print(f"  {'✓ registered' if ok else '✗ not registered (check SUPABASE_URL/KEY)'}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*54}")
    print(f"  BUILD SEALED: {build_id}")
    print(f"  Train pairs : {len(train_sealed):,}")
    print(f"  Eval pairs  : {len(eval_sealed):,}")
    print(f"  R2          : sb-builds/{r2_prefix}/")
    print(f"  Local       : {build_dir}/")
    print(f"{'='*54}")
    print(f"\nNext: python3 scripts/train_v3.py --config configs/train_whale_v3.yaml \\")
    print(f"      --train {train_out} --eval {eval_out}")


if __name__ == "__main__":
    main()
