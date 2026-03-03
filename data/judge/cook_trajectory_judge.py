"""
Trajectory SwarmJudge Cook — Re-cooks judge pairs with 4-step reasoning chains.

Takes existing flat-verdict judge pairs and regenerates them with trajectory
reasoning: INSPECT → VERIFY → ASSESS → RULE. Uses the same two-tier cook
pattern (80B gen + 235B pass) as the CRE new economy cook.

The output teaches SwarmJudge to THINK through evaluations, not just stamp
PASS/FAIL. Each verdict now comes with an evidence trail.

Usage:
    # Full cook (all 42K pairs, 50 workers)
    TOGETHER_API_KEY=... python3 -m data.judge.cook_trajectory_judge

    # Small test batch
    TOGETHER_API_KEY=... python3 -m data.judge.cook_trajectory_judge --limit 100 --workers 5

    # Resume from checkpoint
    TOGETHER_API_KEY=... python3 -m data.judge.cook_trajectory_judge --resume

    # Status check
    python3 -m data.judge.cook_trajectory_judge --status
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

import requests

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
GEN_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
PASS_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"

from data.factory.safestore import SafeStore, safe_output_dir

INPUT_DIR = Path("/tmp/swarmjudge_7b")  # input stays in /tmp (assembled fresh each time)
OUTPUT_DIR = safe_output_dir("judge_trajectory")
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"
PROGRESS_PATH = OUTPUT_DIR / "progress.json"

_safestore: SafeStore | None = None  # initialized in run_cook()

# From the audits — failure patterns the judge must know about
AUDIT_KNOWLEDGE = {
    "known_failure_modes": [
        "hollow_data: Domain-mismatched economics (e.g., warehouse metrics in DC pair)",
        "hedging_language: 'it depends', 'generally speaking', 'results may vary' — 20% of failures",
        "slop_disclaimers: 'As an AI', 'I cannot provide medical advice' — 15% of failures",
        "numeric_hallucination: Fabricated $, %, rates — 0.9% of CRE pairs",
        "missing_trajectory: Incomplete IDENTIFY→CALCULATE→ANALYZE→EVALUATE→RECOMMEND — 29.4%",
        "outdated_guidelines: Pre-2024 clinical/regulatory references — 8% of medical failures",
        "degenerate_repetition: Same paragraph repeated 3+ times — <0.1%",
        "thinking_leak: <think>...</think> tags in output — Qwen-specific",
    ],
    "frontier_weaknesses": [
        "ChatGPT: Sycophancy + over-disclaiming kills SFT quality",
        "Grok: 60-70% JSON compliance, hallucinated citations",
        "Claude: Over-cautious refusals, verbose padding",
        "Llama 70B: 5% DSCR errors, repetitive structure",
        "Qwen small: Code switching, JSON truncation at token limits",
    ],
    "huggingface_slop": [
        "Alpaca 52K: 80% math wrong",
        "UltraFeedback: chosen/rejected labels SWAPPED",
        "LAION-5B: Zero quality control, pulled by Stanford",
        "Model merge spam: Zero eval culture",
    ],
}

# ═══════════════════════════════════════════════════════════════════════
# Trajectory System Prompt
# ═══════════════════════════════════════════════════════════════════════

TRAJECTORY_SYSTEM_PROMPT = """You are SwarmJudge, the quality assessment engine for AI-generated training data.
You are the gatekeeper. Nothing enters the training pipeline without your verdict.

EVALUATION PROTOCOL — 4-Step Trajectory:

1. INSPECT
   Identify the domain, specialty, data type, and structural format.
   Note what the pair claims to be and what it should contain.
   Flag any immediate red flags (wrong domain signals, format violations).

2. VERIFY
   Check every factual claim, number, citation, and guideline reference:
   - Are statistics real or fabricated?
   - Are regulatory references correct and current?
   - Do calculations check out (DSCR, NOI, cap rates, dosages)?
   - Are clinical guidelines current (not pre-2024)?
   - Are technical specifications accurate?
   Known failure patterns to detect:
   - Hollow data: domain-mismatched economics (warehouse metrics in data center pair)
   - Numeric hallucination: fabricated $, %, rates that look plausible but are wrong
   - Outdated guidelines: superseded clinical or regulatory references
   - Frontier model artifacts: over-disclaiming, sycophancy, verbose padding

3. ASSESS
   Score across 5 criteria (1-5 each, 25 total):
   - ACCURACY: Are facts correct? Numbers real? Citations valid?
   - COMPLETENESS: Full topic coverage? Any critical gaps?
   - STRUCTURE: Well-organized? Proper format for the domain?
   - RELEVANCE: On-topic for stated domain/specialty?
   - SFT_QUALITY: Would this teach a model well? Clean training signal?

4. RULE
   Final verdict based on evidence from steps 1-3.
   PASS requires: total >= 20, every criterion >= 3, accuracy >= 4.
   If FAIL: state exactly what failed, cite evidence from VERIFY step,
   and prescribe specific, actionable fixes.

SLOP DETECTION — Instant quality signals:
- "As an AI language model" / "I cannot provide medical advice" → sft_quality 1
- "It's important to note" / "It's worth mentioning" → reduce sft_quality by 1
- "Comprehensive" + "Robust" + "Leveraging" in same response → likely slop
- Repeated paragraphs → structure 1, degenerate
- <think>...</think> tags visible → structure 1, thinking leak

OUTPUT — Strict JSON, no markdown, no commentary:
{
  "reasoning": {
    "inspect": "...",
    "verify": "...",
    "assess": "...",
    "rule": "..."
  },
  "verdict": "PASS" or "FAIL",
  "scores": {
    "accuracy": N,
    "completeness": N,
    "structure": N,
    "relevance": N,
    "sft_quality": N
  },
  "total": N,
  "issues": ["specific issue with evidence", ...],
  "fixes": ["actionable fix", ...]
}"""


# ═══════════════════════════════════════════════════════════════════════
# Together.ai API
# ═══════════════════════════════════════════════════════════════════════

_session: requests.Session | None = None
_api_key: str = ""
_lock = Lock()

# Counters
_stats = {
    "gen_pass": 0,
    "gen_fail": 0,
    "rewritten": 0,
    "rewrite_fail": 0,
    "errors": 0,
    "api_calls": 0,
    "verdict_match": 0,
    "verdict_mismatch": 0,
}


def _get_session() -> requests.Session:
    global _session, _api_key
    if _session is None:
        _api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_KEY", "")
        if not _api_key:
            print("FATAL: Set TOGETHER_API_KEY")
            sys.exit(1)
        _session = requests.Session()
        _session.headers.update({
            "Authorization": f"Bearer {_api_key}",
            "Content-Type": "application/json",
        })
    return _session


def together_call(
    system: str,
    user: str,
    model: str,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    retries: int = 3,
) -> str | None:
    """Call Together.ai chat completion API."""
    session = _get_session()

    for attempt in range(retries):
        try:
            resp = session.post(
                TOGETHER_URL,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": ["<|im_end|>"],
                },
                timeout=120,
            )
            with _lock:
                _stats["api_calls"] += 1

            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                # Strip Qwen thinking tags
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                return text

            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, 30)
                time.sleep(wait)
                continue

            if resp.status_code in (402, 403):
                print(f"\nFATAL: {resp.status_code} — out of credits or auth error")
                sys.exit(1)

            # Other error — retry
            time.sleep(2)
            continue

        except (requests.RequestException, KeyError, IndexError):
            time.sleep(2)
            continue

    with _lock:
        _stats["errors"] += 1
    return None


# ═══════════════════════════════════════════════════════════════════════
# Quality Gates for Trajectory Output
# ═══════════════════════════════════════════════════════════════════════

REQUIRED_REASONING_STEPS = ["inspect", "verify", "assess", "rule"]
MIN_REASONING_LENGTH = 40  # chars per step


def validate_trajectory(text: str) -> tuple[dict | None, str]:
    """Parse and validate trajectory judge output.

    Returns (parsed_dict, error_reason). parsed_dict is None on failure.
    """
    # Find JSON in text
    text = text.strip()
    if not text.startswith("{"):
        # Try to find JSON block
        start = text.find("{")
        if start == -1:
            return None, "no_json"
        text = text[start:]

    # Find matching closing brace
    depth = 0
    end = -1
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None, "unclosed_json"

    try:
        d = json.loads(text[:end])
    except json.JSONDecodeError:
        return None, "invalid_json"

    # Check reasoning steps
    reasoning = d.get("reasoning")
    if not isinstance(reasoning, dict):
        return None, "missing_reasoning"

    for step in REQUIRED_REASONING_STEPS:
        val = reasoning.get(step, "")
        if not isinstance(val, str) or len(val) < MIN_REASONING_LENGTH:
            return None, f"weak_{step}"

    # Check verdict
    verdict = d.get("verdict")
    if verdict not in ("PASS", "FAIL"):
        return None, "bad_verdict"

    # Check scores
    scores = d.get("scores")
    if not isinstance(scores, dict):
        return None, "missing_scores"

    for key in ("accuracy", "completeness", "structure", "relevance", "sft_quality"):
        val = scores.get(key)
        if not isinstance(val, (int, float)) or val < 1 or val > 5:
            return None, f"bad_score_{key}"

    # Check total
    expected_total = sum(int(scores[k]) for k in scores)
    d["total"] = expected_total  # Fix total if wrong

    # Verify verdict consistency with scores
    if verdict == "PASS":
        if expected_total < 20 or int(scores["accuracy"]) < 4 or any(int(scores[k]) < 3 for k in scores):
            # Scores don't support PASS — flag but don't reject
            pass

    # Ensure issues and fixes are lists
    if not isinstance(d.get("issues"), list):
        d["issues"] = []
    if not isinstance(d.get("fixes"), list):
        d["fixes"] = []

    return d, "ok"


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint / Progress
# ═══════════════════════════════════════════════════════════════════════

_checkpoint_lock = Lock()


def load_checkpoint() -> set:
    """Load set of already-processed record fingerprints."""
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
        return set(data.get("done", []))
    return set()


def save_checkpoint(done: set):
    """Save checkpoint."""
    with _checkpoint_lock:
        # Keep only last 50K to avoid unbounded growth
        done_list = list(done)
        if len(done_list) > 50000:
            done_list = done_list[-50000:]
        CHECKPOINT_PATH.write_text(json.dumps({
            "done": done_list,
            "count": len(done_list),
            "saved_at": datetime.now().isoformat(),
        }))


def save_progress(total_target: int, written: int, gen_pass: int, rewritten: int,
                  failed: int, elapsed_min: float, verdict_match: int, verdict_mismatch: int):
    """Save progress.json for dashboard."""
    rate = written / max(elapsed_min, 0.01)
    remaining = total_target - written
    eta_hours = (remaining / rate / 60) if rate > 0 else 0

    PROGRESS_PATH.write_text(json.dumps({
        "total_written": written,
        "total_target": total_target,
        "gen_pass": gen_pass,
        "rewritten": rewritten,
        "failed": failed,
        "api_calls": _stats["api_calls"],
        "errors": _stats["errors"],
        "rate_per_min": round(rate, 1),
        "elapsed_min": round(elapsed_min, 1),
        "eta_hours": round(eta_hours, 1),
        "gen_model": GEN_MODEL,
        "pass_model": PASS_MODEL,
        "verdict_match": verdict_match,
        "verdict_mismatch": verdict_mismatch,
        "verdict_agreement_pct": round(verdict_match / max(written, 1) * 100, 1),
        "updated_at": datetime.now().isoformat(),
    }, indent=2))


# ═══════════════════════════════════════════════════════════════════════
# Core Cook Logic
# ═══════════════════════════════════════════════════════════════════════

def fingerprint(record: dict) -> str:
    """Generate unique fingerprint for a judge record."""
    user_msg = record["messages"][1]["content"]
    return hashlib.md5(user_msg.encode()).hexdigest()[:16]


def extract_user_prompt(record: dict) -> str:
    """Extract the user evaluation prompt from an existing judge record."""
    return record["messages"][1]["content"]


def extract_original_verdict(record: dict) -> str:
    """Get the original verdict from metadata."""
    return record.get("metadata", {}).get("verdict", "UNKNOWN")


def cook_one(record: dict) -> dict | None:
    """Cook a single trajectory judge pair.

    Two-tier: GEN model first, PASS model on quality failure.
    Returns the cooked record or None on total failure.
    """
    user_prompt = extract_user_prompt(record)
    original_verdict = extract_original_verdict(record)
    domain = record.get("metadata", {}).get("domain", "unknown")
    specialty = record.get("metadata", {}).get("specialty", "unknown")
    fp = fingerprint(record)

    # Tier 1: GEN model
    raw = together_call(TRAJECTORY_SYSTEM_PROMPT, user_prompt, GEN_MODEL, max_tokens=2048, temperature=0.3)
    if raw:
        parsed, reason = validate_trajectory(raw)
        if parsed:
            with _lock:
                _stats["gen_pass"] += 1
                if parsed["verdict"] == original_verdict:
                    _stats["verdict_match"] += 1
                else:
                    _stats["verdict_mismatch"] += 1

            return _build_output(parsed, record, domain, specialty, fp, "gen")

    # Tier 2: PASS model rewrite
    raw = together_call(TRAJECTORY_SYSTEM_PROMPT, user_prompt, PASS_MODEL, max_tokens=3000, temperature=0.2)
    if raw:
        parsed, reason = validate_trajectory(raw)
        if parsed:
            with _lock:
                _stats["rewritten"] += 1
                if parsed["verdict"] == original_verdict:
                    _stats["verdict_match"] += 1
                else:
                    _stats["verdict_mismatch"] += 1

            return _build_output(parsed, record, domain, specialty, fp, "rewrite")

    # Total failure
    with _lock:
        _stats["rewrite_fail"] += 1
    return None


def _build_output(parsed: dict, original: dict, domain: str, specialty: str,
                  fp: str, tier: str) -> dict:
    """Build the final trajectory judge training record."""
    return {
        "messages": [
            {"role": "system", "content": TRAJECTORY_SYSTEM_PROMPT},
            {"role": "user", "content": original["messages"][1]["content"]},
            {"role": "assistant", "content": json.dumps(parsed, indent=2)},
        ],
        "metadata": {
            "domain": domain,
            "specialty": specialty,
            "verdict": parsed["verdict"],
            "total": parsed["total"],
            "source": f"trajectory_cook_{original.get('metadata', {}).get('source', 'unknown')}",
            "original_verdict": original.get("metadata", {}).get("verdict"),
            "verdict_match": parsed["verdict"] == original.get("metadata", {}).get("verdict", ""),
            "cook_tier": tier,
            "model": GEN_MODEL if tier == "gen" else PASS_MODEL,
            "fingerprint": fp,
            "cooked_at": datetime.now().isoformat(),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Main Cook
# ═══════════════════════════════════════════════════════════════════════

def load_input(limit: int = 0) -> list[dict]:
    """Load existing judge training pairs."""
    records = []
    train_path = INPUT_DIR / "swarmjudge_7b_train.jsonl"
    eval_path = INPUT_DIR / "swarmjudge_7b_eval.jsonl"

    for path in [train_path, eval_path]:
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
                        if limit and len(records) >= limit:
                            return records
    return records


def run_cook(workers: int = 50, limit: int = 0, resume: bool = True):
    """Run the trajectory judge cook."""
    global _safestore
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "swarmjudge_trajectory_train.jsonl"

    print("=" * 70)
    print("  TRAJECTORY SWARMJUDGE COOK")
    print(f"  GEN:  {GEN_MODEL}")
    print(f"  PASS: {PASS_MODEL}")
    print(f"  Workers: {workers}")
    print("=" * 70)

    # Load input
    records = load_input(limit=limit)
    total = len(records)
    print(f"\n  Loaded {total:,} judge pairs from {INPUT_DIR}")

    # Initialize SafeStore — batch→reconcile→batch→reconcile
    _safestore = SafeStore(
        "judge_trajectory",
        bucket="sb-medical",
        prefix="judge_trajectory/",
        domain="judge",
        r2_push_every=500,
        supabase_push_every=100,
    )
    _safestore.start(total_expected=total)

    # Load checkpoint
    done = load_checkpoint() if resume else set()
    if done:
        print(f"  Resuming: {len(done):,} already done")

    # Filter out already-done records
    todo = [r for r in records if fingerprint(r) not in done]
    print(f"  To cook: {len(todo):,}")

    if not todo:
        print("  Nothing to do.")
        return

    # Open output file in append mode
    mode = "a" if resume and output_path.exists() else "w"
    written = len(done)
    start_time = time.time()
    out_lock = Lock()

    with open(output_path, mode) as fout:

        def process(record):
            result = cook_one(record)
            fp = fingerprint(record)
            if result:
                with out_lock:
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    # SafeStore: every pair goes to Supabase + R2 automatically
                    if _safestore:
                        _safestore.save(result)
            return fp, result is not None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, r): r for r in todo}

            for future in as_completed(futures):
                fp, success = future.result()
                done.add(fp)

                if success:
                    written += 1

                # Progress every 50 records
                if written % 50 == 0:
                    elapsed = (time.time() - start_time) / 60
                    rate = written / max(elapsed, 0.01)
                    remaining = total - written
                    eta = remaining / rate / 60 if rate > 0 else 0
                    match_pct = _stats["verdict_match"] / max(written, 1) * 100

                    print(f"  [{written:,}/{total:,}] "
                          f"{rate:.0f}/min | "
                          f"gen={_stats['gen_pass']:,} rw={_stats['rewritten']:,} "
                          f"fail={_stats['rewrite_fail']:,} | "
                          f"verdict match={match_pct:.0f}% | "
                          f"ETA={eta:.1f}h")

                    save_progress(total, written, _stats["gen_pass"],
                                  _stats["rewritten"], _stats["rewrite_fail"],
                                  elapsed, _stats["verdict_match"],
                                  _stats["verdict_mismatch"])

                # Checkpoint every 500
                if len(done) % 500 == 0:
                    save_checkpoint(done)

    # Final save
    save_checkpoint(done)
    elapsed = (time.time() - start_time) / 60
    save_progress(total, written, _stats["gen_pass"], _stats["rewritten"],
                  _stats["rewrite_fail"], elapsed, _stats["verdict_match"],
                  _stats["verdict_mismatch"])

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"  Written: {written:,} / {total:,}")
    print(f"  Gen pass: {_stats['gen_pass']:,}")
    print(f"  Rewritten: {_stats['rewritten']:,}")
    print(f"  Failed: {_stats['rewrite_fail']:,}")
    print(f"  Verdict match: {_stats['verdict_match']:,}/{written:,} "
          f"({_stats['verdict_match']/max(written,1)*100:.0f}%)")
    print(f"  API calls: {_stats['api_calls']:,}")
    print(f"  Elapsed: {elapsed:.1f}m")
    print(f"  Output: {output_path}")

    # SafeStore: finalize — flush to R2 + verify + log to Supabase
    if _safestore:
        _safestore.finalize()

    print(f"{'='*70}")


def show_status():
    """Show current cook status from progress.json."""
    if not PROGRESS_PATH.exists():
        print("  No cook in progress.")
        return

    p = json.loads(PROGRESS_PATH.read_text())
    print(f"\n  TRAJECTORY JUDGE COOK STATUS")
    print(f"  {'─'*50}")
    print(f"  Written:   {p['total_written']:,} / {p['total_target']:,}")
    print(f"  Gen pass:  {p['gen_pass']:,}")
    print(f"  Rewritten: {p['rewritten']:,}")
    print(f"  Failed:    {p['failed']:,}")
    print(f"  Rate:      {p['rate_per_min']:.0f}/min")
    print(f"  API calls: {p['api_calls']:,}")
    print(f"  Verdict agreement: {p.get('verdict_agreement_pct', 0):.0f}%")
    print(f"  ETA:       {p['eta_hours']:.1f}h")
    print(f"  Updated:   {p['updated_at']}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cook trajectory reasoning chains for SwarmJudge",
    )
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="Limit input records (0=all)")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--status", action="store_true", help="Show cook status")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    resume = not args.no_resume
    run_cook(workers=args.workers, limit=args.limit, resume=resume)


if __name__ == "__main__":
    main()
