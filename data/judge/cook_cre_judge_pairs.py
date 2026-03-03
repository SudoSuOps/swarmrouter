#!/usr/bin/env python3
"""
CRE Judge Pairs Cook
====================

Takes CRE Q&A pairs from the inventory and generates SwarmJudge 5-criterion
evaluation pairs using Qwen3-235B-A22B (stable, 22B active params).

Output teaches SwarmJudge CRE-level discipline:
  - Every number verified
  - Every calculation checked
  - Hollow data caught
  - Inflated scores rejected

Sources:
  - data/swarmcre_dataset/output/swarmcre_train.jsonl     (881K standard CRE)
  - data/swarmcre_dataset/output/swarmcre_v2_100k.jsonl   (100K new era CRE)

Task type distribution (optimized for rubric discipline):
  underwriting_calc   40% — math-heavy, every number checkable
  ic_memo             20% — complex multi-step reasoning
  lease_reasoning     15% — legal precision + clause verification
  market_comp         10% — market knowledge + plausibility
  data_center          5% — new era (blockchain, power, GPU pricing)
  risk_triage          5% — analytical rigor
  t12_normalization    5% — accounting precision

Usage:
  # Dry run (20 pairs)
  TOGETHER_API_KEY=... python3 cook_cre_judge_pairs.py --dry-run

  # Full cook (20K pairs)
  TOGETHER_API_KEY=... python3 cook_cre_judge_pairs.py --target 20000 --workers 50

  # New era only (data_center / DC-heavy)
  TOGETHER_API_KEY=... python3 cook_cre_judge_pairs.py --new-era-only --target 8000 --workers 50

  # Status
  python3 cook_cre_judge_pairs.py --status
"""

import json
import os
import sys
import time
import random
import hashlib
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cre-judge-cook")

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_KEY", "")
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
JUDGE_MODEL  = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"  # default; override with --model

# Together.ai pricing map (input $/1M, output $/1M)
MODEL_PRICING = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput":          (0.90, 3.50),
    "moonshotai/Kimi-K2.5":                              (1.00, 3.00),
    "MiniMaxAI/MiniMax-M2.5":                            (0.80, 2.40),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.85),
}
PRICE_INPUT_PER_1M  = 0.90
PRICE_OUTPUT_PER_1M = 3.50

TIMEOUT     = 120.0
MAX_RETRIES = 4
RETRY_DELAY = 5.0

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "swarmcre_dataset" / "output"

DEFAULT_OUTPUT_DIR = Path("/tmp/cre_judge_cook")

# ── Blockchain / technology detection ──────────────────────────────────
BLOCKCHAIN_SIGNALS = ["blockchain", "erc-1400", "tokeniz", "hts token", "digital asset",
                      "smart contract", "defi", "web3", "reit token"]
TECHNOLOGY_SIGNALS = ["gpu", "nvidia", "h100", "vera rubin", "b200", "pue", "power density",
                      "cooling", "rack", "colocation", "hyperscaler", "inference compute"]

def is_blockchain(content: str) -> bool:
    c = content.lower()
    return any(s in c for s in BLOCKCHAIN_SIGNALS)

def is_technology(content: str) -> bool:
    c = content.lower()
    return any(s in c for s in TECHNOLOGY_SIGNALS)

# ── Task / domain sampling weights (sum to 1.0) ────────────────────────
# Math + systems discipline. Blockchain and tech are first-class buckets.
TASK_WEIGHTS = {
    "underwriting_calc":     0.30,  # math anchor — every number verifiable
    "blockchain_tech":       0.25,  # new era: blockchain + GPU/systems CRE
    "ic_memo":               0.18,  # complex multi-step reasoning
    "t12_normalization":     0.10,  # accounting precision
    "lease_reasoning":       0.10,  # legal + financial precision
    "risk_triage":           0.07,  # analytical rigor
}

# ═══════════════════════════════════════════════════════════════════════
# SwarmJudge system prompt (5-criterion, domain-agnostic)
# ═══════════════════════════════════════════════════════════════════════

SWARMJUDGE_SYSTEM = """You are SwarmJudge, the quality assessment engine for AI-generated training data.
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
- "As an AI language model" / "I cannot provide medical advice" -> sft_quality 1
- "It's important to note" / "It's worth mentioning" -> reduce sft_quality by 1
- "Comprehensive" + "Robust" + "Leveraging" in same response -> likely slop
- Repeated paragraphs -> structure 1, degenerate
- <think>...</think> tags visible -> structure 1, thinking leak

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
# Pair sampling
# ═══════════════════════════════════════════════════════════════════════

def load_and_sample(target: int, new_era_only: bool, data_dir: Path, seed: int = 42) -> list[dict]:
    """
    Sample CRE Q&A pairs from inventory according to task weights.

    Buckets:
      underwriting_calc  — math-heavy standard CRE
      blockchain_tech    — DC file (100% bc+tech) + v2 blockchain signal pairs
      ic_memo            — complex IC memos
      t12_normalization  — accounting precision
      lease_reasoning    — legal/financial precision
      risk_triage        — analytical rigor
    """
    random.seed(seed)
    log.info("Loading CRE inventory...")

    standard_cre  = data_dir / "swarmcre_train.jsonl"
    new_era_cre   = data_dir / "swarmcre_v2_100k.jsonl"
    dc_blockchain = data_dir / "swarmcre_dc_7k.jsonl"

    # Load DC/blockchain file (3,150 pairs, 100% blockchain+tech)
    dc_blockchain_pairs: list[dict] = []
    if dc_blockchain.exists():
        with open(dc_blockchain) as f:
            for line in f:
                line = line.strip()
                if line:
                    dc_blockchain_pairs.append(json.loads(line))
        log.info(f"DC/blockchain file:  {len(dc_blockchain_pairs):,}")
    else:
        log.warning(f"DC file not found: {dc_blockchain}")

    # Load new era v2 (100K)
    new_era_pairs: list[dict] = []
    if new_era_cre.exists():
        with open(new_era_cre) as f:
            for line in f:
                line = line.strip()
                if line:
                    new_era_pairs.append(json.loads(line))
        log.info(f"New era CRE (v2):    {len(new_era_pairs):,}")
    else:
        log.warning(f"New era file not found: {new_era_cre}")

    # Load standard CRE (881K) — skip if new_era_only
    standard_pairs: list[dict] = []
    if not new_era_only and standard_cre.exists():
        log.info("Loading standard CRE — streaming...")
        with open(standard_cre) as f:
            for line in f:
                line = line.strip()
                if line:
                    standard_pairs.append(json.loads(line))
        log.info(f"Standard CRE:        {len(standard_pairs):,}")

    # ── Build buckets ──────────────────────────────────────────────────
    buckets: dict[str, list] = {k: [] for k in TASK_WEIGHTS}

    # blockchain_tech bucket: DC file first, then v2 blockchain-signal pairs
    buckets["blockchain_tech"].extend(dc_blockchain_pairs)
    for p in new_era_pairs:
        content = json.dumps(p)
        if is_blockchain(content) or is_technology(content):
            buckets["blockchain_tech"].append(p)

    log.info(f"blockchain_tech pool: {len(buckets['blockchain_tech']):,}")

    # Standard task type buckets from new_era + standard
    all_standard = new_era_pairs + standard_pairs
    for p in all_standard:
        tt = p.get("task_type", "") or ""
        if tt == "underwriting_calc":
            buckets["underwriting_calc"].append(p)
        elif tt == "ic_memo":
            buckets["ic_memo"].append(p)
        elif tt == "t12_normalization":
            buckets["t12_normalization"].append(p)
        elif tt == "lease_reasoning":
            buckets["lease_reasoning"].append(p)
        elif tt == "risk_triage":
            buckets["risk_triage"].append(p)

    # ── Sample ────────────────────────────────────────────────────────
    sampled: list[dict] = []
    for bucket, weight in TASK_WEIGHTS.items():
        n    = int(target * weight)
        pool = buckets[bucket]
        if not pool:
            log.warning(f"Empty bucket: {bucket}")
            continue
        random.shuffle(pool)
        take = pool[:n]
        sampled.extend(take)
        log.info(f"  {bucket:<22} {len(take):>6,}  (pool: {len(pool):,})")

    random.shuffle(sampled)
    log.info(f"Total sampled: {len(sampled):,} / {target:,} target")
    return sampled[:target]


def build_eval_request(pair: dict) -> tuple[str, str]:
    """
    Extract question and answer from a CRE pair and build the judge eval request.
    Returns (user_message, metadata_str).
    """
    messages = pair.get("messages", [])
    user_msg  = next((m["content"] for m in messages if m["role"] == "user"),  "")
    asst_msg  = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    meta      = pair.get("metadata", {})
    task_type = pair.get("task_type", meta.get("task_type", "unknown"))
    asset_type = meta.get("asset_type", "cre")

    # Truncate very long answers to avoid token overflow
    if len(asst_msg) > 6000:
        asst_msg = asst_msg[:6000] + "\n[...truncated...]"

    user_turn = (
        f"Evaluate this AI-generated Q&A pair for training data quality.\n\n"
        f"DOMAIN: cre\n"
        f"SPECIALTY: {asset_type}\n"
        f"TASK: {task_type}\n\n"
        f"QUESTION:\n{user_msg}\n\n"
        f"ANSWER:\n{asst_msg}"
    )
    return user_turn, f"{asset_type}/{task_type}"


def fingerprint(pair: dict) -> str:
    raw = pair.get("id") or json.dumps(pair.get("messages", [])[:2], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════
# API call
# ═══════════════════════════════════════════════════════════════════════

async def cook_one(
    client: httpx.AsyncClient,
    pair: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Generate a judge evaluation for one CRE pair."""
    fp = fingerprint(pair)
    user_turn, specialty = build_eval_request(pair)

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": SWARMJUDGE_SYSTEM},
            {"role": "user",   "content": user_turn},
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {TOGETHER_KEY}",
        "Content-Type":  "application/json",
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                resp = await client.post(TOGETHER_URL, json=payload, headers=headers, timeout=TIMEOUT)
                resp.raise_for_status()
                data     = resp.json()
                raw_out  = data["choices"][0]["message"]["content"]
                usage    = data.get("usage", {})
                in_toks  = usage.get("prompt_tokens", 0)
                out_toks = usage.get("completion_tokens", 0)

                # Parse judge output
                try:
                    judge_eval = json.loads(raw_out)
                except json.JSONDecodeError:
                    # Try to extract JSON block
                    import re
                    m = re.search(r'\{.*\}', raw_out, re.DOTALL)
                    if m:
                        try:
                            judge_eval = json.loads(m.group())
                        except Exception:
                            judge_eval = None
                    else:
                        judge_eval = None

                if judge_eval is None:
                    return {"fp": fp, "ok": False, "error": f"parse_error: {raw_out[:100]}", "in_toks": in_toks, "out_toks": out_toks, "pair": pair}

                # Validate: must be 5-criterion schema
                scores = judge_eval.get("scores", {})
                required = {"accuracy", "completeness", "structure", "relevance", "sft_quality"}
                wrong    = {"grounding", "coherence", "efficiency", "format", "safety"}
                if not required.issubset(scores.keys()) or any(w in scores for w in wrong):
                    return {"fp": fp, "ok": False, "error": "wrong_schema", "in_toks": in_toks, "out_toks": out_toks, "pair": pair}

                # Validate verdict logic
                total   = judge_eval.get("total", sum(scores.values()))
                verdict = judge_eval.get("verdict", "").upper()
                min_s   = min(scores.values()) if scores else 0
                acc     = scores.get("accuracy", 0)

                expected_pass = (total >= 20 and min_s >= 3 and acc >= 4)
                if verdict == "PASS" and not expected_pass:
                    judge_eval["verdict"] = "FAIL"
                    judge_eval.setdefault("issues", []).append("Corrected: score threshold not met for PASS")
                    verdict = "FAIL"
                elif verdict == "FAIL" and expected_pass and total >= 22:
                    pass  # Keep FAIL — judge may have caught something scores don't reflect

                # Build training pair in SwarmJudge format
                meta = pair.get("metadata", {})
                training_pair = {
                    "messages": [
                        {"role": "system",    "content": SWARMJUDGE_SYSTEM},
                        {"role": "user",      "content": user_turn},
                        {"role": "assistant", "content": json.dumps(judge_eval, ensure_ascii=False)},
                    ],
                    "metadata": {
                        "domain":           "cre",
                        "specialty":        meta.get("asset_type", "cre"),
                        "task_type":        pair.get("task_type", ""),
                        "verdict":          verdict,
                        "total":            total,
                        "source":           "trajectory_cook_cre_judge_v2",
                        "model_judge":      JUDGE_MODEL,
                        "fingerprint":      fp,
                        "cooked_at":        datetime.now(timezone.utc).isoformat(),
                        "new_era":          "data_center" in meta.get("asset_type", ""),
                        "blockchain":       is_blockchain(json.dumps(pair)),
                        "technology":       is_technology(json.dumps(pair)),
                    },
                }

                return {"fp": fp, "ok": True, "verdict": verdict, "total": total, "in_toks": in_toks, "out_toks": out_toks, "pair": training_pair}

            except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.NetworkError) as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    import random as _r
                    # 503 = server overloaded — longer backoff
                    is_503 = "503" in str(e)
                    base   = RETRY_DELAY * (3 ** attempt) if is_503 else RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(base + _r.uniform(0, 5))

    return {"fp": fp, "ok": False, "error": f"api_error: {last_error}", "in_toks": 0, "out_toks": 0, "pair": pair}


# ═══════════════════════════════════════════════════════════════════════
# Main cook loop
# ═══════════════════════════════════════════════════════════════════════

async def run_cook(args):
    if not TOGETHER_KEY:
        log.error("Set TOGETHER_API_KEY or TOGETHER_KEY")
        sys.exit(1)

    global JUDGE_MODEL, PRICE_INPUT_PER_1M, PRICE_OUTPUT_PER_1M
    JUDGE_MODEL = args.model
    PRICE_INPUT_PER_1M, PRICE_OUTPUT_PER_1M = MODEL_PRICING.get(
        JUDGE_MODEL, (1.00, 4.00)  # safe fallback if unknown model
    )
    log.info(f"Model: {JUDGE_MODEL}  pricing: ${PRICE_INPUT_PER_1M}/1M in, ${PRICE_OUTPUT_PER_1M}/1M out")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path   = output_dir / "cre_judge_train.jsonl"
    ckpt_path  = output_dir / "checkpoint.txt"
    stats_path = output_dir / "cook_stats.json"

    # Load checkpoint (resume)
    done_fps: set[str] = set()
    if ckpt_path.exists():
        done_fps = {l.strip() for l in open(ckpt_path) if l.strip()}
        log.info(f"Resume: {len(done_fps):,} already cooked")

    # Sample source pairs
    data_dir = Path(args.data_dir)
    sampled = load_and_sample(args.target, args.new_era_only, data_dir, seed=args.seed)
    pending = [p for p in sampled if fingerprint(p) not in done_fps]
    log.info(f"Pending: {len(pending):,}")

    if args.dry_run:
        pending = pending[:20]
        log.info("DRY RUN — 20 pairs")

    if not pending:
        log.info("Nothing to do.")
        return

    total       = len(pending)
    cooked      = 0
    pass_count  = 0
    fail_count  = 0
    errors      = 0
    in_toks     = 0
    out_toks    = 0
    start       = time.time()

    semaphore = asyncio.Semaphore(args.workers)

    log.info(f"Cook starting — model: {JUDGE_MODEL} — workers: {args.workers}")

    async with httpx.AsyncClient() as client:
        tasks = [cook_one(client, pair, semaphore) for pair in pending]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            cooked += 1
            in_toks  += result.get("in_toks", 0)
            out_toks += result.get("out_toks", 0)

            if result["ok"]:
                with open(out_path, "a") as f:
                    f.write(json.dumps(result["pair"], ensure_ascii=False) + "\n")
                with open(ckpt_path, "a") as f:
                    f.write(result["fp"] + "\n")
                if result.get("verdict") == "PASS":
                    pass_count += 1
                else:
                    fail_count += 1
            else:
                errors += 1
                log.debug(f"Error [{result['fp']}]: {result.get('error','?')}")

            if cooked % 200 == 0 or cooked == total:
                elapsed  = time.time() - start
                rate     = cooked / elapsed if elapsed > 0 else 0
                eta_s    = (total - cooked) / rate if rate > 0 else 0
                cost_in  = in_toks  / 1_000_000 * PRICE_INPUT_PER_1M
                cost_out = out_toks / 1_000_000 * PRICE_OUTPUT_PER_1M
                log.info(
                    f"[{cooked:>6}/{total}] pass={pass_count:,} fail={fail_count:,} err={errors} "
                    f"rate={rate:.1f}/s eta={eta_s/60:.0f}m cost=${cost_in+cost_out:.2f}"
                )

    elapsed  = time.time() - start
    cost_in  = in_toks  / 1_000_000 * PRICE_INPUT_PER_1M
    cost_out = out_toks / 1_000_000 * PRICE_OUTPUT_PER_1M

    stats = {
        "completed_utc":    datetime.now(timezone.utc).isoformat(),
        "model":            JUDGE_MODEL,
        "target":           args.target,
        "cooked":           cooked,
        "pass":             pass_count,
        "fail":             fail_count,
        "errors":           errors,
        "pass_rate_pct":    round(pass_count / max(cooked - errors, 1) * 100, 1),
        "elapsed_s":        round(elapsed, 1),
        "input_tokens":     in_toks,
        "output_tokens":    out_toks,
        "cost_input_usd":   round(cost_in, 2),
        "cost_output_usd":  round(cost_out, 2),
        "cost_total_usd":   round(cost_in + cost_out, 2),
        "output_file":      str(out_path),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("═" * 58)
    print("CRE Judge Cook Complete")
    print("═" * 58)
    print(f"  Cooked:    {cooked:,}  (target: {args.target:,})")
    print(f"  PASS:      {pass_count:,}  ({stats['pass_rate_pct']}%)")
    print(f"  FAIL:      {fail_count:,}")
    print(f"  Errors:    {errors:,}")
    print(f"  Elapsed:   {elapsed/3600:.1f}h")
    print(f"  Cost:      ${cost_in+cost_out:.2f}")
    print(f"  Output:    {out_path}")
    print("═" * 58)


# ═══════════════════════════════════════════════════════════════════════
# Status
# ═══════════════════════════════════════════════════════════════════════

def show_status(args):
    output_dir = Path(args.output_dir)
    stats_path = output_dir / "cook_stats.json"
    out_path   = output_dir / "cre_judge_train.jsonl"
    ckpt_path  = output_dir / "checkpoint.txt"

    if stats_path.exists():
        with open(stats_path) as f:
            s = json.load(f)
        print(f"Completed:  {s['completed_utc']}")
        print(f"Cooked:     {s['cooked']:,}  ({s['pass_rate_pct']}% PASS)")
        print(f"Cost:       ${s['cost_total_usd']:.2f}")
        print(f"Output:     {s['output_file']}")
    elif ckpt_path.exists():
        done = sum(1 for _ in open(ckpt_path))
        cooked = sum(1 for _ in open(out_path)) if out_path.exists() else 0
        print(f"In progress: {done:,} processed, {cooked:,} cooked to output")
    else:
        print("Not started.")


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CRE Judge Pairs Cook — SwarmJudge training data")
    parser.add_argument("--target",       type=int,  default=23000, help="Number of pairs to cook (default: 23000)")
    parser.add_argument("--workers",      type=int,  default=50,    help="Concurrent workers (default: 50)")
    parser.add_argument("--model",        default=JUDGE_MODEL,      help="Together.ai model ID")
    parser.add_argument("--seed",         type=int,  default=42,    help="Random seed for pair sampling (use different seeds per model run)")
    parser.add_argument("--output-dir",   default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-dir",     default=str(_DEFAULT_DATA_DIR), help="Directory containing CRE inventory jsonl files")
    parser.add_argument("--new-era-only", action="store_true", help="Sample only from swarmcre_v2_100k.jsonl")
    parser.add_argument("--dry-run",      action="store_true", help="Cook 20 pairs only")
    parser.add_argument("--status",       action="store_true", help="Show progress and exit")
    args = parser.parse_args()

    if args.status:
        show_status(args)
        return

    asyncio.run(run_cook(args))


if __name__ == "__main__":
    main()
