#!/usr/bin/env python3
"""
Gate-6 Final Approval Validator — Qwen/Qwen3.5-397B-A17B
=========================================================

Runs 103,931 SwarmJudge training pairs through the final approval gate.

The 397B meta-validates each pair: given the original Q&A + the judge's
evaluation, it checks whether the judge evaluation is CORRECT. Only pairs
that receive a 397B PASS with confidence >= 0.75 enter the sealed set.

No pair enters the cook without a 397B seal.

Usage:
  # Dry run — 10 pairs
  TOGETHER_API_KEY=... python3 validate_judge_pairs_397b.py --dry-run

  # Full validation run
  TOGETHER_API_KEY=... python3 validate_judge_pairs_397b.py \\
      --train /data2/swarmjudge_27b_data/swarmjudge_27b_train.jsonl \\
      --eval  /data2/swarmjudge_27b_data/swarmjudge_27b_eval.jsonl \\
      --output-dir /data2/swarmjudge_27b_data/gate6/ \\
      --workers 30

  # Status check
  python3 validate_judge_pairs_397b.py --status --output-dir /data2/swarmjudge_27b_data/gate6/
"""

import json
import os
import sys
import time
import hashlib
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("gate6")

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

TOGETHER_KEY  = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_KEY", "")
TOGETHER_URL  = "https://api.together.xyz/v1/chat/completions"
MODEL_GATE6   = "Qwen/Qwen3.5-397B-A17B"

# Gate-6 acceptance threshold
MIN_CONFIDENCE = 0.75

# Together.ai pricing (USD per 1M tokens)
PRICE_INPUT_PER_1M  = 0.60
PRICE_OUTPUT_PER_1M = 3.60

TIMEOUT     = 180.0  # 397B can be slow on long sequences
MAX_RETRIES = 5
RETRY_DELAY = 10.0   # 397B is in high demand — longer base delay

# ═══════════════════════════════════════════════════════════════════════
# Gate-6 System Prompt
# ═══════════════════════════════════════════════════════════════════════

GATE6_SYSTEM = """You are the Gate-6 Final Approval Validator for AI training data quality assurance.

Your role: verify that a judge model's evaluation of an AI-generated Q&A pair is CORRECT.

You will receive:
1. An original Q&A pair (domain, specialty, question, answer)
2. The judge's evaluation of that pair (reasoning, scores, verdict, issues, fixes)

Your task: determine whether the judge's evaluation is accurate and well-reasoned.

Verification checklist:
- Are the judge's scores (1-5 each) justified by specific evidence from the answer?
- Is the reasoning in each step (inspect, verify, assess, rule) sound and specific — not generic?
- If the judge ruled PASS: would you independently agree this answer is high-quality training data?
- If the judge ruled FAIL: are the cited issues real, specific, and actionable?
- Did the judge MISS any significant issues (hallucinated numbers, outdated guidelines, hollow data, slop markers)?
- Is the total score consistent with the individual scores?
- Is the PASS/FAIL verdict consistent with the scoring thresholds (PASS = total>=20, all>=3, accuracy>=4)?

Score calibration check (critical):
- A perfect 25/25 must be genuinely perfect — no gaps, no hedging, no fabricated numbers
- Accuracy score of 5 means every fact, number, and calculation is verifiably correct
- SFT_QUALITY of 5 means zero slop markers, zero filler, pure clean training signal
- Inflate scores are a training failure — penalize if scores are not earned

Output strict JSON only, no markdown, no preamble:
{
  "verdict": "PASS" or "FAIL",
  "confidence": <0.0 to 1.0>,
  "rationale": "<one clear paragraph: what the judge got right/wrong>",
  "remediation": ["<specific fix>", ...],
  "judge_score_accuracy": "<correct|inflated|deflated|wrong_verdict>"
}

PASS = judge evaluation is correct, pair is quality training data for a judge model.
FAIL = judge made an error: wrong scores, missed issues, false PASS on slop, or flawed reasoning."""


def build_validation_prompt(pair: dict) -> str:
    """Build the user-turn prompt for Gate-6 validation."""
    messages = pair.get("messages", [])
    meta     = pair.get("metadata", {})

    # The user message already contains domain, specialty, question, answer
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

    # The assistant message is the judge's evaluation
    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

    # Try to parse and re-format judge output for clarity
    try:
        judge_eval = json.loads(assistant_msg)
        judge_block = json.dumps(judge_eval, indent=2)
    except Exception:
        judge_block = assistant_msg

    source = meta.get("source", "unknown")
    domain = meta.get("domain", "unknown")

    return (
        f"SOURCE: {source}\n"
        f"DOMAIN: {domain}\n\n"
        f"═══ ORIGINAL Q&A PAIR ═══\n"
        f"{user_msg}\n\n"
        f"═══ JUDGE EVALUATION TO VERIFY ═══\n"
        f"{judge_block}\n\n"
        f"Verify: is this judge evaluation correct? "
        f"Output JSON verdict only. /no_think"
    )


def fingerprint(pair: dict) -> str:
    """Stable fingerprint for a pair — used for resume."""
    fp = pair.get("metadata", {}).get("fingerprint")
    if fp:
        return fp
    raw = json.dumps(pair, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_checkpoint(checkpoint_path: Path) -> set:
    """Load set of already-processed fingerprints."""
    if not checkpoint_path.exists():
        return set()
    done = set()
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(line)
    return done


# ═══════════════════════════════════════════════════════════════════════
# API call
# ═══════════════════════════════════════════════════════════════════════

async def call_gate6(
    client: httpx.AsyncClient,
    pair: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Call Gate-6 validator for one pair. Returns enriched result dict."""
    fp    = fingerprint(pair)
    meta  = pair.get("metadata", {})
    prompt_text = build_validation_prompt(pair)

    payload = {
        "model": MODEL_GATE6,
        "messages": [
            {"role": "system", "content": GATE6_SYSTEM},
            {"role": "user",   "content": prompt_text},
        ],
        "max_tokens": 4096,
        "temperature": 0.1,   # near-deterministic for a gate decision
        # Note: Qwen3.5-397B-A17B is a thinking model — reasoning goes to message.reasoning,
        # actual JSON output goes to message.content. Needs 4096 tokens to clear the think phase.
    }

    headers = {
        "Authorization": f"Bearer {TOGETHER_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                resp = await client.post(
                    TOGETHER_URL,
                    json=payload,
                    headers=headers,
                    timeout=TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()

                raw_output  = data["choices"][0]["message"]["content"]
                usage       = data.get("usage", {})
                input_toks  = usage.get("prompt_tokens", 0)
                output_toks = usage.get("completion_tokens", 0)

                try:
                    gate_result = json.loads(raw_output)
                except json.JSONDecodeError:
                    # Try to extract partial JSON — model may have been cut off
                    import re
                    gate_result = None
                    # Try to recover verdict + confidence from partial output
                    v_match = re.search(r'"verdict"\s*:\s*"(PASS|FAIL)"', raw_output)
                    c_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw_output)
                    r_match = re.search(r'"rationale"\s*:\s*"([^"]{20,})', raw_output)
                    if v_match and c_match:
                        gate_result = {
                            "verdict": v_match.group(1),
                            "confidence": float(c_match.group(1)),
                            "rationale": r_match.group(1)[:300] if r_match else "(truncated)",
                            "remediation": ["(response truncated — partial recovery)"],
                            "judge_score_accuracy": "truncated",
                        }
                    if gate_result is None:
                        gate_result = {
                            "verdict": "FAIL",
                            "confidence": 0.0,
                            "rationale": f"Gate-6 parse error: {raw_output[:200]}",
                            "remediation": ["Gate-6 response parse error — retry"],
                            "judge_score_accuracy": "parse_error",
                        }

                verdict    = gate_result.get("verdict", "FAIL").upper()
                confidence = float(gate_result.get("confidence", 0.0))
                sealed     = (verdict == "PASS" and confidence >= MIN_CONFIDENCE)

                return {
                    "fingerprint":   fp,
                    "sealed":        sealed,
                    "gate6_verdict": verdict,
                    "gate6_confidence": confidence,
                    "gate6_rationale":  gate_result.get("rationale", ""),
                    "gate6_remediation": gate_result.get("remediation", []),
                    "gate6_judge_score_accuracy": gate_result.get("judge_score_accuracy", ""),
                    "input_tokens":  input_toks,
                    "output_tokens": output_toks,
                    "pair":          pair,
                    "error":         None,
                }

            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    import random
                    wait = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 5)
                    await asyncio.sleep(wait)

    return {
        "fingerprint":   fp,
        "sealed":        False,
        "gate6_verdict": "FAIL",
        "gate6_confidence": 0.0,
        "gate6_rationale":  f"API error after {MAX_RETRIES} retries: {last_error}",
        "gate6_remediation": ["API error — needs retry"],
        "gate6_judge_score_accuracy": "api_error",
        "input_tokens":  0,
        "output_tokens": 0,
        "pair":          pair,
        "error":         last_error,
    }


# ═══════════════════════════════════════════════════════════════════════
# Writer — flush sealed pairs to disk in batches
# ═══════════════════════════════════════════════════════════════════════

class Writer:
    def __init__(self, sealed_path: Path, report_path: Path, checkpoint_path: Path, batch: int = 50):
        self.sealed_path     = sealed_path
        self.report_path     = report_path
        self.checkpoint_path = checkpoint_path
        self.batch_size      = batch
        self._sealed_buf     = []
        self._report_buf     = []
        self._ckpt_buf       = []

    def add(self, result: dict):
        fp   = result["fingerprint"]
        pair = result["pair"]

        # Always write to report
        report_entry = {
            "fingerprint":        fp,
            "sealed":             result["sealed"],
            "gate6_verdict":      result["gate6_verdict"],
            "gate6_confidence":   result["gate6_confidence"],
            "gate6_rationale":    result["gate6_rationale"],
            "gate6_remediation":  result["gate6_remediation"],
            "gate6_judge_score_accuracy": result["gate6_judge_score_accuracy"],
            "input_tokens":       result["input_tokens"],
            "output_tokens":      result["output_tokens"],
            "domain":             pair.get("metadata", {}).get("domain", ""),
            "specialty":          pair.get("metadata", {}).get("specialty", ""),
            "source":             pair.get("metadata", {}).get("source", ""),
            "original_verdict":   pair.get("metadata", {}).get("verdict", ""),
            "error":              result.get("error"),
        }
        self._report_buf.append(report_entry)

        # Only sealed pairs enter the training set
        if result["sealed"]:
            # Enrich pair metadata with gate6 stamp
            sealed_pair = dict(pair)
            sealed_pair["metadata"] = {
                **pair.get("metadata", {}),
                "gate6_verdict":     result["gate6_verdict"],
                "gate6_confidence":  result["gate6_confidence"],
                "gate6_model":       MODEL_GATE6,
                "gate6_sealed_at":   datetime.now(timezone.utc).isoformat(),
            }
            self._sealed_buf.append(sealed_pair)

        self._ckpt_buf.append(fp)

        if len(self._report_buf) >= self.batch_size:
            self.flush()

    def flush(self):
        if self._sealed_buf:
            with open(self.sealed_path, "a") as f:
                for p in self._sealed_buf:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
        if self._report_buf:
            with open(self.report_path, "a") as f:
                for r in self._report_buf:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if self._ckpt_buf:
            with open(self.checkpoint_path, "a") as f:
                for fp in self._ckpt_buf:
                    f.write(fp + "\n")
        self._sealed_buf.clear()
        self._report_buf.clear()
        self._ckpt_buf.clear()


# ═══════════════════════════════════════════════════════════════════════
# Main validation loop
# ═══════════════════════════════════════════════════════════════════════

async def run_validation(args):
    if not TOGETHER_KEY:
        log.error("Set TOGETHER_API_KEY or TOGETHER_KEY")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve output paths (separate sealed sets for train vs eval)
    is_eval         = args.eval_mode
    label           = "eval" if is_eval else "train"
    sealed_path     = output_dir / f"judge_{label}_397b_sealed.jsonl"
    report_path     = output_dir / f"validation_report_397b_{label}.jsonl"
    checkpoint_path = output_dir / f"checkpoint_397b_{label}.txt"

    input_path = Path(args.eval if is_eval else args.train)
    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        sys.exit(1)

    # Load all pairs
    log.info(f"Loading pairs from {input_path} ...")
    all_pairs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_pairs.append(json.loads(line))

    log.info(f"Loaded {len(all_pairs):,} pairs")

    # Resume: skip already-processed
    done = load_checkpoint(checkpoint_path)
    if done:
        log.info(f"Resume: {len(done):,} already processed, skipping")

    pending = [p for p in all_pairs if fingerprint(p) not in done]
    log.info(f"Pending: {len(pending):,} pairs to validate")

    if args.dry_run:
        pending = pending[:10]
        log.info("DRY RUN — 10 pairs only")

    if not pending:
        log.info("Nothing to do.")
        return

    # Stats
    total        = len(pending)
    processed    = 0
    sealed_count = 0
    failed_count = 0
    api_errors   = 0
    total_input  = 0
    total_output = 0
    start_time   = time.time()

    writer    = Writer(sealed_path, report_path, checkpoint_path)
    semaphore = asyncio.Semaphore(args.workers)

    log.info(f"Gate-6 validation starting — model: {MODEL_GATE6} — workers: {args.workers}")
    log.info(f"Sealed → {sealed_path}")
    log.info(f"Report → {report_path}")

    async with httpx.AsyncClient() as client:
        tasks = [
            call_gate6(client, pair, semaphore)
            for pair in pending
        ]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            writer.add(result)

            processed    += 1
            total_input  += result["input_tokens"]
            total_output += result["output_tokens"]

            if result["sealed"]:
                sealed_count += 1
            elif result.get("error"):
                api_errors += 1
            else:
                failed_count += 1

            # Progress log every 100 pairs
            if processed % 100 == 0 or processed == total:
                elapsed     = time.time() - start_time
                rate        = processed / elapsed if elapsed > 0 else 0
                eta_s       = (total - processed) / rate if rate > 0 else 0
                accept_rate = (sealed_count / processed * 100) if processed > 0 else 0
                cost_input  = total_input  / 1_000_000 * PRICE_INPUT_PER_1M
                cost_output = total_output / 1_000_000 * PRICE_OUTPUT_PER_1M
                cost_total  = cost_input + cost_output

                log.info(
                    f"[{processed:>6}/{total}] "
                    f"sealed={sealed_count:,} ({accept_rate:.1f}%) "
                    f"failed={failed_count:,} errors={api_errors} "
                    f"rate={rate:.1f}/s eta={eta_s/60:.0f}m "
                    f"cost=${cost_total:.2f}"
                )

                # Hard gate: if acceptance rate < 60% at 1K pairs, warn loudly
                if processed == 1000 and accept_rate < 60.0:
                    log.warning(
                        f"⚠ AT-002 GATE WARNING: acceptance rate {accept_rate:.1f}% < 60% at 1K pairs. "
                        f"Review source data before proceeding to full cook."
                    )

    writer.flush()

    # Final report
    elapsed      = time.time() - start_time
    accept_rate  = (sealed_count / processed * 100) if processed > 0 else 0
    cost_input   = total_input  / 1_000_000 * PRICE_INPUT_PER_1M
    cost_output  = total_output / 1_000_000 * PRICE_OUTPUT_PER_1M
    cost_total   = cost_input + cost_output

    summary = {
        "run_completed_utc":  datetime.now(timezone.utc).isoformat(),
        "model":              MODEL_GATE6,
        "input_file":         str(input_path),
        "label":              label,
        "total_processed":    processed,
        "sealed":             sealed_count,
        "failed":             failed_count,
        "api_errors":         api_errors,
        "acceptance_rate_pct": round(accept_rate, 2),
        "min_confidence":     MIN_CONFIDENCE,
        "elapsed_s":          round(elapsed, 1),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "cost_input_usd":     round(cost_input, 2),
        "cost_output_usd":    round(cost_output, 2),
        "cost_total_usd":     round(cost_total, 2),
        "at_002_pass":        accept_rate >= 60.0,
        "at_003_pass":        sealed_count >= 50000 if not is_eval else True,
    }

    summary_path = output_dir / f"gate6_summary_{label}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("═" * 60)
    print(f"Gate-6 Validation Complete — {label.upper()}")
    print("═" * 60)
    print(f"  Processed:       {processed:,}")
    print(f"  Sealed (PASS):   {sealed_count:,}  ({accept_rate:.1f}%)")
    print(f"  Rejected (FAIL): {failed_count:,}")
    print(f"  API errors:      {api_errors:,}")
    print(f"  Elapsed:         {elapsed/3600:.1f}h")
    print(f"  Cost:            ${cost_total:.2f}  (in=${cost_input:.2f} out=${cost_output:.2f})")
    print()
    print(f"  AT-002 acceptance ≥ 60%:     {'✓ PASS' if summary['at_002_pass'] else '✗ FAIL — review data'}")
    if not is_eval:
        print(f"  AT-003 sealed   ≥ 50K pairs: {'✓ PASS' if summary['at_003_pass'] else '✗ FAIL — insufficient data'}")
    print()
    print(f"  Sealed output:  {sealed_path}")
    print(f"  Report:         {report_path}")
    print(f"  Summary:        {summary_path}")
    print("═" * 60)

    if not summary["at_002_pass"]:
        log.error("AT-002 FAILED — acceptance rate below 60%. DO NOT proceed to cook.")
        sys.exit(2)
    if not is_eval and not summary["at_003_pass"]:
        log.error("AT-003 FAILED — sealed set below 50K pairs. DO NOT proceed to cook.")
        sys.exit(2)


# ═══════════════════════════════════════════════════════════════════════
# Status command
# ═══════════════════════════════════════════════════════════════════════

def show_status(args):
    output_dir = Path(args.output_dir)
    for label in ("train", "eval"):
        summary_path = output_dir / f"gate6_summary_{label}.json"
        ckpt_path    = output_dir / f"checkpoint_397b_{label}.txt"
        sealed_path  = output_dir / f"judge_{label}_397b_sealed.jsonl"

        print(f"\n── {label.upper()} ──")
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            print(f"  Completed:    {s['run_completed_utc']}")
            print(f"  Processed:    {s['total_processed']:,}")
            print(f"  Sealed:       {s['sealed']:,}  ({s['acceptance_rate_pct']}%)")
            print(f"  Cost:         ${s['cost_total_usd']:.2f}")
            print(f"  AT-002:       {'PASS' if s['at_002_pass'] else 'FAIL'}")
            if label == "train":
                print(f"  AT-003:       {'PASS' if s['at_003_pass'] else 'FAIL'}")
        elif ckpt_path.exists():
            done = sum(1 for _ in open(ckpt_path))
            sealed = 0
            if sealed_path.exists():
                sealed = sum(1 for _ in open(sealed_path))
            print(f"  In progress:  {done:,} processed, {sealed:,} sealed")
        else:
            print("  Not started")


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Gate-6 Final Approval Validator — Qwen3.5-397B-A17B"
    )
    parser.add_argument(
        "--train",
        default="/data2/swarmjudge_27b_data/swarmjudge_27b_train.jsonl",
        help="Path to judge training JSONL",
    )
    parser.add_argument(
        "--eval",
        default="/data2/swarmjudge_27b_data/swarmjudge_27b_eval.jsonl",
        help="Path to judge eval JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="/data2/swarmjudge_27b_data/gate6",
        help="Output directory for sealed sets + reports",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Concurrent API workers (default: 12 — 397B is in high demand)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_CONFIDENCE,
        help=f"Minimum gate6 confidence to seal a pair (default: {MIN_CONFIDENCE})",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Validate eval set instead of train set",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test 10 pairs only, no output written",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current progress and exit",
    )

    args = parser.parse_args()

    if args.status:
        show_status(args)
        return

    asyncio.run(run_validation(args))


if __name__ == "__main__":
    main()
