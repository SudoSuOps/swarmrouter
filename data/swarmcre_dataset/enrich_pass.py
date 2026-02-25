#!/usr/bin/env python3
"""Standalone enrichment pass — enrich existing SwarmCRE training pairs via Together.ai.

Reads swarmcre_train.jsonl line by line, enriches eligible records
(ic_memo, lease_reasoning) via Together.ai 70B rewrite + 235B CoVe verification,
writes enriched output to swarmcre_train_enriched.jsonl.

Non-enrichable records pass through unchanged.

Usage:
    TOGETHER_API_KEY=tgp_v1_... python3 -m data.swarmcre_dataset.enrich_pass
    # or
    python3 data/swarmcre_dataset/enrich_pass.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Handle import paths
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.swarmcre_dataset.enrichment import Enricher

# ── Config ────────────────────────────────────────────

INPUT_FILE = _this_dir / "output" / "swarmcre_train.jsonl"
OUTPUT_FILE = _this_dir / "output" / "swarmcre_train_enriched.jsonl"
PROGRESS_FILE = _this_dir / "output" / ".enrich_progress.json"
BATCH_LOG_INTERVAL = 100  # Log progress every N records

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("enrich_pass")


def load_progress():
    """Load progress from checkpoint file."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"processed": 0, "enriched": 0, "skipped": 0, "errors": 0}


def save_progress(stats):
    """Save progress checkpoint."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(stats, f)


def main():
    api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key:
        # Try OPENAI_API_KEY (Together.ai compat)
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.error("No API key found. Set TOGETHER_API_KEY or OPENAI_API_KEY")
        sys.exit(1)

    if not INPUT_FILE.exists():
        log.error("Input file not found: %s", INPUT_FILE)
        sys.exit(1)

    # Load progress for resume support
    progress = load_progress()
    skip_to = progress["processed"]

    enricher = Enricher(enabled=True, api_key=api_key)
    log.info("Enrichment pass starting")
    log.info("  Input:  %s", INPUT_FILE)
    log.info("  Output: %s", OUTPUT_FILE)
    log.info("  Resume from record: %d", skip_to)

    stats = {
        "processed": 0,
        "enriched": 0,
        "skipped": 0,
        "errors": 0,
        "cove_passed": 0,
        "cove_failed": 0,
    }

    t0 = time.time()

    # Open output in append mode if resuming, write mode if fresh
    mode = "a" if skip_to > 0 else "w"
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, mode, encoding="utf-8") as fout:

        for line_num, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            # Skip already-processed records on resume
            if line_num < skip_to:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line + "\n")
                stats["errors"] += 1
                stats["processed"] += 1
                continue

            # Enrich
            enriched_record = enricher.enrich(record)

            # Track enrichment
            if enriched_record.get("enriched"):
                stats["enriched"] += 1
                stats["cove_passed"] += 1
            elif "enriched" in enriched_record and not enriched_record["enriched"]:
                stats["cove_failed"] += 1
            else:
                stats["skipped"] += 1

            fout.write(json.dumps(enriched_record, ensure_ascii=False) + "\n")
            stats["processed"] += 1

            # Progress logging
            if stats["processed"] % BATCH_LOG_INTERVAL == 0:
                elapsed = time.time() - t0
                rate = stats["processed"] / elapsed if elapsed > 0 else 0
                log.info(
                    "Progress: %d processed | %d enriched | %d CoVe failed | "
                    "%d skipped | %d errors | %.1f rec/s",
                    stats["processed"], stats["enriched"], stats["cove_failed"],
                    stats["skipped"], stats["errors"], rate,
                )
                # Save checkpoint
                save_progress({
                    "processed": line_num + 1,
                    "enriched": stats["enriched"],
                    "skipped": stats["skipped"],
                    "errors": stats["errors"],
                })

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("ENRICHMENT COMPLETE")
    log.info("=" * 60)
    log.info("  Total processed: %d", stats["processed"])
    log.info("  Enriched (CoVe pass): %d", stats["enriched"])
    log.info("  CoVe rejected: %d", stats["cove_failed"])
    log.info("  Skipped (non-enrichable): %d", stats["skipped"])
    log.info("  Errors: %d", stats["errors"])
    log.info("  Time: %.1f minutes", elapsed / 60)
    log.info("  Output: %s", OUTPUT_FILE)

    # Enricher internal stats
    log.info("  Enricher stats: %s", enricher.stats)

    # Clean up progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    # Final stats to stdout
    print(json.dumps({
        "status": "complete",
        "processed": stats["processed"],
        "enriched": stats["enriched"],
        "cove_failed": stats["cove_failed"],
        "skipped": stats["skipped"],
        "errors": stats["errors"],
        "elapsed_minutes": round(elapsed / 60, 1),
        "output": str(OUTPUT_FILE),
    }, indent=2))


if __name__ == "__main__":
    main()
