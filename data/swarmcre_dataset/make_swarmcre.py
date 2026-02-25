#!/usr/bin/env python3
"""SwarmCRE Dataset Factory — Build 1M platinum CRE training pairs.

Usage:
    python -m data.swarmcre_dataset.make_swarmcre                       # full build
    python -m data.swarmcre_dataset.make_swarmcre --deals 10000         # smaller build
    python -m data.swarmcre_dataset.make_swarmcre --eval-only           # eval sets only
    python -m data.swarmcre_dataset.make_swarmcre --status              # show progress
    python -m data.swarmcre_dataset.make_swarmcre --sample 5            # show 5 samples
    python -m data.swarmcre_dataset.make_swarmcre --validate            # validate output
    python -m data.swarmcre_dataset.make_swarmcre --enrich              # with Together.ai
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

# Handle both module-relative and direct execution
try:
    from .builder import BuildConfig, DatasetBuilder
    from .eval_builder import EvalBuilder
    from .quality_checks import QualityPipeline
except ImportError:
    # Direct execution: adjust sys.path
    _this_dir = Path(__file__).resolve().parent
    _project_root = _this_dir.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from data.swarmcre_dataset.builder import BuildConfig, DatasetBuilder
    from data.swarmcre_dataset.eval_builder import EvalBuilder
    from data.swarmcre_dataset.quality_checks import QualityPipeline


# ═══════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════


def _setup_logging(verbose: bool = False):
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)


# ═══════════════════════════════════════════════════════════════════
# CLI ACTIONS
# ═══════════════════════════════════════════════════════════════════


def _action_status(config: BuildConfig):
    """Show build progress per shard and exit."""
    builder = DatasetBuilder(config)
    builder.show_status()


def _action_eval_only(config: BuildConfig):
    """Build 3 eval sets and exit."""
    print("=" * 60)
    print("SwarmCRE Eval Set Builder")
    print("=" * 60)

    eval_builder = EvalBuilder(config)
    results = eval_builder.build_all()

    print("\nEval Build Results:")
    print("-" * 60)
    for name, stats in results.items():
        if name == "summary":
            continue
        print(
            f"  {stats.get('eval_set', name):25s}: "
            f"{stats.get('written', 0):>6,} written, "
            f"{stats.get('failed', 0):>4,} failed, "
            f"{stats.get('elapsed_sec', 0):.1f}s"
        )
        print(f"    -> {stats.get('output', 'N/A')}")

    summary = results.get("summary", {})
    print("-" * 60)
    print(
        f"  {'TOTAL':25s}: "
        f"{summary.get('total_written', 0):>6,} written, "
        f"{summary.get('total_failed', 0):>4,} failed"
    )
    print()


def _action_sample(config: BuildConfig, n: int):
    """Show N random samples from the output dataset."""
    final_path = config.output_dir / "swarmcre_train.jsonl"

    # Try shard files if merged file doesn't exist
    if not final_path.exists():
        shard_files = sorted(config.output_dir.glob("shard_*.jsonl"))
        if not shard_files:
            print(f"No dataset files found in {config.output_dir}")
            sys.exit(1)
        print(f"(merged file not found, sampling from {len(shard_files)} shard files)")
        source_files = shard_files
    else:
        source_files = [final_path]

    # Collect all lines using reservoir sampling (memory-safe)
    reservoir = []
    total_lines = 0
    rng = random.Random(42)

    for fpath in source_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                if len(reservoir) < n:
                    reservoir.append(line)
                else:
                    j = rng.randint(0, total_lines - 1)
                    if j < n:
                        reservoir[j] = line

    if not reservoir:
        print("No records found in dataset files.")
        sys.exit(1)

    print(f"\nShowing {len(reservoir)} random sample(s) from {total_lines:,} records:\n")
    print("=" * 80)

    for i, raw_line in enumerate(reservoir):
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            print(f"\n[Sample {i + 1}] INVALID JSON")
            continue

        print(f"\n--- Sample {i + 1} of {len(reservoir)} ---")
        print(f"  ID:         {record.get('id', 'N/A')}")
        print(f"  Deal ID:    {record.get('deal_id', 'N/A')}")
        print(f"  Task type:  {record.get('task_type', 'N/A')}")
        print(f"  Difficulty:  {record.get('difficulty', 'N/A')}")

        meta = record.get("metadata", {})
        print(f"  Asset type: {meta.get('asset_type', 'N/A')}")
        print(f"  Market:     {meta.get('market_name', 'N/A')}")
        print(f"  SF:         {meta.get('sf', 'N/A'):,}" if isinstance(meta.get("sf"), int) else f"  SF:         {meta.get('sf', 'N/A')}")

        messages = record.get("messages", [])
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate long content for display
            display = content[:500] + ("..." if len(content) > 500 else "")
            print(f"\n  [{role.upper()}]")
            for line in display.split("\n"):
                print(f"    {line}")

        targets = record.get("gold", {}).get("numeric_targets", {})
        if targets:
            print(f"\n  Gold numeric targets:")
            for k, v in list(targets.items())[:10]:
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v:,}" if isinstance(v, int) else f"    {k}: {v}")

        print()

    print("=" * 80)


def _action_validate(config: BuildConfig):
    """Run validation on existing output files."""
    print("=" * 60)
    print("SwarmCRE Output Validation")
    print("=" * 60)

    pipeline = QualityPipeline()
    files_to_check = []

    # Check for merged file
    final_path = config.output_dir / "swarmcre_train.jsonl"
    if final_path.exists():
        files_to_check.append(("train", final_path))

    # Check for eval files
    for eval_name in ["eval_gold_2k", "eval_hard_500", "eval_adversarial_500"]:
        eval_path = config.output_dir / f"{eval_name}.jsonl"
        if eval_path.exists():
            files_to_check.append((eval_name, eval_path))

    # Check shard files if no merged file
    if not final_path.exists():
        for shard_file in sorted(config.output_dir.glob("shard_*.jsonl")):
            files_to_check.append((shard_file.stem, shard_file))

    if not files_to_check:
        print(f"\nNo dataset files found in {config.output_dir}")
        sys.exit(1)

    overall_pass = 0
    overall_fail = 0
    overall_total = 0

    for label, fpath in files_to_check:
        pipeline.reset()
        file_total = 0
        file_pass = 0
        file_fail = 0
        file_size = fpath.stat().st_size

        with open(fpath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                file_total += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    file_fail += 1
                    continue

                passed, failures = pipeline.check(record)
                if passed:
                    file_pass += 1
                else:
                    file_fail += 1

        pct = (file_pass / file_total * 100) if file_total > 0 else 0
        print(
            f"\n  {label:30s}: {file_total:>8,} records, "
            f"{file_pass:>8,} pass ({pct:.1f}%), "
            f"{file_fail:>4,} fail, "
            f"{file_size / 1_048_576:.1f} MB"
        )

        # Show gate-level stats
        gate_stats = pipeline.stats.get("gates", {})
        for gate_name, counts in gate_stats.items():
            if counts["fail"] > 0:
                print(f"    {gate_name}: {counts['fail']} failures")

        overall_total += file_total
        overall_pass += file_pass
        overall_fail += file_fail

    overall_pct = (overall_pass / overall_total * 100) if overall_total > 0 else 0
    print("\n" + "-" * 60)
    print(
        f"  {'OVERALL':30s}: {overall_total:>8,} records, "
        f"{overall_pass:>8,} pass ({overall_pct:.1f}%), "
        f"{overall_fail:>4,} fail"
    )
    print()

    if overall_fail > 0:
        print(f"WARNING: {overall_fail:,} records failed validation.")
    else:
        print("All records passed validation.")
    print()


def _action_full_build(config: BuildConfig):
    """Full dataset build: training data + eval sets."""
    print("=" * 60)
    print("SwarmCRE Dataset Factory")
    print("=" * 60)
    print(f"  Deals:          {config.num_deals:,}")
    print(f"  Tasks/deal:     {config.tasks_per_deal}")
    print(f"  Target records: ~{config.num_deals * config.tasks_per_deal:,}")
    print(f"  Shards:         {config.num_shards}")
    print(f"  Seed:           {config.seed}")
    print(f"  Output:         {config.output_dir}")
    print(f"  Enrichment:     {'enabled' if config.enable_enrichment else 'disabled'}")
    print("=" * 60)

    # Phase 1: Build training data
    print("\nPhase 1: Building training data...")
    t0 = time.time()
    builder = DatasetBuilder(config)
    build_stats = builder.build_all()

    print(f"\nTraining build complete in {build_stats['total_elapsed_sec']:.1f}s")
    print(f"  Written:  {build_stats['total_written']:,}")
    print(f"  Failed:   {build_stats['total_failed']:,}")

    # Per-shard summary
    for shard_stat in build_stats.get("shards", []):
        print(
            f"  Shard {shard_stat['shard']:3d}: "
            f"{shard_stat.get('written', 0):>8,} written, "
            f"{shard_stat.get('failed', 0):>4,} failed, "
            f"{shard_stat.get('elapsed_sec', 0):.1f}s"
        )

    # Phase 2: Merge shards
    print("\nPhase 2: Merging shards...")
    merged_count = builder.merge_shards()
    print(f"  Merged {merged_count:,} records into swarmcre_train.jsonl")

    # Phase 3: Build eval sets
    print("\nPhase 3: Building evaluation sets...")
    eval_builder = EvalBuilder(config)
    eval_results = eval_builder.build_all()

    for name, stats in eval_results.items():
        if name == "summary":
            continue
        print(
            f"  {stats.get('eval_set', name):25s}: "
            f"{stats.get('written', 0):>6,} written, "
            f"{stats.get('failed', 0):>4,} failed"
        )

    # Phase 4: Summary
    total_elapsed = time.time() - t0
    eval_summary = eval_results.get("summary", {})

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"  Training records: {build_stats['total_written']:>10,}")
    print(f"  Eval records:     {eval_summary.get('total_written', 0):>10,}")
    print(f"  Total records:    {build_stats['total_written'] + eval_summary.get('total_written', 0):>10,}")
    print(f"  Total time:       {total_elapsed:>10.1f}s")
    print(f"  Output dir:       {config.output_dir}")
    print()

    # List output files
    print("Output files:")
    for fpath in sorted(config.output_dir.glob("*.jsonl")):
        size_mb = fpath.stat().st_size / 1_048_576
        print(f"  {fpath.name:40s} {size_mb:>8.1f} MB")
    print()


# ═══════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════


def _parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SwarmCRE Dataset Factory -- Build 1M platinum CRE training pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                          # full 100K-deal build (~1M records)\n"
            "  %(prog)s --deals 1000 --shards 2  # quick test build\n"
            "  %(prog)s --eval-only              # build 3 eval sets only\n"
            "  %(prog)s --sample 10              # preview 10 random records\n"
            "  %(prog)s --validate               # run QA on existing output\n"
            "  %(prog)s --status                 # check shard progress\n"
        ),
    )

    parser.add_argument(
        "--deals",
        type=int,
        default=100_000,
        help="Number of deals to generate (default: 100,000)",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=8,
        help="Number of parallel shards (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--tasks-per-deal",
        type=int,
        default=10,
        help="Number of training tasks per deal (default: 10)",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        default=False,
        help="Enable Together.ai enrichment for narrative tasks",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Build eval sets only (skip training data)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="Show build progress per shard and exit",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Run validation on existing output files",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Show N random samples from the output dataset",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )

    return parser.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


def main(argv=None):
    """CLI entry point."""
    args = _parse_args(argv)
    _setup_logging(verbose=args.verbose)

    config = BuildConfig(
        seed=args.seed,
        num_deals=args.deals,
        num_shards=args.shards,
        output_dir=args.output_dir,
        enable_enrichment=args.enrich,
        tasks_per_deal=args.tasks_per_deal,
    )

    # Route to the requested action
    if args.status:
        _action_status(config)
    elif args.eval_only:
        _action_eval_only(config)
    elif args.sample > 0:
        _action_sample(config, args.sample)
    elif args.validate:
        _action_validate(config)
    else:
        _action_full_build(config)


if __name__ == "__main__":
    main()
