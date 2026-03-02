#!/usr/bin/env python3
"""
SwarmRouter-4B-0 — Harvest Vetter
====================================
Runs the 38K harvest router pairs through Together.ai 235B to:
  1. Verify the routing decision is correct
  2. Correct wrong domain/model assignments
  3. Enrich with accurate reasoning strings
  4. Flag + drop low-quality queries

This converts derived routing labels into verified routing labels.
Uses 80B-turbo for speed (10 pairs/batch, 40 concurrent = fast).

Usage:
    OPENAI_API_KEY=... python3 -m data.router_v3.vet_harvest
    OPENAI_API_KEY=... python3 -m data.router_v3.vet_harvest --dry-run
    OPENAI_API_KEY=... python3 -m data.router_v3.vet_harvest --input <path> --model 235b
"""

import asyncio
import json
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

from openai import AsyncOpenAI

from data.router_v3.schema import (
    DOMAINS, TASK_TYPES, COMPLEXITY, RISK_LEVELS,
    LATENCY_TIERS, COST_SENSITIVITY, MODELS, TOOLS,
    SYSTEM_PROMPT, get_recommended_model, needs_proposal,
)

API_KEY = os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = "https://api.together.xyz/v1"

MODEL_80B  = "Qwen/Qwen3-Next-80B-A3B-Instruct"     # fast, cheap — default for vetting
MODEL_235B = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"  # quality — use for final pass

OUTPUT_DIR = Path(__file__).parent / "output"

VET_PROMPT = """You are verifying routing decisions for SwarmRouter-4B-0.

Given a user query, output the CORRECT routing decision as JSON. Be strict.

Model fleet:
- router-4b: simple queries, general, basic technical (low complexity)
- research-8b: mid-level business, financial, legal, technical (medium)
- med-14b: clinical medicine, diagnostics, treatment (medical domain)
- swarmpharma-35b: drug safety, DDI, PK/PD, dosing (pharma domain — ALWAYS)
- swarmcre-35b: CRE underwriting, new economy assets (cre medium/high — ALWAYS)
- swarmjudge-27b: quality eval, agent scoring, policy check (judge domain — ALWAYS)
- swarmresearch-32b: deep research, safety/aviation (safety domain — ALWAYS)

Nine domains: medical | pharma | cre | safety | technical | business | financial | legal | judge

Hard rules:
- pharma → swarmpharma-35b (no exceptions)
- safety/aviation → swarmresearch-32b (no exceptions)
- judge/quality-eval → swarmjudge-27b (no exceptions)
- cre medium/high → swarmcre-35b

For each query below, output JSON with ONLY these keys:
{{
  "domain": "<domain>",
  "task_type": "<task_type>",
  "complexity": "low|medium|high",
  "risk_level": "low|medium|high|critical",
  "latency_tier": "realtime|interactive|batch",
  "cost_sensitivity": "low|medium|high",
  "recommended_model": "<model>",
  "escalation_allowed": true|false,
  "proposal_required": true|false,
  "requires_tools": ["none"],
  "reasoning": "<max 120 chars explaining routing>"
}}

Proposal gate rules:
- pharma or safety domain → proposal_required: true (always)
- medical high risk/complexity → true
- cre/legal medium+ complexity → true
- financial high complexity → true
- otherwise → false

Queries to route (respond with a JSON array, one object per query):
{queries}"""


async def vet_batch(
    client: AsyncOpenAI,
    batch: list[dict],
    model: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Vet one batch of records, returning corrected records."""
    async with semaphore:
        queries = [r["messages"][1]["content"] for r in batch]
        queries_text = "\n\n".join(
            f"Query {i+1}: {q[:300]}" for i, q in enumerate(queries)
        )
        prompt = VET_PROMPT.format(queries=queries_text)

        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.3,   # Low temp for accurate routing
                timeout=60,
            )
            raw = resp.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            verdicts = json.loads(raw)
            if not isinstance(verdicts, list):
                return []

            results = []
            for i, verdict in enumerate(verdicts):
                if i >= len(batch):
                    break
                if not isinstance(verdict, dict):
                    continue

                original = batch[i]
                query = original["messages"][1]["content"]

                # Validate and apply verdict
                domain = verdict.get("domain", "business")
                if domain not in DOMAINS:
                    domain = "business"

                task_type = verdict.get("task_type", "qa")
                if task_type not in TASK_TYPES:
                    task_type = "qa"

                complexity = verdict.get("complexity", "medium")
                if complexity not in COMPLEXITY:
                    complexity = "medium"

                risk_level = verdict.get("risk_level", "medium")
                if risk_level not in RISK_LEVELS:
                    risk_level = "medium"

                latency_tier = verdict.get("latency_tier", "interactive")
                if latency_tier not in LATENCY_TIERS:
                    latency_tier = "interactive"

                cost_sensitivity = verdict.get("cost_sensitivity", "medium")
                if cost_sensitivity not in COST_SENSITIVITY:
                    cost_sensitivity = "medium"

                recommended_model = verdict.get("recommended_model", "research-8b")
                if recommended_model not in MODELS:
                    recommended_model = get_recommended_model(domain, complexity, risk_level)

                requires_tools = verdict.get("requires_tools", ["none"])
                if not isinstance(requires_tools, list):
                    requires_tools = ["none"]
                requires_tools = [t for t in requires_tools if t in TOOLS] or ["none"]

                reasoning = verdict.get("reasoning", "")[:120]

                # proposal_required: trust verdict if provided, else derive deterministically
                proposal_required = verdict.get("proposal_required")
                if not isinstance(proposal_required, bool):
                    proposal_required = needs_proposal(domain, complexity, risk_level)

                routing_answer = {
                    "domain": domain,
                    "task_type": task_type,
                    "complexity": complexity,
                    "risk_level": risk_level,
                    "latency_tier": latency_tier,
                    "cost_sensitivity": cost_sensitivity,
                    "recommended_model": recommended_model,
                    "escalation_allowed": bool(verdict.get("escalation_allowed", True)),
                    "proposal_required": proposal_required,
                    "requires_tools": requires_tools,
                    "reasoning": reasoning,
                }

                # Check if verdict changed the original routing
                original_model = original["metadata"].get("recommended_model", "")
                correction_flag = original_model != recommended_model

                results.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": json.dumps(routing_answer, indent=2)},
                    ],
                    "metadata": {
                        "domain": domain,
                        "task_type": task_type,
                        "complexity": complexity,
                        "risk_level": risk_level,
                        "recommended_model": recommended_model,
                        "stream": "vetted_harvest",
                        "vet_model": model.split("/")[-1],
                        "corrected": correction_flag,
                        "original_model": original_model,
                        "generated_at": datetime.utcnow().isoformat(),
                    },
                })

            return results

        except json.JSONDecodeError:
            return []
        except Exception as e:
            print(f"  [warn] vet error: {type(e).__name__}: {e}")
            return []


async def main_async(args):
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    model = MODEL_235B if args.model == "235b" else MODEL_80B

    # Find input file
    if args.input:
        input_path = Path(args.input)
    else:
        # Find latest harvest file
        files = sorted(OUTPUT_DIR.glob("router_v3_harvest_*.jsonl"), reverse=True)
        if not files:
            print("No harvest files found. Run mine_harvest.py first.")
            return
        input_path = files[0]

    print(f"Input: {input_path}")
    print(f"Model: {model.split('/')[-1]}")

    # Load records
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

    print(f"Loaded: {len(records):,} records")

    if args.dry_run:
        print("\n=== DRY RUN — vetting 3 records ===")
        sem = asyncio.Semaphore(1)
        results = await vet_batch(client, records[:3], model, sem)
        for r in results:
            m = r["metadata"]
            print(f"  domain={m['domain']} model={m['recommended_model']}"
                  f" corrected={m['corrected']} original={m['original_model']}")
            print(f"  Q: {r['messages'][1]['content'][:100]}")
        return

    # Batch processing
    batch_size = args.batch_size
    concurrency = args.concurrency
    sem = asyncio.Semaphore(concurrency)

    batches = [records[i:i+batch_size] for i in range(0, len(records), batch_size)]
    total_batches = len(batches)
    total_vetted = 0
    total_corrected = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = OUTPUT_DIR / f"router_v3_vetted_{timestamp}.jsonl"

    print(f"\nVetting {len(records):,} records in {total_batches} batches "
          f"(concurrency={concurrency}, batch_size={batch_size})")
    print("=" * 60)

    start = time.time()

    with open(output_path, "w") as out_f:
        tasks = [vet_batch(client, batch, model, sem) for batch in batches]

        for i, coro in enumerate(asyncio.as_completed(tasks)):
            results = await coro
            if results:
                for r in results:
                    out_f.write(json.dumps(r) + "\n")
                    if r["metadata"].get("corrected"):
                        total_corrected += 1
                total_vetted += len(results)

            if (i + 1) % 50 == 0 or (i + 1) == total_batches:
                elapsed = time.time() - start
                rate = total_vetted / elapsed * 60 if elapsed > 0 else 0
                pct = (i + 1) / total_batches * 100
                print(f"  [{pct:.0f}%] {total_vetted:,} vetted | "
                      f"{total_corrected:,} corrected | {rate:.0f}/min")

    elapsed = time.time() - start
    correction_rate = total_corrected / total_vetted * 100 if total_vetted else 0

    print("\n" + "=" * 60)
    print(f"DONE — {total_vetted:,} pairs vetted in {elapsed:.0f}s")
    print(f"Corrections made: {total_corrected:,} ({correction_rate:.1f}%)")
    print(f"Output: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SwarmRouter-4B-0 Harvest Vetter")
    parser.add_argument("--input", type=str, default="",
                        help="Input harvest file (default: latest router_v3_harvest_*.jsonl)")
    parser.add_argument("--model", choices=["80b", "235b"], default="80b",
                        help="80b = fast/cheap, 235b = quality (default: 80b)")
    parser.add_argument("--concurrency", type=int, default=50,
                        help="Concurrent API requests (default: 50)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Records per API call (default: 10)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
