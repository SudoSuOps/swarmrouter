#!/usr/bin/env python3
"""
Qwen3.5 Benchmark — SwarmJudge Appliance Brain Eval
=====================================================

Tests:
  1. JSON validity        — structured output fidelity
  2. Tool-calling         — correct function selection + arg population
  3. RAG grounding        — answer grounded in provided context only
  4. Tokens/sec           — raw throughput on this rig

Usage:
  # Against Ollama model
  python3 benchmark_qwen35.py --model qwen3.5:9b

  # Against llama-server (GGUF)
  python3 benchmark_qwen35.py --model qwen3.5:9b --base-url http://localhost:8080
"""

import json
import time
import argparse
import statistics
import httpx
from datetime import datetime

# ─── Test suites ───────────────────────────────────────────────────────────

JSON_TESTS = [
    {
        "name": "judge_eval_schema",
        "prompt": """You are SwarmJudge. Evaluate this AI response and output ONLY valid JSON matching the schema exactly.

RESPONSE: "The DSCR for this deal is 1.42x based on NOI of $892,000 and annual debt service of $628,000."

Output JSON:
{
  "verdict": "PASS" or "FAIL",
  "scores": {"accuracy": N, "completeness": N, "structure": N, "relevance": N, "sft_quality": N},
  "total": N,
  "issues": [],
  "fixes": []
}""",
        "required_keys": ["verdict", "scores", "total", "issues", "fixes"],
        "required_score_keys": ["accuracy", "completeness", "structure", "relevance", "sft_quality"],
    },
    {
        "name": "router_schema",
        "prompt": """Output ONLY valid JSON for this routing decision:
Request: "Calculate cap rate for a cold storage facility in Dallas with NOI $1.2M and purchase price $18M"

Schema:
{"domain": str, "complexity": "low|medium|high|expert", "risk": "none|low|medium|high", "model": str, "proposal_required": bool}""",
        "required_keys": ["domain", "complexity", "risk", "model", "proposal_required"],
    },
    {
        "name": "nested_json",
        "prompt": """Output ONLY valid JSON:
{
  "deal": {"address": "123 Industrial Blvd, Houston TX", "sf": 85000, "asking_price": 12500000},
  "metrics": {"noi": 875000, "cap_rate": 0.07, "dscr": 1.35},
  "verdict": "underwrite"
}
Reproduce exactly.""",
        "required_keys": ["deal", "metrics", "verdict"],
    },
]

TOOL_TESTS = [
    {
        "name": "calculate_dscr",
        "prompt": """You have access to this tool:
calculate_dscr(noi: float, annual_debt_service: float) -> float

A warehouse in Dallas has NOI of $1,250,000 and annual debt service of $875,000.
Call the tool to calculate DSCR. Output ONLY:
{"tool": "calculate_dscr", "args": {"noi": <value>, "annual_debt_service": <value>}}""",
        "expected_tool": "calculate_dscr",
        "expected_args": ["noi", "annual_debt_service"],
        "expected_values": {"noi": 1250000, "annual_debt_service": 875000},
    },
    {
        "name": "search_comps",
        "prompt": """Tools available:
- search_comps(asset_type: str, market: str, sf_min: int, sf_max: int) -> list
- get_cap_rate(market: str, asset_type: str) -> float
- calculate_noi(gross_revenue: float, expenses: float) -> float

Find comparable sales for a 120,000 SF cross-dock facility in Phoenix.
Output ONLY: {"tool": "<name>", "args": {<key>: <value>}}""",
        "expected_tool": "search_comps",
        "expected_args": ["asset_type", "market"],
    },
    {
        "name": "wrong_tool_rejection",
        "prompt": """Tools available:
- send_email(to: str, subject: str, body: str) -> bool
- calculate_cap_rate(noi: float, price: float) -> float

A data center asking $45M generates $3.15M NOI. What is the cap rate?
Output ONLY: {"tool": "<name>", "args": {<key>: <value>}}""",
        "expected_tool": "calculate_cap_rate",
        "expected_args": ["noi", "price"],
    },
]

RAG_TESTS = [
    {
        "name": "grounded_answer",
        "context": """LEASE ABSTRACT — Unit 12, Alliance Industrial Park
Tenant: FastShip Logistics LLC
Base Rent: $8.25/SF NNN
Term: 7 years commencing January 1, 2024
Options: Two 5-year renewal options at 95% of fair market value
Escalations: 3% annually
Security Deposit: 3 months base rent
Permitted Use: Last-mile distribution, no hazmat""",
        "question": "What is the annual escalation rate and how many renewal options does the tenant have?",
        "required_facts": ["3%", "two", "5-year"],
        "forbidden": ["hallucination", "I don't know", "not specified"],
    },
    {
        "name": "no_hallucination",
        "context": """PROPERTY RECORD — 8900 Commerce Way, Fort Worth TX
Parcel: 0234-567-890
Year Built: 2019
SF: 425,000
Zoning: I-2 Heavy Industrial
Last Sale: $28,500,000 (March 2022)
Current Owner: Blackstone Industrial REIT""",
        "question": "What is the current cap rate and NOI for this property?",
        "required_facts": [],
        "forbidden_behavior": "must_refuse_or_say_not_in_context",
        "grounding_check": True,
    },
    {
        "name": "math_from_context",
        "context": """T-12 SUMMARY — 450 Logistics Center
Gross Revenue:     $4,200,000
Vacancy (5%):       ($210,000)
Effective Revenue: $3,990,000
OpEx:             ($1,197,000)
NOI:              $2,793,000
Debt Service:     ($1,860,000)""",
        "question": "Calculate DSCR and cap rate if purchase price is $39,900,000",
        "required_math": {"dscr": 1.50, "cap_rate": 0.07},
        "tolerance": 0.02,
    },
]

THROUGHPUT_PROMPTS = [
    "Evaluate this CRE underwriting for a 250,000 SF distribution warehouse in Dallas with NOI $2.1M, asking price $30M, and 5-year WALT. Provide detailed analysis covering cap rate, DSCR assuming 65% LTV at 6.5% rate 25-year amort, and market commentary.",
    "You are SwarmJudge. Rate this response on 5 criteria (1-5 each): accuracy, completeness, structure, relevance, sft_quality. Then provide PASS or FAIL verdict with total score.",
    "Generate an investment committee memo for a cold storage acquisition in Phoenix: 180,000 SF, $52M price, $3.64M NOI, Lineage Logistics tenant, 8 years remaining, 2.5% bumps.",
]


# ─── Ollama client ─────────────────────────────────────────────────────────

def chat(model: str, messages: list, base_url: str, timeout: float = 60.0) -> tuple[str, float, int]:
    """Returns (content, elapsed_s, output_tokens)"""
    t0 = time.time()
    resp = httpx.post(
        f"{base_url}/api/chat",
        json={"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.1}},
        timeout=timeout,
    )
    resp.raise_for_status()
    d = resp.json()
    content = d["message"]["content"]
    elapsed = time.time() - t0
    out_toks = d.get("eval_count", 0)
    return content, elapsed, out_toks


def extract_json(text: str) -> dict | None:
    """Try to parse JSON from model output."""
    import re
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Code block
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Bare JSON object
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


# ─── Test runners ──────────────────────────────────────────────────────────

def run_json_tests(model: str, base_url: str) -> dict:
    results = []
    for t in JSON_TESTS:
        try:
            content, elapsed, toks = chat(model, [{"role": "user", "content": t["prompt"]}], base_url)
            parsed = extract_json(content)
            valid = parsed is not None
            keys_ok = False
            schema_ok = False
            if valid and parsed:
                required = t.get("required_keys", [])
                keys_ok = all(k in parsed for k in required)
                if "required_score_keys" in t and "scores" in parsed:
                    score_keys = t["required_score_keys"]
                    schema_ok = all(k in parsed["scores"] for k in score_keys)
                else:
                    schema_ok = keys_ok
            results.append({
                "test": t["name"], "valid_json": valid, "keys_ok": keys_ok,
                "schema_ok": schema_ok, "elapsed": round(elapsed, 2), "tokens": toks,
                "output_preview": content[:120],
            })
            status = "PASS" if (valid and keys_ok) else "FAIL"
            print(f"  [{status}] {t['name']:<30} valid={valid} keys={keys_ok} ({elapsed:.1f}s)")
        except Exception as e:
            results.append({"test": t["name"], "error": str(e)})
            print(f"  [ERR ] {t['name']:<30} {e}")

    pass_count = sum(1 for r in results if r.get("valid_json") and r.get("keys_ok"))
    return {"tests": results, "pass_rate": round(pass_count / len(JSON_TESTS), 3), "passed": pass_count, "total": len(JSON_TESTS)}


def run_tool_tests(model: str, base_url: str) -> dict:
    results = []
    for t in TOOL_TESTS:
        try:
            content, elapsed, toks = chat(model, [{"role": "user", "content": t["prompt"]}], base_url)
            parsed = extract_json(content)
            tool_ok = False
            args_ok = False
            if parsed:
                tool_ok = parsed.get("tool") == t["expected_tool"]
                args_ok = all(k in parsed.get("args", {}) for k in t.get("expected_args", []))
                # Check values if specified
                if "expected_values" in t and parsed.get("args"):
                    for k, v in t["expected_values"].items():
                        if abs(parsed["args"].get(k, 0) - v) > 1:
                            args_ok = False
            status = "PASS" if (tool_ok and args_ok) else "FAIL"
            results.append({
                "test": t["name"], "tool_correct": tool_ok, "args_correct": args_ok,
                "elapsed": round(elapsed, 2), "got_tool": (parsed or {}).get("tool"),
                "output_preview": content[:120],
            })
            print(f"  [{status}] {t['name']:<30} tool={tool_ok} args={args_ok}  got={parsed.get('tool') if parsed else 'None'} ({elapsed:.1f}s)")
        except Exception as e:
            results.append({"test": t["name"], "error": str(e)})
            print(f"  [ERR ] {t['name']:<30} {e}")

    pass_count = sum(1 for r in results if r.get("tool_correct") and r.get("args_correct"))
    return {"tests": results, "pass_rate": round(pass_count / len(TOOL_TESTS), 3), "passed": pass_count, "total": len(TOOL_TESTS)}


def run_rag_tests(model: str, base_url: str) -> dict:
    results = []
    for t in RAG_TESTS:
        prompt = f"CONTEXT:\n{t['context']}\n\nQUESTION: {t['question']}\n\nAnswer using ONLY information from the context above."
        try:
            content, elapsed, toks = chat(model, [{"role": "user", "content": prompt}], base_url, timeout=90)
            content_lower = content.lower()

            # Check required facts present
            facts_found = [f for f in t.get("required_facts", []) if f.lower() in content_lower]
            facts_ok = len(facts_found) == len(t.get("required_facts", []))

            # Check no hallucination for context-absent info
            grounding_ok = True
            if t.get("grounding_check"):
                # Model should say "not in context" or refuse — not invent numbers
                import re
                has_fake_numbers = bool(re.search(r'\$[\d,]+|\d+\.?\d*%', content))
                grounding_ok = not has_fake_numbers

            # Check math if required
            math_ok = True
            if "required_math" in t:
                import re
                for metric, expected in t["required_math"].items():
                    tolerance = t.get("tolerance", 0.02)
                    # Look for numeric values near expected
                    nums = [float(x.replace(',', '')) for x in re.findall(r'[\d,]+\.?\d*', content) if x]
                    math_ok = any(abs(n - expected) <= tolerance for n in nums)

            passed = facts_ok and grounding_ok and math_ok
            status = "PASS" if passed else "FAIL"
            results.append({
                "test": t["name"], "facts_ok": facts_ok, "grounding_ok": grounding_ok,
                "math_ok": math_ok, "elapsed": round(elapsed, 2), "tokens": toks,
                "output_preview": content[:200],
            })
            print(f"  [{status}] {t['name']:<30} facts={facts_ok} grounding={grounding_ok} math={math_ok} ({elapsed:.1f}s)")
        except Exception as e:
            results.append({"test": t["name"], "error": str(e)})
            print(f"  [ERR ] {t['name']:<30} {e}")

    pass_count = sum(1 for r in results if r.get("facts_ok") and r.get("grounding_ok"))
    return {"tests": results, "pass_rate": round(pass_count / len(RAG_TESTS), 3), "passed": pass_count, "total": len(RAG_TESTS)}


def run_throughput(model: str, base_url: str) -> dict:
    print(f"  Warming up...")
    # Warmup
    try:
        chat(model, [{"role": "user", "content": "Say OK"}], base_url, timeout=30)
    except Exception:
        pass

    rates = []
    for i, prompt in enumerate(THROUGHPUT_PROMPTS):
        try:
            _, elapsed, toks = chat(model, [{"role": "user", "content": prompt}], base_url, timeout=120)
            rate = toks / elapsed if elapsed > 0 else 0
            rates.append(rate)
            print(f"  Prompt {i+1}: {toks} tokens in {elapsed:.1f}s = {rate:.1f} tok/s")
        except Exception as e:
            print(f"  Prompt {i+1}: ERROR {e}")

    avg = statistics.mean(rates) if rates else 0
    return {"tok_per_sec": [round(r, 1) for r in rates], "avg_tok_per_sec": round(avg, 1), "samples": len(rates)}


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="qwen3.5:9b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--output",   default="/tmp/benchmark_results.json")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"Benchmark: {args.model}")
    print(f"Endpoint:  {args.base_url}")
    print(f"Time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*60}\n")

    results = {"model": args.model, "base_url": args.base_url, "timestamp": datetime.now().isoformat()}

    print("── JSON Validity ─────────────────────────────────")
    results["json_validity"] = run_json_tests(args.model, args.base_url)

    print("\n── Tool Calling ──────────────────────────────────")
    results["tool_calling"] = run_tool_tests(args.model, args.base_url)

    print("\n── RAG Grounding ─────────────────────────────────")
    results["rag_grounding"] = run_rag_tests(args.model, args.base_url)

    print("\n── Throughput ────────────────────────────────────")
    results["throughput"] = run_throughput(args.model, args.base_url)

    # ── Summary ──────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"RESULTS — {args.model}")
    print(f"{'═'*60}")
    j = results["json_validity"]
    t = results["tool_calling"]
    r = results["rag_grounding"]
    tp = results["throughput"]
    print(f"  JSON validity:    {j['passed']}/{j['total']}  ({j['pass_rate']*100:.0f}%)")
    print(f"  Tool calling:     {t['passed']}/{t['total']}  ({t['pass_rate']*100:.0f}%)")
    print(f"  RAG grounding:    {r['passed']}/{r['total']}  ({r['pass_rate']*100:.0f}%)")
    print(f"  Throughput:       {tp['avg_tok_per_sec']} tok/s avg")
    print(f"{'═'*60}\n")

    import json as _j
    with open(args.output, "w") as f:
        _j.dump(results, f, indent=2)
    print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
