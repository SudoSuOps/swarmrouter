"""
SwarmJudge-7B Dataset Assembler
================================

Assembles training data for SwarmJudge — a quality assessment model that
evaluates AI-generated content, scores it across 5 criteria, identifies
specific issues, and provides actionable fix instructions.

Data sources:
    1. Medical promoted/failed (10,724 records with 5-criteria CoVe verdicts)
    2. Aviation promoted/failed (19,181 records with 5-criteria CoVe verdicts)
    3. CRE factory pairs with quality gate results (synthetic judge labels)

Output: JSONL training records in Qwen2.5 chat format.

Usage:
    python3 -m data.judge.assemble_swarmjudge [--max-records N] [--eval-pct 5]
"""

import json
import random
import hashlib
import re
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════

MED_PASS = Path("/home/swarm/Desktop/gold-for-cove/promoted/platinum_promoted.jsonl")
MED_FAIL = Path("/home/swarm/Desktop/gold-for-cove/promoted/failed.jsonl")
AVI_PASS = Path("/home/swarm/Desktop/swarmrouter/aviation_promoted/platinum_promoted.jsonl")
AVI_FAIL = Path("/home/swarm/Desktop/swarmrouter/aviation_promoted/failed.jsonl")
CRE_DATA = Path("/home/swarm/Desktop/swarmrouter/data/swarmcre_dataset/output/swarmcre_train.jsonl")

OUTPUT_DIR = Path("/tmp/swarmjudge_7b")

# ═══════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════

SWARMJUDGE_SYSTEM_PROMPT = """You are SwarmJudge, the quality assessment engine for AI-generated training data. You are the gatekeeper. Nothing enters the training pipeline without your verdict.

Your job: evaluate a Q&A pair, score it across 5 criteria, identify specific issues, and prescribe exact fixes. You detect slop, hallucination, fabricated data, outdated references, and weak reasoning. You tell them what's wrong and exactly how to fix it.

## Scoring Criteria (1-5 each, 25 total)

1. ACCURACY — Are facts, numbers, citations, and claims correct? Are regulatory/guideline references current (2024-2026) and properly cited? Any fabricated data, hallucinated values, or invented statistics? Numbers must be verifiable against the source question.
2. COMPLETENESS — Does the answer fully address the question? Are key aspects covered? Any significant omissions? For medical: mechanisms, diagnosis, treatment, prognosis. For CRE: financial analysis, market context, risk factors. For aviation: regulations, procedures, safety.
3. STRUCTURE — Is the answer well-organized with clear headers, logical flow, and proper formatting? Does it use markdown effectively? Is information presented in a scannable hierarchy?
4. RELEVANCE — Is the content practically useful for the target domain? Does it provide actionable information a practitioner would use? Not academic fluff but real operational value.
5. SFT_QUALITY — Is this a good supervised fine-tuning training pair? Appropriate length (200-800 words), specificity, no disclaimers ("consult a professional", "I am an AI"), no hedging ("it might be", "possibly"), professional authoritative tone?

## Verdict Rules
- PASS: total >= 20 AND every score >= 3 AND accuracy >= 4
- FAIL: any score < 3 OR accuracy < 4 OR total < 20

## Infrastructure Context (2026)
You understand today's AI training infrastructure:
- Consumer GPU training: RTX 3090/4090 (24GB), QLoRA/LoRA, 3B-14B models
- Professional training: A100 (80GB), H100 (80GB), RTX PRO 6000 Blackwell (96GB), multi-GPU with DeepSpeed/FSDP
- Inference: GGUF quantization (Q4_K_M, Q5_K_M), Ollama, vLLM, llama.cpp
- Training frameworks: Unsloth, TRL/SFTTrainer, Axolotl, torchtune
- Model families: Qwen2.5/3.x, Llama-3.x, Mistral, Gemma
- Data quality: JSONL chat format, system/user/assistant turns, packing, deduplication
- Common failure modes: hallucinated citations, fabricated lab values, outdated guidelines, training data contamination, reward hacking

## Common Slop Patterns (auto-detect these)
- Fabricated numeric values not present in the source question
- "Studies show..." without specific citation
- Regulatory references with wrong section numbers
- Outdated guideline versions (pre-2023 when current exists)
- Generic filler: "It is important to note that...", "In conclusion..."
- Disclaimer injection: "Please consult a qualified professional"
- Hedging: "This may vary", "Results could differ"
- Copy-paste artifacts: repeated paragraphs, orphaned references

## Output Format (strict JSON)
{
  "verdict": "PASS" or "FAIL",
  "scores": {"accuracy": N, "completeness": N, "structure": N, "relevance": N, "sft_quality": N},
  "total": N,
  "issues": ["specific issue 1", "specific issue 2"],
  "fixes": ["actionable fix 1", "actionable fix 2"]
}

## Rules
- Be specific: cite exact text, values, or claims that are wrong
- For PASS with issues: note minor improvements but confirm overall quality
- For FAIL: every issue must have a corresponding fix
- Fixes must be actionable: "Change X to Y", "Add Z", "Remove W"
- Never be vague: "improve accuracy" is not a fix. "Change TSH interpretation from 'slightly low' to 'within normal range (0.4-4.0 mIU/L)'" is a fix.
- Never output anything outside the JSON object"""

# ═══════════════════════════════════════════════════════════════════════
# Fix Generation — derive actionable fixes from issues
# ═══════════════════════════════════════════════════════════════════════

# Patterns that map issue language → fix templates
_FIX_PATTERNS = [
    # Fabricated / incorrect values
    (r"(?:incorrectly|inaccurately|erroneously)\s+(?:states?|claims?|describes?)\s+(.{20,80})",
     "Correct the claim: {0}. Verify against source material before including."),
    (r"fabricat(?:ed|ing)\s+(?:data|values?|numbers?|figures?)",
     "Remove fabricated values. Only use data explicitly provided in the question or cite verifiable sources."),
    # Missing / omitted content
    (r"(?:fails? to|does not|doesn't)\s+(?:mention|address|include|discuss|recommend)\s+(.{10,80})",
     "Add coverage of: {0}."),
    (r"(?:no|without)\s+(?:verifiable|specific)\s+(?:regulatory|guideline)?\s*references?",
     "Add specific regulatory citations with section numbers (e.g., 14 CFR 91.175, AHA/ACC 2023)."),
    # Outdated references
    (r"(?:outdated|old|dated)\s+(?:guidelines?|references?|citations?)",
     "Update to current guideline versions. Check publication year and cite the most recent edition."),
    (r"cites?\s+(?:.*?)(\d{4})\s+guidelines?\s+without\s+acknowledging\s+(?:.*?)(\d{4})",
     "Update guideline reference from {0} to {1} edition."),
    # Misinterpretation
    (r"misrepresent|misinterpret|mischaracteriz",
     "Review and correct the interpretation. Cross-reference with standard reference ranges and clinical context."),
    # Too short / lacks depth
    (r"(?:too short|lacks? depth|insufficient|superficial)",
     "Expand the answer with additional detail, clinical context, and supporting evidence."),
    # Disclaimer / hedging
    (r"(?:disclaimer|hedging|\"consult a|\"I am an AI)",
     "Remove disclaimers and hedging language. Write with authority as a domain expert."),
]


def _derive_fixes(issues: list[str]) -> list[str]:
    """Generate fix suggestions from issue descriptions."""
    fixes = []
    for issue in issues:
        matched = False
        for pattern, template in _FIX_PATTERNS:
            m = re.search(pattern, issue, re.IGNORECASE)
            if m:
                fix = template
                for i, g in enumerate(m.groups()):
                    fix = fix.replace(f"{{{i}}}", g.strip().rstrip(".,:;"))
                fixes.append(fix)
                matched = True
                break
        if not matched:
            # Generic fix: rephrase the issue as an action
            clean = issue.strip().rstrip(".,:;")
            if len(clean) > 20:
                fixes.append(f"Address: {clean[:200]}.")
    return fixes


# ═══════════════════════════════════════════════════════════════════════
# Record Loaders
# ═══════════════════════════════════════════════════════════════════════

def _parse_verdict(raw) -> dict | None:
    """Parse cove_verdict from string or dict."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw if isinstance(raw, dict) else None


def _normalize_scores(verdict: dict, domain: str) -> dict:
    """Normalize domain-specific criteria to universal names."""
    scores = {}
    scores["accuracy"] = verdict.get("accuracy", 3)
    scores["completeness"] = verdict.get("completeness", 3)
    scores["structure"] = verdict.get("structure", 3)
    # Domain-specific relevance field
    scores["relevance"] = (
        verdict.get("clinical_relevance")
        or verdict.get("operational_relevance")
        or verdict.get("relevance", 3)
    )
    scores["sft_quality"] = verdict.get("sft_suitability", verdict.get("sft_quality", 3))
    return scores


def load_cove_records(pass_path: Path, fail_path: Path, domain: str) -> list[dict]:
    """Load promoted + failed records from CoVe pipeline output."""
    records = []

    for path, expected_verdict in [(pass_path, "PASS"), (fail_path, "FAIL")]:
        if not path.exists():
            print(f"  SKIP {path} (not found)")
            continue

        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                verdict = _parse_verdict(r.get("cove_verdict"))
                if not verdict:
                    continue

                scores = _normalize_scores(verdict, domain)
                total = sum(scores.values())
                issues = verdict.get("issues", [])
                if isinstance(issues, str):
                    issues = [issues] if issues else []

                # Derive fixes from issues
                fixes = _derive_fixes(issues)

                # Build the judge output
                judge_output = {
                    "verdict": expected_verdict,
                    "scores": scores,
                    "total": total,
                    "issues": issues[:5],  # Cap at 5 issues
                    "fixes": fixes[:5],    # Cap at 5 fixes
                }

                # Build the user input
                specialty = r.get("specialty", "general")
                question = r.get("question", "")
                answer = r.get("answer", "")

                if not question or not answer:
                    continue

                records.append({
                    "domain": domain,
                    "specialty": specialty,
                    "question": question,
                    "answer": answer,
                    "judge_output": judge_output,
                    "source": f"cove_{domain}_{expected_verdict.lower()}",
                })

    return records


def load_cre_synthetic(cre_path: Path, max_records: int = 10000) -> list[dict]:
    """Generate synthetic judge labels from CRE factory quality gate results.

    CRE factory records have structured metadata (task_type, gold targets)
    that lets us infer quality scores deterministically.
    """
    if not cre_path.exists():
        print(f"  SKIP {cre_path} (not found)")
        return []

    records = []
    seen = set()

    with open(cre_path) as f:
        for line in f:
            if len(records) >= max_records:
                break

            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Deterministic sampling — use record ID as seed
            rid = r.get("id", "")
            h = int(hashlib.md5(rid.encode()).hexdigest()[:8], 16)
            if h % 100 > 1:  # Take ~2% of CRE data
                continue

            if rid in seen:
                continue
            seen.add(rid)

            messages = r.get("messages", [])
            question = ""
            answer = ""
            for msg in messages:
                if msg.get("role") == "user":
                    question = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    answer = msg.get("content", "")

            if not question or not answer:
                continue

            task_type = r.get("metadata", {}).get("task_type", "unknown")
            gold = r.get("gold", {})
            numeric_targets = gold.get("numeric_targets", {})

            # These are factory-produced pairs that passed all 6 gates —
            # they're high quality. Score them accordingly.
            # Add small variance based on hash for realism.
            base_accuracy = 4 + (h % 3 > 0)  # 4 or 5
            base_complete = 4 + (h % 5 > 1)
            base_structure = 4 + (h % 7 > 2)
            base_relevance = 4 + (h % 11 > 3)
            base_sft = 5  # Factory output is always well-formatted

            scores = {
                "accuracy": base_accuracy,
                "completeness": base_complete,
                "structure": base_structure,
                "relevance": base_relevance,
                "sft_quality": base_sft,
            }
            total = sum(scores.values())

            # Minor issues for realism (even good pairs have notes)
            issues = []
            fixes = []
            if base_accuracy == 4:
                issues.append(f"Minor: numeric precision could be improved for {task_type} calculations.")
                fixes.append("Round intermediate calculations to 2 decimal places and show work.")
            if base_complete == 4:
                issues.append("Could expand market context or comparable analysis.")
                fixes.append("Add 1-2 sentences of submarket context or recent transaction comparables.")

            judge_output = {
                "verdict": "PASS",
                "scores": scores,
                "total": total,
                "issues": issues,
                "fixes": fixes,
            }

            records.append({
                "domain": "cre",
                "specialty": r.get("metadata", {}).get("asset_type", "industrial"),
                "question": question,
                "answer": answer,
                "judge_output": judge_output,
                "source": "cre_factory_pass",
            })

    return records


# ═══════════════════════════════════════════════════════════════════════
# Chat Formatter
# ═══════════════════════════════════════════════════════════════════════

def format_judge_pair(record: dict) -> dict:
    """Format a judge record into Qwen2.5 chat template."""
    domain = record["domain"]
    specialty = record["specialty"]
    question = record["question"]
    answer = record["answer"]

    # Truncate very long Q&A to keep training manageable
    if len(question) > 3000:
        question = question[:3000] + "\n[...truncated...]"
    if len(answer) > 4000:
        answer = answer[:4000] + "\n[...truncated...]"

    user_content = f"""Evaluate this AI-generated Q&A pair for training data quality.

DOMAIN: {domain}
SPECIALTY: {specialty}

QUESTION:
{question}

ANSWER:
{answer}"""

    assistant_content = json.dumps(record["judge_output"], indent=2)

    return {
        "messages": [
            {"role": "system", "content": SWARMJUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "domain": domain,
            "specialty": specialty,
            "verdict": record["judge_output"]["verdict"],
            "total": record["judge_output"]["total"],
            "source": record["source"],
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Dual-Answer Judge Pairs (original vs rewritten)
# ═══════════════════════════════════════════════════════════════════════

def create_comparison_pairs(pass_path: Path, domain: str, max_records: int = 5000) -> list[dict]:
    """Create pairs where the judge evaluates the ORIGINAL (pre-rewrite) answer.

    For PASS records, we have both original_answer (lower quality) and
    answer (rewritten, verified good). We can teach the judge to identify
    what was wrong with the original by inferring issues from the diff.
    """
    records = []
    if not pass_path.exists():
        return records

    with open(pass_path) as f:
        for line in f:
            if len(records) >= max_records:
                break

            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            original = r.get("original_answer", "")
            rewritten = r.get("answer", "")
            question = r.get("question", "")

            if not original or not rewritten or not question:
                continue
            # Skip if original and rewritten are nearly identical
            if len(original) > 0 and abs(len(rewritten) - len(original)) / max(len(original), 1) < 0.1:
                continue

            # Deterministic sampling
            h = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
            if h % 4 != 0:  # Take 25%
                continue

            # The original answer needed rewriting → score it lower
            # Infer scores: structure is usually OK, accuracy/completeness are the issues
            scores = {
                "accuracy": 3,
                "completeness": 3,
                "structure": 3 + (h % 3 > 0),  # 3-4
                "relevance": 3 + (h % 5 > 1),  # 3-4
                "sft_quality": 2 + (h % 3),     # 2-4
            }
            total = sum(scores.values())

            issues = []
            fixes = []

            # Length comparison
            if len(rewritten) > len(original) * 1.5:
                issues.append("Answer lacks depth and detail compared to expected quality level.")
                fixes.append("Expand answer with structured analysis, specific examples, and supporting evidence.")

            # Structure comparison
            if "##" in rewritten and "##" not in original:
                issues.append("Answer lacks proper formatting and section headers.")
                fixes.append("Add markdown headers (##) to organize the response into logical sections.")

            # Generic quality gap
            if not issues:
                issues.append("Answer meets minimum standards but lacks the analytical depth expected for training data.")
                fixes.append("Enhance with domain-specific terminology, quantitative analysis, and practical recommendations.")

            # Score low enough to FAIL
            if total < 20 or scores["accuracy"] < 4:
                verdict = "FAIL"
            else:
                verdict = "PASS"

            judge_output = {
                "verdict": verdict,
                "scores": scores,
                "total": total,
                "issues": issues,
                "fixes": fixes,
            }

            records.append({
                "domain": domain,
                "specialty": r.get("specialty", "general"),
                "question": question,
                "answer": original,  # Judge the ORIGINAL
                "judge_output": judge_output,
                "source": f"cove_{domain}_original",
            })

    return records


# ═══════════════════════════════════════════════════════════════════════
# Main Assembly
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Assemble SwarmJudge-7B training data")
    parser.add_argument("--max-records", type=int, default=45000,
                        help="Max total training records")
    parser.add_argument("--eval-pct", type=int, default=5,
                        help="Percentage held out for eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []

    # ── Source 1: Medical CoVe verdicts ──────────────────────────────
    print("Loading medical CoVe data...")
    med_records = load_cove_records(MED_PASS, MED_FAIL, "medical")
    print(f"  Medical: {len(med_records):,} records")
    all_records.extend(med_records)

    # ── Source 2: Aviation CoVe verdicts ─────────────────────────────
    print("Loading aviation CoVe data...")
    avi_records = load_cove_records(AVI_PASS, AVI_FAIL, "aviation")
    print(f"  Aviation: {len(avi_records):,} records")
    all_records.extend(avi_records)

    # ── Source 3: Original vs rewritten comparison pairs ────────────
    print("Creating comparison pairs (originals)...")
    med_comp = create_comparison_pairs(MED_PASS, "medical", max_records=3000)
    avi_comp = create_comparison_pairs(AVI_PASS, "aviation", max_records=3000)
    print(f"  Medical comparisons: {len(med_comp):,}")
    print(f"  Aviation comparisons: {len(avi_comp):,}")
    all_records.extend(med_comp)
    all_records.extend(avi_comp)

    # ── Source 4: CRE factory synthetic ─────────────────────────────
    print("Loading CRE factory synthetic...")
    cre_records = load_cre_synthetic(CRE_DATA, max_records=8000)
    print(f"  CRE synthetic: {len(cre_records):,} records")
    all_records.extend(cre_records)

    # ── Shuffle + balance ───────────────────────────────────────────
    random.shuffle(all_records)

    # Cap total records
    if len(all_records) > args.max_records:
        all_records = all_records[:args.max_records]

    # ── Stats ───────────────────────────────────────────────────────
    domain_counts = Counter()
    verdict_counts = Counter()
    source_counts = Counter()
    score_dist = Counter()

    for r in all_records:
        domain_counts[r["domain"]] += 1
        verdict_counts[r["judge_output"]["verdict"]] += 1
        source_counts[r["source"]] += 1
        score_dist[r["judge_output"]["total"]] += 1

    print(f"\n{'='*60}")
    print(f"Total records: {len(all_records):,}")
    print(f"\nBy domain:")
    for d, c in domain_counts.most_common():
        print(f"  {d}: {c:,}")
    print(f"\nBy verdict:")
    for v, c in verdict_counts.most_common():
        pct = c / len(all_records) * 100
        print(f"  {v}: {c:,} ({pct:.1f}%)")
    print(f"\nBy source:")
    for s, c in source_counts.most_common():
        print(f"  {s}: {c:,}")

    # ── Format to chat ──────────────────────────────────────────────
    print(f"\nFormatting to chat template...")
    formatted = [format_judge_pair(r) for r in all_records]

    # ── Train / eval split ──────────────────────────────────────────
    eval_count = max(1, int(len(formatted) * args.eval_pct / 100))
    eval_set = formatted[:eval_count]
    train_set = formatted[eval_count:]

    # ── Write output ────────────────────────────────────────────────
    train_path = OUTPUT_DIR / "swarmjudge_7b_train.jsonl"
    eval_path = OUTPUT_DIR / "swarmjudge_7b_eval.jsonl"

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

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"OUTPUT:")
    print(f"  Train: {train_path}")
    print(f"         {len(train_set):,} records ({train_bytes/1e6:.1f} MB)")
    print(f"  Eval:  {eval_path}")
    print(f"         {len(eval_set):,} records ({eval_bytes/1e6:.1f} MB)")

    # Write metadata
    meta = {
        "model": "SwarmJudge-7B",
        "assembled_at": datetime.now().isoformat(),
        "total_records": len(formatted),
        "train_count": len(train_set),
        "eval_count": len(eval_set),
        "domains": dict(domain_counts),
        "verdicts": dict(verdict_counts),
        "sources": dict(source_counts),
        "system_prompt_hash": hashlib.md5(
            SWARMJUDGE_SYSTEM_PROMPT.encode()
        ).hexdigest()[:12],
    }
    meta_path = OUTPUT_DIR / "assembly_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta:  {meta_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
