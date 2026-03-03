#!/usr/bin/env python3
"""
SwarmJudge-9B-B0-Phase1 Eval
==============================
Runs the merged model against phase1_eval.jsonl.
Reports: JSON validity, verdict accuracy, score MAE, PASS/FAIL breakdown.

Run:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python3 eval_swarmjudge_9b.py

    python3 eval_swarmjudge_9b.py --sample 200 --max-new-tokens 1024
"""

import json
import argparse
import sys
import random
from pathlib import Path
from datetime import datetime, timezone

random.seed(42)

MODEL_DIR  = Path("/data2/swarmjudge-9b-b0/merged")
EVAL_FILE  = Path("/data2/swarmjudge_27b_data/phase1/phase1_eval.jsonl")
OUT_DIR    = Path("/data2/swarmjudge-9b-b0/eval")
CRITERIA   = ["accuracy", "completeness", "structure", "relevance", "sft_quality"]


def format_chat(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    return "\n".join(parts) + "\n<|im_start|>assistant\n"


def parse_output(text: str) -> dict | None:
    """Extract JSON from model output."""
    text = text.strip()
    # strip any trailing im_end
    if "<|im_end|>" in text:
        text = text[:text.index("<|im_end|>")].strip()
    try:
        return json.loads(text)
    except Exception:
        # try to find JSON blob
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
    return None


def score_pair(pred: dict | None, gold: dict) -> dict:
    result = {
        "json_valid":      pred is not None,
        "verdict_match":   False,
        "score_mae":       None,
        "pass_precision":  None,
        "fail_recall":     None,
    }
    if pred is None:
        return result

    gold_verdict = gold.get("verdict", "").upper()
    pred_verdict = pred.get("verdict", "").upper()
    result["verdict_match"] = (gold_verdict == pred_verdict)

    # Score MAE across 5 criteria
    gold_scores = gold.get("scores", {})
    pred_scores = pred.get("scores", {})
    maes = []
    for c in CRITERIA:
        if c in gold_scores and c in pred_scores:
            try:
                maes.append(abs(float(pred_scores[c]) - float(gold_scores[c])))
            except (TypeError, ValueError):
                pass
    if maes:
        result["score_mae"] = sum(maes) / len(maes)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample",        type=int,   default=200, help="Eval N samples (default: 200)")
    parser.add_argument("--max-new-tokens", type=int,  default=512)
    parser.add_argument("--batch-size",    type=int,   default=2)
    parser.add_argument("--model",         default=str(MODEL_DIR))
    parser.add_argument("--eval-file",     default=str(EVAL_FILE))
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SwarmJudge-9B-B0-Phase1 Eval")
    print(f"{'='*60}")
    print(f"Model:     {args.model}")
    print(f"Eval file: {args.eval_file}")

    # Load eval set
    eval_path = Path(args.eval_file)
    pairs = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    if args.sample:
        pairs = random.sample(pairs, min(args.sample, len(pairs)))
    print(f"Samples:   {len(pairs)}")

    gold_labels = []
    prompts     = []
    for p in pairs:
        msgs = p["messages"]
        # prompt = system + user only; gold = assistant content
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        gold_asst   = next((m["content"] for m in msgs if m["role"] == "assistant"), "{}")
        try:
            gold_labels.append(json.loads(gold_asst))
        except Exception:
            gold_labels.append({})
        prompts.append(format_chat(prompt_msgs))

    # Load model via transformers (Qwen3.5 Mamba-Transformer hybrid)
    print(f"\nLoading model via transformers...")
    import os, torch
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    stop_ids = []
    for tok in ["<|im_end|>", "<|endoftext|>"]:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if ids:
            stop_ids.append(ids[0])

    print(f"Running inference on {len(prompts)} prompts (batch={args.batch_size})...")
    raw_outputs = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i+args.batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=2048).to("cuda:0")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                eos_token_id=stop_ids if stop_ids else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j, seq in enumerate(out):
            prompt_len = enc["input_ids"].shape[1]
            gen = seq[prompt_len:]
            raw_outputs.append(tokenizer.decode(gen, skip_special_tokens=False))
        done = min(i + args.batch_size, len(prompts))
        print(f"  {done}/{len(prompts)}", end="\r", flush=True)
    print()

    # Score
    results = []
    json_valid    = 0
    verdict_match = 0
    maes          = []
    by_verdict    = {"PASS": {"total":0,"match":0}, "FAIL": {"total":0,"match":0}}

    for text, gold in zip(raw_outputs, gold_labels):
        pred = parse_output(text)
        sc   = score_pair(pred, gold)
        results.append({"gold": gold, "pred": pred, "scores": sc})

        if sc["json_valid"]:
            json_valid += 1
        if sc["verdict_match"]:
            verdict_match += 1
        if sc["score_mae"] is not None:
            maes.append(sc["score_mae"])

        gv = gold.get("verdict", "").upper()
        if gv in by_verdict:
            by_verdict[gv]["total"] += 1
            if sc["verdict_match"]:
                by_verdict[gv]["match"] += 1

    n = len(results)
    print(f"\n{'='*60}")
    print(f"SwarmJudge-9B-B0-Phase1 — Eval Results")
    print(f"{'='*60}")
    print(f"  Samples:         {n}")
    print(f"  JSON valid:      {json_valid}/{n}  ({100*json_valid/n:.1f}%)")
    print(f"  Verdict match:   {verdict_match}/{n}  ({100*verdict_match/n:.1f}%)")
    print(f"  Score MAE:       {sum(maes)/len(maes):.3f}  (n={len(maes)})" if maes else "  Score MAE:       n/a")
    print(f"\n  By verdict:")
    for v, d in by_verdict.items():
        pct = 100*d["match"]/d["total"] if d["total"] else 0
        print(f"    {v:<6}  {d['match']:>4}/{d['total']:<4}  ({pct:.1f}%)")
    print(f"{'='*60}")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_utc":   datetime.now(timezone.utc).isoformat(),
        "model":           args.model,
        "eval_file":       args.eval_file,
        "n_samples":       n,
        "json_valid_pct":  round(100*json_valid/n, 2),
        "verdict_acc_pct": round(100*verdict_match/n, 2),
        "score_mae":       round(sum(maes)/len(maes), 3) if maes else None,
        "by_verdict":      {v: {"acc_pct": round(100*d["match"]/d["total"],2) if d["total"] else 0, **d}
                            for v, d in by_verdict.items()},
    }
    rpt_path = OUT_DIR / "eval_report.json"
    rpt_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report → {rpt_path}")

    # Save failed cases for analysis
    fails = [r for r in results if not r["scores"]["verdict_match"] or not r["scores"]["json_valid"]]
    if fails:
        fail_path = OUT_DIR / "eval_failures.jsonl"
        with open(fail_path, "w") as f:
            for r in fails[:100]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Failures → {fail_path}  ({len(fails)} cases, saved first 100)")


if __name__ == "__main__":
    main()
