"""
Dataset Entropy Report — Proves your data isn't repetitive.

Measures diversity across 4 dimensions:
    1. Vocabulary entropy  (Shannon entropy of word distribution)
    2. Unique sentence structures (POS-tag-like structural fingerprints)
    3. Bigram diversity    (unique word pairs / total word pairs)
    4. Opening diversity   (unique first-10-token patterns)

Usage:
    python3 -m data.factory.entropy path/to/train.jsonl
    python3 -m data.factory.entropy path/to/train.jsonl --top-repeats 20
    python3 -m data.factory.entropy path/to/train.jsonl --json
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# Text extraction
# ═══════════════════════════════════════════════════════════════════════

def _extract_text(record: dict) -> tuple[str, str]:
    """Extract (question, answer) text from a training record."""
    messages = record.get("messages", [])
    question = ""
    answer = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            question = content
        elif role == "assistant":
            answer = content
    return question, answer


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer. No dependencies."""
    return re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?", text.lower())


# ═══════════════════════════════════════════════════════════════════════
# Structural fingerprinting
# ═══════════════════════════════════════════════════════════════════════

# Lightweight POS-like buckets (no spaCy needed)
_NUM_PAT = re.compile(r'^[\d,$%.]+$')
_STOP = frozenset([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
    'not', 'no', 'if', 'then', 'than', 'that', 'this', 'it', 'its',
    'they', 'their', 'them', 'we', 'our', 'you', 'your', 'he', 'she',
])


def _tag(word: str) -> str:
    """Cheap POS-like tag for structural fingerprinting."""
    if _NUM_PAT.match(word):
        return 'NUM'
    if word in _STOP:
        return 'STOP'
    if len(word) <= 2:
        return 'SHORT'
    return 'WORD'


def _structural_fingerprint(text: str) -> str:
    """
    Sentence structure fingerprint.
    Split into sentences, take first 3, tag each word, join.
    """
    sentences = re.split(r'[.!?]\s+', text[:500])[:3]
    parts = []
    for sent in sentences:
        tokens = _tokenize(sent)[:12]
        tags = [_tag(t) for t in tokens]
        parts.append('-'.join(tags))
    return ' | '.join(parts)


def _opening_fingerprint(text: str, n: int = 10) -> str:
    """First n tokens, lowercased. Catches repetitive openings."""
    tokens = _tokenize(text)[:n]
    return ' '.join(tokens)


# ═══════════════════════════════════════════════════════════════════════
# Entropy calculation
# ═══════════════════════════════════════════════════════════════════════

def _shannon_entropy(counter: Counter) -> float:
    """Shannon entropy in bits, normalized to [0, 1]."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    n_unique = len(counter)
    if n_unique <= 1:
        return 0.0
    max_entropy = math.log2(n_unique)
    if max_entropy == 0:
        return 0.0
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in counter.values()
        if count > 0
    )
    return entropy / max_entropy


# ═══════════════════════════════════════════════════════════════════════
# Main report
# ═══════════════════════════════════════════════════════════════════════

def entropy_report(input_path: str, top_repeats: int = 10) -> dict:
    """
    Run full entropy analysis on a JSONL dataset.

    Returns dict with all metrics + optional top repeated patterns.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    word_counter = Counter()
    bigram_counter = Counter()
    structure_counter = Counter()
    opening_counter_q = Counter()
    opening_counter_a = Counter()

    total_pairs = 0
    total_words_q = 0
    total_words_a = 0
    total_bigrams = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            question, answer = _extract_text(record)
            if not answer:
                continue

            total_pairs += 1

            # Tokenize
            q_tokens = _tokenize(question)
            a_tokens = _tokenize(answer)
            all_tokens = q_tokens + a_tokens

            total_words_q += len(q_tokens)
            total_words_a += len(a_tokens)

            # Word frequency
            word_counter.update(all_tokens)

            # Bigrams (answer only — that's what the model learns)
            for i in range(len(a_tokens) - 1):
                bg = (a_tokens[i], a_tokens[i + 1])
                bigram_counter[bg] += 1
                total_bigrams += 1

            # Structural fingerprint (answer)
            fp = _structural_fingerprint(answer)
            structure_counter[fp] += 1

            # Opening patterns
            opening_counter_q[_opening_fingerprint(question)] += 1
            opening_counter_a[_opening_fingerprint(answer)] += 1

    # Compute metrics
    vocab_entropy = _shannon_entropy(word_counter)
    bigram_entropy = _shannon_entropy(bigram_counter)
    structure_entropy = _shannon_entropy(structure_counter)

    unique_structures = len(structure_counter)
    unique_vocab = len(word_counter)
    unique_bigrams = len(bigram_counter)
    unique_openings_q = len(opening_counter_q)
    unique_openings_a = len(opening_counter_a)

    # Bigram diversity ratio
    bigram_diversity = unique_bigrams / total_bigrams if total_bigrams > 0 else 0

    # Top repeated openings (detect template problems)
    top_q_openings = opening_counter_q.most_common(top_repeats)
    top_a_openings = opening_counter_a.most_common(top_repeats)
    top_structures = structure_counter.most_common(top_repeats)

    # Repetition concentration: what % of pairs use the top 10 structures?
    top10_structure_count = sum(c for _, c in structure_counter.most_common(10))
    structure_concentration = top10_structure_count / total_pairs if total_pairs > 0 else 0

    # ── Combined health score (0-100) ──
    conc_score = max(0, 1.0 - (structure_concentration / 0.60))
    health_score = round(
        vocab_entropy * 25
        + structure_entropy * 25
        + conc_score * 30
        + bigram_entropy * 20
    )
    health_score = max(0, min(100, health_score))

    if health_score >= 80:
        health_grade = "EXCELLENT"
    elif health_score >= 60:
        health_grade = "HEALTHY"
    elif health_score >= 40:
        health_grade = "WEAK"
    else:
        health_grade = "CRITICAL"

    return {
        "file": str(path),
        "pairs": total_pairs,
        "health_score": health_score,
        "health_grade": health_grade,
        "total_words": total_words_q + total_words_a,
        "total_words_question": total_words_q,
        "total_words_answer": total_words_a,
        "avg_question_len": total_words_q / total_pairs if total_pairs > 0 else 0,
        "avg_answer_len": total_words_a / total_pairs if total_pairs > 0 else 0,
        "unique_vocab": unique_vocab,
        "vocab_entropy": round(vocab_entropy, 4),
        "unique_bigrams": unique_bigrams,
        "bigram_entropy": round(bigram_entropy, 4),
        "bigram_diversity": round(bigram_diversity, 4),
        "unique_structures": unique_structures,
        "structure_entropy": round(structure_entropy, 4),
        "structure_concentration_top10": round(structure_concentration, 4),
        "unique_openings_question": unique_openings_q,
        "unique_openings_answer": unique_openings_a,
        "top_question_openings": top_q_openings,
        "top_answer_openings": top_a_openings,
        "top_structures": top_structures,
    }


def _use_color() -> bool:
    """Auto-detect color support. Disabled if piped or NO_COLOR set."""
    import os
    if os.environ.get('NO_COLOR'):
        return False
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


# ANSI colors (zero deps)
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _c(code: str, text: str, color: bool) -> str:
    """Wrap text in ANSI code if color enabled."""
    return f"{code}{text}{_RESET}" if color else text


def _bar(value: float, max_value: float = 100, width: int = 40) -> str:
    """ASCII bar for 0-max scale."""
    filled = int(width * value / max_value) if max_value > 0 else 0
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _bar01(value: float, width: int = 40) -> str:
    """ASCII bar for 0.0-1.0 scale."""
    filled = int(width * value)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _grade_color(grade: str) -> str:
    """Return ANSI code for health grade."""
    if grade == "EXCELLENT":
        return _GREEN
    if grade == "HEALTHY":
        return _CYAN
    if grade == "WEAK":
        return _YELLOW
    return _RED


def _metric_color(value: float, good: float, warn: float) -> str:
    """Return ANSI code based on threshold (higher is better)."""
    if value >= good:
        return _GREEN
    if value >= warn:
        return _YELLOW
    return _RED


def _metric_color_inv(value: float, good: float, warn: float) -> str:
    """Return ANSI code based on threshold (lower is better)."""
    if value <= good:
        return _GREEN
    if value <= warn:
        return _YELLOW
    return _RED


def print_report(report: dict, color: bool | None = None) -> None:
    """Pretty-print entropy report with ASCII health bars."""
    if color is None:
        color = _use_color()

    hs = report['health_score']
    hg = report['health_grade']
    gc = _grade_color(hg)
    ve = report['vocab_entropy']
    be = report['bigram_entropy']
    se = report['structure_entropy']
    bd = report['bigram_diversity']
    sc = report['structure_concentration_top10']

    print()
    print(_c(_BOLD, "Dataset Health Report", color))
    print("═" * 56)
    print(f"  {_c(_DIM, 'file:', color)}  {report['file']}")
    print(f"  {_c(_DIM, 'pairs:', color)} {report['pairs']:,}")
    print()

    # ── Health Score Bar ──
    print(_c(_BOLD, "  Health Score", color))
    bar_str = _bar(hs, 100, 44)
    print(f"  [{_c(gc, bar_str, color)}] {hs}/100  {_c(gc, hg, color)}")
    print()

    # ── Entropy Metrics Bars ──
    print(_c(_BOLD, "  Entropy Metrics", color))
    print("  " + "─" * 52)

    vc = _metric_color(ve, 0.80, 0.65)
    print(f"  Vocabulary    [{_c(vc, _bar01(ve, 36), color)}] {ve:.4f}")

    sc2 = _metric_color(se, 0.75, 0.55)
    print(f"  Structure     [{_c(sc2, _bar01(se, 36), color)}] {se:.4f}")

    bc = _metric_color(be, 0.70, 0.50)
    print(f"  Bigram        [{_c(bc, _bar01(be, 36), color)}] {be:.4f}")

    dc = _metric_color(bd, 0.30, 0.15)
    print(f"  Bigram Div    [{_c(dc, _bar01(bd, 36), color)}] {bd:.4f}")
    print()

    # ── Concentration Bar ──
    print(_c(_BOLD, "  Top-10 Concentration", color))
    print("  " + "─" * 52)
    conc_pct = sc * 100
    cc = _metric_color_inv(sc, 0.15, 0.30)
    conc_label = "LOW" if sc <= 0.15 else ("MED" if sc <= 0.30 else "HIGH")
    print(f"  [{_c(cc, _bar(conc_pct, 100, 36), color)}] {conc_pct:.1f}%  ({_c(cc, conc_label, color)})")
    print()

    # ── Vocabulary Stats ──
    print(_c(_BOLD, "  Vocabulary", color))
    print("  " + "─" * 52)
    print(f"  unique words:              {report['unique_vocab']:,}")
    print(f"  unique bigrams:            {report['unique_bigrams']:,}")
    print(f"  unique structures:         {report['unique_structures']:,}")
    print(f"  unique question openings:  {report['unique_openings_question']:,}")
    print(f"  unique answer openings:    {report['unique_openings_answer']:,}")
    print(f"  avg question length:       {report['avg_question_len']:.0f} words")
    print(f"  avg answer length:         {report['avg_answer_len']:.0f} words")
    print()

    # ── Top Repeated Openings ──
    print(_c(_BOLD, "  Top Repeated Answer Openings", color))
    print("  " + "─" * 52)
    pairs = report['pairs'] or 1
    for opening, count in report['top_answer_openings'][:10]:
        pct = count / pairs * 100
        bw = min(int(pct * 0.8), 24)
        bar_str = "█" * bw + "░" * (24 - bw)
        oc = _RED if pct > 10 else (_YELLOW if pct > 5 else _DIM)
        print(f"  {count:5d} ({pct:4.1f}%) {_c(oc, bar_str, color)} {opening[:45]}")

    print()
    print("═" * 56)
    print()


def print_compare(reports: list[dict], color: bool | None = None) -> None:
    """Side-by-side comparison of multiple dataset reports with bars."""
    if color is None:
        color = _use_color()

    # Find max name length for alignment
    names = []
    for r in reports:
        name = Path(r['file']).stem
        if len(name) > 20:
            name = name[:17] + "..."
        names.append(name)
    pad = max(len(n) for n in names) + 2

    print()
    print(_c(_BOLD, "Dataset Comparison", color))
    print("═" * (pad + 52))
    print()

    # Health score bars
    print(_c(_BOLD, "  Health Score", color))
    print("  " + "─" * (pad + 48))
    for name, r in zip(names, reports):
        hs = r['health_score']
        gc = _grade_color(r['health_grade'])
        bar_str = _bar(hs, 100, 36)
        print(f"  {name:<{pad}} [{_c(gc, bar_str, color)}] {hs:3d}  {_c(gc, r['health_grade'], color)}")
    print()

    # Entropy comparison
    for label, key, good, warn in [
        ("Vocabulary Entropy", "vocab_entropy", 0.80, 0.65),
        ("Structure Entropy", "structure_entropy", 0.75, 0.55),
        ("Bigram Entropy", "bigram_entropy", 0.70, 0.50),
    ]:
        print(f"  {_c(_BOLD, label, color)}")
        for name, r in zip(names, reports):
            v = r[key]
            mc = _metric_color(v, good, warn)
            print(f"  {name:<{pad}} [{_c(mc, _bar01(v, 36), color)}] {v:.4f}")
        print()

    # Concentration
    print(f"  {_c(_BOLD, 'Top-10 Concentration', color)}")
    for name, r in zip(names, reports):
        sc = r['structure_concentration_top10']
        pct = sc * 100
        cc = _metric_color_inv(sc, 0.15, 0.30)
        label = "LOW" if sc <= 0.15 else ("MED" if sc <= 0.30 else "HIGH")
        print(f"  {name:<{pad}} [{_c(cc, _bar(pct, 100, 36), color)}] {pct:4.1f}% {_c(cc, label, color)}")

    print()
    print("═" * (pad + 52))
    print()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Dataset entropy report — zero deps, visual health bars"
    )
    parser.add_argument("input", nargs="+", help="Path to JSONL file(s)")
    parser.add_argument("--top-repeats", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--compare", action="store_true",
                        help="Side-by-side comparison (when multiple files given)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    args = parser.parse_args()

    use_color = not args.no_color

    reports = []
    for path in args.input:
        reports.append(entropy_report(path, top_repeats=args.top_repeats))

    if args.json:
        out = []
        for report in reports:
            r = dict(report)
            r["top_question_openings"] = [
                {"opening": o, "count": c} for o, c in r["top_question_openings"]
            ]
            r["top_answer_openings"] = [
                {"opening": o, "count": c} for o, c in r["top_answer_openings"]
            ]
            r["top_structures"] = [
                {"structure": s, "count": c} for s, c in r["top_structures"]
            ]
            out.append(r)
        print(json.dumps(out[0] if len(out) == 1 else out, indent=2))
    elif args.compare and len(reports) > 1:
        print_compare(reports, color=use_color)
    else:
        for report in reports:
            print_report(report, color=use_color)


if __name__ == "__main__":
    main()
