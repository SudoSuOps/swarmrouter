"""
SwarmRouter-4B-0 — V3 Routing Schema
=====================================
Three primary verticals: SwarmMed | SwarmCRE | SwarmJudge
Nine domains. Ship with confidence, not guesswork.
"""

# ── Domains ────────────────────────────────────────────────────────
DOMAINS = [
    "medical",      # Clinical medicine, diagnostics, treatment — SwarmMed
    "pharma",       # Drug safety, DDI, PK/PD, dosing — SwarmPharma branch
    "cre",          # Commercial real estate, new economy — SwarmCRE
    "safety",       # Aviation ops, industrial safety, critical risk — SwarmAVA
    "technical",    # Software, engineering, infrastructure, ML/compute
    "business",     # Strategy, planning, operations, management
    "financial",    # Financial modeling, valuation, analysis (non-CRE)
    "legal",        # Contracts, regulations, compliance, case research
    "judge",        # Quality evaluation, scoring, drift, policy — SwarmJudge
]

# ── Task Types ─────────────────────────────────────────────────────
TASK_TYPES = [
    "qa",               # Question answering
    "summarization",    # Summarize documents/data
    "reasoning",        # Multi-step reasoning, analysis
    "generation",       # Content generation, writing
    "planning",         # Strategic planning, workflow design
    "triage",           # Urgency classification, risk triage
    "evaluation",       # Evaluate output quality, score results — routes to judge
    "extraction",       # Extract structured data from unstructured text
]

COMPLEXITY = ["low", "medium", "high"]
RISK_LEVELS = ["low", "medium", "high", "critical"]
LATENCY_TIERS = ["realtime", "interactive", "batch"]
COST_SENSITIVITY = ["low", "medium", "high"]

# ── Model Fleet ────────────────────────────────────────────────────
MODELS = [
    "router-4b",            # Self — fast, simple queries, technical/general low
    "research-8b",          # Mid-tier: business, financial, legal, technical medium
    "med-14b",              # SwarmMed — clinical, diagnostic, treatment
    "swarmpharma-35b",      # SwarmPharma — drug safety, DDI, PK/PD (always)
    "swarmcre-35b",         # SwarmCRE — CRE underwriting, new economy (always medium+)
    "swarmjudge-27b",       # SwarmJudge — quality eval, scoring, policy (always)
    "swarmresearch-32b",    # Deep research + safety/aviation (always)
]

TOOLS = [
    "none",
    "web_search",
    "file_search",
    "calculator",
    "code_exec",
    "vector_db",
    "sensor_read",
    "ledger_lookup",
    "document_parser",
]

# ── Routing Rules ──────────────────────────────────────────────────
ROUTING_RULES = {
    "medical": {
        "low":    {"risk_low": "research-8b",     "risk_high": "med-14b"},
        "medium": {"risk_low": "med-14b",          "risk_high": "med-14b"},
        "high":   {"risk_low": "med-14b",          "risk_high": "med-14b"},
    },
    "pharma": {
        # Always swarmpharma-35b — drug safety is always specialized
        "low":    {"risk_low": "swarmpharma-35b",  "risk_high": "swarmpharma-35b"},
        "medium": {"risk_low": "swarmpharma-35b",  "risk_high": "swarmpharma-35b"},
        "high":   {"risk_low": "swarmpharma-35b",  "risk_high": "swarmpharma-35b"},
    },
    "cre": {
        "low":    {"risk_low": "research-8b",      "risk_high": "swarmcre-35b"},
        "medium": {"risk_low": "swarmcre-35b",     "risk_high": "swarmcre-35b"},
        "high":   {"risk_low": "swarmcre-35b",     "risk_high": "swarmcre-35b"},
    },
    "safety": {
        # Aviation, industrial, critical safety — always elevated, never below 32b
        "low":    {"risk_low": "swarmresearch-32b","risk_high": "swarmresearch-32b"},
        "medium": {"risk_low": "swarmresearch-32b","risk_high": "swarmresearch-32b"},
        "high":   {"risk_low": "swarmresearch-32b","risk_high": "swarmresearch-32b"},
    },
    "technical": {
        "low":    {"risk_low": "router-4b",        "risk_high": "research-8b"},
        "medium": {"risk_low": "research-8b",      "risk_high": "research-8b"},
        "high":   {"risk_low": "research-8b",      "risk_high": "swarmresearch-32b"},
    },
    "business": {
        "low":    {"risk_low": "router-4b",        "risk_high": "research-8b"},
        "medium": {"risk_low": "research-8b",      "risk_high": "research-8b"},
        "high":   {"risk_low": "research-8b",      "risk_high": "swarmresearch-32b"},
    },
    "financial": {
        "low":    {"risk_low": "research-8b",      "risk_high": "research-8b"},
        "medium": {"risk_low": "research-8b",      "risk_high": "swarmresearch-32b"},
        "high":   {"risk_low": "swarmresearch-32b","risk_high": "swarmresearch-32b"},
    },
    "legal": {
        "low":    {"risk_low": "research-8b",      "risk_high": "research-8b"},
        "medium": {"risk_low": "research-8b",      "risk_high": "swarmresearch-32b"},
        "high":   {"risk_low": "swarmresearch-32b","risk_high": "swarmresearch-32b"},
    },
    "judge": {
        # Always swarmjudge-27b — quality eval, scoring, policy
        "low":    {"risk_low": "swarmjudge-27b",   "risk_high": "swarmjudge-27b"},
        "medium": {"risk_low": "swarmjudge-27b",   "risk_high": "swarmjudge-27b"},
        "high":   {"risk_low": "swarmjudge-27b",   "risk_high": "swarmjudge-27b"},
    },
}

SYSTEM_PROMPT = """You are SwarmRouter-4B-0, the intelligent routing brain of the Swarm & Bee AI platform.

Your ONLY job is to analyze user requests and output STRICT JSON routing decisions.

Three primary verticals:
- SwarmMed: clinical medicine (med-14b) + drug safety/pharma (swarmpharma-35b)
- SwarmCRE: commercial real estate + new economy assets (swarmcre-35b)
- SwarmJudge: quality evaluation, agent scoring, drift detection, policy (swarmjudge-27b)

Model fleet:
- router-4b: fast routing, simple technical/business queries
- research-8b: mid-tier business, financial, legal, technical
- med-14b: clinical medicine, diagnostics, treatment protocols
- swarmpharma-35b: drug safety, DDI, pharmacokinetics, dosing — ALWAYS for pharma
- swarmcre-35b: CRE underwriting, new economy (data centers, blockchain, last-mile)
- swarmjudge-27b: quality evaluation, agent scoring, output assessment, policy — ALWAYS for judge
- swarmresearch-32b: deep research, safety/aviation (all risk levels), complex multi-domain

Nine domains:
- medical: clinical medicine, diagnostics, treatment
- pharma: drug safety, interactions, dosing, pharmacokinetics
- cre: commercial real estate, property analysis, new economy assets
- safety: aviation operations, industrial safety, critical risk scenarios
- technical: software, engineering, ML/compute infrastructure
- business: strategy, operations, planning, management
- financial: financial modeling, valuation, analysis (non-CRE)
- legal: contracts, compliance, regulations, case research
- judge: quality evaluation, scoring, drift detection, policy checks

Output ONLY valid JSON matching this exact schema:

{
  "domain": "medical|pharma|cre|safety|technical|business|financial|legal|judge",
  "task_type": "qa|summarization|reasoning|generation|planning|triage|evaluation|extraction",
  "complexity": "low|medium|high",
  "risk_level": "low|medium|high|critical",
  "latency_tier": "realtime|interactive|batch",
  "cost_sensitivity": "low|medium|high",
  "recommended_model": "router-4b|research-8b|med-14b|swarmpharma-35b|swarmcre-35b|swarmjudge-27b|swarmresearch-32b",
  "escalation_allowed": true|false,
  "proposal_required": true|false,
  "requires_tools": ["none|web_search|file_search|calculator|code_exec|vector_db|sensor_read|ledger_lookup|document_parser"],
  "reasoning": "explanation under 120 chars"
}

Hard routing rules:
- pharma domain → ALWAYS swarmpharma-35b
- safety domain → ALWAYS swarmresearch-32b (no exceptions)
- judge domain → ALWAYS swarmjudge-27b
- cre medium/high complexity → ALWAYS swarmcre-35b
- Output ONLY valid JSON, no markdown, no extra text
- requires_tools must be array (use ["none"] if no tools needed)
- reasoning must be ≤120 characters

Proposal gate rules (proposal_required):
- pharma → ALWAYS true (drug safety, patient risk)
- safety → ALWAYS true (aviation, industrial, life risk)
- medical high risk_level or high complexity → true
- cre medium/high complexity → true (capital at stake)
- legal medium/high complexity → true (compliance risk)
- financial high complexity → true (capital at stake)
- technical/business/judge low/medium → false"""


def get_recommended_model(domain: str, complexity: str, risk_level: str) -> str:
    """Deterministic model selection from routing rules."""
    rules = ROUTING_RULES.get(domain, ROUTING_RULES["business"])
    comp_rules = rules.get(complexity, rules["medium"])
    key = "risk_high" if risk_level in ("high", "critical") else "risk_low"
    return comp_rules[key]


def needs_proposal(domain: str, complexity: str, risk_level: str) -> bool:
    """Deterministic proposal gate.

    Returns True if this job must be pre-approved by SwarmJudge-27B
    before execution. Hard gates for high-risk / high-cost domains.
    """
    # Hard always-true domains
    if domain in ("pharma", "safety"):
        return True

    # High-risk flag overrides
    if risk_level in ("high", "critical"):
        if domain in ("medical", "cre", "legal", "financial"):
            return True

    # Complexity-based gates
    if complexity in ("medium", "high"):
        if domain in ("cre", "legal"):
            return True
        if complexity == "high" and domain in ("medical", "financial", "technical"):
            return True

    return False
