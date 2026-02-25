"""
Routing Physics - Labeling Rules for Dataset Generation
Deterministic rules for assigning routing labels
"""
from typing import Dict, List
import random


def apply_routing_physics(
    domain: str,
    user_message: str,
    task_type: str,
    complexity: str,
    keywords: List[str] = None
) -> Dict:
    """
    Apply SwarmRouter routing physics to generate ground truth labels

    Args:
        domain: Primary domain
        user_message: The user's request
        task_type: Type of task
        complexity: Task complexity
        keywords: Optional keywords from message

    Returns:
        Complete routing dict matching RouterOutput schema
    """

    # Initialize
    keywords = keywords or []
    risk_level = "low"
    recommended_model = "router-3b"
    requires_tools = ["none"]
    latency_tier = "interactive"
    cost_sensitivity = "medium"

    # Domain-specific routing physics
    if domain == "medical":
        recommended_model = "med-14b"
        risk_level = "high" if complexity in ["medium", "high"] else "medium"
        if any(kw in user_message.lower() for kw in ["diagnosis", "treatment", "emergency", "critical"]):
            risk_level = "critical"
        latency_tier = "interactive"
        cost_sensitivity = "low"  # Don't optimize cost in medical
        if task_type in ["summarization", "triage"]:
            requires_tools = ["vector_db"]

    elif domain == "aviation":
        recommended_model = "research-32b" if complexity == "high" else "research-8b"
        risk_level = "high" if complexity in ["medium", "high"] else "medium"
        if any(kw in user_message.lower() for kw in ["safety", "emergency", "failure", "crash"]):
            risk_level = "critical"
        latency_tier = "interactive"
        cost_sensitivity = "low"

    elif domain == "cre":
        recommended_model = "research-32b" if complexity == "high" else "research-8b"
        risk_level = "medium" if complexity in ["medium", "high"] else "low"
        if any(kw in user_message.lower() for kw in ["underwriting", "acquisition", "$"]):
            cost_sensitivity = "low"
        latency_tier = "batch" if task_type in ["summarization", "planning"] else "interactive"
        if task_type == "planning":
            requires_tools = ["calculator", "vector_db"]

    elif domain == "compute":
        if complexity == "high":
            recommended_model = "research-32b"
        elif complexity == "medium":
            recommended_model = "research-8b"
        else:
            recommended_model = "router-3b"

        risk_level = "high" if any(kw in user_message.lower() for kw in ["production", "deploy", "delete", "drop"]) else "medium"
        latency_tier = "interactive"
        cost_sensitivity = "medium"

        if task_type == "planning":
            requires_tools = ["calculator"]

    elif domain == "research":
        recommended_model = "research-32b" if complexity == "high" else "research-8b"
        risk_level = "low"
        latency_tier = "batch" if task_type in ["summarization", "reasoning"] else "interactive"
        cost_sensitivity = "high"  # Research can tolerate cost optimization

        if task_type in ["summarization", "research"]:
            requires_tools = ["web_search", "vector_db"]

    elif domain == "coding":
        if complexity == "high" or task_type == "generation":
            recommended_model = "research-8b"
        else:
            recommended_model = "router-3b"

        risk_level = "medium" if any(kw in user_message.lower() for kw in ["production", "deploy", "database"]) else "low"
        latency_tier = "interactive"
        cost_sensitivity = "medium"

        if task_type == "generation":
            requires_tools = ["code_exec"]

    elif domain == "operations":
        if complexity == "high":
            recommended_model = "research-8b"
        else:
            recommended_model = "router-3b"

        risk_level = "medium" if any(kw in user_message.lower() for kw in ["production", "critical", "outage"]) else "low"
        latency_tier = "realtime" if risk_level in ["high", "critical"] else "interactive"
        cost_sensitivity = "medium"

        if any(kw in user_message.lower() for kw in ["sensor", "temperature", "status"]):
            requires_tools = ["sensor_read"]
        elif any(kw in user_message.lower() for kw in ["ledger", "transaction", "block"]):
            requires_tools = ["ledger_lookup"]

    elif domain == "general":
        recommended_model = "router-3b"
        risk_level = "low"
        latency_tier = "interactive"
        cost_sensitivity = "high"
        requires_tools = ["none"]

    # Escalation logic
    escalation_allowed = recommended_model != "router-3b" or risk_level in ["high", "critical"]

    # Generate reasoning
    reasoning = generate_reasoning(
        domain=domain,
        recommended_model=recommended_model,
        complexity=complexity,
        risk_level=risk_level
    )

    return {
        "domain": domain,
        "task_type": task_type,
        "complexity": complexity,
        "risk_level": risk_level,
        "latency_tier": latency_tier,
        "cost_sensitivity": cost_sensitivity,
        "recommended_model": recommended_model,
        "escalation_allowed": escalation_allowed,
        "requires_tools": requires_tools,
        "reasoning": reasoning
    }


def generate_reasoning(domain: str, recommended_model: str, complexity: str, risk_level: str) -> str:
    """Generate brief routing reasoning (<120 chars)"""

    reasons = {
        "medical": [
            "Medical query requires specialist model",
            "Clinical reasoning needs med-14b",
            "Healthcare domain, route to medical specialist",
        ],
        "aviation": [
            "Aviation safety requires specialized model",
            "Flight ops query needs domain expert",
            "Regulatory/safety concern, use research model",
        ],
        "cre": [
            "Commercial real estate analysis task",
            "CRE underwriting requires research model",
            "Property analysis needs deep reasoning",
        ],
        "compute": [
            f"{complexity.capitalize()} compute task, use {recommended_model}",
            "Infrastructure query for compute specialist",
            "Technical depth requires larger model",
        ],
        "research": [
            "Research synthesis task",
            "Deep analysis requires research model",
            "Multi-source reasoning needed",
        ],
        "coding": [
            f"{complexity.capitalize()} coding task",
            "Code generation needs research model",
            "Simple code query for router",
        ],
        "operations": [
            f"Ops task, {risk_level} risk",
            "Operational query",
            "Real-time ops response needed",
        ],
        "general": [
            "General QA, router handles",
            "Simple query for router-3b",
            "Low complexity general task",
        ],
    }

    reason = random.choice(reasons.get(domain, ["Route to appropriate model"]))
    return reason[:119]  # Enforce <120 char limit


def infer_complexity(user_message: str, task_type: str) -> str:
    """Infer complexity from message characteristics"""

    msg_lower = user_message.lower()
    word_count = len(user_message.split())

    # High complexity indicators
    if any(kw in msg_lower for kw in ["design", "architecture", "implement", "comprehensive", "detailed", "analyze deeply"]):
        return "high"

    if word_count > 50:
        return "high"

    if task_type in ["planning", "reasoning", "generation"]:
        if word_count > 30:
            return "high"
        elif word_count > 15:
            return "medium"

    # Medium complexity
    if any(kw in msg_lower for kw in ["compare", "explain", "how does", "why", "multi"]):
        return "medium"

    if word_count > 20:
        return "medium"

    # Default low
    return "low"


def infer_task_type(user_message: str) -> str:
    """Infer task type from user message"""

    msg_lower = user_message.lower()

    if any(kw in msg_lower for kw in ["summarize", "summary", "tldr", "brief"]):
        return "summarization"

    if any(kw in msg_lower for kw in ["plan", "design", "strategy", "roadmap"]):
        return "planning"

    if any(kw in msg_lower for kw in ["generate", "create", "write", "build"]):
        return "generation"

    if any(kw in msg_lower for kw in ["why", "how does", "explain", "analyze"]):
        return "reasoning"

    if any(kw in msg_lower for kw in ["triage", "route", "classify", "categorize"]):
        return "triage"

    # Default
    return "qa"
