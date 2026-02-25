"""
SwarmRouter JSON Schema & Validators
Pydantic models for strict routing output validation
"""
from typing import List, Literal
from pydantic import BaseModel, Field, validator


class RouterOutput(BaseModel):
    """Strict JSON schema for SwarmRouter output"""

    domain: Literal[
        "general", "coding", "operations", "research",
        "medical", "cre", "aviation", "compute"
    ] = Field(..., description="Primary domain of the request")

    task_type: Literal[
        "qa", "summarization", "reasoning",
        "generation", "planning", "triage"
    ] = Field(..., description="Type of task requested")

    complexity: Literal["low", "medium", "high"] = Field(
        ..., description="Task complexity level"
    )

    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        ..., description="Risk/safety level of task"
    )

    latency_tier: Literal["realtime", "interactive", "batch"] = Field(
        ..., description="Expected response latency tier"
    )

    cost_sensitivity: Literal["low", "medium", "high"] = Field(
        ..., description="Cost optimization priority"
    )

    recommended_model: Literal[
        "router-3b", "research-8b", "med-14b", "research-32b"
    ] = Field(..., description="Model to handle this request")

    escalation_allowed: bool = Field(
        ..., description="Whether escalation to larger model is permitted"
    )

    requires_tools: List[Literal[
        "none", "web_search", "file_search", "calculator",
        "code_exec", "vector_db", "sensor_read", "ledger_lookup"
    ]] = Field(..., description="Required external tools")

    reasoning: str = Field(
        ..., max_length=120, description="Brief routing explanation"
    )

    @validator('reasoning')
    def validate_reasoning_length(cls, v):
        if len(v) > 120:
            raise ValueError("reasoning must be ≤120 characters")
        return v

    @validator('requires_tools')
    def validate_tools_not_empty(cls, v):
        if not v:
            raise ValueError("requires_tools must have at least one item")
        if len(v) == 0:
            return ["none"]
        return v

    class Config:
        extra = "forbid"  # No additional keys allowed
        json_schema_extra = {
            "example": {
                "domain": "medical",
                "task_type": "triage",
                "complexity": "high",
                "risk_level": "critical",
                "latency_tier": "interactive",
                "cost_sensitivity": "low",
                "recommended_model": "med-14b",
                "escalation_allowed": False,
                "requires_tools": ["vector_db"],
                "reasoning": "Complex medical query requires specialist model"
            }
        }


def validate_router_output(json_str: str) -> tuple[bool, RouterOutput | dict]:
    """
    Validate raw JSON string against RouterOutput schema

    Returns:
        (is_valid, parsed_output_or_error_dict)
    """
    import json

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, {"error": "invalid_json", "details": str(e)}

    try:
        output = RouterOutput(**data)
        return True, output
    except Exception as e:
        return False, {"error": "schema_validation_failed", "details": str(e)}


def format_system_prompt() -> str:
    """System prompt for router model"""
    return """You are SwarmRouter, a specialized routing model that analyzes user requests and outputs STRICT JSON routing decisions.

Your ONLY job is to output valid JSON matching this exact schema:

{
  "domain": "general|coding|operations|research|medical|cre|aviation|compute",
  "task_type": "qa|summarization|reasoning|generation|planning|triage",
  "complexity": "low|medium|high",
  "risk_level": "low|medium|high|critical",
  "latency_tier": "realtime|interactive|batch",
  "cost_sensitivity": "low|medium|high",
  "recommended_model": "router-3b|research-8b|med-14b|research-32b",
  "escalation_allowed": true|false,
  "requires_tools": ["none|web_search|file_search|calculator|code_exec|vector_db|sensor_read|ledger_lookup"],
  "reasoning": "explanation under 120 chars"
}

Rules:
- Output ONLY valid JSON, no markdown, no extra text
- All keys required
- requires_tools must be array (use ["none"] if no tools needed)
- reasoning must be ≤120 characters

Routing logic:
- Medical → med-14b (risk ≥ high)
- Aviation → research-8b/32b (risk ≥ high)
- CRE → research-8b/32b (risk medium/high)
- Research → research-8b/32b
- Coding → router-3b or research-8b
- Compute/infra → router-3b/8b/32b by complexity
- General → router-3b"""


def format_user_prompt(user_message: str) -> str:
    """Format user message for routing"""
    return f"Route this request:\n\n{user_message}"
