"""
SwarmCRE Dataset Factory — Pydantic Output Schemas

12+ schemas for structured extraction tasks.
Shared by templates (rendering) and quality gates (validation).
"""

from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# RENT ROLL
# ═══════════════════════════════════════════════════════════════════


class RentRollLineItem(BaseModel):
    tenant_name: str
    suite: str
    sf: int = Field(ge=0)
    rent_psf: float = Field(ge=0, le=50.0)
    annual_rent: int = Field(ge=0)
    lease_start: str  # "YYYY-MM-DD"
    lease_end: str
    escalation_type: Literal["fixed", "cpi", "flat"]
    escalation_rate: float = Field(ge=0.0, le=0.10)
    lease_type: Literal["NNN", "modified_gross", "gross"]
    credit_tier: Literal["investment_grade", "national_credit", "local_credit", "startup"]


class RentRollJSON(BaseModel):
    property_name: str
    as_of_date: str
    total_sf: int = Field(ge=0)
    occupied_sf: int = Field(ge=0)
    vacancy_rate: float = Field(ge=0.0, le=1.0)
    num_tenants: int = Field(ge=0)
    walt_years: float = Field(ge=0.0)
    tenants: list[RentRollLineItem]


# ═══════════════════════════════════════════════════════════════════
# T-12 OPERATING STATEMENT
# ═══════════════════════════════════════════════════════════════════


class T12Revenue(BaseModel):
    base_rent: int
    cam_reimbursements: int = 0
    other_income: int = 0
    potential_gross_income: int
    vacancy_loss: int
    effective_gross_income: int


class T12Expenses(BaseModel):
    property_tax: int
    insurance: int
    cam_maintenance: int
    management_fee: int
    reserves: int
    utilities: int = 0
    total: int


class T12JSON(BaseModel):
    property_name: str
    period: str  # "Jan 2025 - Dec 2025"
    gross_sf: int = Field(ge=0)
    revenue: T12Revenue
    expenses: T12Expenses
    noi: int
    noi_per_sf: float


# ═══════════════════════════════════════════════════════════════════
# LEASE ABSTRACT
# ═══════════════════════════════════════════════════════════════════


class LeaseAbstractJSON(BaseModel):
    tenant_name: str
    premises: str
    sf: int = Field(ge=0)
    lease_type: Literal["NNN", "modified_gross", "gross"]
    term_start: str
    term_end: str
    base_rent_psf: float = Field(ge=0)
    annual_rent: int = Field(ge=0)
    escalation_type: Literal["fixed", "cpi", "flat"]
    escalation_rate: float = Field(ge=0.0, le=0.10)
    renewal_options: str
    tenant_improvements: str
    guarantor: str
    termination_rights: str
    use_restrictions: str
    key_provisions: list[str]


# ═══════════════════════════════════════════════════════════════════
# UNDERWRITING SUMMARY
# ═══════════════════════════════════════════════════════════════════


class UnderwritingSummaryJSON(BaseModel):
    property_name: str
    asset_type: str
    sf: int = Field(ge=0)
    year_built: int = Field(ge=0)
    clear_height_ft: int = Field(ge=0)
    dock_doors: int = Field(ge=0)
    potential_gross_income: int
    vacancy_rate: float = Field(ge=0.0, le=1.0)
    effective_gross_income: int
    total_opex: int
    noi: int
    cap_rate: float = Field(ge=0.0, le=0.20)
    value: int
    price_per_sf: float
    loan_amount: int
    ltv: float = Field(ge=0.0, le=1.0)
    annual_debt_service: int
    dscr: float = Field(ge=0.0)
    debt_yield: float = Field(ge=0.0)
    cash_on_cash: float


# ═══════════════════════════════════════════════════════════════════
# IC MEMO
# ═══════════════════════════════════════════════════════════════════


class ICMemoFinancials(BaseModel):
    purchase_price: int
    price_per_sf: float
    going_in_cap: float
    noi: int
    dscr: float
    ltv: float
    debt_yield: float
    cash_on_cash: float


class ICMemoJSON(BaseModel):
    deal_name: str
    executive_summary: str
    property_overview: str
    market_analysis: str
    tenant_analysis: str
    financial_summary: ICMemoFinancials
    assumptions: list[str]
    risk_factors: list[str]
    open_questions: list[str]
    recommendation: Literal["proceed", "caution", "kill"]
    recommendation_rationale: str


# ═══════════════════════════════════════════════════════════════════
# LOI / TERM SHEET
# ═══════════════════════════════════════════════════════════════════


class LOITermSheet(BaseModel):
    buyer: str
    seller: str
    property_name: str
    property_address: str
    purchase_price: int
    price_per_sf: float
    earnest_money: int
    earnest_money_pct: float
    due_diligence_days: int = Field(ge=0, le=180)
    closing_days: int = Field(ge=0, le=365)
    financing_contingency: bool
    inspection_contingency: bool
    title_contingency: bool
    environmental_contingency: bool
    conditions: list[str]


# ═══════════════════════════════════════════════════════════════════
# MARKET COMP
# ═══════════════════════════════════════════════════════════════════


class CompTransaction(BaseModel):
    property_name: str
    sf: int
    sale_price: int
    price_per_sf: float
    cap_rate: float
    sale_date: str
    distance_miles: float
    adjustment_notes: str


class MarketCompJSON(BaseModel):
    subject_property: str
    subject_sf: int
    subject_asking_price: int
    comps: list[CompTransaction]
    adjusted_value_range: str  # e.g. "$7.2M - $7.8M"
    analysis: str


# ═══════════════════════════════════════════════════════════════════
# RISK TRIAGE
# ═══════════════════════════════════════════════════════════════════


class RiskFactor(BaseModel):
    category: Literal[
        "environmental", "structural", "financial", "market",
        "tenant", "legal", "zoning", "operational"
    ]
    description: str
    severity: Literal["low", "medium", "high", "critical"]
    mitigation: str


class RiskTriageJSON(BaseModel):
    deal_name: str
    risk_score: Literal["low", "medium", "high", "critical"]
    deal_killers: list[str]
    risk_factors: list[RiskFactor]
    recommendation: Literal["proceed", "caution", "kill"]
    recommendation_rationale: str


# ═══════════════════════════════════════════════════════════════════
# SENSITIVITY TABLE
# ═══════════════════════════════════════════════════════════════════


class SensitivityScenario(BaseModel):
    scenario: str
    noi: int
    value: int
    dscr: float
    cash_on_cash: float


class SensitivityTableJSON(BaseModel):
    property_name: str
    base_case: SensitivityScenario
    scenarios: list[SensitivityScenario]


# ═══════════════════════════════════════════════════════════════════
# STRUCTURED AGENT OUTPUT
# ═══════════════════════════════════════════════════════════════════


class StructuredAgentOutput(BaseModel):
    task_type: str
    input_summary: str
    output: dict
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_chain: list[str]
    follow_up_actions: list[str]
    risk_flags: list[str] = []


# ═══════════════════════════════════════════════════════════════════
# DEBT SIZING SUMMARY
# ═══════════════════════════════════════════════════════════════════


class DebtSizingJSON(BaseModel):
    property_name: str
    noi: int
    lender_type: str
    loan_amount: int
    ltv: float
    rate: float
    amort_years: int
    io_years: int
    term_years: int
    annual_debt_service: int
    dscr: float
    debt_yield: float
    max_loan_by_ltv: int
    max_loan_by_dscr: int
    max_loan_by_debt_yield: int
    binding_constraint: str


# ═══════════════════════════════════════════════════════════════════
# BROKER OM SUMMARY
# ═══════════════════════════════════════════════════════════════════


class BrokerOMSummary(BaseModel):
    property_name: str
    property_type: str
    sf: int
    year_built: int
    location: str
    submarket: str
    highlights: list[str]
    noi: int
    cap_rate: float
    asking_price: int
    price_per_sf: float
    tenant_summary: str
    market_overview: str
    investment_thesis: str


# ═══════════════════════════════════════════════════════════════════
# 1031 EXCHANGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════


class Exchange1031JSON(BaseModel):
    exchange_type: Literal["simultaneous", "delayed", "reverse", "improvement"]
    relinquished_property: str
    relinquished_sale_price: int
    relinquished_adjusted_basis: int
    total_gain: int
    mortgage_boot: int = 0
    cash_boot: int = 0
    total_boot: int = 0
    recognized_gain: int = 0
    deferred_gain: int
    replacement_property: str
    replacement_cost: int
    replacement_new_basis: int
    identification_deadline: str
    closing_deadline: str


# ═══════════════════════════════════════════════════════════════════
# TAX ANALYSIS
# ═══════════════════════════════════════════════════════════════════


class TaxAnalysisJSON(BaseModel):
    property_name: str
    analysis_type: Literal[
        "depreciation_schedule", "cost_segregation",
        "sale_tax_liability", "depreciation_recapture"
    ]
    original_basis: int
    accumulated_depreciation: int = 0
    adjusted_basis: int = 0
    total_gain: int = 0
    recapture_amount: int = 0
    recapture_tax: int = 0
    ltcg_tax: int = 0
    niit_tax: int = 0
    total_tax: int = 0


# ═══════════════════════════════════════════════════════════════════
# SCHEMA REGISTRY — maps string names to classes for quality gates
# ═══════════════════════════════════════════════════════════════════

SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "RentRollJSON": RentRollJSON,
    "T12JSON": T12JSON,
    "LeaseAbstractJSON": LeaseAbstractJSON,
    "UnderwritingSummaryJSON": UnderwritingSummaryJSON,
    "ICMemoJSON": ICMemoJSON,
    "LOITermSheet": LOITermSheet,
    "MarketCompJSON": MarketCompJSON,
    "RiskTriageJSON": RiskTriageJSON,
    "SensitivityTableJSON": SensitivityTableJSON,
    "StructuredAgentOutput": StructuredAgentOutput,
    "DebtSizingJSON": DebtSizingJSON,
    "BrokerOMSummary": BrokerOMSummary,
    "Exchange1031JSON": Exchange1031JSON,
    "TaxAnalysisJSON": TaxAnalysisJSON,
}
