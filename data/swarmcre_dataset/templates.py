"""
SwarmCRE Dataset Factory — Template Registry

Extensible template system for generating Q&A training pairs.
Each template maps a deal skeleton to a (question, answer) pair.

The builder.py has its own inline task builders for all 10 task types.
This module provides a TemplateRegistry for tasks.py as an alternative
extensible approach. Templates are grouped into 7 buckets (A-G).

Bucket A: Underwriting calculations
Bucket B: Rent roll extraction
Bucket C: T-12 normalization
Bucket D: IC memo / risk triage
Bucket E: Market comps
Bucket F: Deal structure (lease reasoning, LOI)
Bucket G: Agent ops (structured output)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Callable, Optional


# ═══════════════════════════════════════════════════════════════════
# TEMPLATE DATACLASS
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Template:
    """A single Q&A template that renders against a deal skeleton."""

    template_id: str
    task_type: str
    difficulty: str  # "low", "medium", "high"
    gold_fields: list[str] = field(default_factory=list)
    output_schema: Optional[str] = None
    requires_enrichment: bool = False
    _render_fn: Optional[Callable] = field(default=None, repr=False)

    def render(self, deal_dict: dict) -> tuple[str, str]:
        """Render the template against a deal skeleton.

        Returns:
            (question, answer) tuple of strings.
        """
        if self._render_fn is None:
            return "", ""
        return self._render_fn(deal_dict)


# ═══════════════════════════════════════════════════════════════════
# TEMPLATE REGISTRY
# ═══════════════════════════════════════════════════════════════════


class TemplateRegistry:
    """Registry of templates indexed by (task_type, difficulty)."""

    def __init__(self):
        self._by_type_diff: dict[tuple[str, str], list[Template]] = defaultdict(list)
        self._all: list[Template] = []

    def register(self, template: Template) -> None:
        key = (template.task_type, template.difficulty)
        self._by_type_diff[key].append(template)
        self._all.append(template)

    def get_templates(self, task_type: str, difficulty: str) -> list[Template]:
        return self._by_type_diff.get((task_type, difficulty), [])

    @property
    def all_templates(self) -> list[Template]:
        return list(self._all)

    def __len__(self) -> int:
        return len(self._all)

    def summary(self) -> dict[str, dict[str, int]]:
        result = defaultdict(lambda: defaultdict(int))
        for t in self._all:
            result[t.task_type][t.difficulty] += 1
        return {k: dict(v) for k, v in result.items()}


# ═══════════════════════════════════════════════════════════════════
# HELPER — safe deal field access
# ═══════════════════════════════════════════════════════════════════


def _g(d: dict, *keys, default=0):
    """Nested get with default."""
    v = d
    for k in keys:
        if isinstance(v, dict):
            v = v.get(k, default)
        else:
            return default
    return v


def _rent_lines(deal: dict) -> str:
    return "\n".join(
        f"- {t['name']}: {t['sf']:,} SF at ${t['rent_psf']:.2f}/SF"
        for t in deal.get("rent_roll", [])
    )


def _rent_lines_detail(deal: dict) -> str:
    return "\n".join(
        f"- {t['name']} ({t.get('suite', 'N/A')}): {t['sf']:,} SF at "
        f"${t['rent_psf']:.2f}/SF, {t['lease_type']}, "
        f"{t['escalation_type']} escalation"
        + (f" @ {t['escalation_rate']:.1%}" if t['escalation_type'] == 'fixed' else "")
        + f", expires {t['lease_end']}"
        for t in deal.get("rent_roll", [])
    )


def _expense_summary(deal: dict) -> str:
    exp = _g(deal, "gold", "expense_lines", default={})
    if not isinstance(exp, dict):
        exp = deal.get("expense_lines", {})
    parts = []
    for k in ["property_tax", "insurance", "cam_maintenance", "reserves", "utilities"]:
        v = exp.get(k, 0)
        if v:
            parts.append(f"{k.replace('_', ' ').title()}: ${v:,}")
    return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# BUCKET A — UNDERWRITING CALCULATIONS
# ═══════════════════════════════════════════════════════════════════


def _a001_pgi(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    q = (
        f"{d['property_name']} is a {d['sf']:,} SF {d['asset_type_display']} "
        f"in {d['submarket']}, {d['market_name']}. Tenants:\n"
        f"{_rent_lines(d)}\n\nCalculate the Potential Gross Income (PGI)."
    )
    lines = "\n".join(
        f"  {t['name']}: {t['sf']:,} SF x ${t['rent_psf']:.2f}/SF = ${t['annual_rent']:,}"
        for t in d.get("rent_roll", [])
    )
    a = f"PGI = sum of all annual rents\n{lines}\n\nPGI = ${g.get('pgi', 0):,}"
    return q, a


def _a002_noi_simple(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    q = (
        f"{d['property_name']} has an EGI of ${g.get('egi', 0):,} and total operating "
        f"expenses of ${g.get('total_opex', 0):,}. Calculate the NOI."
    )
    a = (
        f"NOI = EGI - Total OpEx\n"
        f"NOI = ${g.get('egi', 0):,} - ${g.get('total_opex', 0):,}\n"
        f"NOI = ${g.get('noi', 0):,}"
    )
    return q, a


def _a003_value(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    q = (
        f"{d['property_name']} generates NOI of ${g.get('noi', 0):,}. "
        f"Using a {g.get('cap_rate', 0.06):.2%} cap rate, what is the value?"
    )
    a = (
        f"Value = NOI / Cap Rate\n"
        f"Value = ${g.get('noi', 0):,} / {g.get('cap_rate', 0.06):.4f}\n"
        f"Value = ${g.get('value', 0):,}"
    )
    return q, a


def _a004_vacancy_loss(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    q = (
        f"The PGI for {d['property_name']} is ${g.get('pgi', 0):,}. "
        f"The market vacancy rate is {d['vacancy_rate']:.1%}. "
        f"Calculate the vacancy/credit loss and EGI."
    )
    a = (
        f"Vacancy loss = PGI x vacancy rate\n"
        f"Vacancy loss = ${g.get('pgi', 0):,} x {d['vacancy_rate']:.3f} = ${g.get('vacancy_loss', 0):,}\n\n"
        f"EGI = PGI - Vacancy loss\n"
        f"EGI = ${g.get('pgi', 0):,} - ${g.get('vacancy_loss', 0):,} = ${g.get('egi', 0):,}"
    )
    return q, a


def _a005_dscr(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    q = (
        f"{d['property_name']} has NOI of ${g.get('noi', 0):,} and annual debt service "
        f"of ${g.get('annual_debt_service', 0):,}. Calculate the DSCR."
    )
    a = (
        f"DSCR = NOI / Annual Debt Service\n"
        f"DSCR = ${g.get('noi', 0):,} / ${g.get('annual_debt_service', 0):,}\n"
        f"DSCR = {g.get('dscr', 0):.2f}x"
    )
    return q, a


def _a006_price_psf(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    q = (
        f"A {d['sf']:,} SF property is valued at ${g.get('value', 0):,}. "
        f"What is the price per square foot?"
    )
    a = (
        f"Price/SF = Value / SF\n"
        f"Price/SF = ${g.get('value', 0):,} / {d['sf']:,}\n"
        f"Price/SF = ${g.get('price_per_sf', 0):.2f}"
    )
    return q, a


def _a010_multistep(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    vacancy = d.get("vacancy_rate", 0.05)
    q = (
        f"{d['property_name']} is a {d['sf']:,} SF {d['asset_type_display']}.\n\n"
        f"Rent roll:\n{_rent_lines(d)}\n\n"
        f"Vacancy: {vacancy:.1%}\n"
        f"Expenses: {_expense_summary(d)}\n"
        f"Management fee: {d.get('management_fee_pct', 0.04):.1%} of EGI\n"
        f"Cap rate: {g.get('cap_rate', 0.06):.2%}\n\n"
        f"Calculate: PGI, EGI, total OpEx, NOI, and property value."
    )
    a = (
        f"Step 1: PGI = ${g.get('pgi', 0):,}\n"
        f"Step 2: EGI = PGI x (1 - {vacancy:.3f}) = ${g.get('egi', 0):,}\n"
        f"Step 3: Management fee = EGI x {d.get('management_fee_pct', 0.04):.3f} = ${g.get('management_fee', 0):,}\n"
        f"Step 4: Total OpEx = ${g.get('total_opex', 0):,}\n"
        f"Step 5: NOI = EGI - OpEx = ${g.get('noi', 0):,}\n"
        f"Step 6: Value = NOI / Cap Rate = ${g.get('value', 0):,}\n"
        f"Price per SF = ${g.get('price_per_sf', 0):.2f}"
    )
    return q, a


def _a011_debt_yield(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    q = (
        f"{d['property_name']} has NOI of ${g.get('noi', 0):,} and a loan of "
        f"${g.get('loan_amount', 0):,}. Calculate the debt yield and LTV."
    )
    a = (
        f"Debt Yield = NOI / Loan Amount\n"
        f"Debt Yield = ${g.get('noi', 0):,} / ${g.get('loan_amount', 0):,}\n"
        f"Debt Yield = {g.get('debt_yield', 0):.2%}\n\n"
        f"LTV = Loan / Value\n"
        f"LTV = ${g.get('loan_amount', 0):,} / ${g.get('value', 0):,}\n"
        f"LTV = {g.get('ltv', 0):.1%}"
    )
    return q, a


def _a020_full_underwriting(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    debt = d.get("debt", {})
    vacancy = d.get("vacancy_rate", 0.05)
    q = (
        f"Perform a complete underwriting for {d['property_name']}, "
        f"a {d['sf']:,} SF {d['asset_type_display']} in {d['submarket']}, {d['market_name']}.\n\n"
        f"Rent roll:\n{_rent_lines_detail(d)}\n\n"
        f"Vacancy: {vacancy:.1%}\n"
        f"Expenses: {_expense_summary(d)}\n"
        f"Management fee: {d.get('management_fee_pct', 0.04):.1%} of EGI\n"
        f"Cap rate: {g.get('cap_rate', 0.06):.2%}\n\n"
        f"Debt: {debt.get('lender_display', 'N/A')} — "
        f"${debt.get('loan_amount', 0):,} at {debt.get('rate', 0):.2%}, "
        f"{debt.get('amort_years', 0)}-yr amort, {debt.get('io_years', 0)}-yr IO\n\n"
        f"Calculate all metrics: PGI, EGI, NOI, value, price/SF, "
        f"annual debt service, DSCR, debt yield, LTV, equity, cash-on-cash."
    )
    a = (
        f"REVENUE\n"
        f"  PGI: ${g.get('pgi', 0):,}\n"
        f"  Vacancy loss ({vacancy:.1%}): ${g.get('vacancy_loss', 0):,}\n"
        f"  EGI: ${g.get('egi', 0):,}\n\n"
        f"EXPENSES\n"
        f"  Management fee: ${g.get('management_fee', 0):,}\n"
        f"  Total OpEx: ${g.get('total_opex', 0):,}\n\n"
        f"NOI: ${g.get('noi', 0):,} (${g.get('noi_per_sf', 0):.2f}/SF)\n\n"
        f"VALUATION\n"
        f"  Cap rate: {g.get('cap_rate', 0):.2%}\n"
        f"  Value: ${g.get('value', 0):,}\n"
        f"  Price/SF: ${g.get('price_per_sf', 0):.2f}\n\n"
        f"DEBT METRICS\n"
        f"  Loan: ${g.get('loan_amount', 0):,}\n"
        f"  Annual DS: ${g.get('annual_debt_service', 0):,}\n"
        f"  DSCR: {g.get('dscr', 0):.2f}x\n"
        f"  Debt yield: {g.get('debt_yield', 0):.2%}\n"
        f"  LTV: {g.get('ltv', 0):.1%}\n\n"
        f"RETURNS\n"
        f"  Equity: ${g.get('equity', 0):,}\n"
        f"  Cash-on-cash: {g.get('cash_on_cash', 0):.2%}"
    )
    return q, a


def _a021_coc_return(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    q = (
        f"An investor acquires {d['property_name']} for ${g.get('value', 0):,} with "
        f"a ${g.get('loan_amount', 0):,} loan. NOI is ${g.get('noi', 0):,} and annual "
        f"debt service is ${g.get('annual_debt_service', 0):,}. "
        f"Calculate the equity investment and cash-on-cash return."
    )
    a = (
        f"Equity = Value - Loan = ${g.get('value', 0):,} - ${g.get('loan_amount', 0):,} "
        f"= ${g.get('equity', 0):,}\n\n"
        f"Before-tax cash flow = NOI - DS = ${g.get('noi', 0):,} - "
        f"${g.get('annual_debt_service', 0):,} = "
        f"${g.get('noi', 0) - g.get('annual_debt_service', 0):,}\n\n"
        f"Cash-on-cash = BTCF / Equity = {g.get('cash_on_cash', 0):.2%}"
    )
    return q, a


# ═══════════════════════════════════════════════════════════════════
# BUCKET B — RENT ROLL EXTRACTION
# ═══════════════════════════════════════════════════════════════════


def _b001_extract_roll(d: dict) -> tuple[str, str]:
    narrative = d.get("rent_roll_narrative", "")
    if not narrative or not d.get("rent_roll"):
        return "", ""
    roll = d.get("rent_roll", [])
    sf = d.get("sf", 0)
    structured = {
        "property_name": d["property_name"],
        "as_of_date": "2025-01-01",
        "total_sf": sf,
        "occupied_sf": sum(t["sf"] for t in roll),
        "vacancy_rate": d.get("vacancy_rate", 0.05),
        "num_tenants": len(roll),
        "tenants": roll,
    }
    q = (
        f"Extract the rent roll from the following property description into "
        f"structured JSON matching the RentRollJSON schema.\n\n{narrative}"
    )
    return q, json.dumps(structured, indent=2)


def _b002_walt_calc(d: dict) -> tuple[str, str]:
    roll = d.get("rent_roll", [])
    if not roll:
        return "", ""
    q = (
        f"From the following rent roll for {d['property_name']}, "
        f"calculate the Weighted Average Lease Term (WALT).\n\n"
        + "\n".join(
            f"- {t['name']}: {t['sf']:,} SF, expires {t['lease_end']}"
            for t in roll
        )
    )
    total_sf = sum(t["sf"] for t in roll)
    lines = []
    for t in roll:
        end_yr = int(t["lease_end"].split("-")[0])
        rem = max(0, end_yr - 2025)
        lines.append(f"  {t['name']}: {t['sf']:,} SF x {rem} years = {t['sf'] * rem:,}")
    walt = d.get("walt_years", 0)
    a = (
        f"WALT = sum(SF x remaining years) / total SF\n\n"
        + "\n".join(lines)
        + f"\n\nTotal SF: {total_sf:,}\nWALT = {walt:.1f} years"
    )
    return q, a


def _b003_occupancy(d: dict) -> tuple[str, str]:
    roll = d.get("rent_roll", [])
    if not roll:
        return "", ""
    occ_sf = sum(t["sf"] for t in roll)
    total_sf = d.get("sf", occ_sf)
    occ_rate = occ_sf / total_sf if total_sf > 0 else 0
    q = (
        f"{d['property_name']} is a {total_sf:,} SF building. Current tenants:\n"
        + "\n".join(f"- {t['name']}: {t['sf']:,} SF" for t in roll)
        + "\n\nWhat is the occupancy rate and vacant SF?"
    )
    a = (
        f"Occupied SF: {occ_sf:,}\n"
        f"Total SF: {total_sf:,}\n"
        f"Vacant SF: {total_sf - occ_sf:,}\n"
        f"Occupancy: {occ_rate:.1%}\n"
        f"Vacancy: {1 - occ_rate:.1%}"
    )
    return q, a


# ═══════════════════════════════════════════════════════════════════
# BUCKET C — T-12 NORMALIZATION
# ═══════════════════════════════════════════════════════════════════


def _c001_normalize(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    sf = d.get("sf", 0)
    if sf <= 0:
        return "", ""
    exp = g.get("expense_lines", d.get("expense_lines", {}))
    q = (
        f"Normalize the T-12 for {d['property_name']} ({sf:,} SF).\n\n"
        f"Reported:\n"
        f"  PGI: ${g.get('pgi', 0):,}\n"
        f"  Vacancy loss: ${g.get('vacancy_loss', 0):,}\n"
        f"  EGI: ${g.get('egi', 0):,}\n"
        f"  Property tax: ${exp.get('property_tax', 0):,}\n"
        f"  Insurance: ${exp.get('insurance', 0):,}\n"
        f"  CAM: ${exp.get('cam_maintenance', 0):,}\n"
        f"  Management: ${exp.get('management_fee', 0):,}\n"
        f"  Reserves: ${exp.get('reserves', 0):,}\n"
        f"  Total OpEx: ${g.get('total_opex', 0):,}\n"
        f"  NOI: ${g.get('noi', 0):,}\n\n"
        f"Verify the math and produce a stabilized T-12 summary."
    )
    a = (
        f"T-12 VERIFICATION: {d['property_name']}\n\n"
        f"REVENUE\n"
        f"  PGI: ${g.get('pgi', 0):,}\n"
        f"  Vacancy ({d.get('vacancy_rate', 0.05):.1%}): -${g.get('vacancy_loss', 0):,}\n"
        f"  EGI: ${g.get('egi', 0):,}\n\n"
        f"EXPENSES\n"
        f"  Total OpEx: ${g.get('total_opex', 0):,}\n\n"
        f"STABILIZED NOI: ${g.get('noi', 0):,}\n"
        f"NOI/SF: ${g.get('noi_per_sf', 0):.2f}"
    )
    return q, a


def _c002_expense_ratio(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    egi = g.get("egi", 0)
    opex = g.get("total_opex", 0)
    if egi <= 0:
        return "", ""
    ratio = opex / egi
    q = (
        f"{d['property_name']} reports EGI of ${egi:,} and total operating expenses "
        f"of ${opex:,}. Calculate the operating expense ratio and NOI margin."
    )
    a = (
        f"OpEx Ratio = Total OpEx / EGI = ${opex:,} / ${egi:,} = {ratio:.1%}\n"
        f"NOI Margin = 1 - OpEx Ratio = {1 - ratio:.1%}\n"
        f"NOI = ${g.get('noi', 0):,}"
    )
    return q, a


# ═══════════════════════════════════════════════════════════════════
# BUCKET D — IC MEMO / RISK TRIAGE
# ═══════════════════════════════════════════════════════════════════


def _d001_ic_memo(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    roll = d.get("rent_roll", [])
    debt = d.get("debt", {})
    dscr = g.get("dscr", 0)

    if dscr >= 1.30:
        rec = "PROCEED"
        rationale = f"Strong DSCR of {dscr:.2f}x with stable tenancy supports acquisition."
    elif dscr >= 1.15:
        rec = "CAUTION"
        rationale = f"Adequate DSCR of {dscr:.2f}x but limited margin of safety."
    else:
        rec = "KILL"
        rationale = f"DSCR of {dscr:.2f}x below minimum threshold."

    q = (
        f"Prepare an IC memo for {d['property_name']}, "
        f"a {d['sf']:,} SF {d['asset_type_display']} in {d['submarket']}, {d['market_name']}.\n\n"
        f"Tenants:\n" + "\n".join(
            f"- {t['name']}: {t['sf']:,} SF, ${t['rent_psf']:.2f}/SF, expires {t['lease_end']}"
            for t in roll
        )
        + f"\n\nNOI: ${g.get('noi', 0):,} | Cap: {g.get('cap_rate', 0):.2%} | "
        f"Value: ${g.get('value', 0):,}\n"
        f"Debt: {debt.get('lender_display', 'N/A')} — ${debt.get('loan_amount', 0):,} | "
        f"DSCR: {dscr:.2f}x | CoC: {g.get('cash_on_cash', 0):.2%}\n\n"
        f"Provide recommendation: Proceed, Caution, or Kill."
    )
    a = (
        f"INVESTMENT COMMITTEE MEMO\n{'=' * 50}\n\n"
        f"DEAL: {d['property_name']}\n\n"
        f"EXECUTIVE SUMMARY\n"
        f"{d['property_name']} is a {d['sf']:,} SF {d['asset_type_display']} in {d['submarket']}, "
        f"{d['market_name']}. "
        + (f"100% leased to {roll[0]['name']}" if len(roll) == 1
           else f"Occupied by {len(roll)} tenants")
        + f" generating ${g.get('noi', 0):,} NOI ({g.get('cap_rate', 0):.2%} cap).\n\n"
        f"FINANCIAL SUMMARY\n"
        f"  Purchase price: ${g.get('value', 0):,} (${g.get('price_per_sf', 0):.2f}/SF)\n"
        f"  NOI: ${g.get('noi', 0):,}\n"
        f"  DSCR: {dscr:.2f}x | LTV: {g.get('ltv', 0):.1%}\n"
        f"  Debt yield: {g.get('debt_yield', 0):.2%}\n"
        f"  Cash-on-cash: {g.get('cash_on_cash', 0):.2%}\n\n"
        f"RECOMMENDATION: {rec}\n{rationale}"
    )
    return q, a


def _d002_risk_triage(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    dscr = g.get("dscr", 0)
    vacancy = d.get("vacancy_rate", 0.05)
    roll = d.get("rent_roll", [])

    risks = []
    if dscr < 1.25:
        risks.append(f"[HIGH] Financial: DSCR of {dscr:.2f}x below target")
    if vacancy > 0.08:
        risks.append(f"[MEDIUM] Market: Elevated vacancy at {vacancy:.1%}")
    if d.get("year_built", 2000) < 1990:
        risks.append(f"[MEDIUM] Structural: Building age ({d.get('year_built', 'N/A')})")
    if any(t.get("credit_tier") == "startup" for t in roll):
        risks.append("[MEDIUM] Tenant: Startup credit tenant(s)")
    if not risks:
        risks.append("[LOW] Standard market cycle risk")

    rec = "KILL" if dscr < 1.10 else "CAUTION" if dscr < 1.25 or vacancy > 0.10 else "PROCEED"

    q = (
        f"Risk triage for {d['property_name']}: {d['sf']:,} SF {d['asset_type_display']}, "
        f"vacancy {vacancy:.1%}, DSCR {dscr:.2f}x, year built {d.get('year_built', 'N/A')}. "
        f"Recommend Proceed/Caution/Kill."
    )
    a = (
        f"RISK TRIAGE: {d['property_name']}\n\n"
        f"RISK FACTORS:\n" + "\n".join(f"  {r}" for r in risks)
        + f"\n\nRECOMMENDATION: {rec}"
    )
    return q, a


# ═══════════════════════════════════════════════════════════════════
# BUCKET E — MARKET COMPS
# ═══════════════════════════════════════════════════════════════════


def _e001_comp_analysis(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    sf = d.get("sf", 0)
    value = g.get("value", 0)
    ppsf = g.get("price_per_sf", 0)
    cap = g.get("cap_rate", 0.06)
    if sf <= 0 or value <= 0:
        return "", ""

    q = (
        f"Provide comp analysis for {d['property_name']}, {sf:,} SF in "
        f"{d['submarket']}, {d['market_name']}. "
        f"Asking: ${value:,} (${ppsf:.2f}/SF, {cap:.2%} cap). "
        f"Is this pricing reasonable for the submarket?"
    )
    a = (
        f"COMP ANALYSIS: {d['property_name']}\n\n"
        f"Subject: {sf:,} SF, ${value:,} (${ppsf:.2f}/SF, {cap:.2%} cap)\n"
        f"Market: {d['submarket']}, {d['market_name']} ({d['market_tier']} tier)\n\n"
        f"At ${ppsf:.2f}/SF and a {cap:.2%} cap rate, the pricing is "
        + ("competitive" if cap >= 0.055 else "premium")
        + f" for the {d['market_tier']} market tier."
    )
    return q, a


# ═══════════════════════════════════════════════════════════════════
# BUCKET F — DEAL STRUCTURE (LEASE REASONING, LOI)
# ═══════════════════════════════════════════════════════════════════


def _f001_lease_reasoning(d: dict) -> tuple[str, str]:
    roll = d.get("rent_roll", [])
    if not roll:
        return "", ""
    t = roll[0]
    mkt = d.get("market_rent_psf", 7.0)
    delta = t["rent_psf"] - mkt
    pos = "above" if delta > 0 else "below" if delta < 0 else "at"

    q = (
        f"Analyze {t['name']}'s lease at {d['property_name']}:\n"
        f"  Rent: ${t['rent_psf']:.2f}/SF | Market: ${mkt:.2f}/SF\n"
        f"  Term: {t['lease_start']} to {t['lease_end']}\n"
        f"  Type: {t['lease_type']} | Credit: {t['credit_tier']}\n\n"
        f"How does this compare to market? Renewal strategy?"
    )
    a = (
        f"LEASE ANALYSIS: {t['name']}\n\n"
        f"Current: ${t['rent_psf']:.2f}/SF is {pos} market (${mkt:.2f}/SF, "
        f"delta ${delta:+.2f}/SF).\n"
        f"Annual rent: ${t['annual_rent']:,}\n"
        f"Credit: {t['credit_tier']}\n\n"
        f"STRATEGY: "
        + ("Favorable spread — monitor for rollover risk." if delta > 0
           else "Below market — mark-to-market upside at renewal." if delta < 0
           else "At market — standard renewal terms apply.")
    )
    return q, a


def _f002_loi(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    value = g.get("value", 0)
    if value <= 0:
        return "", ""
    em = round(value * 0.02)
    q = (
        f"Draft an LOI for {d['property_name']} at {d.get('property_address', 'N/A')}. "
        f"Purchase price: ${value:,}. Earnest money: 2%. DD: 45 days. Closing: 90 days."
    )
    loi = {
        "buyer": "Apex Industrial Investors LLC",
        "seller": "Legacy Properties LLC",
        "property_name": d["property_name"],
        "property_address": d.get("property_address", ""),
        "purchase_price": value,
        "price_per_sf": g.get("price_per_sf", 0),
        "earnest_money": em,
        "earnest_money_pct": 0.02,
        "due_diligence_days": 45,
        "closing_days": 90,
        "financing_contingency": True,
        "inspection_contingency": True,
        "title_contingency": True,
        "environmental_contingency": True,
        "conditions": [
            "Phase I ESA", "PCA", "Tenant estoppels within 15 days", "Clear title",
        ],
    }
    return q, json.dumps(loi, indent=2)


# ═══════════════════════════════════════════════════════════════════
# BUCKET G — AGENT OPS
# ═══════════════════════════════════════════════════════════════════


def _g001_agent_screen(d: dict) -> tuple[str, str]:
    g = d.get("gold", {})
    noi = g.get("noi", 0)
    dscr = g.get("dscr", 0)
    value = g.get("value", 0)
    cap = g.get("cap_rate", 0.06)

    conf = 0.85 if dscr >= 1.25 else 0.60 if dscr >= 1.10 else 0.35
    q = (
        f"Agent mode: Screen {d['property_name']}, {d['sf']:,} SF, "
        f"NOI ${noi:,}, {cap:.2%} cap, DSCR {dscr:.2f}x. Return structured JSON."
    )
    out = {
        "task_type": "underwriting_screen",
        "input_summary": f"{d['property_name']}, {d['sf']:,} SF, NOI ${noi:,}",
        "output": {"value": value, "noi": noi, "cap_rate": cap, "dscr": dscr,
                   "pass_screen": dscr >= 1.15},
        "confidence": conf,
        "reasoning_chain": [
            f"NOI ${noi:,} at {cap:.2%} cap = ${value:,}",
            f"DSCR {dscr:.2f}x {'meets' if dscr >= 1.20 else 'below'} 1.20x target",
        ],
        "follow_up_actions": ["Order Phase I ESA", "Obtain estoppels"],
        "risk_flags": [f"Thin DSCR {dscr:.2f}x"] if dscr < 1.25 else [],
    }
    return q, json.dumps(out, indent=2)


# ═══════════════════════════════════════════════════════════════════
# REGISTRY POPULATION
# ═══════════════════════════════════════════════════════════════════


def _build_registry() -> TemplateRegistry:
    """Build and populate the default template registry."""
    reg = TemplateRegistry()

    # Bucket A: Underwriting — low difficulty
    for tid, fn, fields in [
        ("A_001", _a001_pgi, ["pgi"]),
        ("A_002", _a002_noi_simple, ["noi", "egi", "total_opex"]),
        ("A_003", _a003_value, ["value", "noi", "cap_rate"]),
        ("A_004", _a004_vacancy_loss, ["pgi", "vacancy_loss", "egi"]),
        ("A_005", _a005_dscr, ["noi", "annual_debt_service", "dscr"]),
        ("A_006", _a006_price_psf, ["value", "price_per_sf"]),
    ]:
        reg.register(Template(
            template_id=tid, task_type="underwriting_calc",
            difficulty="low", gold_fields=fields, _render_fn=fn,
        ))

    # Bucket A: Underwriting — medium difficulty
    for tid, fn, fields in [
        ("A_010", _a010_multistep, ["pgi", "egi", "total_opex", "noi", "value"]),
        ("A_011", _a011_debt_yield, ["noi", "loan_amount", "debt_yield", "ltv", "value"]),
    ]:
        reg.register(Template(
            template_id=tid, task_type="underwriting_calc",
            difficulty="medium", gold_fields=fields, _render_fn=fn,
        ))

    # Bucket A: Underwriting — high difficulty
    for tid, fn, fields in [
        ("A_020", _a020_full_underwriting, [
            "pgi", "egi", "total_opex", "noi", "value", "annual_debt_service",
            "dscr", "debt_yield", "ltv", "equity", "cash_on_cash",
        ]),
        ("A_021", _a021_coc_return, ["value", "loan_amount", "noi",
                                      "annual_debt_service", "equity", "cash_on_cash"]),
    ]:
        reg.register(Template(
            template_id=tid, task_type="underwriting_calc",
            difficulty="high", gold_fields=fields, _render_fn=fn,
        ))

    # Bucket B: Rent roll extraction
    reg.register(Template(
        template_id="B_001", task_type="rent_roll_extraction",
        difficulty="medium", gold_fields=["pgi"],
        output_schema="RentRollJSON", _render_fn=_b001_extract_roll,
    ))
    reg.register(Template(
        template_id="B_002", task_type="rent_roll_extraction",
        difficulty="low", gold_fields=["pgi"], _render_fn=_b002_walt_calc,
    ))
    reg.register(Template(
        template_id="B_003", task_type="rent_roll_extraction",
        difficulty="low", gold_fields=[], _render_fn=_b003_occupancy,
    ))

    # Bucket C: T-12 normalization
    reg.register(Template(
        template_id="C_001", task_type="t12_normalization",
        difficulty="medium", gold_fields=["noi", "egi", "total_opex"],
        output_schema="T12JSON", _render_fn=_c001_normalize,
    ))
    reg.register(Template(
        template_id="C_002", task_type="t12_normalization",
        difficulty="low", gold_fields=["noi", "egi", "total_opex"],
        _render_fn=_c002_expense_ratio,
    ))

    # Bucket D: IC memo
    reg.register(Template(
        template_id="D_001", task_type="ic_memo",
        difficulty="high", gold_fields=["noi", "dscr", "cap_rate", "value"],
        requires_enrichment=True, _render_fn=_d001_ic_memo,
    ))
    reg.register(Template(
        template_id="D_002", task_type="risk_triage",
        difficulty="medium", gold_fields=["dscr"],
        _render_fn=_d002_risk_triage,
    ))

    # Bucket E: Market comps
    reg.register(Template(
        template_id="E_001", task_type="market_comp_narrative",
        difficulty="medium", gold_fields=["value", "price_per_sf", "cap_rate"],
        requires_enrichment=True, _render_fn=_e001_comp_analysis,
    ))

    # Bucket F: Deal structure
    reg.register(Template(
        template_id="F_001", task_type="lease_reasoning",
        difficulty="medium", gold_fields=[],
        requires_enrichment=True, _render_fn=_f001_lease_reasoning,
    ))
    reg.register(Template(
        template_id="F_002", task_type="loi_deliverable",
        difficulty="medium", gold_fields=["value", "price_per_sf"],
        output_schema="LOITermSheet", _render_fn=_f002_loi,
    ))

    # Bucket G: Agent ops
    reg.register(Template(
        template_id="G_001", task_type="structured_agent_output",
        difficulty="medium", gold_fields=["noi", "value", "dscr"],
        output_schema="StructuredAgentOutput", _render_fn=_g001_agent_screen,
    ))

    return reg


# Module-level singleton
_registry: Optional[TemplateRegistry] = None


def get_registry() -> TemplateRegistry:
    """Get or create the default template registry."""
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return _registry
