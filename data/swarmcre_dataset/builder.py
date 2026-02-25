"""
SwarmCRE Dataset Factory — Sharded Multiprocessing Orchestrator

Builds 1M platinum CRE training pairs across N shards using ProcessPoolExecutor.
Memory-safe: streaming writes (append mode), no full dataset in RAM.
Resume-safe: progress files per shard track processed deal indices.
"""

import json
import hashlib
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from .deal_generator import DealGenerator
from .underwriting_engine import UnderwritingEngine
from .quality_checks import QualityPipeline

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# BUILD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════


@dataclass
class BuildConfig:
    """Configuration for a dataset build run."""

    seed: int = 42
    num_deals: int = 100_000
    num_shards: int = 8
    output_dir: Path = field(default_factory=lambda: Path("output"))
    enable_enrichment: bool = False
    tasks_per_deal: int = 10

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


# ═══════════════════════════════════════════════════════════════════
# TASK BUILDER — turns a deal skeleton + task type into a record
# ═══════════════════════════════════════════════════════════════════

from .constants import (
    TASK_DISTRIBUTION,
    TASK_SYSTEM_PROMPTS,
    DIFFICULTY_WEIGHTS,
    SWARMCRE_SYSTEM_PROMPT,
    EXCHANGE_1031,
    EXCHANGE_SCENARIOS,
    CRE_TAXATION,
)

import random as _random_mod
import hashlib as _hashlib_mod


def _task_rng(master_seed: int, deal_idx: int, task_idx: int) -> _random_mod.Random:
    """Per-task deterministic RNG using SHA-256 hash chain."""
    payload = f"{master_seed}:{deal_idx}:task:{task_idx}"
    h = _hashlib_mod.sha256(payload.encode()).hexdigest()
    return _random_mod.Random(int(h[:16], 16))


def _weighted_choice(rng, items: list, weights: list):
    """Weighted random choice using cumulative distribution."""
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if r <= cumulative:
            return item
    return items[-1]


def _pick_task_type(rng) -> str:
    """Select a task type from the distribution."""
    types = list(TASK_DISTRIBUTION.keys())
    weights = [TASK_DISTRIBUTION[t] for t in types]
    return _weighted_choice(rng, types, weights)


def _pick_difficulty(rng) -> str:
    """Select a difficulty level from the distribution."""
    levels = list(DIFFICULTY_WEIGHTS.keys())
    weights = [DIFFICULTY_WEIGHTS[d] for d in levels]
    return _weighted_choice(rng, levels, weights)


def _build_underwriting_calc_task(deal_dict: dict, difficulty: str, rng) -> dict:
    """Build an underwriting calculation Q&A task."""
    gold = deal_dict.get("gold", {})
    sf = deal_dict.get("sf", 0)
    property_name = deal_dict.get("property_name", "Subject Property")
    asset_display = deal_dict.get("asset_type_display", "industrial property")
    market = deal_dict.get("market_name", "")
    submarket = deal_dict.get("submarket", "")

    if difficulty == "low":
        # Simple single-step calculations
        calc_options = []
        if gold.get("pgi", 0) > 0:
            calc_options.append({
                "question": (
                    f"{property_name} is a {sf:,} SF {asset_display} in {submarket}, {market}. "
                    f"The property has the following tenants:\n"
                    + "\n".join(
                        f"- {t['name']}: {t['sf']:,} SF at ${t['rent_psf']:.2f}/SF"
                        for t in deal_dict.get("rent_roll", [])
                    )
                    + "\n\nCalculate the Potential Gross Income (PGI)."
                ),
                "answer": (
                    f"PGI = sum of all annual rents\n"
                    + "\n".join(
                        f"  {t['name']}: {t['sf']:,} SF x ${t['rent_psf']:.2f}/SF = ${t['annual_rent']:,}"
                        for t in deal_dict.get("rent_roll", [])
                    )
                    + f"\n\nPGI = ${gold['pgi']:,}"
                ),
                "numeric_targets": {"pgi": gold["pgi"]},
            })
        if gold.get("noi", 0) != 0 and gold.get("egi", 0) > 0:
            calc_options.append({
                "question": (
                    f"{property_name} has an Effective Gross Income of ${gold['egi']:,} "
                    f"and total operating expenses of ${gold['total_opex']:,}. "
                    f"What is the Net Operating Income?"
                ),
                "answer": (
                    f"NOI = EGI - Total OpEx\n"
                    f"NOI = ${gold['egi']:,} - ${gold['total_opex']:,}\n"
                    f"NOI = ${gold['noi']:,}"
                ),
                "numeric_targets": {"noi": gold["noi"]},
            })
        if gold.get("value", 0) > 0:
            calc_options.append({
                "question": (
                    f"{property_name} generates a NOI of ${gold['noi']:,}. "
                    f"Using a {gold['cap_rate']:.2%} cap rate, what is the value "
                    f"via direct capitalization?"
                ),
                "answer": (
                    f"Value = NOI / Cap Rate\n"
                    f"Value = ${gold['noi']:,} / {gold['cap_rate']:.4f}\n"
                    f"Value = ${gold['value']:,}"
                ),
                "numeric_targets": {"value": gold["value"]},
            })
        if not calc_options:
            calc_options.append({
                "question": f"What is the land value of {property_name}?",
                "answer": f"Land value = ${gold.get('land_value', 0):,}",
                "numeric_targets": {"land_value": gold.get("land_value", 0)},
            })
        chosen = rng.choice(calc_options)

    elif difficulty == "medium":
        # Multi-step: NOI through debt service
        rent_roll = deal_dict.get("rent_roll", [])
        vacancy = deal_dict.get("vacancy_rate", 0.05)
        debt = deal_dict.get("debt", {})
        chosen = {
            "question": (
                f"{property_name} is a {sf:,} SF {asset_display} in {submarket}, {market}.\n\n"
                f"Rent roll:\n"
                + "\n".join(
                    f"- {t['name']}: {t['sf']:,} SF at ${t['rent_psf']:.2f}/SF ({t['lease_type']})"
                    for t in rent_roll
                )
                + f"\n\nVacancy rate: {vacancy:.1%}\n"
                f"Operating expenses: property tax ${gold.get('expense_lines', {}).get('property_tax', 0):,}, "
                f"insurance ${gold.get('expense_lines', {}).get('insurance', 0):,}, "
                f"CAM ${gold.get('expense_lines', {}).get('cam_maintenance', 0):,}, "
                f"reserves ${gold.get('expense_lines', {}).get('reserves', 0):,}\n"
                f"Management fee: {deal_dict.get('management_fee_pct', 0.04):.1%} of EGI\n"
                f"Cap rate: {gold.get('cap_rate', 0.06):.2%}\n\n"
                f"Calculate: PGI, EGI, total OpEx, NOI, and property value."
            ),
            "answer": (
                f"Step 1: PGI = ${gold.get('pgi', 0):,}\n"
                f"Step 2: EGI = PGI x (1 - {vacancy:.3f}) = ${gold.get('egi', 0):,}\n"
                f"Step 3: Management fee = EGI x {deal_dict.get('management_fee_pct', 0.04):.3f} "
                f"= ${gold.get('management_fee', 0):,}\n"
                f"Step 4: Total OpEx = ${gold.get('total_opex', 0):,}\n"
                f"Step 5: NOI = EGI - OpEx = ${gold.get('noi', 0):,}\n"
                f"Step 6: Value = NOI / Cap Rate = ${gold.get('value', 0):,}\n"
                f"Price per SF = ${gold.get('price_per_sf', 0):.2f}"
            ),
            "numeric_targets": {
                "pgi": gold.get("pgi", 0),
                "egi": gold.get("egi", 0),
                "total_opex": gold.get("total_opex", 0),
                "noi": gold.get("noi", 0),
                "value": gold.get("value", 0),
            },
        }

    else:  # high
        # Full underwriting: revenue through cash-on-cash, plus debt sizing
        debt = deal_dict.get("debt", {})
        vacancy = deal_dict.get("vacancy_rate", 0.05)
        rent_roll = deal_dict.get("rent_roll", [])
        chosen = {
            "question": (
                f"Perform a complete underwriting analysis for {property_name}, "
                f"a {sf:,} SF {asset_display} in {submarket}, {market}.\n\n"
                f"Rent roll:\n"
                + "\n".join(
                    f"- {t['name']} ({t.get('suite', 'N/A')}): {t['sf']:,} SF at "
                    f"${t['rent_psf']:.2f}/SF, {t['lease_type']}, "
                    f"{t['escalation_type']} escalation"
                    + (f" @ {t['escalation_rate']:.1%}" if t['escalation_type'] == 'fixed' else "")
                    + f", expires {t['lease_end']}"
                    for t in rent_roll
                )
                + f"\n\nVacancy: {vacancy:.1%}\n"
                f"Expenses (per SF): property tax ${gold.get('expense_lines', {}).get('property_tax', 0):,}, "
                f"insurance ${gold.get('expense_lines', {}).get('insurance', 0):,}, "
                f"CAM ${gold.get('expense_lines', {}).get('cam_maintenance', 0):,}, "
                f"utilities ${gold.get('expense_lines', {}).get('utilities', 0):,}, "
                f"reserves ${gold.get('expense_lines', {}).get('reserves', 0):,}\n"
                f"Management fee: {deal_dict.get('management_fee_pct', 0.04):.1%} of EGI\n"
                f"Cap rate: {gold.get('cap_rate', 0.06):.2%}\n\n"
                f"Debt: {debt.get('lender_display', 'N/A')} — "
                f"${debt.get('loan_amount', 0):,} at {debt.get('rate', 0):.2%}, "
                f"{debt.get('amort_years', 0)}-year amort, "
                f"{debt.get('io_years', 0)}-year IO, "
                f"{debt.get('term_years', 0)}-year term\n\n"
                f"Calculate all metrics: PGI, EGI, NOI, value, price/SF, "
                f"annual debt service, DSCR, debt yield, LTV, equity, and cash-on-cash return."
            ),
            "answer": (
                f"REVENUE\n"
                f"  PGI: ${gold.get('pgi', 0):,}\n"
                f"  Vacancy loss ({vacancy:.1%}): ${gold.get('vacancy_loss', 0):,}\n"
                f"  EGI: ${gold.get('egi', 0):,}\n\n"
                f"EXPENSES\n"
                f"  Management fee: ${gold.get('management_fee', 0):,}\n"
                f"  Total OpEx: ${gold.get('total_opex', 0):,}\n\n"
                f"NOI: ${gold.get('noi', 0):,} (${gold.get('noi_per_sf', 0):.2f}/SF)\n\n"
                f"VALUATION\n"
                f"  Cap rate: {gold.get('cap_rate', 0):.2%}\n"
                f"  Value: ${gold.get('value', 0):,}\n"
                f"  Price/SF: ${gold.get('price_per_sf', 0):.2f}\n\n"
                f"DEBT METRICS\n"
                f"  Loan amount: ${gold.get('loan_amount', 0):,}\n"
                f"  Annual debt service: ${gold.get('annual_debt_service', 0):,}\n"
                f"  DSCR: {gold.get('dscr', 0):.2f}x\n"
                f"  Debt yield: {gold.get('debt_yield', 0):.2%}\n"
                f"  LTV: {gold.get('ltv', 0):.1%}\n\n"
                f"RETURNS\n"
                f"  Equity: ${gold.get('equity', 0):,}\n"
                f"  Cash-on-cash: {gold.get('cash_on_cash', 0):.2%}"
            ),
            "numeric_targets": {
                "pgi": gold.get("pgi", 0),
                "egi": gold.get("egi", 0),
                "total_opex": gold.get("total_opex", 0),
                "noi": gold.get("noi", 0),
                "value": gold.get("value", 0),
                "annual_debt_service": gold.get("annual_debt_service", 0),
                "dscr": gold.get("dscr", 0),
                "debt_yield": gold.get("debt_yield", 0),
                "ltv": gold.get("ltv", 0),
                "equity": gold.get("equity", 0),
                "cash_on_cash": gold.get("cash_on_cash", 0),
            },
        }

    return chosen


def _build_rent_roll_extraction_task(deal_dict: dict, difficulty: str, rng) -> dict:
    """Build a rent roll extraction from narrative task."""
    narrative = deal_dict.get("rent_roll_narrative", "")
    rent_roll = deal_dict.get("rent_roll", [])
    sf = deal_dict.get("sf", 0)
    vacancy = deal_dict.get("vacancy_rate", 0.05)
    property_name = deal_dict.get("property_name", "Subject Property")

    if not narrative or not rent_roll:
        return None

    occupied_sf = sum(t["sf"] for t in rent_roll)
    num_tenants = len(rent_roll)

    structured_roll = {
        "property_name": property_name,
        "as_of_date": "2025-01-01",
        "total_sf": sf,
        "occupied_sf": occupied_sf,
        "vacancy_rate": vacancy,
        "num_tenants": num_tenants,
        "tenants": rent_roll,
    }

    question = (
        f"Extract the rent roll from the following property description into "
        f"structured JSON matching the RentRollJSON schema.\n\n{narrative}"
    )

    return {
        "question": question,
        "answer": json.dumps(structured_roll, indent=2),
        "numeric_targets": {
            "total_sf": sf,
            "occupied_sf": occupied_sf,
            "num_tenants": num_tenants,
        },
    }


def _build_ic_memo_task(deal_dict: dict, difficulty: str, rng) -> dict:
    """Build an IC memo writing task."""
    gold = deal_dict.get("gold", {})
    property_name = deal_dict.get("property_name", "Subject Property")
    sf = deal_dict.get("sf", 0)
    asset_display = deal_dict.get("asset_type_display", "industrial")
    market = deal_dict.get("market_name", "")
    submarket = deal_dict.get("submarket", "")
    rent_roll = deal_dict.get("rent_roll", [])
    debt = deal_dict.get("debt", {})
    vacancy = deal_dict.get("vacancy_rate", 0.05)

    dscr = gold.get("dscr", 0)
    if dscr >= 1.30:
        recommendation = "proceed"
        rationale = f"Strong DSCR of {dscr:.2f}x with stable tenancy supports acquisition."
    elif dscr >= 1.15:
        recommendation = "caution"
        rationale = f"Adequate DSCR of {dscr:.2f}x but limited margin of safety."
    else:
        recommendation = "kill"
        rationale = f"DSCR of {dscr:.2f}x is below minimum threshold for prudent lending."

    risk_factors = []
    if vacancy > 0.08:
        risk_factors.append(f"Elevated vacancy at {vacancy:.1%} above market average")
    if dscr < 1.25:
        risk_factors.append(f"Thin debt service coverage at {dscr:.2f}x")
    if any(t.get("credit_tier") == "startup" for t in rent_roll):
        risk_factors.append("Startup credit tenant(s) with limited operating history")
    if deal_dict.get("year_built", 2000) < 1990:
        risk_factors.append(f"Aged building (built {deal_dict.get('year_built', 'N/A')}), potential deferred maintenance")
    if not risk_factors:
        risk_factors.append("Limited near-term lease rollover risk")

    question = (
        f"Prepare an Investment Committee memo for the following acquisition:\n\n"
        f"Property: {property_name}\n"
        f"Type: {sf:,} SF {asset_display}\n"
        f"Location: {submarket}, {market}\n"
        f"Year built: {deal_dict.get('year_built', 'N/A')}\n"
        f"Clear height: {deal_dict.get('clear_height_ft', 0)}' clear\n"
        f"Dock doors: {deal_dict.get('dock_doors', 0)}\n\n"
        f"Tenants:\n"
        + "\n".join(
            f"- {t['name']}: {t['sf']:,} SF, ${t['rent_psf']:.2f}/SF, "
            f"{t['lease_type']}, expires {t['lease_end']}"
            for t in rent_roll
        )
        + f"\n\nFinancials:\n"
        f"  NOI: ${gold.get('noi', 0):,}\n"
        f"  Cap rate: {gold.get('cap_rate', 0):.2%}\n"
        f"  Value: ${gold.get('value', 0):,}\n"
        f"  Debt: {debt.get('lender_display', 'N/A')} — ${debt.get('loan_amount', 0):,}\n"
        f"  DSCR: {dscr:.2f}x\n"
        f"  Cash-on-cash: {gold.get('cash_on_cash', 0):.2%}\n\n"
        f"Provide recommendation: Proceed, Caution, or Kill."
    )

    answer = (
        f"INVESTMENT COMMITTEE MEMO\n"
        f"{'=' * 50}\n\n"
        f"DEAL: {property_name}\n\n"
        f"EXECUTIVE SUMMARY\n"
        f"{property_name} is a {sf:,} SF {asset_display} located in the {submarket} "
        f"submarket of {market}. The property is "
        + (f"100% leased to {rent_roll[0]['name']}" if len(rent_roll) == 1
           else f"occupied by {len(rent_roll)} tenants")
        + f" generating ${gold.get('noi', 0):,} NOI "
        f"({gold.get('cap_rate', 0):.2%} going-in cap).\n\n"
        f"FINANCIAL SUMMARY\n"
        f"  Purchase price: ${gold.get('value', 0):,} (${gold.get('price_per_sf', 0):.2f}/SF)\n"
        f"  NOI: ${gold.get('noi', 0):,}\n"
        f"  Going-in cap: {gold.get('cap_rate', 0):.2%}\n"
        f"  DSCR: {dscr:.2f}x\n"
        f"  LTV: {gold.get('ltv', 0):.1%}\n"
        f"  Debt yield: {gold.get('debt_yield', 0):.2%}\n"
        f"  Cash-on-cash: {gold.get('cash_on_cash', 0):.2%}\n\n"
        f"RISK FACTORS\n"
        + "\n".join(f"  - {r}" for r in risk_factors)
        + f"\n\nRECOMMENDATION: {recommendation.upper()}\n"
        f"{rationale}"
    )

    return {
        "question": question,
        "answer": answer,
        "numeric_targets": {
            "noi": gold.get("noi", 0),
            "dscr": dscr,
            "cap_rate": gold.get("cap_rate", 0),
            "value": gold.get("value", 0),
        },
    }


def _build_lease_reasoning_task(deal_dict: dict, difficulty: str, rng) -> dict:
    """Build a lease analysis/reasoning task."""
    rent_roll = deal_dict.get("rent_roll", [])
    if not rent_roll:
        return None

    tenant = rng.choice(rent_roll)
    market_rent = deal_dict.get("market_rent_psf", 7.00)
    property_name = deal_dict.get("property_name", "Subject Property")
    market = deal_dict.get("market_name", "")
    submarket = deal_dict.get("submarket", "")

    rent_delta = tenant["rent_psf"] - market_rent
    pct_delta = (rent_delta / market_rent * 100) if market_rent > 0 else 0

    if rent_delta > 0:
        position = "above"
        strategy = (
            f"The tenant is paying ${rent_delta:.2f}/SF ({abs(pct_delta):.1f}%) above market. "
            f"This is favorable for the landlord but creates rollover risk at lease expiration. "
            f"Consider offering a modest renewal concession to retain the tenant and avoid "
            f"downtime and TI costs."
        )
    elif rent_delta < 0:
        position = "below"
        strategy = (
            f"The tenant is paying ${abs(rent_delta):.2f}/SF ({abs(pct_delta):.1f}%) below market. "
            f"This represents mark-to-market upside at lease expiration. "
            f"If the tenant seeks renewal, push for market-rate escalation."
        )
    else:
        position = "at"
        strategy = "The tenant is paying at market rate. Standard renewal terms apply."

    question = (
        f"Analyze the lease for {tenant['name']} at {property_name} "
        f"({submarket}, {market}):\n\n"
        f"  Tenant: {tenant['name']}\n"
        f"  Space: {tenant['sf']:,} SF\n"
        f"  Current rent: ${tenant['rent_psf']:.2f}/SF ({tenant['lease_type']})\n"
        f"  Annual rent: ${tenant['annual_rent']:,}\n"
        f"  Lease term: {tenant['lease_start']} to {tenant['lease_end']}\n"
        f"  Escalation: {tenant['escalation_type']}"
        + (f" at {tenant['escalation_rate']:.1%}" if tenant['escalation_type'] == 'fixed' else "")
        + f"\n  Credit tier: {tenant['credit_tier']}\n"
        f"  Market rent: ${market_rent:.2f}/SF\n\n"
        f"How does this lease compare to market? What is the optimal renewal strategy?"
    )

    answer = (
        f"LEASE ANALYSIS: {tenant['name']} at {property_name}\n\n"
        f"The current rent of ${tenant['rent_psf']:.2f}/SF is {position} the market rate "
        f"of ${market_rent:.2f}/SF (delta: ${rent_delta:+.2f}/SF, {pct_delta:+.1f}%).\n\n"
        f"Annual rent: ${tenant['annual_rent']:,}\n"
        f"Remaining term through {tenant['lease_end']}\n"
        f"Credit tier: {tenant['credit_tier']}\n\n"
        f"STRATEGY\n{strategy}"
    )

    return {
        "question": question,
        "answer": answer,
        "numeric_targets": {
            "current_rent_psf": tenant["rent_psf"],
            "market_rent_psf": market_rent,
            "rent_delta_psf": round(rent_delta, 2),
            "annual_rent": tenant["annual_rent"],
        },
    }


def _build_market_comp_task(deal_dict: dict, difficulty: str, rng) -> dict:
    """Build a market comp / narrative analysis task."""
    gold = deal_dict.get("gold", {})
    property_name = deal_dict.get("property_name", "Subject Property")
    sf = deal_dict.get("sf", 0)
    market = deal_dict.get("market_name", "")
    submarket = deal_dict.get("submarket", "")
    cap_rate = gold.get("cap_rate", 0.06)
    value = gold.get("value", 0)
    price_per_sf = gold.get("price_per_sf", 0)

    # Generate synthetic comps
    num_comps = rng.randint(3, 5)
    comps = []
    for c in range(num_comps):
        comp_sf = round(sf * rng.uniform(0.7, 1.3) / 1000) * 1000
        comp_sf = max(5000, comp_sf)
        comp_cap = round(cap_rate + rng.uniform(-0.015, 0.015), 4)
        comp_cap = max(0.030, comp_cap)
        comp_ppsf = round(price_per_sf * rng.uniform(0.80, 1.20), 2)
        comp_price = round(comp_sf * comp_ppsf)
        comp_dist = round(rng.uniform(0.3, 8.0), 1)
        sale_year = rng.choice([2023, 2024, 2025])
        sale_month = rng.randint(1, 12)
        comp_name = f"Comp {c + 1} — {rng.choice(['Commerce', 'Logistics', 'Industrial', 'Gateway', 'Enterprise'])} {rng.choice(['Center', 'Park', 'Hub', 'Plaza'])}"

        adj_notes = []
        if abs(comp_sf - sf) / max(sf, 1) > 0.15:
            adj_notes.append(f"size adjustment ({comp_sf:,} vs {sf:,} SF)")
        if comp_dist > 5.0:
            adj_notes.append(f"location adjustment ({comp_dist} mi)")
        if sale_year < 2025:
            adj_notes.append("time adjustment for market movement")
        if not adj_notes:
            adj_notes.append("minimal adjustment required")

        comps.append({
            "property_name": comp_name,
            "sf": comp_sf,
            "sale_price": comp_price,
            "price_per_sf": comp_ppsf,
            "cap_rate": comp_cap,
            "sale_date": f"{sale_year}-{sale_month:02d}-15",
            "distance_miles": comp_dist,
            "adjustment_notes": "; ".join(adj_notes),
        })

    avg_ppsf = round(sum(c["price_per_sf"] for c in comps) / len(comps), 2)
    lo_val = round(sf * avg_ppsf * 0.95)
    hi_val = round(sf * avg_ppsf * 1.05)

    question = (
        f"Provide a comparable transaction analysis for {property_name}, "
        f"a {sf:,} SF property in {submarket}, {market}.\n"
        f"Subject asking price: ${value:,} (${price_per_sf:.2f}/SF, {cap_rate:.2%} cap).\n\n"
        f"Recent comparable sales:\n"
        + "\n".join(
            f"- {c['property_name']}: {c['sf']:,} SF, ${c['sale_price']:,} "
            f"(${c['price_per_sf']:.2f}/SF, {c['cap_rate']:.2%} cap), "
            f"sold {c['sale_date']}, {c['distance_miles']} mi away"
            for c in comps
        )
        + "\n\nAnalyze the comps, note adjustments, and provide a supported value range."
    )

    answer = (
        f"COMPARABLE TRANSACTION ANALYSIS: {property_name}\n\n"
        f"Subject: {sf:,} SF, ${value:,} asking (${price_per_sf:.2f}/SF)\n\n"
        f"COMP SUMMARY\n"
        + "\n".join(
            f"  {c['property_name']}: ${c['price_per_sf']:.2f}/SF ({c['cap_rate']:.2%} cap) — "
            f"{c['adjustment_notes']}"
            for c in comps
        )
        + f"\n\nAverage comp price/SF: ${avg_ppsf:.2f}\n"
        f"Adjusted value range: ${lo_val:,} - ${hi_val:,}\n\n"
        f"The subject's asking price of ${value:,} (${price_per_sf:.2f}/SF) "
        + ("falls within" if lo_val <= value <= hi_val else "falls outside")
        + f" the adjusted comp range of ${lo_val:,} - ${hi_val:,}."
    )

    return {
        "question": question,
        "answer": answer,
        "numeric_targets": {
            "subject_value": value,
            "subject_price_per_sf": price_per_sf,
            "avg_comp_ppsf": avg_ppsf,
        },
    }


def _build_t12_normalization_task(deal_dict: dict, difficulty: str, rng) -> dict:
    """Build a T-12 normalization task."""
    gold = deal_dict.get("gold", {})
    sf = deal_dict.get("sf", 0)
    property_name = deal_dict.get("property_name", "Subject Property")

    if sf <= 0:
        return None

    pgi = gold.get("pgi", 0)
    egi = gold.get("egi", 0)
    vacancy = deal_dict.get("vacancy_rate", 0.05)
    expense_lines = gold.get("expense_lines", {})
    noi = gold.get("noi", 0)

    # Add non-recurring items for normalization
    non_recurring_roof = round(sf * rng.uniform(1.50, 4.00))
    non_recurring_legal = rng.randint(5000, 25000)
    reported_opex = gold.get("total_opex", 0) + non_recurring_roof + non_recurring_legal
    reported_noi = egi - reported_opex

    question = (
        f"Normalize the following trailing 12-month operating statement for {property_name} "
        f"({sf:,} SF).\n\n"
        f"REPORTED T-12:\n"
        f"  Base rent: ${pgi:,}\n"
        f"  Vacancy/credit loss: ${gold.get('vacancy_loss', 0):,}\n"
        f"  EGI: ${egi:,}\n\n"
        f"  Expenses:\n"
        f"    Property tax: ${expense_lines.get('property_tax', 0):,}\n"
        f"    Insurance: ${expense_lines.get('insurance', 0):,}\n"
        f"    CAM/maintenance: ${expense_lines.get('cam_maintenance', 0):,}\n"
        f"    Management fee: ${expense_lines.get('management_fee', 0):,}\n"
        f"    Reserves: ${expense_lines.get('reserves', 0):,}\n"
        f"    Roof repair (one-time): ${non_recurring_roof:,}\n"
        f"    Legal/litigation (one-time): ${non_recurring_legal:,}\n"
        f"  Total expenses: ${reported_opex:,}\n\n"
        f"  Reported NOI: ${reported_noi:,}\n\n"
        f"Identify non-recurring items and produce a stabilized NOI."
    )

    answer = (
        f"T-12 NORMALIZATION: {property_name}\n\n"
        f"NON-RECURRING ITEMS REMOVED:\n"
        f"  - Roof repair: ${non_recurring_roof:,} (capital expenditure, not recurring OpEx)\n"
        f"  - Legal/litigation: ${non_recurring_legal:,} (one-time expense)\n"
        f"  Total removed: ${non_recurring_roof + non_recurring_legal:,}\n\n"
        f"STABILIZED T-12:\n"
        f"  EGI: ${egi:,}\n"
        f"  Normalized OpEx: ${gold.get('total_opex', 0):,}\n"
        f"  Stabilized NOI: ${noi:,}\n"
        f"  NOI/SF: ${gold.get('noi_per_sf', 0):.2f}\n\n"
        f"Adjustment: +${non_recurring_roof + non_recurring_legal:,} vs reported NOI"
    )

    return {
        "question": question,
        "answer": answer,
        "numeric_targets": {
            "stabilized_noi": noi,
            "non_recurring_total": non_recurring_roof + non_recurring_legal,
            "normalized_opex": gold.get("total_opex", 0),
        },
    }


def _build_lease_abstract_task(deal_dict: dict, difficulty: str, rng) -> dict:
    """Build a lease abstract extraction task."""
    rent_roll = deal_dict.get("rent_roll", [])
    if not rent_roll:
        return None

    tenant = rng.choice(rent_roll)
    property_name = deal_dict.get("property_name", "Subject Property")
    property_address = deal_dict.get("property_address", "")

    renewal_options = rng.choice([
        "Two (2) consecutive five (5) year renewal options at 95% of then-prevailing market rate",
        "One (1) five (5) year renewal option at fair market value with 9-month notice",
        "Three (3) consecutive three (3) year renewal options with 3% annual bumps",
        "No renewal options",
    ])

    termination_rights = rng.choice([
        "Tenant may terminate after year 5 with 12-month notice and 6-month penalty",
        "No early termination rights",
        "Mutual termination option at year 7 with 9-month notice",
        "Tenant termination option at month 36 with unamortized TI payback",
    ])

    use_restrictions = rng.choice([
        "General warehouse, distribution, and light manufacturing; no hazardous materials",
        "E-commerce fulfillment and distribution only",
        "General industrial use consistent with M-1 zoning",
        "Cold storage and food distribution; must maintain FSMA compliance",
    ])

    narrative = (
        f"The following is a summary of the lease between {tenant['name']} (Tenant) "
        f"and the landlord for premises at {property_name}, {property_address}.\n\n"
        f"PREMISES: {tenant.get('suite', 'Entire Building')}, comprising approximately "
        f"{tenant['sf']:,} rentable square feet.\n\n"
        f"TERM: Commencing {tenant['lease_start']} and expiring {tenant['lease_end']}.\n\n"
        f"BASE RENT: ${tenant['rent_psf']:.2f} per square foot per annum "
        f"(${tenant['annual_rent']:,} annually), payable in equal monthly installments "
        f"of ${round(tenant['annual_rent'] / 12):,}.\n\n"
        f"LEASE TYPE: {tenant['lease_type']}\n\n"
        f"ESCALATIONS: {tenant['escalation_type'].capitalize()}"
        + (f" at {tenant['escalation_rate']:.1%} per annum" if tenant['escalation_type'] == 'fixed' else "")
        + f".\n\n"
        f"RENEWAL OPTIONS: {renewal_options}\n\n"
        f"GUARANTOR: {tenant.get('guarantee_type', 'corporate').replace('_', ' ').title()}\n\n"
        f"TERMINATION RIGHTS: {termination_rights}\n\n"
        f"USE RESTRICTIONS: {use_restrictions}\n\n"
        f"TI ALLOWANCE: ${tenant.get('ti_allowance_psf', 0):.2f}/SF\n"
    )

    question = (
        f"Extract key lease terms from the following into structured JSON "
        f"matching the LeaseAbstractJSON schema.\n\n{narrative}"
    )

    abstract = {
        "tenant_name": tenant["name"],
        "premises": tenant.get("suite", "Entire Building"),
        "sf": tenant["sf"],
        "lease_type": tenant["lease_type"],
        "term_start": tenant["lease_start"],
        "term_end": tenant["lease_end"],
        "base_rent_psf": tenant["rent_psf"],
        "annual_rent": tenant["annual_rent"],
        "escalation_type": tenant["escalation_type"],
        "escalation_rate": tenant["escalation_rate"],
        "renewal_options": renewal_options,
        "tenant_improvements": f"${tenant.get('ti_allowance_psf', 0):.2f}/SF",
        "guarantor": tenant.get("guarantee_type", "corporate").replace("_", " ").title(),
        "termination_rights": termination_rights,
        "use_restrictions": use_restrictions,
        "key_provisions": [
            f"Lease type: {tenant['lease_type']}",
            f"Escalation: {tenant['escalation_type']}"
            + (f" at {tenant['escalation_rate']:.1%}" if tenant['escalation_type'] == 'fixed' else ""),
        ],
    }

    return {
        "question": question,
        "answer": json.dumps(abstract, indent=2),
        "numeric_targets": {
            "sf": tenant["sf"],
            "base_rent_psf": tenant["rent_psf"],
            "annual_rent": tenant["annual_rent"],
        },
    }


def _build_risk_triage_task(deal_dict: dict, difficulty: str, rng) -> dict:
    """Build a risk triage task."""
    gold = deal_dict.get("gold", {})
    property_name = deal_dict.get("property_name", "Subject Property")
    sf = deal_dict.get("sf", 0)
    asset_display = deal_dict.get("asset_type_display", "industrial")
    market = deal_dict.get("market_name", "")
    submarket = deal_dict.get("submarket", "")
    year_built = deal_dict.get("year_built", 2000)
    vacancy = deal_dict.get("vacancy_rate", 0.05)
    dscr = gold.get("dscr", 0)
    rent_roll = deal_dict.get("rent_roll", [])
    debt = deal_dict.get("debt", {})

    risk_factors = []
    deal_killers = []

    if dscr < 1.10:
        deal_killers.append(f"DSCR of {dscr:.2f}x below minimum underwriting threshold")
        risk_factors.append({
            "category": "financial",
            "description": f"Dangerously thin debt coverage at {dscr:.2f}x",
            "severity": "critical",
            "mitigation": "Seek rate buy-down, additional equity, or lower leverage structure",
        })
    elif dscr < 1.25:
        risk_factors.append({
            "category": "financial",
            "description": f"Below-target DSCR at {dscr:.2f}x",
            "severity": "high",
            "mitigation": "Negotiate rate reduction or increase equity contribution",
        })

    if vacancy > 0.10:
        risk_factors.append({
            "category": "market",
            "description": f"Elevated vacancy at {vacancy:.1%}",
            "severity": "high" if vacancy > 0.15 else "medium",
            "mitigation": "Verify submarket demand drivers and absorption trends",
        })

    if year_built < 1985:
        risk_factors.append({
            "category": "structural",
            "description": f"Building constructed in {year_built}, potential deferred maintenance",
            "severity": "medium",
            "mitigation": "Obtain PCA and budget for capital reserves",
        })

    if year_built < 1980:
        risk_factors.append({
            "category": "environmental",
            "description": "Pre-1980 construction may have asbestos or lead paint",
            "severity": "high",
            "mitigation": "Phase I ESA required; budget for Phase II if RECs identified",
        })

    startup_tenants = [t for t in rent_roll if t.get("credit_tier") == "startup"]
    if startup_tenants:
        risk_factors.append({
            "category": "tenant",
            "description": f"{len(startup_tenants)} startup-credit tenant(s) with limited history",
            "severity": "medium",
            "mitigation": "Require personal guarantee plus additional security deposit",
        })

    if not risk_factors:
        risk_factors.append({
            "category": "market",
            "description": "Standard market cycle risk",
            "severity": "low",
            "mitigation": "Monitor submarket fundamentals quarterly",
        })

    if deal_killers:
        risk_score = "critical"
        recommendation = "kill"
    elif any(r["severity"] == "high" for r in risk_factors):
        risk_score = "high"
        recommendation = "caution"
    elif any(r["severity"] == "medium" for r in risk_factors):
        risk_score = "medium"
        recommendation = "caution"
    else:
        risk_score = "low"
        recommendation = "proceed"

    question = (
        f"Perform a risk triage for the acquisition of {property_name}:\n\n"
        f"  Type: {sf:,} SF {asset_display}\n"
        f"  Location: {submarket}, {market}\n"
        f"  Year built: {year_built}\n"
        f"  Vacancy: {vacancy:.1%}\n"
        f"  NOI: ${gold.get('noi', 0):,}\n"
        f"  DSCR: {dscr:.2f}x\n"
        f"  LTV: {gold.get('ltv', 0):.1%}\n"
        f"  Debt: {debt.get('lender_display', 'N/A')}\n"
        f"  Tenants: {len(rent_roll)} ({', '.join(t.get('credit_tier', 'N/A') for t in rent_roll[:5])})\n\n"
        f"Identify risks, severity, mitigations, and provide a Proceed/Caution/Kill recommendation."
    )

    answer = (
        f"RISK TRIAGE: {property_name}\n\n"
        f"KEY METRICS\n"
        f"  DSCR: {dscr:.2f}x | Vacancy: {vacancy:.1%} | LTV: {gold.get('ltv', 0):.1%}\n"
        f"  NOI: ${gold.get('noi', 0):,} | Value: ${gold.get('value', 0):,}\n\n"
        f"Risk score: {risk_score.upper()}\n\n"
        + (f"DEAL KILLERS:\n" + "\n".join(f"  ** {dk}" for dk in deal_killers) + "\n\n" if deal_killers else "")
        + f"RISK FACTORS:\n"
        + "\n".join(
            f"  [{r['severity'].upper()}] {r['category']}: {r['description']}\n"
            f"    Mitigation: {r['mitigation']}"
            for r in risk_factors
        )
        + f"\n\nRECOMMENDATION: {recommendation.upper()}"
    )

    return {
        "question": question,
        "answer": answer,
        "numeric_targets": {
            "dscr": dscr,
            "vacancy_rate": vacancy,
        },
    }


def _build_loi_task(deal_dict: dict, difficulty: str, rng) -> dict:
    """Build an LOI / term sheet drafting task."""
    gold = deal_dict.get("gold", {})
    property_name = deal_dict.get("property_name", "Subject Property")
    property_address = deal_dict.get("property_address", "")
    value = gold.get("value", 0)
    price_per_sf = gold.get("price_per_sf", 0)

    earnest_pct = round(rng.uniform(0.01, 0.03), 3)
    earnest_money = round(value * earnest_pct)
    dd_days = rng.choice([30, 45, 60, 90])
    closing_days = rng.choice([30, 45, 60, 90]) + dd_days

    buyer_names = [
        "Apex Industrial Investors LLC", "Summit Capital Partners",
        "Vanguard CRE Fund III", "Metro Logistics Acquisitions",
        "Pinnacle Industrial REIT", "Atlas Distribution Holdings",
    ]
    seller_names = [
        "Legacy Properties LLC", "Heritage Industrial Trust",
        "Regional Investment Group", "Family Office Holdings LP",
    ]

    buyer = rng.choice(buyer_names)
    seller = rng.choice(seller_names)

    question = (
        f"Draft an LOI / term sheet for the acquisition of {property_name} "
        f"at {property_address}.\n\n"
        f"  Buyer: {buyer}\n"
        f"  Seller: {seller}\n"
        f"  Purchase price: ${value:,}\n"
        f"  Earnest money: {earnest_pct:.1%} of purchase price\n"
        f"  DD period: {dd_days} days\n"
        f"  Closing: {closing_days} days from execution\n\n"
        f"Include standard contingencies."
    )

    loi = {
        "buyer": buyer,
        "seller": seller,
        "property_name": property_name,
        "property_address": property_address,
        "purchase_price": value,
        "price_per_sf": price_per_sf,
        "earnest_money": earnest_money,
        "earnest_money_pct": earnest_pct,
        "due_diligence_days": dd_days,
        "closing_days": closing_days,
        "financing_contingency": True,
        "inspection_contingency": True,
        "title_contingency": True,
        "environmental_contingency": True,
        "conditions": [
            "Satisfactory Phase I Environmental Site Assessment",
            "Satisfactory Property Condition Assessment",
            "Tenant estoppel certificates within 15 days of execution",
            "Clear title with standard industrial exceptions",
            f"Financing commitment within {dd_days} days",
        ],
    }

    return {
        "question": question,
        "answer": json.dumps(loi, indent=2),
        "numeric_targets": {
            "purchase_price": value,
            "earnest_money": earnest_money,
        },
    }


def _build_structured_agent_task(deal_dict: dict, difficulty: str, rng) -> dict:
    """Build a structured agent output task."""
    gold = deal_dict.get("gold", {})
    property_name = deal_dict.get("property_name", "Subject Property")
    sf = deal_dict.get("sf", 0)
    noi = gold.get("noi", 0)
    value = gold.get("value", 0)
    dscr = gold.get("dscr", 0)
    cap_rate = gold.get("cap_rate", 0)

    if dscr >= 1.25:
        confidence = round(rng.uniform(0.80, 0.95), 2)
    elif dscr >= 1.10:
        confidence = round(rng.uniform(0.55, 0.75), 2)
    else:
        confidence = round(rng.uniform(0.30, 0.50), 2)

    question = (
        f"Process in agent mode: Quick underwriting screen for {property_name}, "
        f"{sf:,} SF, NOI ${noi:,}, {cap_rate:.2%} cap, DSCR {dscr:.2f}x. "
        f"Return structured JSON for downstream consumption."
    )

    agent_output = {
        "task_type": "underwriting_screen",
        "input_summary": f"{property_name}, {sf:,} SF, NOI ${noi:,}",
        "output": {
            "value": value,
            "noi": noi,
            "cap_rate": cap_rate,
            "dscr": dscr,
            "pass_screen": dscr >= 1.15,
        },
        "confidence": confidence,
        "reasoning_chain": [
            f"NOI of ${noi:,} at {cap_rate:.2%} cap yields ${value:,} value",
            f"DSCR of {dscr:.2f}x " + ("meets" if dscr >= 1.20 else "below") + " target 1.20x threshold",
            f"{'Proceed' if dscr >= 1.15 else 'Flag for review'} based on preliminary screen",
        ],
        "follow_up_actions": [
            "Order Phase I ESA" if dscr >= 1.15 else "Request revised debt terms",
            "Obtain tenant estoppels",
            "Verify rent roll with lease abstracts",
        ],
        "risk_flags": (
            [f"Thin DSCR at {dscr:.2f}x"] if dscr < 1.25 else []
        ),
    }

    return {
        "question": question,
        "answer": json.dumps(agent_output, indent=2),
        "numeric_targets": {
            "value": value,
            "noi": noi,
            "dscr": dscr,
        },
    }


def _build_1031_exchange_task(deal_dict: dict, difficulty: str, rng) -> dict:
    """Build a 1031 exchange analysis task."""
    gold = deal_dict.get("gold", {})
    property_name = deal_dict.get("property_name", "Subject Property")
    sf = deal_dict.get("sf", 0)
    asset_display = deal_dict.get("asset_type_display", "industrial property")
    market = deal_dict.get("market_name", "")
    submarket = deal_dict.get("submarket", "")
    value = gold.get("value", 0)

    if value <= 0:
        return None

    # Generate exchange scenario parameters
    years_held = rng.randint(5, 20)
    # Original purchase price (below current value to create gain)
    appreciation = rng.uniform(0.20, 0.80)
    original_purchase = round(value / (1 + appreciation))
    # Building basis ~80-85% of purchase (rest is land)
    building_pct = rng.uniform(0.75, 0.85)
    original_basis = round(original_purchase * building_pct)
    land_value = original_purchase - original_basis

    # Depreciation: 39-year straight-line, capped at years_held
    annual_depr = round(original_basis / 39)
    accum_depr = min(annual_depr * years_held, original_basis)
    adjusted_basis = original_basis - accum_depr + land_value  # land is not depreciated

    # Gain
    sale_price = value
    total_gain = sale_price - adjusted_basis

    if total_gain <= 0:
        # Ensure there's a gain for the exercise
        total_gain = round(value * rng.uniform(0.10, 0.30))
        adjusted_basis = sale_price - total_gain

    # Recapture
    recapture = min(accum_depr, total_gain)

    # Debt on relinquished
    relin_debt = gold.get("loan_amount", round(value * 0.60))

    if difficulty == "low":
        # Simple delayed exchange — full deferral, no boot
        replacement_cost = round(sale_price * rng.uniform(1.05, 1.30))
        replacement_debt = round(replacement_cost * rng.uniform(0.60, 0.75))
        mortgage_boot = max(0, relin_debt - replacement_debt)
        # Ensure no boot for low difficulty
        if mortgage_boot > 0:
            replacement_debt = relin_debt + rng.randint(10000, 50000)
            mortgage_boot = 0
        cash_boot = 0
        total_boot = 0
        recognized_gain = 0
        deferred_gain = total_gain
        new_basis = replacement_cost - deferred_gain
        exchange_type = "delayed"

        question = (
            f"A client is selling {property_name}, a {sf:,} SF {asset_display} in "
            f"{submarket}, {market} for ${sale_price:,}.\n\n"
            f"Original purchase: ${original_purchase:,} ({years_held} years ago)\n"
            f"Building basis: ${original_basis:,}\n"
            f"Land value: ${land_value:,}\n"
            f"Accumulated depreciation: ${accum_depr:,} (39-year straight-line)\n"
            f"Adjusted basis: ${adjusted_basis:,}\n"
            f"Existing mortgage: ${relin_debt:,}\n\n"
            f"They want to do a 1031 exchange into a ${replacement_cost:,} replacement property "
            f"with ${replacement_debt:,} in new debt.\n\n"
            f"Calculate the total gain, deferred gain, and new basis in the replacement property. "
            f"Note the identification and closing deadlines."
        )

        answer = (
            f"1031 EXCHANGE ANALYSIS: {property_name}\n\n"
            f"EXCHANGE TYPE: Delayed (Starker) Exchange\n\n"
            f"RELINQUISHED PROPERTY\n"
            f"  Sale price: ${sale_price:,}\n"
            f"  Original basis: ${original_basis:,} (building) + ${land_value:,} (land)\n"
            f"  Accumulated depreciation: ${accum_depr:,} ({years_held} years x ${annual_depr:,}/yr)\n"
            f"  Adjusted basis: ${adjusted_basis:,}\n"
            f"  Total gain: ${total_gain:,}\n\n"
            f"BOOT ANALYSIS\n"
            f"  Mortgage boot: $0 (replacement debt >= relinquished debt)\n"
            f"  Cash boot: $0\n"
            f"  Total boot: $0\n\n"
            f"GAIN RECOGNITION\n"
            f"  Recognized gain: $0 (no boot received)\n"
            f"  Deferred gain: ${deferred_gain:,}\n\n"
            f"REPLACEMENT PROPERTY\n"
            f"  Cost: ${replacement_cost:,}\n"
            f"  New basis: ${replacement_cost:,} - ${deferred_gain:,} = ${new_basis:,}\n\n"
            f"DEADLINES\n"
            f"  45-day identification deadline: identify up to 3 replacement properties\n"
            f"  180-day closing deadline: close on replacement property\n"
            f"  Qualified Intermediary (QI) must hold all exchange proceeds"
        )

    elif difficulty == "medium":
        # Exchange with boot — mortgage boot or partial exchange
        scenario = rng.choice(["mortgage_boot", "partial_exchange", "dst_replacement"])

        if scenario == "mortgage_boot":
            replacement_cost = round(sale_price * rng.uniform(0.85, 1.05))
            replacement_debt = round(replacement_cost * rng.uniform(0.50, 0.65))
            mortgage_boot = max(0, relin_debt - replacement_debt)
            cash_boot = 0
            total_boot = mortgage_boot
            recognized_gain = min(total_gain, total_boot)
            deferred_gain = total_gain - recognized_gain
            new_basis = replacement_cost - deferred_gain
            exchange_type = "delayed"

            question = (
                f"A client is exchanging {property_name} ({sf:,} SF {asset_display}, "
                f"{submarket}, {market}) via a 1031 exchange.\n\n"
                f"RELINQUISHED PROPERTY:\n"
                f"  Sale price: ${sale_price:,}\n"
                f"  Adjusted basis: ${adjusted_basis:,}\n"
                f"  Existing mortgage: ${relin_debt:,}\n\n"
                f"REPLACEMENT PROPERTY:\n"
                f"  Purchase price: ${replacement_cost:,}\n"
                f"  New mortgage: ${replacement_debt:,}\n\n"
                f"The replacement property has less debt than the relinquished. "
                f"Calculate the mortgage boot, recognized gain, deferred gain, and new basis."
            )

            answer = (
                f"1031 EXCHANGE WITH MORTGAGE BOOT\n\n"
                f"GAIN CALCULATION\n"
                f"  Sale price: ${sale_price:,}\n"
                f"  Adjusted basis: ${adjusted_basis:,}\n"
                f"  Total gain: ${total_gain:,}\n\n"
                f"BOOT ANALYSIS\n"
                f"  Relinquished debt: ${relin_debt:,}\n"
                f"  Replacement debt: ${replacement_debt:,}\n"
                f"  Mortgage boot (debt relief): ${mortgage_boot:,}\n"
                f"  Cash boot: $0\n"
                f"  Total boot: ${total_boot:,}\n\n"
                f"GAIN RECOGNITION\n"
                f"  Recognized gain: min(${total_gain:,}, ${total_boot:,}) = ${recognized_gain:,}\n"
                f"  Deferred gain: ${total_gain:,} - ${recognized_gain:,} = ${deferred_gain:,}\n\n"
                f"REPLACEMENT PROPERTY BASIS\n"
                f"  Cost: ${replacement_cost:,}\n"
                f"  Less deferred gain: ${deferred_gain:,}\n"
                f"  New basis: ${new_basis:,}\n\n"
                f"NOTE: Recognized gain of ${recognized_gain:,} is taxable in current year.\n"
                f"Section 1250 recapture applies to ${min(recapture, recognized_gain):,} at 25% rate."
            )

        elif scenario == "partial_exchange":
            cash_boot = round(sale_price * rng.uniform(0.05, 0.15))
            replacement_cost = round((sale_price - cash_boot) * rng.uniform(0.95, 1.10))
            replacement_debt = round(replacement_cost * rng.uniform(0.60, 0.75))
            mortgage_boot = max(0, relin_debt - replacement_debt)
            total_boot = mortgage_boot + cash_boot
            recognized_gain = min(total_gain, total_boot)
            deferred_gain = total_gain - recognized_gain
            new_basis = replacement_cost - deferred_gain
            exchange_type = "delayed"

            question = (
                f"A client is selling {property_name} for ${sale_price:,} and wants to take "
                f"${cash_boot:,} in cash while deferring the rest via 1031 exchange.\n\n"
                f"  Adjusted basis: ${adjusted_basis:,}\n"
                f"  Existing mortgage: ${relin_debt:,}\n"
                f"  Cash to be received: ${cash_boot:,}\n"
                f"  Replacement property: ${replacement_cost:,}\n"
                f"  New mortgage: ${replacement_debt:,}\n\n"
                f"Calculate the total boot, recognized gain, deferred gain, new basis, "
                f"and tax impact of the partial exchange."
            )

            answer = (
                f"PARTIAL 1031 EXCHANGE ANALYSIS\n\n"
                f"GAIN CALCULATION\n"
                f"  Sale price: ${sale_price:,}\n"
                f"  Adjusted basis: ${adjusted_basis:,}\n"
                f"  Total gain: ${total_gain:,}\n\n"
                f"BOOT ANALYSIS\n"
                f"  Cash boot received: ${cash_boot:,}\n"
                f"  Mortgage boot: ${mortgage_boot:,}\n"
                f"  Total boot: ${total_boot:,}\n\n"
                f"GAIN RECOGNITION\n"
                f"  Recognized gain: min(${total_gain:,}, ${total_boot:,}) = ${recognized_gain:,}\n"
                f"  Deferred gain: ${deferred_gain:,}\n\n"
                f"REPLACEMENT PROPERTY\n"
                f"  Cost: ${replacement_cost:,}\n"
                f"  New basis: ${replacement_cost:,} - ${deferred_gain:,} = ${new_basis:,}\n\n"
                f"TAX ON RECOGNIZED GAIN\n"
                f"  Section 1250 recapture (25%): applies to ${min(recapture, recognized_gain):,}\n"
                f"  LTCG (20%) + NIIT (3.8%): applies to remaining gain"
            )

        else:  # dst_replacement
            replacement_cost = round(sale_price * rng.uniform(1.00, 1.20))
            replacement_debt = round(replacement_cost * rng.uniform(0.50, 0.65))
            mortgage_boot = max(0, relin_debt - replacement_debt)
            if mortgage_boot > 0:
                replacement_debt = relin_debt + rng.randint(1000, 20000)
                mortgage_boot = 0
            cash_boot = 0
            total_boot = 0
            recognized_gain = 0
            deferred_gain = total_gain
            new_basis = replacement_cost - deferred_gain
            exchange_type = "delayed"

            question = (
                f"A client is selling {property_name} for ${sale_price:,} and exchanging "
                f"into a Delaware Statutory Trust (DST) interest.\n\n"
                f"  Adjusted basis: ${adjusted_basis:,}\n"
                f"  Total gain: ${total_gain:,}\n"
                f"  DST investment: ${replacement_cost:,}\n"
                f"  DST debt (pro-rata): ${replacement_debt:,}\n\n"
                f"Analyze the 1031 exchange into DST including basis, gain deferral, "
                f"and passive income implications."
            )

            answer = (
                f"1031 EXCHANGE INTO DST\n\n"
                f"RELINQUISHED PROPERTY\n"
                f"  Sale price: ${sale_price:,}\n"
                f"  Adjusted basis: ${adjusted_basis:,}\n"
                f"  Total gain: ${total_gain:,}\n\n"
                f"DST REPLACEMENT\n"
                f"  Investment: ${replacement_cost:,}\n"
                f"  Pro-rata debt: ${replacement_debt:,}\n"
                f"  Boot: $0 (replacement value >= relinquished)\n"
                f"  Deferred gain: ${deferred_gain:,}\n"
                f"  New basis: ${new_basis:,}\n\n"
                f"DST CONSIDERATIONS\n"
                f"  - Passive investment — no active management required\n"
                f"  - 1031-eligible as real property interest\n"
                f"  - Fractional ownership structure — beneficial interests\n"
                f"  - Fixed income stream — limited upside but predictable cash flow\n"
                f"  - Future 1031 exchange from DST may be limited at maturity\n\n"
                f"DEADLINES\n"
                f"  45-day identification: identify DST offering(s)\n"
                f"  180-day closing: complete subscription and fund exchange"
            )

    else:  # high difficulty
        # Full multi-step: compute gain, depreciation recapture, boot, tax comparison
        replacement_cost = round(sale_price * rng.uniform(0.90, 1.15))
        replacement_debt = round(replacement_cost * rng.uniform(0.55, 0.70))
        mortgage_boot = max(0, relin_debt - replacement_debt)
        cash_boot = round(sale_price * rng.uniform(0.0, 0.08)) if rng.random() > 0.5 else 0
        total_boot = mortgage_boot + cash_boot
        recognized_gain = min(total_gain, total_boot)
        deferred_gain = total_gain - recognized_gain
        new_basis = replacement_cost - deferred_gain
        exchange_type = rng.choice(["delayed", "reverse"])

        # Tax comparison: what if they DON'T do 1031?
        ltcg_portion = max(0, total_gain - recapture)
        recapture_tax = round(recapture * 0.25)
        ltcg_tax = round(ltcg_portion * 0.20)
        niit_tax = round(total_gain * 0.038)
        total_tax_no_exchange = recapture_tax + ltcg_tax + niit_tax

        # Tax on recognized gain (partial)
        recapture_on_recognized = min(recapture, recognized_gain)
        ltcg_on_recognized = max(0, recognized_gain - recapture_on_recognized)
        tax_on_recognized = (
            round(recapture_on_recognized * 0.25)
            + round(ltcg_on_recognized * 0.20)
            + round(recognized_gain * 0.038)
        )
        tax_savings = total_tax_no_exchange - tax_on_recognized

        question = (
            f"Provide a complete 1031 exchange analysis with tax comparison for "
            f"{property_name}, a {sf:,} SF {asset_display} in {submarket}, {market}.\n\n"
            f"RELINQUISHED PROPERTY:\n"
            f"  Sale price: ${sale_price:,}\n"
            f"  Original purchase: ${original_purchase:,} ({years_held} years ago)\n"
            f"  Building basis: ${original_basis:,}\n"
            f"  Land value: ${land_value:,}\n"
            f"  Accumulated depreciation: ${accum_depr:,}\n"
            f"  Adjusted basis: ${adjusted_basis:,}\n"
            f"  Existing mortgage: ${relin_debt:,}\n\n"
            f"REPLACEMENT PROPERTY:\n"
            f"  Purchase price: ${replacement_cost:,}\n"
            f"  New mortgage: ${replacement_debt:,}\n"
            + (f"  Cash received: ${cash_boot:,}\n" if cash_boot > 0 else "")
            + f"\nCalculate:\n"
            f"1. Total gain and depreciation recapture\n"
            f"2. Boot (mortgage + cash), recognized gain, deferred gain\n"
            f"3. New basis in replacement property\n"
            f"4. Tax comparison: full sale vs. 1031 exchange\n"
            f"5. Total tax savings from the exchange"
        )

        answer = (
            f"1031 EXCHANGE ANALYSIS: {property_name}\n"
            f"{'=' * 50}\n\n"
            f"1. GAIN & DEPRECIATION RECAPTURE\n"
            f"  Sale price: ${sale_price:,}\n"
            f"  Adjusted basis: ${adjusted_basis:,}\n"
            f"  Total gain: ${total_gain:,}\n"
            f"  Depreciation recapture (Section 1250): ${recapture:,}\n"
            f"  Capital gain above recapture: ${ltcg_portion:,}\n\n"
            f"2. BOOT & GAIN RECOGNITION\n"
            f"  Relinquished debt: ${relin_debt:,}\n"
            f"  Replacement debt: ${replacement_debt:,}\n"
            f"  Mortgage boot: ${mortgage_boot:,}\n"
            f"  Cash boot: ${cash_boot:,}\n"
            f"  Total boot: ${total_boot:,}\n"
            f"  Recognized gain: min(${total_gain:,}, ${total_boot:,}) = ${recognized_gain:,}\n"
            f"  Deferred gain: ${deferred_gain:,}\n\n"
            f"3. REPLACEMENT PROPERTY BASIS\n"
            f"  Replacement cost: ${replacement_cost:,}\n"
            f"  Less deferred gain: ${deferred_gain:,}\n"
            f"  New basis: ${new_basis:,}\n\n"
            f"4. TAX COMPARISON\n"
            f"  WITHOUT 1031 EXCHANGE:\n"
            f"    Section 1250 recapture (25%): ${recapture_tax:,}\n"
            f"    LTCG (20%): ${ltcg_tax:,}\n"
            f"    NIIT (3.8%): ${niit_tax:,}\n"
            f"    Total tax liability: ${total_tax_no_exchange:,}\n\n"
            f"  WITH 1031 EXCHANGE:\n"
            f"    Tax on recognized gain: ${tax_on_recognized:,}\n\n"
            f"5. TAX SAVINGS: ${tax_savings:,}\n\n"
            f"DEADLINES\n"
            f"  45-day identification: identify up to 3 replacement properties\n"
            f"  180-day closing: close on replacement\n"
            f"  Qualified Intermediary must hold all exchange proceeds"
        )

    return {
        "question": question,
        "answer": answer,
        "numeric_targets": {
            "total_gain": total_gain,
            "deferred_gain": deferred_gain,
            "recognized_gain": recognized_gain,
            "new_basis": new_basis,
            "total_boot": total_boot,
        },
    }


def _build_tax_analysis_task(deal_dict: dict, difficulty: str, rng) -> dict:
    """Build a CRE tax analysis task (depreciation, cost seg, sale analysis)."""
    gold = deal_dict.get("gold", {})
    property_name = deal_dict.get("property_name", "Subject Property")
    sf = deal_dict.get("sf", 0)
    asset_display = deal_dict.get("asset_type_display", "industrial property")
    value = gold.get("value", 0)

    if value <= 0:
        return None

    # Generate purchase/basis params
    purchase_price = value
    building_pct = rng.uniform(0.75, 0.85)
    building_basis = round(purchase_price * building_pct)
    land_value = purchase_price - building_basis

    if difficulty == "low":
        # Simple depreciation schedule
        annual_depr = round(building_basis / 39)
        year_5_accum = annual_depr * 5
        year_10_accum = annual_depr * 10
        year_5_basis = building_basis - year_5_accum + land_value
        year_10_basis = building_basis - year_10_accum + land_value

        question = (
            f"Calculate the depreciation schedule for {property_name}, a {sf:,} SF "
            f"{asset_display} purchased for ${purchase_price:,}.\n\n"
            f"  Purchase price: ${purchase_price:,}\n"
            f"  Land value: ${land_value:,}\n"
            f"  Building basis: ${building_basis:,}\n"
            f"  Depreciation method: 39-year straight-line (non-residential commercial)\n\n"
            f"Calculate the annual depreciation deduction and adjusted basis at years 5 and 10."
        )

        answer = (
            f"DEPRECIATION SCHEDULE: {property_name}\n\n"
            f"BASIS ALLOCATION\n"
            f"  Purchase price: ${purchase_price:,}\n"
            f"  Land (not depreciable): ${land_value:,}\n"
            f"  Building basis: ${building_basis:,}\n\n"
            f"STRAIGHT-LINE DEPRECIATION (39 years)\n"
            f"  Annual deduction: ${building_basis:,} / 39 = ${annual_depr:,}/year\n\n"
            f"ADJUSTED BASIS OVER TIME\n"
            f"  Year 5: ${building_basis:,} - ${year_5_accum:,} + ${land_value:,} = ${year_5_basis:,}\n"
            f"  Year 10: ${building_basis:,} - ${year_10_accum:,} + ${land_value:,} = ${year_10_basis:,}\n\n"
            f"NOTE: Land value of ${land_value:,} is never depreciated. "
            f"Only the building basis of ${building_basis:,} is subject to 39-year depreciation."
        )

        return {
            "question": question,
            "answer": answer,
            "numeric_targets": {
                "annual_depreciation": annual_depr,
                "building_basis": building_basis,
            },
        }

    elif difficulty == "medium":
        # Cost segregation analysis
        accel_pct = round(rng.uniform(0.20, 0.35), 2)
        bonus_depr_pct = rng.choice([0.60, 0.40, 0.20])
        bonus_year_label = {0.60: "2024", 0.40: "2025", 0.20: "2026"}.get(bonus_depr_pct, "2024")

        accelerated_basis = round(building_basis * accel_pct)
        remaining_39yr = building_basis - accelerated_basis
        year_1_bonus = round(accelerated_basis * bonus_depr_pct)
        remaining_accel = accelerated_basis - year_1_bonus
        annual_39yr = round(remaining_39yr / 39)

        # Breakdown of accelerated components
        five_yr_pct = round(accel_pct * 0.45, 3)
        seven_yr_pct = round(accel_pct * 0.20, 3)
        fifteen_yr_pct = round(accel_pct * 0.35, 3)
        five_yr_basis = round(building_basis * five_yr_pct)
        seven_yr_basis = round(building_basis * seven_yr_pct)
        fifteen_yr_basis = round(building_basis * fifteen_yr_pct)

        # Compare: with vs without cost seg
        standard_annual = round(building_basis / 39)
        year_1_with_costseg = year_1_bonus + annual_39yr
        year_1_benefit = year_1_with_costseg - standard_annual
        tax_rate = 0.37
        year_1_tax_savings = round(year_1_benefit * tax_rate)

        question = (
            f"Perform a cost segregation analysis for {property_name}, a {sf:,} SF "
            f"{asset_display} purchased for ${purchase_price:,}.\n\n"
            f"  Building basis: ${building_basis:,}\n"
            f"  Land value: ${land_value:,}\n"
            f"  Cost seg study finds {accel_pct:.0%} of basis can be reclassified\n"
            f"  Bonus depreciation rate: {bonus_depr_pct:.0%} ({bonus_year_label})\n\n"
            f"Calculate year-1 depreciation with cost segregation vs. standard 39-year, "
            f"and the tax savings at a 37% marginal rate."
        )

        answer = (
            f"COST SEGREGATION ANALYSIS: {property_name}\n\n"
            f"RECLASSIFICATION\n"
            f"  Total building basis: ${building_basis:,}\n"
            f"  Accelerated ({accel_pct:.0%}): ${accelerated_basis:,}\n"
            f"    5-year property: ${five_yr_basis:,}\n"
            f"    7-year property: ${seven_yr_basis:,}\n"
            f"    15-year property: ${fifteen_yr_basis:,}\n"
            f"  Remaining 39-year: ${remaining_39yr:,}\n\n"
            f"YEAR 1 DEPRECIATION WITH COST SEGREGATION\n"
            f"  Bonus depreciation ({bonus_depr_pct:.0%} of ${accelerated_basis:,}): ${year_1_bonus:,}\n"
            f"  39-year component (${remaining_39yr:,} / 39): ${annual_39yr:,}\n"
            f"  Total year 1 deduction: ${year_1_with_costseg:,}\n\n"
            f"WITHOUT COST SEGREGATION\n"
            f"  Standard 39-year deduction: ${standard_annual:,}\n\n"
            f"BENEFIT\n"
            f"  Additional year 1 deduction: ${year_1_benefit:,}\n"
            f"  Tax savings at 37% rate: ${year_1_tax_savings:,}"
        )

        return {
            "question": question,
            "answer": answer,
            "numeric_targets": {
                "year_1_bonus": year_1_bonus,
                "annual_39yr": annual_39yr,
                "accelerated_basis": accelerated_basis,
                "year_1_tax_savings": year_1_tax_savings,
            },
        }

    else:  # high difficulty
        # Full sale tax analysis with depreciation recapture
        years_held = rng.randint(7, 20)
        annual_depr = round(building_basis / 39)
        accum_depr = min(annual_depr * years_held, building_basis)
        adjusted_basis = building_basis - accum_depr + land_value

        # Appreciation
        sale_price = round(purchase_price * rng.uniform(1.15, 1.80))
        total_gain = sale_price - adjusted_basis

        if total_gain <= 0:
            total_gain = round(sale_price * 0.20)
            adjusted_basis = sale_price - total_gain

        recapture = min(accum_depr, total_gain)
        ltcg_portion = max(0, total_gain - recapture)

        recapture_tax = round(recapture * 0.25)
        ltcg_tax = round(ltcg_portion * 0.20)
        niit_tax = round(total_gain * 0.038)
        total_tax = recapture_tax + ltcg_tax + niit_tax
        effective_rate = round(total_tax / total_gain * 100, 1) if total_gain > 0 else 0
        after_tax_proceeds = sale_price - total_tax

        question = (
            f"Calculate the complete tax liability on the sale of {property_name}, "
            f"a {sf:,} SF {asset_display}.\n\n"
            f"  Original purchase: ${purchase_price:,} ({years_held} years ago)\n"
            f"  Building basis: ${building_basis:,}\n"
            f"  Land value: ${land_value:,}\n"
            f"  Annual depreciation: ${annual_depr:,}/year (39-year straight-line)\n"
            f"  Accumulated depreciation: ${accum_depr:,}\n"
            f"  Adjusted basis: ${adjusted_basis:,}\n"
            f"  Sale price: ${sale_price:,}\n\n"
            f"Calculate:\n"
            f"1. Total gain\n"
            f"2. Section 1250 depreciation recapture amount and tax (25%)\n"
            f"3. Long-term capital gains tax (20%)\n"
            f"4. Net Investment Income Tax (3.8%)\n"
            f"5. Total tax liability and effective tax rate\n"
            f"6. After-tax proceeds"
        )

        answer = (
            f"TAX ANALYSIS ON SALE: {property_name}\n"
            f"{'=' * 50}\n\n"
            f"1. TOTAL GAIN\n"
            f"  Sale price: ${sale_price:,}\n"
            f"  Adjusted basis: ${adjusted_basis:,}\n"
            f"  Total gain: ${total_gain:,}\n\n"
            f"2. SECTION 1250 DEPRECIATION RECAPTURE\n"
            f"  Accumulated depreciation: ${accum_depr:,}\n"
            f"  Recapture amount: min(${accum_depr:,}, ${total_gain:,}) = ${recapture:,}\n"
            f"  Recapture tax (25%): ${recapture_tax:,}\n\n"
            f"3. LONG-TERM CAPITAL GAINS\n"
            f"  Gain above recapture: ${total_gain:,} - ${recapture:,} = ${ltcg_portion:,}\n"
            f"  LTCG tax (20%): ${ltcg_tax:,}\n\n"
            f"4. NET INVESTMENT INCOME TAX\n"
            f"  Total gain subject to NIIT: ${total_gain:,}\n"
            f"  NIIT (3.8%): ${niit_tax:,}\n\n"
            f"5. TOTAL TAX LIABILITY\n"
            f"  Section 1250 recapture: ${recapture_tax:,}\n"
            f"  LTCG: ${ltcg_tax:,}\n"
            f"  NIIT: ${niit_tax:,}\n"
            f"  Total: ${total_tax:,}\n"
            f"  Effective tax rate: {effective_rate}%\n\n"
            f"6. AFTER-TAX PROCEEDS\n"
            f"  Sale price: ${sale_price:,}\n"
            f"  Less total tax: ${total_tax:,}\n"
            f"  After-tax proceeds: ${after_tax_proceeds:,}"
        )

        return {
            "question": question,
            "answer": answer,
            "numeric_targets": {
                "total_gain": total_gain,
                "recapture": recapture,
                "recapture_tax": recapture_tax,
                "ltcg_tax": ltcg_tax,
                "niit_tax": niit_tax,
                "total_tax": total_tax,
                "after_tax_proceeds": after_tax_proceeds,
            },
        }


# Task type -> builder function mapping
_TASK_BUILDERS = {
    "underwriting_calc": _build_underwriting_calc_task,
    "rent_roll_extraction": _build_rent_roll_extraction_task,
    "t12_normalization": _build_t12_normalization_task,
    "lease_abstract_extraction": _build_lease_abstract_task,
    "ic_memo": _build_ic_memo_task,
    "lease_reasoning": _build_lease_reasoning_task,
    "market_comp_narrative": _build_market_comp_task,
    "risk_triage": _build_risk_triage_task,
    "exchange_1031": _build_1031_exchange_task,
    "tax_analysis": _build_tax_analysis_task,
    "loi_deliverable": _build_loi_task,
    "structured_agent_output": _build_structured_agent_task,
}


def build_tasks_for_deal(
    deal_dict: dict,
    deal_idx: int,
    master_seed: int,
    tasks_per_deal: int,
) -> list[dict]:
    """Build N training records from a single deal skeleton."""
    records = []
    for t in range(tasks_per_deal):
        rng = _task_rng(master_seed, deal_idx, t)
        task_type = _pick_task_type(rng)
        difficulty = _pick_difficulty(rng)

        builder_fn = _TASK_BUILDERS.get(task_type)
        if builder_fn is None:
            continue

        result = builder_fn(deal_dict, difficulty, rng)
        if result is None:
            continue

        system_prompt = TASK_SYSTEM_PROMPTS.get(task_type, SWARMCRE_SYSTEM_PROMPT)

        record_id_payload = f"{master_seed}:{deal_idx}:{t}:{task_type}"
        record_id = hashlib.sha256(record_id_payload.encode()).hexdigest()[:16]

        record = {
            "id": f"swarmcre-{record_id}",
            "deal_id": deal_dict.get("deal_id", ""),
            "task_type": task_type,
            "difficulty": difficulty,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": result["question"]},
                {"role": "assistant", "content": result["answer"]},
            ],
            "gold": {
                "numeric_targets": result.get("numeric_targets", {}),
            },
            "metadata": {
                "task_type": task_type,
                "difficulty": difficulty,
                "asset_type": deal_dict.get("asset_type", ""),
                "market_tier": deal_dict.get("market_tier", ""),
                "market_name": deal_dict.get("market_name", ""),
                "sf": deal_dict.get("sf", 0),
                "seed": master_seed,
                "deal_index": deal_idx,
                "task_index": t,
            },
        }
        records.append(record)

    return records


# ═══════════════════════════════════════════════════════════════════
# DATASET BUILDER
# ═══════════════════════════════════════════════════════════════════


def _build_shard_worker(
    shard: int,
    seed: int,
    num_deals: int,
    num_shards: int,
    tasks_per_deal: int,
    output_dir: str,
) -> dict:
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    output_path = Path(output_dir)
    builder = DatasetBuilder.__new__(DatasetBuilder)
    config = BuildConfig(
        seed=seed,
        num_deals=num_deals,
        num_shards=num_shards,
        output_dir=output_path,
        tasks_per_deal=tasks_per_deal,
    )
    builder.config = config
    builder.generator = DealGenerator(seed=seed)
    builder.quality = QualityPipeline()
    builder.enricher = None
    return builder.build_shard(shard)


class DatasetBuilder:
    """Sharded multiprocessing orchestrator for the SwarmCRE dataset."""

    def __init__(self, config: BuildConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy init of sub-components
        self.generator = DealGenerator(seed=config.seed)
        self.quality = QualityPipeline()
        self.enricher = None

        if config.enable_enrichment:
            try:
                from .enrichment import Enricher
                import os
                api_key = os.environ.get("TOGETHER_API_KEY", "")
                self.enricher = Enricher(enabled=True, api_key=api_key)
                log.info("Together.ai enrichment enabled")
            except Exception as e:
                log.warning("Enrichment requested but unavailable: %s", e)
                self.enricher = None

    # ── Progress / Resume ────────────────────────────────────

    def _progress_path(self, shard: int) -> Path:
        return self.config.output_dir / f".progress_shard_{shard:03d}.json"

    def _shard_path(self, shard: int) -> Path:
        return self.config.output_dir / f"shard_{shard:03d}.jsonl"

    def _load_progress(self, shard: int) -> set[int]:
        """Load set of already-processed deal indices for resume."""
        path = self._progress_path(shard)
        if not path.exists():
            return set()
        try:
            data = json.loads(path.read_text())
            return set(data.get("completed_indices", []))
        except (json.JSONDecodeError, KeyError):
            return set()

    def _save_progress(self, shard: int, completed: set[int]):
        """Save progress checkpoint for resume."""
        path = self._progress_path(shard)
        data = {
            "shard": shard,
            "completed_indices": sorted(completed),
            "count": len(completed),
            "timestamp": time.time(),
        }
        path.write_text(json.dumps(data))

    # ── Shard Builder ────────────────────────────────────────

    def build_shard(self, shard: int) -> dict:
        """Build one shard of the dataset. Returns stats dict."""
        t0 = time.time()
        shard_file = self._shard_path(shard)
        completed = self._load_progress(shard)
        deal_indices = list(range(shard, self.config.num_deals, self.config.num_shards))

        written = 0
        failed = 0
        skipped = len(completed)

        # Open in append mode for resume safety
        with open(shard_file, "a", encoding="utf-8") as f:
            for count, deal_idx in enumerate(deal_indices):
                if deal_idx in completed:
                    continue

                try:
                    # Generate deal skeleton
                    deal = self.generator.generate_deal(deal_idx)
                    deal_dict = deal.to_dict()

                    # Build tasks
                    records = build_tasks_for_deal(
                        deal_dict=deal_dict,
                        deal_idx=deal_idx,
                        master_seed=self.config.seed,
                        tasks_per_deal=self.config.tasks_per_deal,
                    )

                    # Quality check and write each record
                    for record in records:
                        passed, failures = self.quality.check(record)
                        if not passed:
                            failed += 1
                            log.debug(
                                "Record %s failed QA: %s",
                                record.get("id", ""),
                                "; ".join(failures),
                            )
                            continue

                        # Optional enrichment
                        if self.enricher is not None:
                            try:
                                record = self.enricher.enrich(record)
                            except Exception as e:
                                log.debug("Enrichment failed for %s: %s", record.get("id", ""), e)

                        # Stream write — one JSON line at a time
                        f.write(json.dumps(record, separators=(",", ":")) + "\n")
                        written += 1

                    completed.add(deal_idx)

                    # Save progress every 100 deals
                    if len(completed) % 100 == 0:
                        self._save_progress(shard, completed)

                except Exception as e:
                    log.error("Shard %d, deal %d failed: %s", shard, deal_idx, e)
                    failed += 1

        # Final progress save
        self._save_progress(shard, completed)

        elapsed = time.time() - t0
        stats = {
            "shard": shard,
            "written": written,
            "failed": failed,
            "skipped": skipped,
            "total_deals": len(deal_indices),
            "elapsed_sec": round(elapsed, 2),
        }
        log.info("Shard %d complete: %s", shard, stats)
        return stats

    # ── Full Build ───────────────────────────────────────────

    def build_all(self) -> dict:
        """Run all shards via ProcessPoolExecutor. Returns aggregate stats."""
        t0 = time.time()
        all_stats = []

        log.info(
            "Starting build: %d deals, %d shards, %d tasks/deal, seed=%d",
            self.config.num_deals,
            self.config.num_shards,
            self.config.tasks_per_deal,
            self.config.seed,
        )

        with ProcessPoolExecutor(max_workers=self.config.num_shards) as executor:
            futures = {}
            for shard in range(self.config.num_shards):
                future = executor.submit(
                    _build_shard_worker,
                    shard=shard,
                    seed=self.config.seed,
                    num_deals=self.config.num_deals,
                    num_shards=self.config.num_shards,
                    tasks_per_deal=self.config.tasks_per_deal,
                    output_dir=str(self.config.output_dir),
                )
                futures[future] = shard

            for future in as_completed(futures):
                shard = futures[future]
                try:
                    stats = future.result()
                    all_stats.append(stats)
                    log.info("Shard %d returned: written=%d, failed=%d, elapsed=%.1fs",
                             stats["shard"], stats["written"], stats["failed"],
                             stats["elapsed_sec"])
                except Exception as e:
                    log.error("Shard %d raised exception: %s", shard, e)
                    all_stats.append({
                        "shard": shard, "written": 0, "failed": 0,
                        "error": str(e), "elapsed_sec": 0,
                    })

        total_elapsed = time.time() - t0
        total_written = sum(s.get("written", 0) for s in all_stats)
        total_failed = sum(s.get("failed", 0) for s in all_stats)

        return {
            "shards": sorted(all_stats, key=lambda s: s["shard"]),
            "total_written": total_written,
            "total_failed": total_failed,
            "total_elapsed_sec": round(total_elapsed, 2),
        }

    # ── Merge Shards ─────────────────────────────────────────

    def merge_shards(self):
        """Concatenate shard files into final swarmcre_train.jsonl with deterministic shuffle."""
        final_path = self.config.output_dir / "swarmcre_train.jsonl"
        log.info("Merging shards into %s", final_path)

        # Collect all record IDs with their file positions for deterministic shuffle
        # We read line-by-line to avoid loading everything into RAM at once
        index = []  # list of (shard_file, line_offset, record_id_hash)

        for shard in range(self.config.num_shards):
            shard_file = self._shard_path(shard)
            if not shard_file.exists():
                log.warning("Shard file missing: %s", shard_file)
                continue

            offset = 0
            with open(shard_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    # Deterministic sort key from content hash
                    sort_key = hashlib.sha256(
                        f"{self.config.seed}:merge:{line[:64]}".encode()
                    ).hexdigest()[:16]
                    index.append((str(shard_file), line_num, sort_key))

        # Deterministic shuffle using seed
        rng = _random_mod.Random(self.config.seed)
        rng.shuffle(index)

        # Build a lookup: shard_file -> list of lines
        # To avoid repeated file opens, read each shard once into a line list
        shard_lines: dict[str, list[str]] = {}
        for shard in range(self.config.num_shards):
            shard_file = self._shard_path(shard)
            if not shard_file.exists():
                continue
            with open(shard_file, "r", encoding="utf-8") as f:
                shard_lines[str(shard_file)] = [
                    l.strip() for l in f if l.strip()
                ]

        # Write merged output
        written = 0
        with open(final_path, "w", encoding="utf-8") as out:
            for shard_file_str, line_num, _ in index:
                lines = shard_lines.get(shard_file_str)
                if lines is None or line_num >= len(lines):
                    continue
                out.write(lines[line_num] + "\n")
                written += 1

        log.info("Merged %d records into %s", written, final_path)
        return written

    # ── Status ───────────────────────────────────────────────

    def show_status(self):
        """Print progress per shard."""
        print(f"\nSwarmCRE Build Status — {self.config.output_dir}")
        print("=" * 70)
        total_complete = 0
        total_deals = self.config.num_deals

        for shard in range(self.config.num_shards):
            progress_path = self._progress_path(shard)
            shard_file = self._shard_path(shard)
            expected_deals = len(range(shard, self.config.num_deals, self.config.num_shards))

            if progress_path.exists():
                try:
                    data = json.loads(progress_path.read_text())
                    completed = data.get("count", 0)
                    ts = data.get("timestamp", 0)
                    ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "N/A"
                except (json.JSONDecodeError, KeyError):
                    completed = 0
                    ts_str = "N/A"
            else:
                completed = 0
                ts_str = "not started"

            shard_size = 0
            if shard_file.exists():
                shard_size = shard_file.stat().st_size

            pct = (completed / expected_deals * 100) if expected_deals > 0 else 0
            total_complete += completed

            print(
                f"  Shard {shard:3d}: {completed:>7,} / {expected_deals:>7,} deals "
                f"({pct:5.1f}%)  |  file: {shard_size / 1_048_576:.1f} MB  |  {ts_str}"
            )

        overall_pct = (total_complete / total_deals * 100) if total_deals > 0 else 0
        final_path = self.config.output_dir / "swarmcre_train.jsonl"
        final_size = final_path.stat().st_size if final_path.exists() else 0

        print("-" * 70)
        print(f"  Overall: {total_complete:>7,} / {total_deals:>7,} deals ({overall_pct:.1f}%)")
        if final_path.exists():
            print(f"  Final dataset: {final_path} ({final_size / 1_048_576:.1f} MB)")
        else:
            print("  Final dataset: not yet merged")
        print()
