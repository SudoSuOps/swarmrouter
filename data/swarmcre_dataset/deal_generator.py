"""
SwarmCRE Dataset Factory — Deal Universe Engine

Generates 100,000 unique deal skeletons from a single seed.
Every field is deterministic given (master_seed, deal_index).
SHA-256 per-deal seeding for collision resistance across 100K deals.
"""

import hashlib
import random
from dataclasses import dataclass, field
from typing import Iterator

from .constants import (
    ASSET_TYPES, MARKET_TIERS, CREDIT_TIERS, DEBT_TEMPLATES,
    EXPENSE_LINES, LEASE_TYPES, ESCALATION_TYPES,
    TENANT_PREFIXES, TENANT_SUFFIXES,
    PROPERTY_PREFIXES, PROPERTY_SUFFIXES,
)
from .underwriting_engine import UnderwritingEngine


@dataclass
class TenantInfo:
    name: str
    suite: str
    sf: int
    rent_psf: float
    annual_rent: int
    lease_start: str
    lease_end: str
    escalation_type: str
    escalation_rate: float
    lease_type: str
    credit_tier: str
    ti_allowance_psf: float
    guarantee_type: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "suite": self.suite,
            "sf": self.sf,
            "rent_psf": self.rent_psf,
            "annual_rent": self.annual_rent,
            "lease_start": self.lease_start,
            "lease_end": self.lease_end,
            "escalation_type": self.escalation_type,
            "escalation_rate": self.escalation_rate,
            "lease_type": self.lease_type,
            "credit_tier": self.credit_tier,
            "ti_allowance_psf": self.ti_allowance_psf,
            "guarantee_type": self.guarantee_type,
        }


@dataclass
class DealSkeleton:
    """A single CRE deal with all physical, financial, and market parameters."""

    deal_id: str
    seed: int

    # Asset
    asset_type: str
    asset_type_display: str

    # Physical
    sf: int
    clear_height_ft: int
    dock_doors: int
    grade_doors: int
    office_pct: float
    office_sf: int
    warehouse_sf: int
    truck_court_depth_ft: int
    trailer_spots: int
    column_spacing: str
    sprinkler_type: str
    year_built: int
    land_acres: float

    # IOS / Land specifics
    yard_acres: float = 0.0
    yard_spots: int = 0
    land_price_per_acre: int = 0
    entitlement_status: str = ""
    zoning: str = ""

    # Cold storage specifics
    cold_chain: bool = False
    temp_zones: list = field(default_factory=list)

    # Tenancy
    tenancy_type: str = "single"
    num_tenants: int = 1
    lease_type: str = "NNN"
    walt_years: float = 0.0
    credit_tier: str = "local_credit"

    # Rent roll
    rent_roll: list = field(default_factory=list)
    rent_roll_narrative: str = ""

    # Market
    market_tier: str = "primary"
    market_name: str = ""
    submarket: str = ""
    vacancy_rate: float = 0.05
    market_rent_psf: float = 7.00
    cap_rate: float = 0.060

    # Property identity
    property_name: str = ""
    property_address: str = ""

    # Financial (populated by underwriting engine)
    expense_lines: dict = field(default_factory=dict)
    management_fee_pct: float = 0.04
    debt: dict = field(default_factory=dict)
    gold: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for template rendering and JSONL output."""
        return {
            "deal_id": self.deal_id,
            "seed": self.seed,
            "asset_type": self.asset_type,
            "asset_type_display": self.asset_type_display,
            "sf": self.sf,
            "clear_height_ft": self.clear_height_ft,
            "dock_doors": self.dock_doors,
            "grade_doors": self.grade_doors,
            "office_pct": self.office_pct,
            "office_sf": self.office_sf,
            "warehouse_sf": self.warehouse_sf,
            "truck_court_depth_ft": self.truck_court_depth_ft,
            "trailer_spots": self.trailer_spots,
            "column_spacing": self.column_spacing,
            "sprinkler_type": self.sprinkler_type,
            "year_built": self.year_built,
            "land_acres": self.land_acres,
            "yard_acres": self.yard_acres,
            "yard_spots": self.yard_spots,
            "cold_chain": self.cold_chain,
            "temp_zones": self.temp_zones,
            "tenancy_type": self.tenancy_type,
            "num_tenants": self.num_tenants,
            "lease_type": self.lease_type,
            "walt_years": self.walt_years,
            "credit_tier": self.credit_tier,
            "rent_roll": [t.to_dict() if hasattr(t, "to_dict") else t for t in self.rent_roll],
            "market_tier": self.market_tier,
            "market_name": self.market_name,
            "submarket": self.submarket,
            "vacancy_rate": self.vacancy_rate,
            "market_rent_psf": self.market_rent_psf,
            "cap_rate": self.cap_rate,
            "property_name": self.property_name,
            "property_address": self.property_address,
            "expense_lines": self.expense_lines,
            "management_fee_pct": self.management_fee_pct,
            "debt": self.debt,
            "gold": self.gold,
        }


class DealGenerator:
    """Deterministic deal skeleton factory.

    Usage:
        gen = DealGenerator(seed=42)
        for deal in gen.generate_all(100_000):
            print(deal.deal_id, deal.sf, deal.gold["noi"])
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.uw = UnderwritingEngine()
        # Precompute asset type weights for weighted selection
        self._asset_types = list(ASSET_TYPES.keys())
        self._asset_weights = [ASSET_TYPES[k]["weight"] for k in self._asset_types]
        # Market tier weights
        self._market_tiers = list(MARKET_TIERS.keys())
        self._market_weights = [MARKET_TIERS[k]["weight"] for k in self._market_tiers]
        # Credit tier weights
        self._credit_tiers = list(CREDIT_TIERS.keys())
        self._credit_weights = [CREDIT_TIERS[k]["weight"] for k in self._credit_tiers]
        # Lease type weights
        self._lease_types = list(LEASE_TYPES.keys())
        self._lease_weights = [LEASE_TYPES[k]["weight"] for k in self._lease_types]
        # Escalation weights
        self._esc_types = list(ESCALATION_TYPES.keys())
        self._esc_weights = [ESCALATION_TYPES[k]["weight"] for k in self._esc_types]

    def _deal_rng(self, idx: int) -> random.Random:
        """Per-deal deterministic RNG using SHA-256 hash chain."""
        h = hashlib.sha256(f"{self.seed}:{idx}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def _weighted_choice(self, rng: random.Random, items: list, weights: list):
        """Weighted random choice using cumulative distribution."""
        total = sum(weights)
        r = rng.random() * total
        cumulative = 0.0
        for item, weight in zip(items, weights):
            cumulative += weight
            if r <= cumulative:
                return item
        return items[-1]

    def _rand_range(self, rng: random.Random, lo: float, hi: float) -> float:
        """Uniform random in [lo, hi]."""
        return lo + rng.random() * (hi - lo)

    def _rand_int_range(self, rng: random.Random, lo: int, hi: int) -> int:
        """Uniform random integer in [lo, hi]."""
        if lo >= hi:
            return lo
        return rng.randint(lo, hi)

    def _generate_property_name(self, rng: random.Random, market_name: str) -> str:
        prefix = rng.choice(PROPERTY_PREFIXES)
        suffix = rng.choice(PROPERTY_SUFFIXES)
        return f"{prefix} {suffix}"

    def _generate_address(self, rng: random.Random, market_name: str) -> str:
        number = rng.randint(100, 9999)
        streets = [
            "Industrial Blvd", "Commerce Dr", "Logistics Way", "Distribution Ave",
            "Enterprise Pkwy", "Warehouse Rd", "Freight Ln", "Airport Rd",
            "Gateway Blvd", "Business Park Dr", "Technology Dr", "Parkway Blvd",
        ]
        street = rng.choice(streets)
        # Extract city from market_name
        city = market_name.split("/")[0].split("(")[0].strip()
        return f"{number} {street}, {city}"

    def _generate_tenant_name(self, rng: random.Random, credit_tier: str) -> str:
        tier_info = CREDIT_TIERS[credit_tier]
        # 40% chance of using a real-ish example name, 60% procedural
        if rng.random() < 0.4:
            return rng.choice(tier_info["examples"])
        return f"{rng.choice(TENANT_PREFIXES)} {rng.choice(TENANT_SUFFIXES)}"

    def _generate_rent_roll(
        self, rng: random.Random, deal: DealSkeleton, market_rent_psf: float
    ) -> list[TenantInfo]:
        """Generate realistic rent roll for a deal."""
        tenants = []
        remaining_sf = deal.sf

        if deal.tenancy_type == "single":
            num_tenants = 1
        else:
            num_tenants = self._rand_int_range(rng, 2, min(12, max(2, deal.sf // 5000)))

        for i in range(num_tenants):
            if i == num_tenants - 1:
                tenant_sf = remaining_sf
            else:
                # Allocate 10-40% of remaining to this tenant
                pct = self._rand_range(rng, 0.10, 0.40)
                tenant_sf = max(1000, round(remaining_sf * pct / 1000) * 1000)
                tenant_sf = min(tenant_sf, remaining_sf - (num_tenants - i - 1) * 1000)

            remaining_sf -= tenant_sf

            # Credit tier for this tenant
            tenant_credit = self._weighted_choice(rng, self._credit_tiers, self._credit_weights)
            credit_info = CREDIT_TIERS[tenant_credit]

            # Rent with credit adjustment
            rent_psf = round(market_rent_psf * credit_info["rent_premium"], 2)
            # Add some noise (+/- 15%)
            rent_psf = round(rent_psf * self._rand_range(rng, 0.85, 1.15), 2)
            rent_psf = max(1.50, rent_psf)  # floor

            annual_rent = round(tenant_sf * rent_psf)

            # Lease dates
            start_year = self._rand_int_range(rng, 2020, 2025)
            start_month = self._rand_int_range(rng, 1, 12)
            term_years = self._rand_int_range(rng, 3, 15)
            end_year = start_year + term_years
            end_month = start_month

            # Escalation
            esc_type = self._weighted_choice(rng, self._esc_types, self._esc_weights)
            esc_info = ESCALATION_TYPES[esc_type]
            if esc_type == "fixed":
                esc_rate = round(self._rand_range(rng, esc_info["range"][0], esc_info["range"][1]), 3)
            else:
                esc_rate = 0.0

            # Lease type (inherits from deal unless multi-tenant with variation)
            if deal.num_tenants > 1 and rng.random() < 0.2:
                tenant_lease = self._weighted_choice(rng, self._lease_types, self._lease_weights)
            else:
                tenant_lease = deal.lease_type

            # TI allowance
            ti_range = credit_info["ti_allowance_psf_range"]
            ti_psf = round(self._rand_range(rng, ti_range[0], ti_range[1]), 2)

            tenants.append(TenantInfo(
                name=self._generate_tenant_name(rng, tenant_credit),
                suite=f"Suite {chr(65 + i)}" if num_tenants > 1 else "Entire Building",
                sf=tenant_sf,
                rent_psf=rent_psf,
                annual_rent=annual_rent,
                lease_start=f"{start_year}-{start_month:02d}-01",
                lease_end=f"{end_year}-{end_month:02d}-01",
                escalation_type=esc_type,
                escalation_rate=esc_rate,
                lease_type=tenant_lease,
                credit_tier=tenant_credit,
                ti_allowance_psf=ti_psf,
                guarantee_type=credit_info["guarantee"],
            ))

        return tenants

    def _generate_expenses(
        self, rng: random.Random, sf: int, lease_type: str
    ) -> dict[str, int]:
        """Generate operating expense line items."""
        expenses = {}
        for line_name, info in EXPENSE_LINES.items():
            if line_name == "management_fee_pct":
                continue  # handled separately via EGI
            rate_psf = round(self._rand_range(rng, info["range"][0], info["range"][1]), 2)
            expenses[line_name] = round(sf * rate_psf)
        return expenses

    def _compute_walt(self, rent_roll: list[TenantInfo], as_of_year: int = 2025) -> float:
        """Weighted average lease term (years remaining)."""
        total_sf = sum(t.sf for t in rent_roll)
        if total_sf == 0:
            return 0.0
        weighted_years = 0.0
        for t in rent_roll:
            end_year = int(t.lease_end.split("-")[0])
            remaining = max(0, end_year - as_of_year)
            weighted_years += remaining * t.sf
        return round(weighted_years / total_sf, 1)

    def _generate_rent_roll_narrative(self, deal: DealSkeleton) -> str:
        """Generate a text narrative of the rent roll for extraction tasks."""
        parts = []
        parts.append(
            f"{deal.property_name} is a {deal.sf:,} SF {deal.asset_type_display} "
            f"located at {deal.property_address} in the {deal.submarket} submarket "
            f"of {deal.market_name}."
        )

        if deal.num_tenants == 1:
            t = deal.rent_roll[0]
            parts.append(
                f"The property is 100% leased to {t.name} on a "
                f"{deal.lease_type} lease at ${t.rent_psf:.2f}/SF "
                f"(${t.annual_rent:,} annually). The lease runs from "
                f"{t.lease_start} to {t.lease_end} with "
                f"{'a ' + str(round(t.escalation_rate * 100, 1)) + '% annual escalation' if t.escalation_type == 'fixed' else 'CPI-based escalations' if t.escalation_type == 'cpi' else 'flat rent'}."
            )
        else:
            total_sf = sum(t.sf for t in deal.rent_roll)
            occupied_sf = total_sf  # all tenants in roll are occupying
            parts.append(
                f"The property is currently {occupied_sf / deal.sf * 100:.0f}% occupied "
                f"by {deal.num_tenants} tenants with a {deal.walt_years:.1f}-year WALT."
            )
            parts.append("Current rent roll:")
            for t in deal.rent_roll:
                parts.append(
                    f"- {t.name} ({t.suite}): {t.sf:,} SF at ${t.rent_psf:.2f}/SF "
                    f"{t.lease_type}, {t.escalation_type} escalation"
                    + (f" at {t.escalation_rate:.1%}" if t.escalation_type == "fixed" else "")
                    + f", expires {t.lease_end}"
                )

        return "\n".join(parts)

    def generate_deal(self, idx: int) -> DealSkeleton:
        """Generate one deal skeleton deterministically from index."""
        rng = self._deal_rng(idx)
        deal_id = f"DEAL-{self.seed}-{idx:06d}"

        # 1. Asset type
        asset_type = self._weighted_choice(rng, self._asset_types, self._asset_weights)
        asset_info = ASSET_TYPES[asset_type]

        # 2. Market context
        market_tier = self._weighted_choice(rng, self._market_tiers, self._market_weights)
        tier_info = MARKET_TIERS[market_tier]
        market_tuple = rng.choice(tier_info["markets"])
        market_name = market_tuple[0]
        submarket = rng.choice(market_tuple[1])

        # 3. Physical specs
        if asset_type == "industrial_land":
            # Land deals have no building
            sf = 0
            clear_height = 0
            dock_doors = 0
            grade_doors = 0
            office_pct = 0.0
            truck_court = 0
            trailer_spots = 0
            column_spacing = "N/A"
            sprinkler = "N/A"
            year_built = 0
            land_acres = round(self._rand_range(
                rng, asset_info["land_acres_range"][0], asset_info["land_acres_range"][1]
            ), 1)
            land_price = self._rand_int_range(
                rng,
                asset_info["price_per_acre_range"][0],
                asset_info["price_per_acre_range"][1],
            )
        else:
            sf_lo, sf_hi = asset_info["sf_range"]
            sf = round(self._rand_int_range(rng, sf_lo, sf_hi) / 1000) * 1000  # round to nearest 1000
            sf = max(sf_lo, sf)

            ch_lo, ch_hi = asset_info["clear_height_range"]
            clear_height = self._rand_int_range(rng, ch_lo, ch_hi)

            office_pct = round(self._rand_range(
                rng, asset_info["office_pct_range"][0], asset_info["office_pct_range"][1]
            ), 2)

            dock_lo, dock_hi = asset_info["dock_doors_per_10k_sf"]
            dock_doors = max(0, round(sf / 10000 * self._rand_range(rng, dock_lo, dock_hi)))

            grade_lo, grade_hi = asset_info["grade_doors_per_10k_sf"]
            grade_doors = max(0, round(sf / 10000 * self._rand_range(rng, grade_lo, grade_hi)))

            tc_lo, tc_hi = asset_info["truck_court_range"]
            truck_court = self._rand_int_range(rng, tc_lo, tc_hi) if tc_hi > 0 else 0

            ts_lo, ts_hi = asset_info["trailer_spots_per_10k_sf"]
            trailer_spots = max(0, round(sf / 10000 * self._rand_range(rng, ts_lo, ts_hi)))

            column_spacing = rng.choice(asset_info["column_spacing_options"])
            sprinkler = rng.choice(asset_info["sprinkler_options"])

            yb_lo, yb_hi = asset_info["year_built_range"]
            year_built = self._rand_int_range(rng, yb_lo, yb_hi)

            lr_lo, lr_hi = asset_info["land_ratio_range"]
            land_acres = round(sf / 100000 * self._rand_range(rng, lr_lo, lr_hi), 1)
            land_price = 0

        office_sf = round(sf * office_pct) if sf > 0 else 0
        warehouse_sf = sf - office_sf

        # 4. IOS specifics
        yard_acres = 0.0
        yard_spots = 0
        if asset_type == "ios_truck_yard":
            ya_lo, ya_hi = asset_info["yard_acres_range"]
            yard_acres = round(self._rand_range(rng, ya_lo, ya_hi), 1)
            ys_lo, ys_hi = asset_info["yard_spots_range"]
            yard_spots = self._rand_int_range(rng, ys_lo, ys_hi)

        # 5. Cold storage specifics
        cold_chain = asset_info.get("cold_chain", False)
        temp_zones = []
        if cold_chain:
            temp_zones = rng.sample(asset_info["temp_zones"], k=rng.randint(1, len(asset_info["temp_zones"])))

        # 6. Entitlement (land only)
        entitlement = ""
        zoning = ""
        if asset_type == "industrial_land":
            entitlement = rng.choice(asset_info["entitlement_options"])
            zoning = rng.choice(asset_info["zoning_options"])

        # 7. Tenancy + lease type
        tenancy_type = rng.choice(asset_info["typical_tenancy"])
        if tenancy_type == "none":
            tenancy_type = "none"
        lease_type = self._weighted_choice(rng, self._lease_types, self._lease_weights)
        credit_tier = self._weighted_choice(rng, self._credit_tiers, self._credit_weights)

        # 8. Market rents and cap rate
        rent_lo, rent_hi = tier_info["rent_psf_range"]
        market_rent_psf = round(self._rand_range(rng, rent_lo, rent_hi), 2)

        vac_lo, vac_hi = tier_info["vacancy_range"]
        vacancy_rate = round(self._rand_range(rng, vac_lo, vac_hi), 3)

        cap_lo, cap_hi = asset_info["cap_rate_range"]
        base_cap = self._rand_range(rng, cap_lo, cap_hi)
        cap_rate = round(base_cap + tier_info["cap_rate_adjustment"] + CREDIT_TIERS[credit_tier]["cap_discount"], 4)
        cap_rate = max(0.030, cap_rate)  # floor at 3%

        # Property identity
        property_name = self._generate_property_name(rng, market_name)
        property_address = self._generate_address(rng, market_name)

        # Create skeleton
        deal = DealSkeleton(
            deal_id=deal_id,
            seed=idx,
            asset_type=asset_type,
            asset_type_display=asset_info["display"],
            sf=sf,
            clear_height_ft=clear_height,
            dock_doors=dock_doors,
            grade_doors=grade_doors,
            office_pct=office_pct,
            office_sf=office_sf,
            warehouse_sf=warehouse_sf,
            truck_court_depth_ft=truck_court,
            trailer_spots=trailer_spots,
            column_spacing=column_spacing,
            sprinkler_type=sprinkler,
            year_built=year_built,
            land_acres=land_acres,
            yard_acres=yard_acres,
            yard_spots=yard_spots,
            land_price_per_acre=land_price,
            entitlement_status=entitlement,
            zoning=zoning,
            cold_chain=cold_chain,
            temp_zones=temp_zones,
            tenancy_type=tenancy_type,
            num_tenants=1 if tenancy_type == "single" else 0,
            lease_type=lease_type,
            credit_tier=credit_tier,
            market_tier=market_tier,
            market_name=market_name,
            submarket=submarket,
            vacancy_rate=vacancy_rate,
            market_rent_psf=market_rent_psf,
            cap_rate=cap_rate,
            property_name=property_name,
            property_address=property_address,
        )

        # Skip rent roll and financials for land deals
        if asset_type == "industrial_land":
            deal.gold = {
                "land_value": round(land_acres * land_price),
                "price_per_acre": land_price,
                "land_acres": land_acres,
                "entitlement": entitlement,
                "zoning": zoning,
            }
            return deal

        # 9. Generate rent roll
        rent_roll = self._generate_rent_roll(rng, deal, market_rent_psf)
        deal.rent_roll = rent_roll
        deal.num_tenants = len(rent_roll)
        deal.walt_years = self._compute_walt(rent_roll)

        # 10. Generate expenses
        expense_lines = self._generate_expenses(rng, sf, lease_type)
        deal.expense_lines = expense_lines
        deal.management_fee_pct = round(self._rand_range(rng, 0.03, 0.06), 3)

        # 11. Debt terms
        debt_template = rng.choice(DEBT_TEMPLATES)
        # Compute preliminary value for LTV-based loan sizing
        rent_roll_dicts = [t.to_dict() for t in rent_roll]
        pgi = UnderwritingEngine.compute_pgi(rent_roll_dicts)
        egi = UnderwritingEngine.compute_egi(pgi, vacancy_rate)
        mgmt_fee = UnderwritingEngine.compute_management_fee(egi, deal.management_fee_pct)
        temp_expenses = dict(expense_lines)
        temp_expenses["management_fee"] = mgmt_fee
        total_opex = UnderwritingEngine.compute_total_opex(temp_expenses)
        noi = UnderwritingEngine.compute_noi(egi, total_opex)
        preliminary_value = UnderwritingEngine.compute_value(noi, cap_rate)

        ltv = round(self._rand_range(
            rng, debt_template["ltv_range"][0], debt_template["ltv_range"][1]
        ), 3)
        loan_amount = UnderwritingEngine.compute_loan_amount_from_ltv(preliminary_value, ltv)
        rate = round(self._rand_range(
            rng, debt_template["rate_range"][0], debt_template["rate_range"][1]
        ), 4)

        deal.debt = {
            "lender_type": debt_template["name"],
            "lender_display": debt_template["display"],
            "loan_amount": loan_amount,
            "ltv": ltv,
            "rate": rate,
            "amort_years": debt_template["amort_years"],
            "io_years": debt_template["io_years"],
            "term_years": debt_template["term_years"],
            "min_dscr": debt_template["min_dscr"],
        }

        # 12. Compute gold values
        gold_input = {
            "rent_roll": rent_roll_dicts,
            "vacancy_rate": vacancy_rate,
            "expense_lines": expense_lines,
            "cap_rate": cap_rate,
            "sf": sf,
            "management_fee_pct": deal.management_fee_pct,
            "debt": deal.debt,
        }
        deal.gold = UnderwritingEngine.compute_gold(gold_input)

        # 13. Rent roll narrative
        deal.rent_roll_narrative = self._generate_rent_roll_narrative(deal)

        return deal

    def generate_all(self, count: int = 100_000) -> Iterator[DealSkeleton]:
        """Yield all deal skeletons."""
        for idx in range(count):
            yield self.generate_deal(idx)

    def generate_shard(
        self, shard: int, num_shards: int, count: int = 100_000
    ) -> Iterator[DealSkeleton]:
        """Yield deals for a specific shard."""
        for idx in range(shard, count, num_shards):
            yield self.generate_deal(idx)
