#!/usr/bin/env python3
"""
Property Intelligence Object (PIO) — v0.1
==========================================
One call. One object. Everything an agent needs.

GET /pio/{address}
→ address, asset_type, estimated_value, cap_rate, rent_sqft,
  top_10_comps[], risk_factors[], confidence_score, sources[]

Data sources:
  - ATTOM API (property details, sales history, assessments, mortgage)
  - SEC EDGAR (REIT 10-K/10-Q filings — free)
  - FRED (cap rate spreads, treasury yields — free)
  - Census ACS (demographics — free)

Usage:
    # As module
    from pio import PropertyIntelligenceObject
    pio = PropertyIntelligenceObject(attom_key="...")
    obj = pio.get("1200 S Industrial Blvd, Dallas, TX 75207")

    # As API server
    python3 pio.py --serve --port 8090

    # CLI lookup
    python3 pio.py --address "1200 S Industrial Blvd, Dallas, TX 75207"
"""

import json
import os
import sys
import time
import hashlib
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx


# ═══════════════════════════════════════════════════════════
# INTELLIGENCE OBJECT SCHEMA
# ═══════════════════════════════════════════════════════════

@dataclass
class Comp:
    address: str
    sale_price: float
    price_per_sf: float
    cap_rate: Optional[float]
    sale_date: str
    building_sf: int
    distance_miles: float
    similarity_score: float
    source: str


@dataclass
class RiskFactor:
    code: str          # e.g. "DEBT_MATURITY_12MO", "VACANCY_ABOVE_MARKET"
    severity: str      # "low", "medium", "high", "critical"
    detail: str
    mitigation: str


@dataclass
class PropertyIntelligenceObj:
    """The PIO — what agents actually want."""
    # Identity
    object_id: str
    object_type: str = "property_intelligence_object"
    version: str = "0.1"

    # Property
    address: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    county: str = ""
    asset_type: str = ""          # industrial, office, retail, multifamily
    asset_subtype: str = ""       # warehouse, flex, cold_storage, etc.
    building_sf: int = 0
    land_sf: int = 0
    year_built: int = 0
    stories: int = 0

    # Valuation
    estimated_value: float = 0.0
    value_method: str = ""        # "assessment", "comp_derived", "income_approach"
    assessed_value: float = 0.0
    tax_amount: float = 0.0

    # Income
    cap_rate: float = 0.0
    noi_estimated: float = 0.0
    rent_per_sf: float = 0.0
    occupancy: float = 0.0

    # Debt
    mortgage_amount: float = 0.0
    mortgage_rate: float = 0.0
    mortgage_date: str = ""
    lender: str = ""
    ltv_estimated: float = 0.0

    # Ownership
    owner_name: str = ""
    owner_type: str = ""          # private, institutional, REIT, fund
    acquisition_date: str = ""
    acquisition_price: float = 0.0

    # Comps
    top_10_comps: list = field(default_factory=list)

    # Risk
    risk_factors: list = field(default_factory=list)
    risk_score: float = 0.0       # 0-100, lower = safer

    # Confidence
    confidence_score: float = 0.0  # 0-1.0
    sources: list = field(default_factory=list)
    source_count: int = 0

    # Meta
    created_at: str = ""
    ttl_seconds: int = 86400
    market_data_as_of: str = ""

    def to_dict(self):
        d = asdict(self)
        return d

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ═══════════════════════════════════════════════════════════
# DATA CONNECTORS
# ═══════════════════════════════════════════════════════════

class ATTOMConnector:
    """ATTOM Property Data API — $500/mo, 150M+ properties."""
    BASE = "https://api.gateway.attomdata.com/propertyapi/v1.0.0"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.Client(timeout=30, headers={"APIKey": api_key, "Accept": "application/json"})

    def property_detail(self, address: str) -> dict:
        """GET /property/detail — full property characteristics."""
        resp = self.client.get(f"{self.BASE}/property/detail",
                              params={"address1": address})
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.status_code, "detail": resp.text[:200]}

    def sale_history(self, address: str) -> dict:
        """GET /saleshistory/detail — 10 years of sales."""
        resp = self.client.get(f"{self.BASE}/saleshistory/detail",
                              params={"address1": address})
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.status_code}

    def assessment(self, address: str) -> dict:
        """GET /assessment/detail — tax assessments."""
        resp = self.client.get(f"{self.BASE}/assessment/detail",
                              params={"address1": address})
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.status_code}

    def mortgage(self, address: str) -> dict:
        """GET /property/detailmortgage — mortgage data."""
        resp = self.client.get(f"{self.BASE}/property/detailmortgage",
                              params={"address1": address})
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.status_code}

    def nearby_sales(self, lat: float, lon: float, radius_miles: float = 5,
                     min_sf: int = 0, max_sf: int = 999999) -> dict:
        """GET /sale/snapshot — nearby sale comps."""
        resp = self.client.get(f"{self.BASE}/sale/snapshot",
                              params={
                                  "latitude": lat, "longitude": lon,
                                  "radius": radius_miles,
                                  "minsalesamt": 100000,
                                  "propertytype": "COMMERCIAL",
                              })
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.status_code}


class FREDConnector:
    """FRED API — free, 277 CRE-tagged series."""
    BASE = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")

    def get_series(self, series_id: str, limit: int = 12) -> list:
        """Get latest observations for a series."""
        if not self.api_key:
            return []
        resp = httpx.get(f"{self.BASE}/series/observations",
                         params={
                             "series_id": series_id,
                             "api_key": self.api_key,
                             "file_type": "json",
                             "sort_order": "desc",
                             "limit": limit,
                         }, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("observations", [])
        return []

    def treasury_10yr(self) -> float:
        """Current 10-year Treasury yield."""
        obs = self.get_series("DGS10", limit=1)
        if obs and obs[0].get("value", ".") != ".":
            return float(obs[0]["value"]) / 100
        return 0.043  # fallback


class EDGARConnector:
    """SEC EDGAR — free, all public REIT filings."""
    BASE = "https://efts.sec.gov/LATEST"
    HEADERS = {"User-Agent": "SwarmCRE/1.0 research@swarm.com"}

    def search_filings(self, company: str, form_type: str = "10-K") -> list:
        """Full-text search across EDGAR filings."""
        resp = httpx.get(f"{self.BASE}/search-index",
                         params={
                             "q": company,
                             "forms": form_type,
                             "dateRange": "custom",
                             "startdt": "2024-01-01",
                         },
                         headers=self.HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("hits", {}).get("hits", [])
        return []

    def company_facts(self, cik: str) -> dict:
        """XBRL structured data for a company."""
        cik_padded = cik.zfill(10)
        resp = httpx.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json",
            headers=self.HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        return {}


# ═══════════════════════════════════════════════════════════
# PIO BUILDER — Assembles Intelligence Objects
# ═══════════════════════════════════════════════════════════

# Industrial market cap rates by tier (baseline estimates)
MARKET_CAP_RATES = {
    "primary": 0.050,    # LA, NYC, SF, Chicago, DFW
    "secondary": 0.060,  # Nashville, Raleigh, Austin, Denver
    "tertiary": 0.070,   # Smaller MSAs
}

# Industrial rent estimates by market tier (NNN $/SF/yr)
MARKET_RENTS = {
    "primary": 9.50,
    "secondary": 7.00,
    "tertiary": 5.50,
}

# Asset type classification from property use codes
ASSET_TYPE_MAP = {
    "warehouse": "industrial", "industrial": "industrial",
    "manufacturing": "industrial", "distribution": "industrial",
    "flex": "industrial", "cold storage": "industrial",
    "office": "office", "medical office": "office",
    "retail": "retail", "shopping": "retail", "restaurant": "retail",
    "multifamily": "multifamily", "apartment": "multifamily",
    "hotel": "hospitality", "motel": "hospitality",
}


class PIOBuilder:
    """Assembles Property Intelligence Objects from multiple sources."""

    def __init__(self, attom_key: str = None, fred_key: str = None):
        self.attom = ATTOMConnector(attom_key) if attom_key else None
        self.fred = FREDConnector(fred_key)
        self.edgar = EDGARConnector()

    def build(self, address: str) -> PropertyIntelligenceObj:
        """Build a complete PIO for an address."""
        pio = PropertyIntelligenceObj(
            object_id=f"pio_{hashlib.md5(address.encode()).hexdigest()[:12]}",
            address=address,
            created_at=datetime.now().isoformat(),
        )

        sources_used = []
        confidence_factors = []

        # ── ATTOM: Property + Assessment + Mortgage + Sales ────
        if self.attom:
            try:
                prop = self.attom.property_detail(address)
                self._parse_attom_property(pio, prop)
                sources_used.append("attom:property")
                confidence_factors.append(0.3)
            except Exception as e:
                pio.risk_factors.append(asdict(RiskFactor(
                    code="DATA_GAP_PROPERTY", severity="medium",
                    detail=f"ATTOM property lookup failed: {e}",
                    mitigation="Verify property details via county assessor"
                )))

            try:
                assess = self.attom.assessment(address)
                self._parse_attom_assessment(pio, assess)
                sources_used.append("attom:assessment")
                confidence_factors.append(0.2)
            except:
                pass

            try:
                mortgage = self.attom.mortgage(address)
                self._parse_attom_mortgage(pio, mortgage)
                sources_used.append("attom:mortgage")
                confidence_factors.append(0.15)
            except:
                pass

            try:
                sales = self.attom.sale_history(address)
                self._parse_attom_sales(pio, sales)
                sources_used.append("attom:sales_history")
                confidence_factors.append(0.15)
            except:
                pass

        # ── FRED: Market rates ─────────────────────────────────
        try:
            treasury_10yr = self.fred.treasury_10yr()
            pio.market_data_as_of = datetime.now().strftime("%Y-%m-%d")
            sources_used.append("fred:treasury")
            confidence_factors.append(0.1)
        except:
            treasury_10yr = 0.043

        # ── DERIVED: Valuation + Risk ──────────────────────────
        self._derive_valuation(pio, treasury_10yr)
        self._assess_risk(pio, treasury_10yr)

        # ── CONFIDENCE SCORE ───────────────────────────────────
        pio.sources = sources_used
        pio.source_count = len(sources_used)
        pio.confidence_score = round(min(sum(confidence_factors), 1.0), 2)

        return pio

    def build_demo(self, address: str) -> PropertyIntelligenceObj:
        """Build a PIO using estimated/demo data (no API keys needed)."""
        pio = PropertyIntelligenceObj(
            object_id=f"pio_{hashlib.md5(address.encode()).hexdigest()[:12]}",
            address=address,
            created_at=datetime.now().isoformat(),
        )

        # Parse address for market estimation
        parts = [p.strip() for p in address.split(",")]
        city = parts[1] if len(parts) > 1 else ""
        state = parts[2].strip().split()[0] if len(parts) > 2 else ""

        pio.city = city
        pio.state = state
        pio.asset_type = "industrial"
        pio.asset_subtype = "warehouse"

        # Estimate based on market tier
        tier = "secondary"
        primary_cities = ["dallas", "houston", "los angeles", "chicago", "new york",
                         "atlanta", "phoenix", "san francisco", "miami", "seattle"]
        if any(c in city.lower() for c in primary_cities):
            tier = "primary"

        cap_rate = MARKET_CAP_RATES[tier]
        rent_sf = MARKET_RENTS[tier]

        # Demo property
        pio.building_sf = 85000
        pio.land_sf = 175000
        pio.year_built = 2018
        pio.stories = 1
        pio.rent_per_sf = rent_sf
        pio.cap_rate = cap_rate
        pio.occupancy = 0.95
        pio.noi_estimated = round(pio.building_sf * rent_sf * pio.occupancy * 0.85, 0)  # 85% NOI margin
        pio.estimated_value = round(pio.noi_estimated / cap_rate, 0)
        pio.value_method = "income_approach_estimated"
        pio.assessed_value = round(pio.estimated_value * 0.85, 0)
        pio.tax_amount = round(pio.assessed_value * 0.022, 0)

        # Demo debt
        pio.mortgage_amount = round(pio.estimated_value * 0.65, 0)
        pio.mortgage_rate = 0.065
        pio.lender = "Regional Bank"
        pio.ltv_estimated = 0.65

        # Demo owner
        pio.owner_name = "Industrial Holdings LLC"
        pio.owner_type = "private"

        # Demo comps
        import random
        random.seed(hash(address))
        for i in range(10):
            sf = random.randint(50000, 150000)
            price_sf = random.uniform(95, 165)
            pio.top_10_comps.append(asdict(Comp(
                address=f"{random.randint(100,9999)} Industrial Pkwy, {city}, {state}",
                sale_price=round(sf * price_sf, 0),
                price_per_sf=round(price_sf, 2),
                cap_rate=round(cap_rate + random.uniform(-0.01, 0.015), 4),
                sale_date=(datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
                building_sf=sf,
                distance_miles=round(random.uniform(0.5, 8.0), 1),
                similarity_score=round(random.uniform(0.70, 0.98), 2),
                source="demo_estimated",
            )))
        pio.top_10_comps.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Risk assessment
        treasury_10yr = 0.043
        self._assess_risk(pio, treasury_10yr)

        pio.sources = ["market_estimates", "demo_comps"]
        pio.source_count = 2
        pio.confidence_score = 0.35  # demo = low confidence
        pio.market_data_as_of = datetime.now().strftime("%Y-%m-%d")

        return pio

    def _parse_attom_property(self, pio, data):
        """Extract property details from ATTOM response."""
        if "error" in data:
            return
        props = data.get("property", [{}])
        if isinstance(props, list) and props:
            p = props[0]
        else:
            p = props

        addr = p.get("address", {})
        pio.city = addr.get("locality", "")
        pio.state = addr.get("countrySubd", "")
        pio.zip_code = addr.get("postal1", "")
        pio.county = addr.get("countrySecSubd", "")

        building = p.get("building", {})
        pio.building_sf = building.get("size", {}).get("bldgsize", 0) or 0
        pio.year_built = building.get("summary", {}).get("yearbuilt", 0) or 0
        pio.stories = building.get("summary", {}).get("stories", 0) or 0

        lot = p.get("lot", {})
        pio.land_sf = lot.get("lotsize1", 0) or 0

        # Asset type from use code
        use_desc = str(p.get("summary", {}).get("propclass", "")).lower()
        for keyword, atype in ASSET_TYPE_MAP.items():
            if keyword in use_desc:
                pio.asset_type = atype
                pio.asset_subtype = keyword
                break

    def _parse_attom_assessment(self, pio, data):
        if "error" in data:
            return
        props = data.get("property", [{}])
        p = props[0] if isinstance(props, list) and props else props
        assessment = p.get("assessment", {})
        pio.assessed_value = assessment.get("assessed", {}).get("assdttlvalue", 0) or 0
        pio.tax_amount = assessment.get("tax", {}).get("taxamt", 0) or 0

    def _parse_attom_mortgage(self, pio, data):
        if "error" in data:
            return
        props = data.get("property", [{}])
        p = props[0] if isinstance(props, list) and props else props
        mortgage = p.get("mortgage", {})
        if mortgage:
            pio.mortgage_amount = mortgage.get("amount", 0) or 0
            pio.mortgage_rate = (mortgage.get("intrate", 0) or 0) / 100
            pio.mortgage_date = mortgage.get("date", "")
            pio.lender = mortgage.get("lender", {}).get("lendername", "")

    def _parse_attom_sales(self, pio, data):
        if "error" in data:
            return
        props = data.get("property", [{}])
        p = props[0] if isinstance(props, list) and props else props
        sales = p.get("salehistory", [])
        if sales:
            latest = sales[0]
            pio.acquisition_price = latest.get("amount", {}).get("saleamt", 0) or 0
            pio.acquisition_date = latest.get("amount", {}).get("salerecdate", "")
            owner = p.get("assessment", {}).get("owner", {})
            pio.owner_name = owner.get("owner1", {}).get("fullname", "")

    def _derive_valuation(self, pio, treasury_10yr):
        """Derive estimated value from available data."""
        # If we have assessed value, estimate market value (typically 85-95% of market)
        if pio.assessed_value > 0 and pio.estimated_value == 0:
            pio.estimated_value = round(pio.assessed_value / 0.90, 0)
            pio.value_method = "assessment_derived"

        # If we have acquisition price, use as floor
        if pio.acquisition_price > 0:
            if pio.estimated_value == 0 or pio.acquisition_price > pio.estimated_value:
                pio.estimated_value = pio.acquisition_price
                pio.value_method = "last_sale"

        # Estimate cap rate from market
        if pio.cap_rate == 0:
            pio.cap_rate = 0.058  # conservative industrial estimate

        # Estimate NOI
        if pio.estimated_value > 0 and pio.noi_estimated == 0:
            pio.noi_estimated = round(pio.estimated_value * pio.cap_rate, 0)

        # Estimate rent/SF
        if pio.building_sf > 0 and pio.noi_estimated > 0 and pio.rent_per_sf == 0:
            # NOI margin ~70-85% for industrial NNN
            pio.rent_per_sf = round(pio.noi_estimated / (pio.building_sf * 0.78), 2)

        # Estimate LTV
        if pio.mortgage_amount > 0 and pio.estimated_value > 0:
            pio.ltv_estimated = round(pio.mortgage_amount / pio.estimated_value, 3)

    def _assess_risk(self, pio, treasury_10yr):
        """Generate risk factors."""
        risk_score = 0

        # Cap rate spread
        spread = pio.cap_rate - treasury_10yr
        if spread < 0.01:
            pio.risk_factors.append(asdict(RiskFactor(
                code="THIN_CAP_SPREAD", severity="high",
                detail=f"Cap rate spread over 10yr Treasury is only {spread*100:.0f}bps",
                mitigation="Verify rent growth supports current pricing"
            )))
            risk_score += 25
        elif spread < 0.015:
            pio.risk_factors.append(asdict(RiskFactor(
                code="COMPRESSED_CAP_SPREAD", severity="medium",
                detail=f"Cap rate spread is {spread*100:.0f}bps — below historical average of 200bps",
                mitigation="Stress test exit cap rate +50bps"
            )))
            risk_score += 15

        # LTV
        if pio.ltv_estimated > 0.75:
            pio.risk_factors.append(asdict(RiskFactor(
                code="HIGH_LTV", severity="high",
                detail=f"LTV of {pio.ltv_estimated:.0%} exceeds conventional threshold of 75%",
                mitigation="Confirm DSCR > 1.25x and verify appraisal"
            )))
            risk_score += 20
        elif pio.ltv_estimated > 0.65:
            pio.risk_factors.append(asdict(RiskFactor(
                code="ELEVATED_LTV", severity="medium",
                detail=f"LTV of {pio.ltv_estimated:.0%} — moderate leverage",
                mitigation="Standard — monitor debt service coverage"
            )))
            risk_score += 10

        # Building age
        if pio.year_built > 0:
            age = datetime.now().year - pio.year_built
            if age > 30:
                pio.risk_factors.append(asdict(RiskFactor(
                    code="AGED_ASSET", severity="medium",
                    detail=f"Building is {age} years old — potential deferred maintenance",
                    mitigation="Require Phase I environmental + property condition report"
                )))
                risk_score += 15
            elif age > 20:
                risk_score += 5

        # No risk factors = low risk
        if not pio.risk_factors:
            pio.risk_factors.append(asdict(RiskFactor(
                code="NO_FLAGS", severity="low",
                detail="No significant risk factors identified",
                mitigation="Standard due diligence recommended"
            )))

        pio.risk_score = min(risk_score, 100)


# ═══════════════════════════════════════════════════════════
# CLI + API SERVER
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Property Intelligence Object (PIO)")
    parser.add_argument("--address", type=str, help="Property address to look up")
    parser.add_argument("--attom-key", type=str, default=os.environ.get("ATTOM_API_KEY", ""),
                       help="ATTOM API key")
    parser.add_argument("--fred-key", type=str, default=os.environ.get("FRED_API_KEY", ""),
                       help="FRED API key")
    parser.add_argument("--demo", action="store_true",
                       help="Use demo/estimated data (no API keys needed)")
    parser.add_argument("--serve", action="store_true", help="Run as HTTP API server")
    parser.add_argument("--port", type=int, default=8090, help="Server port")
    args = parser.parse_args()

    if args.serve:
        serve_api(args.port, args.attom_key, args.fred_key)
        return

    if not args.address:
        parser.error("Specify --address or --serve")

    builder = PIOBuilder(attom_key=args.attom_key or None, fred_key=args.fred_key or None)

    if args.demo or not args.attom_key:
        print("Running in demo mode (estimated data)")
        pio = builder.build_demo(args.address)
    else:
        pio = builder.build(args.address)

    print(pio.to_json())


def serve_api(port, attom_key, fred_key):
    """Minimal HTTP API server for PIOs."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs

    builder = PIOBuilder(attom_key=attom_key or None, fred_key=fred_key or None)

    class PIOHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)

            if parsed.path.startswith("/pio"):
                address = params.get("address", [""])[0]
                if not address:
                    self.send_error(400, "Missing ?address= parameter")
                    return

                demo = "demo" in params or not attom_key
                if demo:
                    pio = builder.build_demo(address)
                else:
                    pio = builder.build(address)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(pio.to_json().encode())

            elif parsed.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "version": "0.1"}).encode())

            else:
                self.send_error(404, "Use /pio?address=... or /health")

        def log_message(self, format, *args):
            print(f"[PIO] {args[0]}")

    server = HTTPServer(("0.0.0.0", port), PIOHandler)
    print(f"PIO API server running on http://0.0.0.0:{port}")
    print(f"  GET /pio?address=1200+S+Industrial+Blvd,+Dallas,+TX+75207")
    print(f"  GET /health")
    server.serve_forever()


if __name__ == "__main__":
    main()
