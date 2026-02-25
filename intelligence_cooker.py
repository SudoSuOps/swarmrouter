#!/usr/bin/env python3
"""
Intelligence Object Cooker — The Ocean
=======================================
Pulls public feeds → runs through edge model (Qwen 3-4B) → mints
Intelligence Objects → pushes to R2 sb-intelligence bucket.

NOT a grinder. NOT a batch job. It's the ocean — continuous feed.

Public data sources (FREE):
  - SEC EDGAR: REIT 10-K/10-Q filings (property portfolios, NOI, rent rolls)
  - FRED: Treasury yields, cap rate benchmarks, economic indicators
  - Census ACS: Demographics, income, population growth
  - County assessor: Property records (where available)
  - OpenStreetMap: Parcel geometry, POI proximity

Edge model (3-4B, runs anywhere):
  - Together.ai: Qwen/Qwen3-4B or Qwen/Qwen2.5-3B-Instruct
  - Local: ollama run qwen2.5:3b

The cooker structures raw data into Intelligence Objects.
The model doesn't CREATE data — it STRUCTURES it.

Usage:
    # Cook from SEC EDGAR REIT filings
    python3 intelligence_cooker.py --source edgar --ticker PLD

    # Cook from FRED market data
    python3 intelligence_cooker.py --source fred --market dallas

    # Cook all REIT tickers continuously
    python3 intelligence_cooker.py --source edgar --all-reits --continuous

    # Cook and push to R2
    python3 intelligence_cooker.py --source edgar --ticker PLD --push-r2

    # Status
    python3 intelligence_cooker.py --status
"""

import json
import os
import sys
import time
import hashlib
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import httpx
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("cooker")

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
CLOUDFLARE_ACCOUNT_ID = "6abec5e82728df0610a98be9364918e4"
R2_BUCKET = "sb-intelligence"

# Edge model — the whole point: small model structures data, doesn't hallucinate it
EDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # 3B, runs on any edge box
EDGE_MODEL_FALLBACK = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"  # if 3B not serverless

OUTPUT_DIR = Path(os.path.expanduser("~/Desktop/intelligence-objects"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SEC EDGAR headers (required)
EDGAR_HEADERS = {
    "User-Agent": "SwarmIntelligence research@swarmandbee.com",
    "Accept": "application/json",
}

# Major industrial/logistics REITs
REIT_TICKERS = {
    "PLD":  "Prologis",
    "DRE":  "Duke Realty",      # acquired by PLD but filings still exist
    "STAG": "STAG Industrial",
    "FR":   "First Industrial",
    "EGP":  "EastGroup Properties",
    "REXR": "Rexford Industrial",
    "LPT":  "Liberty Property",  # historical
    "TRNO": "Terreno Realty",
    "GTY":  "Getty Realty",
    "IIPR": "Innovative Industrial",
    "COLD": "Americold Realty",  # cold storage REIT
    "PSA":  "Public Storage",
    "EXR":  "Extra Space Storage",
    "CUBE": "CubeSmart",
    "NSA":  "National Storage",
}

FRED_SERIES = {
    "GS10":     "10-Year Treasury Yield",
    "GS5":      "5-Year Treasury Yield",
    "MORTGAGE30US": "30-Year Mortgage Rate",
    "CPIAUCSL": "CPI (Urban Consumers)",
    "UNRATE":   "Unemployment Rate",
    "INDPRO":   "Industrial Production Index",
    "PERMIT":   "Building Permits",
    "HOUST":    "Housing Starts",
    "GDP":      "Gross Domestic Product",
    "FEDFUNDS": "Federal Funds Rate",
}


# ═══════════════════════════════════════════════════════
# SEC EDGAR FEED — REIT 10-K/10-Q PROPERTY DATA
# ═══════════════════════════════════════════════════════

class EdgarFeed:
    """Pull REIT property portfolio data from SEC EDGAR."""

    BASE = "https://efts.sec.gov/LATEST"
    SUBMISSIONS = "https://data.sec.gov/submissions"
    FACTS = "https://data.sec.gov/api/xbrl/companyfacts"

    def __init__(self):
        self.client = httpx.Client(headers=EDGAR_HEADERS, timeout=30)

    def get_cik(self, ticker: str) -> str:
        """Resolve ticker to CIK."""
        r = self.client.get(f"{self.BASE}/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2024-01-01&forms=10-K")
        if r.status_code == 200:
            data = r.json()
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                return hits[0].get("_source", {}).get("entity_id", "")
        # Fallback: try company tickers JSON
        r2 = self.client.get("https://www.sec.gov/files/company_tickers.json")
        if r2.status_code == 200:
            tickers = r2.json()
            for entry in tickers.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)
        return ""

    def get_filings(self, cik: str, form_type: str = "10-K", count: int = 5) -> list:
        """Get recent filings for a CIK."""
        url = f"{self.SUBMISSIONS}/CIK{cik}.json"
        r = self.client.get(url)
        if r.status_code != 200:
            return []

        data = r.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])

        results = []
        for i, form in enumerate(forms):
            if form == form_type and len(results) < count:
                results.append({
                    "form": form,
                    "date": dates[i] if i < len(dates) else "",
                    "accession": accessions[i] if i < len(accessions) else "",
                    "document": docs[i] if i < len(docs) else "",
                })
        return results

    def get_company_facts(self, cik: str) -> dict:
        """Get XBRL company facts — the gold: revenue, NOI, assets, debt."""
        url = f"{self.FACTS}/CIK{cik}.json"
        r = self.client.get(url)
        if r.status_code != 200:
            return {}
        return r.json()

    def extract_financials(self, facts: dict) -> dict:
        """Extract key CRE financials from XBRL facts."""
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        def latest_val(concept: str) -> Optional[float]:
            entries = us_gaap.get(concept, {}).get("units", {})
            for unit_type, vals in entries.items():
                if vals:
                    # Get most recent annual
                    annual = [v for v in vals if v.get("form") in ("10-K", "10-K/A")]
                    if annual:
                        return annual[-1].get("val")
                    return vals[-1].get("val")
            return None

        return {
            "revenue": latest_val("Revenues") or latest_val("RevenueFromContractWithCustomerExcludingAssessedTax"),
            "noi": latest_val("OperatingIncomeLoss"),
            "total_assets": latest_val("Assets"),
            "total_debt": latest_val("LongTermDebt") or latest_val("LongTermDebtNoncurrent"),
            "equity": latest_val("StockholdersEquity"),
            "depreciation": latest_val("DepreciationAndAmortization"),
            "interest_expense": latest_val("InterestExpense"),
            "rental_revenue": latest_val("OperatingLeaseLeaseIncome") or latest_val("OperatingLeasesIncomeStatementLeaseRevenue"),
            "shares_outstanding": latest_val("CommonStockSharesOutstanding"),
            "dividends_per_share": latest_val("CommonStockDividendsPerShareDeclared"),
            "total_properties": latest_val("NumberOfRealEstateProperties"),
            "total_sf": latest_val("AreaOfRealEstateProperty"),
            "occupancy_rate": latest_val("OperatingLeaseWeightedAverageRemainingLeaseTerm1"),
        }

    def get_filing_text(self, accession: str, document: str, cik: str) -> str:
        """Get the actual filing text (truncated for edge model)."""
        acc_clean = accession.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_clean}/{document}"
        try:
            r = self.client.get(url, follow_redirects=True)
            if r.status_code == 200:
                text = r.text
                # Strip HTML tags for text extraction
                clean = re.sub(r'<[^>]+>', ' ', text)
                clean = re.sub(r'\s+', ' ', clean)
                return clean[:15000]  # 15K chars — enough for edge model
        except Exception as e:
            log.warning(f"Failed to get filing text: {e}")
        return ""


# ═══════════════════════════════════════════════════════
# FRED FEED — MARKET DATA
# ═══════════════════════════════════════════════════════

class FredFeed:
    """Pull economic indicators from FRED (no API key needed for basic)."""

    BASE = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self.client = httpx.Client(timeout=15)

    def get_series(self, series_id: str, limit: int = 12) -> list:
        """Get latest observations for a series."""
        if not self.api_key:
            # No key — use scraping fallback
            return self._scrape_series(series_id, limit)

        url = f"{self.BASE}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        r = self.client.get(url, params=params)
        if r.status_code == 200:
            obs = r.json().get("observations", [])
            return [{"date": o["date"], "value": float(o["value"])} for o in obs if o["value"] != "."]
        return []

    def _scrape_series(self, series_id: str, limit: int) -> list:
        """Fallback: scrape FRED page for data."""
        url = f"https://fred.stlouisfed.org/series/{series_id}"
        try:
            r = self.client.get(url, follow_redirects=True)
            # Extract the latest value from meta tags
            match = re.search(r'"observationValue":"([^"]+)"', r.text)
            if match:
                return [{"date": datetime.now().strftime("%Y-%m-%d"), "value": float(match.group(1))}]
        except Exception:
            pass
        return []

    def get_market_snapshot(self) -> dict:
        """Get a full market snapshot — all key series."""
        snapshot = {}
        for series_id, name in FRED_SERIES.items():
            data = self.get_series(series_id, limit=1)
            if data:
                snapshot[series_id] = {
                    "name": name,
                    "value": data[0]["value"],
                    "date": data[0]["date"],
                }
        return snapshot


# ═══════════════════════════════════════════════════════
# EDGE MODEL — STRUCTURES DATA INTO INTELLIGENCE OBJECTS
# ═══════════════════════════════════════════════════════

class EdgeModel:
    """Small model that STRUCTURES raw data into Intelligence Objects.
    Doesn't hallucinate. Doesn't create. Structures."""

    def __init__(self):
        self.client = httpx.Client(timeout=60)
        self.api_key = TOGETHER_API_KEY
        self.model = EDGE_MODEL

    def structure(self, raw_data: dict, context: str, output_schema: str) -> dict:
        """Feed raw data + schema → get structured Intelligence Object."""
        prompt = f"""You are an Intelligence Object structuring engine. Your job is to extract and structure
raw data into the exact JSON schema provided. Do NOT hallucinate or invent data. Only use what's in the raw data.
If a field cannot be determined from the data, set it to null.

RAW DATA:
{json.dumps(raw_data, indent=2)[:8000]}

CONTEXT: {context}

OUTPUT SCHEMA (fill every field from raw data, null if unknown):
{output_schema}

Return ONLY valid JSON. No explanation. No markdown."""

        try:
            r = self.client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                    "temperature": 0.1,  # Low temp — structuring, not creating
                },
            )
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                # Extract JSON from response
                content = content.strip()
                if content.startswith("```"):
                    content = re.sub(r'^```(?:json)?\s*', '', content)
                    content = re.sub(r'\s*```$', '', content)
                return json.loads(content)
            else:
                log.warning(f"Edge model {r.status_code}: {r.text[:200]}")
                # If model not serverless, try fallback
                if r.status_code == 400 and "non-serverless" in r.text:
                    log.info(f"Switching to fallback model: {EDGE_MODEL_FALLBACK}")
                    self.model = EDGE_MODEL_FALLBACK
                    return self.structure(raw_data, context, output_schema)
        except json.JSONDecodeError:
            log.warning("Edge model returned non-JSON, extracting...")
            # Try to find JSON in response
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            log.error(f"Edge model error: {e}")

        return {}


# ═══════════════════════════════════════════════════════
# PIO SCHEMA — WHAT GETS MINTED
# ═══════════════════════════════════════════════════════

PIO_SCHEMA = """{
  "object_id": "pio_{hash}",
  "object_type": "property_intelligence_object",
  "version": "1.0",
  "source_type": "edgar|fred|assessor|composite",

  "entity": {
    "name": "company or property name",
    "ticker": "if REIT",
    "cik": "SEC CIK",
    "entity_type": "reit|private|institutional"
  },

  "portfolio": {
    "total_properties": null,
    "total_sf": null,
    "occupancy_rate": null,
    "markets": []
  },

  "financials": {
    "revenue": null,
    "noi": null,
    "total_assets": null,
    "total_debt": null,
    "equity": null,
    "interest_expense": null,
    "rental_revenue": null,
    "dividends_per_share": null,
    "shares_outstanding": null,
    "implied_cap_rate": null,
    "debt_to_assets": null,
    "interest_coverage": null
  },

  "market_context": {
    "treasury_10yr": null,
    "cap_rate_spread": null,
    "industrial_vacancy": null,
    "rent_growth_yoy": null
  },

  "risk_factors": [],
  "confidence_score": 0.0,
  "sources": [],
  "created_at": "ISO timestamp",
  "ttl_seconds": 86400
}"""

PROPERTY_PIO_SCHEMA = """{
  "object_id": "pio_{hash}",
  "object_type": "property_intelligence_object",
  "version": "1.0",
  "source_type": "edgar_property",

  "address": "full address",
  "city": "",
  "state": "",
  "asset_type": "industrial|warehouse|flex|cold_storage|data_center",
  "asset_subtype": "",

  "building_sf": null,
  "land_acres": null,
  "year_built": null,
  "clear_height_ft": null,
  "dock_doors": null,

  "tenant": "if disclosed",
  "lease_expiry": "if disclosed",
  "annual_rent": null,
  "rent_per_sf": null,
  "occupancy": null,

  "estimated_value": null,
  "cap_rate": null,
  "noi": null,

  "risk_factors": [],
  "confidence_score": 0.0,
  "sources": [],
  "created_at": "ISO timestamp"
}"""


# ═══════════════════════════════════════════════════════
# COOKER — THE OCEAN
# ═══════════════════════════════════════════════════════

class IntelligenceCooker:
    """The ocean. Feeds flow in, Intelligence Objects flow out."""

    def __init__(self, push_r2: bool = False):
        self.edgar = EdgarFeed()
        self.fred = FredFeed()
        self.edge = EdgeModel()
        self.push_r2 = push_r2
        self.output_file = OUTPUT_DIR / "intelligence_objects.jsonl"
        self.stats = {"minted": 0, "failed": 0, "pushed": 0}

    def cook_reit(self, ticker: str) -> list:
        """Cook a REIT → mint Intelligence Objects from 10-K/10-Q data."""
        name = REIT_TICKERS.get(ticker, ticker)
        log.info(f"Cooking {ticker} ({name})...")

        # Step 1: Get CIK
        cik = self.edgar.get_cik(ticker)
        if not cik:
            log.warning(f"Could not resolve CIK for {ticker}")
            return []

        log.info(f"  CIK: {cik}")

        # Step 2: Get XBRL financial facts
        facts = self.edgar.get_company_facts(cik)
        financials = self.edgar.extract_financials(facts)
        log.info(f"  Revenue: {financials.get('revenue')}, NOI: {financials.get('noi')}")

        # Step 3: Get recent filings
        filings_10k = self.edgar.get_filings(cik, "10-K", count=2)
        filings_10q = self.edgar.get_filings(cik, "10-Q", count=2)

        # Step 4: Get market data
        market = self.fred.get_market_snapshot()

        # Step 5: Structure into Intelligence Object via edge model
        raw_data = {
            "ticker": ticker,
            "company_name": name,
            "cik": cik,
            "financials": financials,
            "filings_10k": filings_10k,
            "filings_10q": filings_10q,
            "market_snapshot": {k: v["value"] for k, v in market.items()} if market else {},
        }

        # Mint the portfolio-level PIO
        obj = self.edge.structure(raw_data, f"REIT portfolio analysis for {name} ({ticker})", PIO_SCHEMA)

        if not obj:
            # Fallback: structure without model
            obj = self._structure_reit_deterministic(ticker, name, cik, financials, market)

        if obj:
            obj["object_id"] = f"pio_{hashlib.sha256(f'{ticker}-{datetime.now().date()}'.encode()).hexdigest()[:12]}"
            obj["created_at"] = datetime.now().isoformat()
            obj["sources"] = [f"sec-edgar-{cik}", "fred-market-data"]
            self._emit(obj)

        # Step 6: If we have filing text, try to extract individual properties
        property_objects = []
        for filing in filings_10k[:1]:  # Latest 10-K only
            text = self.edgar.get_filing_text(filing["accession"], filing["document"], cik)
            if text and len(text) > 1000:
                log.info(f"  Extracting properties from 10-K ({len(text)} chars)...")
                props = self._extract_properties_from_filing(ticker, text, cik)
                property_objects.extend(props)

        objects = [obj] if obj else []
        objects.extend(property_objects)

        log.info(f"  Minted {len(objects)} Intelligence Objects for {ticker}")
        return objects

    def _structure_reit_deterministic(self, ticker, name, cik, financials, market) -> dict:
        """Fallback: structure PIO without edge model (pure deterministic)."""
        revenue = financials.get("revenue")
        noi = financials.get("noi")
        assets = financials.get("total_assets")
        debt = financials.get("total_debt")
        equity = financials.get("equity")
        interest = financials.get("interest_expense")

        # Derived metrics
        implied_cap = None
        if noi and assets and assets > 0:
            implied_cap = round(noi / assets, 4)

        debt_ratio = None
        if debt and assets and assets > 0:
            debt_ratio = round(debt / assets, 4)

        coverage = None
        if noi and interest and interest > 0:
            coverage = round(noi / interest, 2)

        treasury = None
        spread = None
        if market:
            gs10 = market.get("GS10", {})
            if isinstance(gs10, dict):
                treasury = gs10.get("value")
            elif isinstance(gs10, (int, float)):
                treasury = gs10
            if treasury and implied_cap:
                spread = round(implied_cap * 100 - treasury, 2)

        risk_factors = []
        if debt_ratio and debt_ratio > 0.6:
            risk_factors.append({"code": "HIGH_LEVERAGE", "severity": "high",
                                 "detail": f"Debt/assets ratio {debt_ratio:.1%}"})
        if coverage and coverage < 2.0:
            risk_factors.append({"code": "LOW_COVERAGE", "severity": "medium",
                                 "detail": f"Interest coverage {coverage:.1f}x"})
        if spread and spread < 100:
            risk_factors.append({"code": "THIN_SPREAD", "severity": "high",
                                 "detail": f"Cap rate spread {spread:.0f}bps over 10yr"})

        confidence = 0.3
        if revenue: confidence += 0.15
        if noi: confidence += 0.15
        if assets: confidence += 0.1
        if debt: confidence += 0.1
        if market: confidence += 0.1

        return {
            "object_id": "",
            "object_type": "property_intelligence_object",
            "version": "1.0",
            "source_type": "edgar",
            "entity": {
                "name": name,
                "ticker": ticker,
                "cik": cik,
                "entity_type": "reit",
            },
            "portfolio": {
                "total_properties": financials.get("total_properties"),
                "total_sf": financials.get("total_sf"),
                "occupancy_rate": financials.get("occupancy_rate"),
                "markets": [],
            },
            "financials": {
                "revenue": revenue,
                "noi": noi,
                "total_assets": assets,
                "total_debt": debt,
                "equity": equity,
                "interest_expense": interest,
                "rental_revenue": financials.get("rental_revenue"),
                "dividends_per_share": financials.get("dividends_per_share"),
                "shares_outstanding": financials.get("shares_outstanding"),
                "implied_cap_rate": implied_cap,
                "debt_to_assets": debt_ratio,
                "interest_coverage": coverage,
            },
            "market_context": {
                "treasury_10yr": treasury,
                "cap_rate_spread": spread,
                "industrial_vacancy": None,
                "rent_growth_yoy": None,
            },
            "risk_factors": risk_factors,
            "confidence_score": round(min(confidence, 0.95), 2),
            "sources": [],
            "created_at": "",
            "ttl_seconds": 86400,
        }

    def _extract_properties_from_filing(self, ticker: str, text: str, cik: str) -> list:
        """Use edge model to extract individual properties from filing text."""
        # Find property-related sections
        prop_section = ""
        for marker in ["Property Portfolio", "Properties", "Our Properties",
                       "Property List", "Schedule of Real Estate", "Major Properties"]:
            idx = text.lower().find(marker.lower())
            if idx >= 0:
                prop_section = text[idx:idx + 8000]
                break

        if not prop_section:
            # Take a chunk that likely has property data
            prop_section = text[2000:10000]

        if len(prop_section) < 200:
            return []

        prompt = f"""Extract individual properties from this REIT filing text.
Return a JSON array of properties. Each property should have:
address, city, state, asset_type, building_sf, tenant, rent_per_sf, occupancy

Only include properties you can clearly identify. Return [] if none found.

TEXT:
{prop_section[:6000]}

Return ONLY a JSON array. No explanation."""

        try:
            r = self.edge.client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.edge.api_key}"},
                json={
                    "model": self.edge.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                    "temperature": 0.1,
                },
            )
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = re.sub(r'^```(?:json)?\s*', '', content)
                    content = re.sub(r'\s*```$', '', content)
                props = json.loads(content)

                objects = []
                for prop in props[:50]:  # Max 50 per filing
                    addr = prop.get("address", "")
                    if not addr:
                        continue

                    obj_id = hashlib.sha256(f"{ticker}-{addr}-{datetime.now().date()}".encode()).hexdigest()[:12]
                    pio = {
                        "object_id": f"pio_{obj_id}",
                        "object_type": "property_intelligence_object",
                        "version": "1.0",
                        "source_type": "edgar_property",
                        "address": addr,
                        "city": prop.get("city", ""),
                        "state": prop.get("state", ""),
                        "asset_type": prop.get("asset_type", "industrial"),
                        "building_sf": prop.get("building_sf"),
                        "tenant": prop.get("tenant"),
                        "rent_per_sf": prop.get("rent_per_sf"),
                        "occupancy": prop.get("occupancy"),
                        "parent_reit": ticker,
                        "sources": [f"sec-edgar-10K-{cik}"],
                        "confidence_score": 0.65,
                        "created_at": datetime.now().isoformat(),
                    }
                    self._emit(pio)
                    objects.append(pio)

                return objects
        except Exception as e:
            log.warning(f"Property extraction failed: {e}")

        return []

    def cook_market(self, market_name: str = "national") -> dict:
        """Cook market-level Intelligence Object from FRED data."""
        log.info(f"Cooking market snapshot: {market_name}...")
        snapshot = self.fred.get_market_snapshot()

        if not snapshot:
            log.warning("No FRED data available")
            return {}

        obj_id = hashlib.sha256(f"market-{market_name}-{datetime.now().date()}".encode()).hexdigest()[:12]
        obj = {
            "object_id": f"mio_{obj_id}",
            "object_type": "market_intelligence_object",
            "version": "1.0",
            "source_type": "fred",
            "market": market_name,
            "indicators": snapshot,
            "risk_factors": [],
            "confidence_score": 0.9,
            "sources": ["fred.stlouisfed.org"],
            "created_at": datetime.now().isoformat(),
            "ttl_seconds": 3600,
        }

        # Assess market risk
        gs10 = snapshot.get("GS10", {}).get("value", 0) if isinstance(snapshot.get("GS10"), dict) else 0
        fedfunds = snapshot.get("FEDFUNDS", {}).get("value", 0) if isinstance(snapshot.get("FEDFUNDS"), dict) else 0
        unrate = snapshot.get("UNRATE", {}).get("value", 0) if isinstance(snapshot.get("UNRATE"), dict) else 0

        if gs10 and gs10 > 4.5:
            obj["risk_factors"].append({"code": "HIGH_RATES", "severity": "high",
                                        "detail": f"10yr Treasury at {gs10}%"})
        if unrate and unrate > 5.0:
            obj["risk_factors"].append({"code": "ELEVATED_UNEMPLOYMENT", "severity": "medium",
                                        "detail": f"Unemployment at {unrate}%"})

        self._emit(obj)
        return obj

    def cook_all_reits(self) -> list:
        """Cook all REITs in parallel."""
        all_objects = []
        log.info(f"Cooking {len(REIT_TICKERS)} REITs...")

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(self.cook_reit, ticker): ticker for ticker in REIT_TICKERS}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    objects = future.result()
                    all_objects.extend(objects)
                    log.info(f"  {ticker}: {len(objects)} objects")
                except Exception as e:
                    log.error(f"  {ticker} FAILED: {e}")
                    self.stats["failed"] += 1

        return all_objects

    def _emit(self, obj: dict):
        """Write Intelligence Object to JSONL and optionally push to R2."""
        with open(self.output_file, "a") as f:
            f.write(json.dumps(obj) + "\n")
        self.stats["minted"] += 1

        if self.push_r2:
            self._push_to_r2(obj)

    def _push_to_r2(self, obj: dict):
        """Push Intelligence Object to R2 via wrangler or API."""
        obj_id = obj.get("object_id", "unknown")
        key = f"pio/{obj_id}.json"

        # Write to temp file then upload via wrangler
        tmp = OUTPUT_DIR / f"_tmp_{obj_id}.json"
        tmp.write_text(json.dumps(obj, indent=2))

        try:
            result = subprocess.run(
                ["npx", "wrangler", "r2", "object", "put",
                 f"{R2_BUCKET}/{key}", "--file", str(tmp),
                 "--content-type", "application/json"],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "CLOUDFLARE_ACCOUNT_ID": CLOUDFLARE_ACCOUNT_ID},
            )
            if result.returncode == 0:
                self.stats["pushed"] += 1
                log.info(f"  → R2: {key}")
            else:
                log.warning(f"  R2 push failed: {result.stderr[:200]}")
        except Exception as e:
            log.warning(f"  R2 push error: {e}")
        finally:
            tmp.unlink(missing_ok=True)

        # Also push type index
        asset_type = obj.get("asset_type", obj.get("source_type", "unknown"))
        type_key = f"pio/by-type/{asset_type}/{obj_id}.json"
        tmp2 = OUTPUT_DIR / f"_tmp2_{obj_id}.json"
        tmp2.write_text(json.dumps(obj, indent=2))
        try:
            subprocess.run(
                ["npx", "wrangler", "r2", "object", "put",
                 f"{R2_BUCKET}/{type_key}", "--file", str(tmp2),
                 "--content-type", "application/json"],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "CLOUDFLARE_ACCOUNT_ID": CLOUDFLARE_ACCOUNT_ID},
            )
        except Exception:
            pass
        finally:
            tmp2.unlink(missing_ok=True)

    def status(self):
        """Print current status."""
        if self.output_file.exists():
            count = sum(1 for _ in open(self.output_file))
            size = self.output_file.stat().st_size
            print(f"\nIntelligence Objects: {count}")
            print(f"File: {self.output_file} ({size / 1024:.1f} KB)")
        else:
            print("\nNo Intelligence Objects minted yet.")

        print(f"\nSession: minted={self.stats['minted']} failed={self.stats['failed']} pushed={self.stats['pushed']}")
        print(f"R2 bucket: {R2_BUCKET}")
        print(f"Edge model: {self.edge.model}")


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Intelligence Object Cooker — The Ocean")
    parser.add_argument("--source", choices=["edgar", "fred", "all"], default="all",
                        help="Data source to cook from")
    parser.add_argument("--ticker", type=str, help="Specific REIT ticker (e.g. PLD)")
    parser.add_argument("--all-reits", action="store_true", help="Cook all industrial REITs")
    parser.add_argument("--market", type=str, default="national", help="Market for FRED data")
    parser.add_argument("--push-r2", action="store_true", help="Push to R2 sb-intelligence")
    parser.add_argument("--continuous", action="store_true", help="Run continuously (hourly refresh)")
    parser.add_argument("--status", action="store_true", help="Show status")
    args = parser.parse_args()

    cooker = IntelligenceCooker(push_r2=args.push_r2)

    if args.status:
        cooker.status()
        return

    while True:
        log.info("=" * 60)
        log.info("INTELLIGENCE COOKER — feeding the ocean")
        log.info("=" * 60)

        if args.source in ("edgar", "all"):
            if args.ticker:
                cooker.cook_reit(args.ticker.upper())
            elif args.all_reits or args.source == "all":
                cooker.cook_all_reits()

        if args.source in ("fred", "all"):
            cooker.cook_market(args.market)

        cooker.status()

        if not args.continuous:
            break

        log.info("Sleeping 1 hour before next cook cycle...")
        time.sleep(3600)


if __name__ == "__main__":
    main()
