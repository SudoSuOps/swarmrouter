#!/usr/bin/env python3
"""
EDGAR Event Pump — Feed SEC Filings Through the Event Machine
==============================================================
Pulls REIT filings from SEC EDGAR → detects deal/ownership events →
pushes through router.swarmandbee.com/events/process → R2 finality.

The pump. EDGAR → Event Machine → Intelligence Objects.

15 industrial REIT tickers. XBRL financial facts. Filing text extraction.
Event detection: acquisitions, dispositions, NOI changes, occupancy shifts,
debt changes, dividend changes, new developments.

Usage:
    # Pump all 15 REITs
    python3 edgar_pump.py

    # Single ticker
    python3 edgar_pump.py --ticker PLD

    # Push through event machine (live API)
    python3 edgar_pump.py --live

    # Dry run (detect events, don't push)
    python3 edgar_pump.py --dry-run
"""

import json
import os
import sys
import re
import time
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("edgar_pump")

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

API_BASE = "https://router.swarmandbee.com"
EDGAR_HEADERS = {
    "User-Agent": "SwarmIntelligence research@swarmandbee.com",
    "Accept": "application/json",
}

# Industrial/logistics REITs — the ones that matter
REIT_TICKERS = {
    "PLD":  {"name": "Prologis",              "type": "logistics",    "tier": 1},
    "STAG": {"name": "STAG Industrial",       "type": "industrial",   "tier": 1},
    "FR":   {"name": "First Industrial",      "type": "industrial",   "tier": 1},
    "EGP":  {"name": "EastGroup Properties",  "type": "industrial",   "tier": 1},
    "REXR": {"name": "Rexford Industrial",    "type": "industrial",   "tier": 1},
    "TRNO": {"name": "Terreno Realty",        "type": "industrial",   "tier": 1},
    "COLD": {"name": "Americold Realty",      "type": "cold_storage", "tier": 2},
    "PSA":  {"name": "Public Storage",        "type": "storage",      "tier": 2},
    "EXR":  {"name": "Extra Space Storage",   "type": "storage",      "tier": 2},
    "CUBE": {"name": "CubeSmart",             "type": "storage",      "tier": 2},
    "GTY":  {"name": "Getty Realty",          "type": "net_lease",    "tier": 2},
    "IIPR": {"name": "Innovative Industrial", "type": "specialty",    "tier": 2},
    "NSA":  {"name": "National Storage",      "type": "storage",      "tier": 2},
    "LXP":  {"name": "LXP Industrial Trust",  "type": "industrial",   "tier": 1},
    "PLYM": {"name": "Plymouth Industrial",   "type": "industrial",   "tier": 2},
}

OUTPUT_DIR = Path(os.path.expanduser("~/Desktop/intelligence-objects"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════
# EDGAR PULLER
# ═══════════════════════════════════════════════════════

class EdgarPuller:
    """Pull REIT data from SEC EDGAR."""

    def __init__(self):
        self.client = httpx.Client(headers=EDGAR_HEADERS, timeout=30)
        self._cik_cache = {}

    def get_cik(self, ticker: str) -> str:
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        try:
            r = self.client.get("https://www.sec.gov/files/company_tickers.json")
            if r.status_code == 200:
                for entry in r.json().values():
                    if entry.get("ticker", "").upper() == ticker.upper():
                        cik = str(entry["cik_str"]).zfill(10)
                        self._cik_cache[ticker] = cik
                        return cik
        except Exception as e:
            log.warning(f"CIK lookup failed for {ticker}: {e}")
        return ""

    def get_company_facts(self, cik: str) -> dict:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            r = self.client.get(url)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            log.warning(f"Company facts failed for {cik}: {e}")
        return {}

    def get_recent_filings(self, cik: str) -> dict:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            r = self.client.get(url)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            log.warning(f"Submissions failed for {cik}: {e}")
        return {}

    def extract_financials(self, facts: dict) -> dict:
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        def latest(concept, form_filter=None):
            entries = us_gaap.get(concept, {}).get("units", {})
            for unit_type, vals in entries.items():
                if not vals:
                    continue
                if form_filter:
                    filtered = [v for v in vals if v.get("form") in form_filter]
                    if filtered:
                        return filtered[-1].get("val"), filtered[-1].get("end", ""), filtered[-1].get("form", "")
                return vals[-1].get("val"), vals[-1].get("end", ""), vals[-1].get("form", "")
            return None, "", ""

        def latest_val(concept, form_filter=None):
            v, _, _ = latest(concept, form_filter)
            return v

        def latest_two(concept, form_filter=None):
            """Get last 2 values for comparison (YoY change detection)."""
            entries = us_gaap.get(concept, {}).get("units", {})
            for unit_type, vals in entries.items():
                if not vals:
                    continue
                if form_filter:
                    vals = [v for v in vals if v.get("form") in form_filter]
                if len(vals) >= 2:
                    return vals[-1].get("val"), vals[-2].get("val"), vals[-1].get("end", ""), vals[-2].get("end", "")
            return None, None, "", ""

        revenue_now, revenue_prev, rev_date, rev_prev_date = latest_two("Revenues", ["10-K", "10-K/A"])
        noi_now, noi_prev, noi_date, noi_prev_date = latest_two("OperatingIncomeLoss", ["10-K", "10-K/A"])
        assets_now, assets_prev, _, _ = latest_two("Assets", ["10-K", "10-K/A"])
        debt_now, debt_prev, _, _ = latest_two("LongTermDebt", ["10-K", "10-K/A"])
        if debt_now is None:
            debt_now, debt_prev, _, _ = latest_two("LongTermDebtNoncurrent", ["10-K", "10-K/A"])

        return {
            "revenue": revenue_now,
            "revenue_prev": revenue_prev,
            "revenue_date": rev_date,
            "noi": noi_now,
            "noi_prev": noi_prev,
            "noi_date": noi_date,
            "total_assets": assets_now,
            "total_assets_prev": assets_prev,
            "total_debt": debt_now,
            "total_debt_prev": debt_prev,
            "equity": latest_val("StockholdersEquity", ["10-K", "10-K/A"]),
            "interest_expense": latest_val("InterestExpense", ["10-K", "10-K/A"]),
            "rental_revenue": latest_val("OperatingLeaseLeaseIncome", ["10-K", "10-K/A"]) or latest_val("OperatingLeasesIncomeStatementLeaseRevenue"),
            "total_properties": latest_val("NumberOfRealEstateProperties"),
            "total_sf": latest_val("AreaOfRealEstateProperty"),
            "shares_outstanding": latest_val("CommonStockSharesOutstanding"),
            "dividends_per_share": latest_val("CommonStockDividendsPerShareDeclared"),
        }

    def get_recent_8k_events(self, cik: str) -> list:
        """Get recent 8-K filings — these are the EVENT filings (acquisitions, dispositions, etc.)."""
        submissions = self.get_recent_filings(cik)
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        descriptions = recent.get("primaryDocDescription", [])
        accessions = recent.get("accessionNumber", [])

        events = []
        cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        for i, form in enumerate(forms):
            if form in ("8-K", "8-K/A") and i < len(dates) and dates[i] >= cutoff:
                events.append({
                    "form": form,
                    "date": dates[i],
                    "description": descriptions[i] if i < len(descriptions) else "",
                    "accession": accessions[i] if i < len(accessions) else "",
                })
            if len(events) >= 10:
                break
        return events


# ═══════════════════════════════════════════════════════
# EVENT DETECTOR — EDGAR → Events
# ═══════════════════════════════════════════════════════

class EventDetector:
    """Detect CRE events from EDGAR financial data."""

    @staticmethod
    def detect_events(ticker: str, info: dict, financials: dict, filings_8k: list) -> list:
        name = info["name"]
        events = []

        # ── REIT Filing Event (always create) ─────────
        events.append({
            "event_type": "reit_filing",
            "confidence": 0.95,
            "property": f"{name} ({ticker}) — Industrial REIT portfolio",
            "state": None,
            "market": "national",
            "source": f"SEC EDGAR — CIK {financials.get('_cik', '')}",
            "source_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K",
            "ticker": ticker,
            "entity_name": name,
            "financials": {
                "revenue": financials.get("revenue"),
                "noi": financials.get("noi"),
                "total_assets": financials.get("total_assets"),
                "total_debt": financials.get("total_debt"),
                "total_properties": financials.get("total_properties"),
                "total_sf": financials.get("total_sf"),
            },
        })

        # ── NOI Change Detection ──────────────────────
        noi_now = financials.get("noi")
        noi_prev = financials.get("noi_prev")
        if noi_now and noi_prev and noi_prev != 0:
            noi_change = (noi_now - noi_prev) / abs(noi_prev)
            if abs(noi_change) > 0.05:  # >5% change
                direction = "expansion" if noi_change > 0 else "contraction"
                events.append({
                    "event_type": direction,
                    "confidence": 0.88,
                    "property": f"{name} ({ticker}) — NOI {'increased' if noi_change > 0 else 'decreased'} {abs(noi_change):.1%} YoY",
                    "state": None,
                    "market": "national",
                    "source": f"SEC EDGAR XBRL — {ticker}",
                    "noi_current": noi_now,
                    "noi_previous": noi_prev,
                    "noi_change_pct": round(noi_change * 100, 1),
                })

        # ── Revenue Change Detection ──────────────────
        rev_now = financials.get("revenue")
        rev_prev = financials.get("revenue_prev")
        if rev_now and rev_prev and rev_prev != 0:
            rev_change = (rev_now - rev_prev) / abs(rev_prev)
            if abs(rev_change) > 0.10:  # >10% change
                events.append({
                    "event_type": "expansion" if rev_change > 0 else "contraction",
                    "confidence": 0.85,
                    "property": f"{name} ({ticker}) — Revenue {'grew' if rev_change > 0 else 'declined'} {abs(rev_change):.1%} YoY",
                    "state": None,
                    "market": "national",
                    "source": f"SEC EDGAR XBRL — {ticker}",
                    "revenue_current": rev_now,
                    "revenue_previous": rev_prev,
                    "revenue_change_pct": round(rev_change * 100, 1),
                })

        # ── Debt Change Detection ─────────────────────
        debt_now = financials.get("total_debt")
        debt_prev = financials.get("total_debt_prev")
        if debt_now and debt_prev and debt_prev != 0:
            debt_change = (debt_now - debt_prev) / abs(debt_prev)
            if abs(debt_change) > 0.15:  # >15% change — big refinance or acquisition debt
                events.append({
                    "event_type": "reit_filing",
                    "confidence": 0.82,
                    "property": f"{name} ({ticker}) — Debt {'increased' if debt_change > 0 else 'decreased'} {abs(debt_change):.1%} — {'possible acquisition financing' if debt_change > 0 else 'possible disposition/paydown'}",
                    "state": None,
                    "market": "national",
                    "source": f"SEC EDGAR XBRL — {ticker}",
                    "debt_current": debt_now,
                    "debt_previous": debt_prev,
                    "debt_change_pct": round(debt_change * 100, 1),
                })

        # ── Asset Growth Detection (proxy for acquisitions) ───
        assets_now = financials.get("total_assets")
        assets_prev = financials.get("total_assets_prev")
        if assets_now and assets_prev and assets_prev != 0:
            asset_change = (assets_now - assets_prev) / abs(assets_prev)
            if asset_change > 0.10:  # >10% growth — acquisitions
                events.append({
                    "event_type": "just_sold",
                    "confidence": 0.75,
                    "property": f"{name} ({ticker}) — Total assets grew {asset_change:.1%} YoY (${(assets_now - assets_prev)/1e6:.0f}M increase) — likely acquisition activity",
                    "state": None,
                    "market": "national",
                    "source": f"SEC EDGAR XBRL — {ticker}",
                    "buyer": name,
                    "price": assets_now - assets_prev,
                })
            elif asset_change < -0.05:  # Shrinkage — dispositions
                events.append({
                    "event_type": "just_sold",
                    "confidence": 0.70,
                    "property": f"{name} ({ticker}) — Total assets declined {abs(asset_change):.1%} YoY — likely disposition activity",
                    "state": None,
                    "market": "national",
                    "source": f"SEC EDGAR XBRL — {ticker}",
                    "seller": name,
                    "price": abs(assets_now - assets_prev),
                })

        # ── 8-K Event Detection ───────────────────────
        for filing_8k in filings_8k:
            desc = (filing_8k.get("description") or "").lower()
            # Detect acquisition/disposition keywords
            if any(kw in desc for kw in ["acqui", "purchase", "closing", "completed"]):
                events.append({
                    "event_type": "just_sold",
                    "confidence": 0.88,
                    "property": f"{name} ({ticker}) — 8-K: {filing_8k.get('description', 'Acquisition/disposition')}",
                    "state": None,
                    "market": "national",
                    "source": f"SEC EDGAR 8-K — {filing_8k.get('date', '')}",
                    "buyer": name if "acqui" in desc or "purchase" in desc else None,
                    "seller": name if "dispos" in desc or "sale" in desc else None,
                })
            elif any(kw in desc for kw in ["dispos", "sale of", "sold"]):
                events.append({
                    "event_type": "just_sold",
                    "confidence": 0.85,
                    "property": f"{name} ({ticker}) — 8-K: {filing_8k.get('description', 'Disposition')}",
                    "state": None,
                    "market": "national",
                    "source": f"SEC EDGAR 8-K — {filing_8k.get('date', '')}",
                    "seller": name,
                })

        return events


# ═══════════════════════════════════════════════════════
# EVENT MACHINE PUSHER
# ═══════════════════════════════════════════════════════

class EventMachinePusher:
    """Push events through router.swarmandbee.com/events/process."""

    def __init__(self, live: bool = False):
        self.live = live
        self.client = httpx.Client(timeout=30)
        self.stats = {"pushed": 0, "failed": 0, "events_total": 0}

    def push_events(self, events: list) -> dict:
        self.stats["events_total"] += len(events)

        if not self.live:
            log.info(f"  [DRY RUN] Would push {len(events)} events to Event Machine")
            for e in events:
                log.info(f"    → {e['event_type']} ({e.get('confidence', 0):.0%}) — {e.get('property', 'unknown')[:80]}")
            return {"dry_run": True, "events": len(events)}

        # Push in batches of 25
        results = []
        for i in range(0, len(events), 25):
            batch = events[i:i+25]
            try:
                r = self.client.post(
                    f"{API_BASE}/events/process",
                    json={"events": batch},
                    headers={"Content-Type": "application/json"},
                )
                if r.status_code in (200, 201):
                    data = r.json()
                    self.stats["pushed"] += data.get("processed", 0)
                    self.stats["failed"] += data.get("failed", 0)
                    results.append(data)
                    log.info(f"  → Event Machine: {data.get('processed', 0)} processed, {data.get('failed', 0)} failed")
                else:
                    log.warning(f"  → Event Machine error {r.status_code}: {r.text[:200]}")
                    self.stats["failed"] += len(batch)
            except Exception as e:
                log.error(f"  → Event Machine push failed: {e}")
                self.stats["failed"] += len(batch)

            # SEC rate limiting respect
            time.sleep(0.2)

        return {"batches": len(results), "pushed": self.stats["pushed"], "failed": self.stats["failed"]}

    def push_to_cook_endpoint(self, ticker: str, financials: dict) -> dict:
        """Also cook a portfolio Intelligence Object via /cook/edgar."""
        if not self.live:
            log.info(f"  [DRY RUN] Would cook {ticker} via /cook/edgar")
            return {"dry_run": True}

        try:
            r = self.client.post(
                f"{API_BASE}/cook/edgar",
                json={
                    "ticker": ticker,
                    "financials": {k: v for k, v in financials.items() if not k.startswith("_") and "prev" not in k and "date" not in k},
                },
                headers={"Content-Type": "application/json"},
            )
            if r.status_code in (200, 201):
                data = r.json()
                if data.get("cooked"):
                    log.info(f"  → Cooked {ticker}: {data.get('object', {}).get('object_id', 'unknown')}")
                    return data
                else:
                    log.warning(f"  → Cook failed for {ticker}: {data.get('error', 'unknown')}")
            else:
                log.warning(f"  → Cook endpoint {r.status_code}")
        except Exception as e:
            log.error(f"  → Cook error for {ticker}: {e}")

        return {}


# ═══════════════════════════════════════════════════════
# THE PUMP
# ═══════════════════════════════════════════════════════

class EdgarPump:
    """The pump. EDGAR → Event Detection → Event Machine → R2."""

    def __init__(self, live: bool = False, ticker_filter: str = None):
        self.edgar = EdgarPuller()
        self.detector = EventDetector()
        self.pusher = EventMachinePusher(live=live)
        self.ticker_filter = ticker_filter.upper() if ticker_filter else None
        self.output_file = OUTPUT_DIR / "edgar_events.jsonl"
        self.stats = {
            "tickers_processed": 0,
            "events_detected": 0,
            "objects_cooked": 0,
            "errors": 0,
        }

    def pump(self):
        """Run the full pump."""
        tickers = {self.ticker_filter: REIT_TICKERS[self.ticker_filter]} if self.ticker_filter else REIT_TICKERS

        log.info("=" * 70)
        log.info(f"EDGAR PUMP — {len(tickers)} REITs → Event Machine")
        log.info(f"API: {API_BASE}")
        log.info(f"Live: {self.pusher.live}")
        log.info("=" * 70)

        all_events = []

        for ticker, info in tickers.items():
            log.info(f"\n{'─' * 50}")
            log.info(f"[{ticker}] {info['name']} ({info['type']}, tier {info['tier']})")
            log.info(f"{'─' * 50}")

            try:
                # Step 1: Resolve CIK
                cik = self.edgar.get_cik(ticker)
                if not cik:
                    log.warning(f"  Could not resolve CIK for {ticker}")
                    self.stats["errors"] += 1
                    continue
                log.info(f"  CIK: {cik}")

                # Step 2: Pull XBRL financial facts
                facts = self.edgar.get_company_facts(cik)
                if not facts:
                    log.warning(f"  No XBRL facts for {ticker}")
                    self.stats["errors"] += 1
                    continue

                financials = self.edgar.extract_financials(facts)
                financials["_cik"] = cik
                rev = financials.get("revenue")
                noi = financials.get("noi")
                assets = financials.get("total_assets")
                log.info(f"  Revenue: ${rev/1e6:.0f}M" if rev else "  Revenue: N/A")
                log.info(f"  NOI: ${noi/1e6:.0f}M" if noi else "  NOI: N/A")
                log.info(f"  Assets: ${assets/1e6:.0f}M" if assets else "  Assets: N/A")

                # Step 3: Get recent 8-K events
                filings_8k = self.edgar.get_recent_8k_events(cik)
                log.info(f"  8-K filings (last 90d): {len(filings_8k)}")

                # Step 4: Detect events
                events = self.detector.detect_events(ticker, info, financials, filings_8k)
                log.info(f"  Events detected: {len(events)}")
                for e in events:
                    log.info(f"    [{e['event_type']}] ({e.get('confidence', 0):.0%}) {e.get('property', '')[:70]}")

                all_events.extend(events)
                self.stats["events_detected"] += len(events)
                self.stats["tickers_processed"] += 1

                # Step 5: Save events locally
                with open(self.output_file, "a") as f:
                    for e in events:
                        e["_ticker"] = ticker
                        e["_pump_ts"] = datetime.now().isoformat()
                        f.write(json.dumps(e) + "\n")

                # Step 6: Push events through Event Machine
                if events:
                    self.pusher.push_events(events)

                # Step 7: Cook portfolio object via /cook/edgar
                self.pusher.push_to_cook_endpoint(ticker, financials)
                self.stats["objects_cooked"] += 1

                # SEC rate limiting — 10 req/sec max
                time.sleep(1)

            except Exception as e:
                log.error(f"  FAILED: {e}")
                self.stats["errors"] += 1
                continue

        # Final summary
        log.info(f"\n{'=' * 70}")
        log.info("EDGAR PUMP COMPLETE")
        log.info(f"{'=' * 70}")
        log.info(f"Tickers processed: {self.stats['tickers_processed']}/{len(tickers)}")
        log.info(f"Events detected:   {self.stats['events_detected']}")
        log.info(f"Objects cooked:    {self.stats['objects_cooked']}")
        log.info(f"Events pushed:     {self.pusher.stats['pushed']}")
        log.info(f"Push failures:     {self.pusher.stats['failed']}")
        log.info(f"Errors:            {self.stats['errors']}")
        log.info(f"Output:            {self.output_file}")

        return {
            "tickers": self.stats["tickers_processed"],
            "events": self.stats["events_detected"],
            "cooked": self.stats["objects_cooked"],
            "pushed": self.pusher.stats["pushed"],
            "failed": self.pusher.stats["failed"],
            "errors": self.stats["errors"],
        }


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="EDGAR Event Pump — Feed SEC filings through the Event Machine")
    parser.add_argument("--ticker", type=str, help="Single REIT ticker (e.g., PLD)")
    parser.add_argument("--live", action="store_true", help="Push events to live API (router.swarmandbee.com)")
    parser.add_argument("--dry-run", action="store_true", help="Detect events but don't push (default)")
    args = parser.parse_args()

    live = args.live and not args.dry_run

    if args.ticker and args.ticker.upper() not in REIT_TICKERS:
        print(f"Unknown ticker: {args.ticker}")
        print(f"Available: {', '.join(sorted(REIT_TICKERS.keys()))}")
        sys.exit(1)

    pump = EdgarPump(live=live, ticker_filter=args.ticker)
    result = pump.pump()

    print(f"\nResult: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
