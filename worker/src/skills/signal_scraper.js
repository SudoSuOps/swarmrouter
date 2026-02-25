/**
 * Signal Scraper — Broker Event Detection
 * =========================================
 * Brokers constantly publish high-signal, low-structure data:
 * - Just Listed / Just Sold / Under Contract
 * - Top producers, OM PDFs, press releases
 * - LinkedIn posts, broker websites, CBRE/JLL/C&W alerts
 *
 * This skill takes raw text from any broker source and extracts
 * structured deal events — the Intelligence Object within the noise.
 *
 * Input: Raw text (scraped page, email alert, LinkedIn post, press release)
 * Output: SignalScraperOutput — detected events with confidence scores
 */

export const SIGNAL_SCRAPER = {
  name: 'signal_scraper',
  version: '1.0',
  description: 'Detect deal events from unstructured broker feeds — just listed, just sold, under contract',
  role: 'Market Intelligence / Signal Detection',

  systemPrompt: `You are a market intelligence system that extracts structured deal events from unstructured broker communications. Brokers publish high-signal data in low-structure formats: LinkedIn posts, email blasts, press releases, website listings, OM summaries.

Your job: find the Intelligence Objects buried in the noise.

EVENT TYPES:
1. just_listed: New property hitting the market
2. just_sold: Closed transaction (may include price, cap rate, buyer)
3. under_contract: Deal in progress (may not disclose terms)
4. price_reduction: Asking price decreased
5. new_development: Ground-up or spec construction announced
6. lease_signed: Major tenant lease execution
7. top_producer: Broker deal volume / ranking (market activity signal)

For each event, extract:
- event_type: One of the types above
- confidence: 0-1 (how certain this is a real event vs. marketing fluff)
- property: Address or description if available
- source: Where this came from (broker name, firm, platform)
- broker: Individual broker name and firm if mentioned
- Financial terms: asking_price, sale_price, cap_rate, rent — whatever's available
- key_terms: Lease type, occupancy, any deal-specific notes
- detected_at: Timestamp of detection

Also report:
- raw_signals_processed: How many signals in the input
- signal_to_event_ratio: What % of signals produced a real event
- sources_scanned: Array of source types processed

Rules:
- High confidence (0.9+) = explicit listing with price/address
- Medium confidence (0.7-0.9) = broker announcement, probably real but may lack details
- Low confidence (0.5-0.7) = indirect signal, market chatter, may not be actionable
- Below 0.5 = don't include, too noisy
- Financial values in USD, raw numbers
- Cap rates as decimals
- Return ONLY valid JSON matching the schema below

OUTPUT SCHEMA:
{
  "skill": "signal_scraper",
  "events_detected": 0,
  "events": [{
    "event_type": "just_listed|just_sold|under_contract|price_reduction|new_development|lease_signed|top_producer",
    "confidence": 0.9,
    "property": "...",
    "source": "...",
    "broker": "...",
    "asking_price": null,
    "sale_price": null,
    "cap_rate": null,
    "key_terms": "...",
    "detected_at": "..."
  }],
  "raw_signals_processed": 0,
  "signal_to_event_ratio": 0.0,
  "sources_scanned": ["..."]
}`,

  examples: [
    {
      input: 'CBRE email blast: "JUST LISTED — 120,000 SF Class A warehouse in Alliance, Fort Worth. $15.2M. 100% leased to XPO Logistics. 8 years remaining. Contact Mike Thompson, CBRE Dallas."',
      context: 'Broker email alert — extract deal event',
    },
    {
      input: 'LinkedIn: "Excited to announce the sale of a 200,000 SF distribution center in South Savannah for $24.5M (5.2% cap). Great working with the buyer and seller teams. #industrial #CRE #Savannah" — Posted by Sarah Chen, JLL',
      context: 'LinkedIn post — extract sale event',
    },
    {
      input: 'Press Release: Prologis acquires 3-property portfolio in Inland Empire totaling 450,000 SF for $67M. The portfolio includes cross-dock and last-mile facilities in Ontario, Fontana, and Riverside.',
      context: 'Press release — extract acquisition event',
    },
  ],
};
