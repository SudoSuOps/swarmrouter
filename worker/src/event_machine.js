/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║              THE CRE EVENT MACHINE                              ║
 * ║     Raw Signals → Intelligence Objects → Agent Consumption      ║
 * ║                                                                  ║
 * ║  Why this is 10x CoStar:                                       ║
 * ║  CoStar = human researchers → stale database → $30K/seat GUI   ║
 * ║  This    = live signals → edge AI → structured objects → API   ║
 * ║                                                                  ║
 * ║  Speed:  Signal → Object in seconds, not weeks                  ║
 * ║  Cost:   $0.0002/object vs $30K/seat/year                      ║
 * ║  Access: API-first, agent-native, no walled garden              ║
 * ║  Depth:  Skills reason over objects, not just store them        ║
 * ╚══════════════════════════════════════════════════════════════════╝
 *
 * THE ARCHITECTURE (5 layers):
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │ LAYER 1: SIGNAL SOURCES (raw, unstructured, free/cheap)     │
 * │                                                              │
 * │  PUBLIC (free):                                              │
 * │  ├── SEC EDGAR ─── REIT 10-K/10-Q, proxy statements        │
 * │  │   └── PLD files 10-Q → we detect: 847 properties,       │
 * │  │       $5.1B revenue, 96.5% occupancy change              │
 * │  ├── FRED ──────── Treasury yields, CPI, unemployment       │
 * │  │   └── 10yr hits 4.8% → cap rate spread compression      │
 * │  ├── County recorder ── Deed transfers, liens, tax rolls    │
 * │  │   └── Blackstone entity buys 120K SF in Savannah         │
 * │  ├── Permit filings ── New construction, renovations        │
 * │  │   └── 500K SF spec warehouse permitted in Alliance TX    │
 * │  ├── Court records ─── Foreclosures, bankruptcies           │
 * │  │   └── Industrial tenant files Ch.11 → vacancy signal     │
 * │  └── USPS NCOA ────── Address change = tenant movement      │
 * │                                                              │
 * │  BROKER FEEDS (scrape/parse):                                │
 * │  ├── CBRE, JLL, Cushman listings → just listed/price/terms  │
 * │  ├── Marcus & Millichap → 1031 exchange deal flow           │
 * │  ├── LinkedIn broker posts → under contract, just sold      │
 * │  ├── Press releases → portfolio sales, new development      │
 * │  ├── OM PDFs → full deal packages (extract with AI)         │
 * │  └── Broker websites → inventory, market reports            │
 * │                                                              │
 * │  PAID APIs ($500-$2K/mo):                                    │
 * │  ├── ATTOM ─── 150M property records, tax, deed, AVM       │
 * │  ├── Regrid ── Parcel geometry, zoning, flood zones         │
 * │  └── FRED API ─ Real-time economic indicators               │
 * │                                                              │
 * │  PREMIUM ($25K+/yr — later):                                 │
 * │  ├── CompStak ── Lease comps (actual rents, not asking)     │
 * │  ├── Yardi Matrix ── Rent rolls, ownership, debt            │
 * │  └── Real Capital Analytics ── Institutional transaction    │
 * └─────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 * ┌─────────────────────────────────────────────────────────────┐
 * │ LAYER 2: EVENT DETECTION (signal_scraper + specialized)     │
 * │                                                              │
 * │  Raw signal → classify → extract → confidence score          │
 * │                                                              │
 * │  EVENT TYPES:                                                │
 * │  ├── DEAL EVENTS (direct market activity)                   │
 * │  │   ├── just_listed ──── New supply. Who's selling? Why?   │
 * │  │   ├── just_sold ────── Market pricing. Cap rate signal.  │
 * │  │   ├── under_contract ─ Demand confirmation. Velocity.    │
 * │  │   ├── price_reduction ─ Distress/repricing signal.       │
 * │  │   ├── lease_signed ─── Absorption. Rent achievement.     │
 * │  │   └── loi_submitted ── Early deal pipeline signal.       │
 * │  │                                                          │
 * │  ├── SUPPLY EVENTS (what's coming)                          │
 * │  │   ├── permit_filed ─── Future supply, 18-24mo forward   │
 * │  │   ├── new_development ─ Spec or BTS announcement         │
 * │  │   ├── entitlement ──── Zoning approval = green light     │
 * │  │   └── demolition ───── Existing supply removal           │
 * │  │                                                          │
 * │  ├── OWNERSHIP EVENTS (who owns what)                       │
 * │  │   ├── deed_transfer ── Ownership change (recorder)       │
 * │  │   ├── entity_formation ─ New LLC = acquisition vehicle   │
 * │  │   ├── foreclosure ──── Distressed opportunity            │
 * │  │   └── reit_filing ──── Institutional portfolio shift     │
 * │  │                                                          │
 * │  ├── MACRO EVENTS (market-wide signals)                     │
 * │  │   ├── rate_change ──── Fed, treasury, mortgage rates     │
 * │  │   ├── jobs_report ──── Employment → absorption           │
 * │  │   ├── cpi_release ──── Inflation → rent escalations      │
 * │  │   └── port_volume ──── Trade flow → logistics demand     │
 * │  │                                                          │
 * │  └── TENANT EVENTS (demand-side signals)                    │
 * │      ├── expansion ────── Tenant growing footprint          │
 * │      ├── contraction ──── Tenant shrinking/subletting       │
 * │      ├── bankruptcy ───── Credit event → vacancy risk       │
 * │      └── relocation ───── NCOA/permit = tenant movement     │
 * │                                                              │
 * │  CONFIDENCE SCORING:                                         │
 * │  0.95+ = Verified (deed recorded, EDGAR filing)              │
 * │  0.80+ = High (CBRE listing, press release with numbers)    │
 * │  0.65+ = Medium (LinkedIn post, market chatter)              │
 * │  0.50+ = Low (indirect signal, needs confirmation)           │
 * │  <0.50 = Noise (discard)                                     │
 * └─────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 * ┌─────────────────────────────────────────────────────────────┐
 * │ LAYER 3: STRUCTURING (edge cooker → Intelligence Objects)   │
 * │                                                              │
 * │  Qwen3-30B-A3B on Cloudflare edge. $0.0002/object.          │
 * │                                                              │
 * │  EVENT → one or more Intelligence Objects:                   │
 * │                                                              │
 * │  just_listed event:                                          │
 * │  ├── Property IO (address, SF, price, cap rate, tenant)     │
 * │  ├── Market IO (submarket vacancy, comparable listings)     │
 * │  └── Investor IO (likely buyer types, pricing expectations) │
 * │                                                              │
 * │  just_sold event:                                            │
 * │  ├── Deal IO (buyer, seller, price, cap rate, terms)        │
 * │  ├── Property IO (updated value based on sale)              │
 * │  └── Market IO (comp data point for submarket)              │
 * │                                                              │
 * │  reit_filing event:                                          │
 * │  ├── Portfolio IO (holdings, financials, strategy)           │
 * │  ├── Market IO (geographic exposure, concentration)         │
 * │  └── Deal IO (acquisitions/dispositions from filing)        │
 * │                                                              │
 * │  OBJECT TYPES:                                               │
 * │  ├── property_intelligence_object (PIO) ── Single asset     │
 * │  ├── deal_intelligence_object ──────────── Transaction      │
 * │  ├── market_intelligence_object ────────── Market snapshot  │
 * │  ├── portfolio_intelligence_object ─────── REIT/fund        │
 * │  ├── investor_intelligence_object ──────── Buyer profile    │
 * │  ├── event_intelligence_object ─────────── Raw event record │
 * │  └── skill_output ─────────────────────── Skill result      │
 * │                                                              │
 * │  STATE EXPERT INJECTION:                                     │
 * │  Every object cooked through POST /cook/{STATE}              │
 * │  gets local knowledge: tax rates, cap rate ranges,           │
 * │  assessment ratios, key operators, infrastructure.            │
 * │  The model doesn't guess — we TELL it.                       │
 * │                                                              │
 * │  INFRASTRUCTURE INJECTION:                                   │
 * │  Ports, rail, power, last-mile, air cargo context            │
 * │  auto-injected into every state expert prompt.               │
 * │  A Savannah warehouse knows about JAXPORT and CSX.           │
 * └─────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 * ┌─────────────────────────────────────────────────────────────┐
 * │ LAYER 4: REASONING (skills — the brain on top of objects)   │
 * │                                                              │
 * │  Objects are facts. Skills are judgment.                      │
 * │                                                              │
 * │  9 SKILLS (live):                                            │
 * │  ├── broker_senior ────── "Is this a buy?"                  │
 * │  ├── broker_junior ────── "Pull comps, prep the tour"       │
 * │  ├── intelligence_query ── "Find me everything in FL"       │
 * │  ├── bookmaker ─────────── "Build the OM"                   │
 * │  ├── deal_tracker ──────── "Where's the deal pipeline?"     │
 * │  ├── developer ─────────── "Wire the API integration"       │
 * │  ├── signal_scraper ────── "What just happened?"            │
 * │  ├── investor ──────────── "Who buys this?"                 │
 * │  └── exchange_1031 ─────── "Can I defer the taxes?"         │
 * │                                                              │
 * │  SKILL CHAINING (composable intelligence):                   │
 * │                                                              │
 * │  signal_scraper detects "just_listed" event                  │
 * │    → broker_senior: "Is this a buy?" (pursue/pass/watch)    │
 * │      → if pursue:                                            │
 * │        → investor: "Who buys this?" (1031, REIT, PE)        │
 * │        → exchange_1031: "Does it work for our 1031 buyer?"  │
 * │        → bookmaker: "Build the OM"                          │
 * │        → deal_tracker: "Add to pipeline"                    │
 * │        → broker_junior: "Prep the tour"                     │
 * │                                                              │
 * │  ONE SIGNAL → 6 INTELLIGENCE OBJECTS → DEAL IN MOTION       │
 * │                                                              │
 * │  This is what CoStar can't do. CoStar shows you a listing.  │
 * │  The Event Machine detects the listing, analyzes it,         │
 * │  matches it to a buyer, builds the OM, and tracks the deal  │
 * │  — all before a human broker checks their email.             │
 * └─────────────────────────────────────────────────────────────┘
 *                              │
 *                              ▼
 * ┌─────────────────────────────────────────────────────────────┐
 * │ LAYER 5: DISTRIBUTION (agents consume Intelligence Objects) │
 * │                                                              │
 * │  THE API IS THE PRODUCT. NOT A GUI.                          │
 * │                                                              │
 * │  Any agent, any platform, one API call:                      │
 * │                                                              │
 * │  BUY-SIDE AGENTS:                                            │
 * │  "Alert me when NNN industrial > 100K SF hits market in TX"  │
 * │  → Event Machine detects just_listed → matches criteria      │
 * │  → Pushes PIO + broker_senior analysis to agent              │
 * │                                                              │
 * │  SELL-SIDE AGENTS:                                           │
 * │  "Who should I market this 200K SF to?"                      │
 * │  → investor skill matches buyer types                        │
 * │  → bookmaker builds OM → signal_scraper finds comps          │
 * │                                                              │
 * │  LENDER AGENTS:                                              │
 * │  "What's the risk on this loan request?"                     │
 * │  → broker_senior risk assessment + market context            │
 * │  → intelligence_query pulls comparable defaults               │
 * │                                                              │
 * │  1031 AGENTS:                                                │
 * │  "I have $8.5M to deploy by August 28"                       │
 * │  → exchange_1031 calculates timeline + boot                  │
 * │  → intelligence_query finds matching properties               │
 * │  → deal_tracker monitors pipeline                             │
 * │                                                              │
 * │  MARKET AGENTS:                                              │
 * │  "What's happening in Savannah industrial?"                  │
 * │  → intelligence_query aggregates all Savannah objects         │
 * │  → signal_scraper shows recent events                         │
 * │  → market heat shows Tier 1 status                            │
 * │                                                              │
 * │  DISTRIBUTION METHODS:                                       │
 * │  ├── REST API (current) ─── Pull model, any agent calls     │
 * │  ├── Webhooks (next) ────── Push model, event-driven         │
 * │  ├── WebSocket (later) ──── Real-time stream                 │
 * │  └── RSS/Atom (later) ───── Feed readers, email digests      │
 * └─────────────────────────────────────────────────────────────┘
 *
 *
 * ═══════════════════════════════════════════════════════════════
 * WHY THIS IS 10X COSTAR
 * ═══════════════════════════════════════════════════════════════
 *
 * 1. SPEED
 *    CoStar: Human researcher sees listing → enters data → QA → published (days/weeks)
 *    Event Machine: Signal detected → structured → stored → available (seconds)
 *    10x = real-time vs batch. The signal IS the object.
 *
 * 2. COST
 *    CoStar: $30,000/seat/year. Fortune 500 budget only.
 *    Event Machine: $0.0002/object. Any agent, any size.
 *    10x = democratized access. A solo broker gets the same intelligence as CBRE.
 *
 * 3. AGENT-NATIVE
 *    CoStar: GUI-first. Built for humans clicking through a web app.
 *    Event Machine: API-first. Built for agents that consume JSON.
 *    10x = every AI agent in CRE can plug in. No screen scraping.
 *
 * 4. COMPOSABLE
 *    CoStar: Search → results → done. Static output.
 *    Event Machine: Signal → analyze → match investor → build OM → track deal.
 *    10x = skills CHAIN. One signal triggers a cascade of intelligence.
 *
 * 5. OPEN
 *    CoStar: Walled garden. Your data stays inside their platform.
 *    Event Machine: Open source objects. MIT license. Anyone builds on top.
 *    10x = network effect. More agents = more objects = more value.
 *
 * 6. MARKET-AWARE
 *    CoStar: Same interface for TX and VT. No market weighting.
 *    Event Machine: Market heat index. TX gets 10x cook cycles. VT gets 1x.
 *    10x = resources follow deal flow. 80% of cycles on 20% of markets.
 *
 * 7. INFRASTRUCTURE-AWARE
 *    CoStar: Property data in a vacuum. No port/rail/power context.
 *    Event Machine: Every object knows its infrastructure moat.
 *    10x = a Savannah warehouse knows about the port. A NoVA flex knows about data center power.
 *
 *
 * ═══════════════════════════════════════════════════════════════
 * THE FEED LOOP — HOW THE OCEAN FEEDS ITSELF
 * ═══════════════════════════════════════════════════════════════
 *
 * Phase 1 (NOW — live):
 * ├── Manual cook: POST /cook with property data
 * ├── State experts: POST /cook/{STATE}
 * ├── Skills: POST /skill/{name}
 * ├── Python cooker: intelligence_cooker.py (EDGAR, FRED)
 * └── Object count: growing
 *
 * Phase 2 (NEXT — automated feeds):
 * ├── Cron trigger: Cloudflare Worker cron → EDGAR every 6 hours
 * ├── Cron trigger: FRED indicators daily
 * ├── Webhook: Broker alert emails → signal_scraper
 * ├── Scraper: CBRE/JLL listing pages → new events
 * └── Object count: thousands/day
 *
 * Phase 3 (SCALE — the flywheel):
 * ├── ATTOM API integration → 150M property records
 * ├── County recorder feeds → deed transfers in real-time
 * ├── Permit monitoring → new construction pipeline
 * ├── Court records → foreclosure/bankruptcy detection
 * ├── Agent subscriptions → "alert me when X"
 * └── Object count: millions
 *
 * Phase 4 (MOAT — network effects):
 * ├── Agents write back → "I toured this, here's what I found"
 * ├── Deal outcomes → "We closed at 5.2% cap" → market pricing signal
 * ├── Comp verification → objects get MORE accurate over time
 * ├── Broker contribution → brokers push their deals into the ocean
 * └── The more agents use it, the better the objects get.
 *     The better the objects, the more agents use it.
 *     This is the flywheel CoStar can't build because
 *     they charge $30K/seat to touch the data.
 *
 *
 * ═══════════════════════════════════════════════════════════════
 * IMPLEMENTATION: EVENT PROCESSOR (next build)
 * ═══════════════════════════════════════════════════════════════
 */

import { MARKET_HEAT } from './market_heat.js';
import { STATES } from './states.js';

// ── Event Types ─────────────────────────────────────────

export const EVENT_TYPES = {
  // Deal events
  just_listed:      { category: 'deal',      priority: 10, creates: ['property', 'market'] },
  just_sold:        { category: 'deal',      priority: 10, creates: ['deal', 'property', 'market'] },
  under_contract:   { category: 'deal',      priority: 8,  creates: ['deal', 'property'] },
  price_reduction:  { category: 'deal',      priority: 7,  creates: ['property', 'market'] },
  lease_signed:     { category: 'deal',      priority: 8,  creates: ['property', 'market'] },
  loi_submitted:    { category: 'deal',      priority: 6,  creates: ['deal'] },

  // Supply events
  permit_filed:     { category: 'supply',    priority: 7,  creates: ['market'] },
  new_development:  { category: 'supply',    priority: 8,  creates: ['property', 'market'] },
  entitlement:      { category: 'supply',    priority: 6,  creates: ['market'] },
  demolition:       { category: 'supply',    priority: 5,  creates: ['market'] },

  // Ownership events
  deed_transfer:    { category: 'ownership', priority: 9,  creates: ['deal', 'property'] },
  entity_formation: { category: 'ownership', priority: 5,  creates: ['investor'] },
  foreclosure:      { category: 'ownership', priority: 9,  creates: ['deal', 'property', 'market'] },
  reit_filing:      { category: 'ownership', priority: 8,  creates: ['portfolio', 'market'] },

  // Macro events
  rate_change:      { category: 'macro',     priority: 10, creates: ['market'] },
  jobs_report:      { category: 'macro',     priority: 7,  creates: ['market'] },
  cpi_release:      { category: 'macro',     priority: 6,  creates: ['market'] },
  port_volume:      { category: 'macro',     priority: 7,  creates: ['market'] },

  // Tenant events
  expansion:        { category: 'tenant',    priority: 7,  creates: ['property', 'market'] },
  contraction:      { category: 'tenant',    priority: 7,  creates: ['property', 'market'] },
  bankruptcy:       { category: 'tenant',    priority: 9,  creates: ['property', 'market'] },
  relocation:       { category: 'tenant',    priority: 6,  creates: ['property'] },
};

// ── Event Object Schema ─────────────────────────────────

export function createEventObject(event) {
  return {
    object_id: `eio_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`,
    object_type: 'event_intelligence_object',
    event_type: event.event_type,
    category: EVENT_TYPES[event.event_type]?.category || 'unknown',
    priority: EVENT_TYPES[event.event_type]?.priority || 5,
    confidence: event.confidence || 0,
    // What happened
    property: event.property || null,
    state: event.state || null,
    market: event.market || null,
    // Financial signals
    price: event.price || event.asking_price || event.sale_price || null,
    cap_rate: event.cap_rate || null,
    rent_psf: event.rent_psf || null,
    sf: event.sf || null,
    // Who
    buyer: event.buyer || null,
    seller: event.seller || null,
    tenant: event.tenant || null,
    broker: event.broker || null,
    // Source
    source: event.source || null,
    source_url: event.source_url || null,
    detected_at: event.detected_at || new Date().toISOString(),
    // Chain: what objects should this event create?
    creates_objects: EVENT_TYPES[event.event_type]?.creates || [],
    // Processing
    processed: false,
    skill_chain: [],
    created_at: new Date().toISOString(),
  };
}

// ── Event Processor (processes events → Intelligence Objects) ──

export async function processEvent(event, env) {
  const eventObj = createEventObject(event);
  const results = { event: eventObj, objects_created: [], skills_triggered: [] };

  // Store the event itself
  await env.INTELLIGENCE.put(
    `pio/events/${eventObj.event_type}/${eventObj.object_id}`,
    JSON.stringify(eventObj),
    { customMetadata: { event_type: eventObj.event_type, state: eventObj.state || '', confidence: String(eventObj.confidence) } }
  );

  // Determine state for area expert routing
  const state = eventObj.state || detectState(eventObj);

  // Market heat weighting — hot markets get full processing, cold markets get basic
  const heat = state ? (MARKET_HEAT[state] || { tier: 4, cook_weight: 1 }) : { tier: 4, cook_weight: 1 };

  // For Tier 1-2 deal events with high confidence: trigger skill chain
  if (heat.tier <= 2 && eventObj.confidence >= 0.7 && eventObj.category === 'deal') {
    eventObj.skill_chain = ['broker_senior', 'investor'];
    if (eventObj.event_type === 'just_listed') {
      eventObj.skill_chain.push('exchange_1031'); // Check 1031 fit
    }
  }

  // For any deal event with confidence >= 0.8: create property object
  if (eventObj.confidence >= 0.8 && eventObj.creates_objects.includes('property')) {
    results.objects_created.push('property_intelligence_object');
  }

  // Mark processed
  eventObj.processed = true;

  return results;
}

// ── State Detection (from event data) ───────────────────

function detectState(event) {
  const text = [event.property, event.market, event.source].filter(Boolean).join(' ').toUpperCase();
  for (const code of Object.keys(STATES)) {
    const name = STATES[code].name.toUpperCase();
    if (text.includes(`, ${code}`) || text.includes(` ${code} `) || text.includes(name)) {
      return code;
    }
  }
  return null;
}

// ── Feed Schedule (what to cook and when) ───────────────

export function getFeedSchedule() {
  return {
    every_6_hours: [
      { source: 'edgar', description: 'SEC EDGAR REIT filings', event_types: ['reit_filing'] },
    ],
    every_day: [
      { source: 'fred', description: 'FRED economic indicators', event_types: ['rate_change', 'jobs_report', 'cpi_release'] },
    ],
    every_hour: [
      { source: 'broker_feeds', description: 'CBRE, JLL, C&W listing pages', event_types: ['just_listed', 'price_reduction'] },
      { source: 'linkedin', description: 'Broker LinkedIn posts', event_types: ['just_sold', 'under_contract', 'lease_signed'] },
    ],
    every_15_min: [
      { source: 'news', description: 'CRE press releases', event_types: ['just_sold', 'new_development', 'expansion'] },
    ],
    real_time: [
      { source: 'webhooks', description: 'Broker email alerts, listing notifications', event_types: ['just_listed', 'price_reduction'] },
      { source: 'county_recorder', description: 'Deed transfer recordings', event_types: ['deed_transfer', 'foreclosure'] },
    ],
  };
}

// ── Metrics ─────────────────────────────────────────────

export function eventMachineStats() {
  return {
    architecture: 'CRE Event Machine v1.0',
    layers: {
      signal_sources: '10+ source types (EDGAR, FRED, broker feeds, county records, permits, court records)',
      event_detection: `${Object.keys(EVENT_TYPES).length} event types across 5 categories`,
      structuring: 'Qwen3-30B-A3B edge inference → 7 object types',
      reasoning: '9 skills (broker_senior, broker_junior, intelligence_query, bookmaker, deal_tracker, developer, signal_scraper, investor, exchange_1031)',
      distribution: 'REST API (live), webhooks (next), WebSocket (later)',
    },
    event_categories: {
      deal: Object.entries(EVENT_TYPES).filter(([_, v]) => v.category === 'deal').map(([k]) => k),
      supply: Object.entries(EVENT_TYPES).filter(([_, v]) => v.category === 'supply').map(([k]) => k),
      ownership: Object.entries(EVENT_TYPES).filter(([_, v]) => v.category === 'ownership').map(([k]) => k),
      macro: Object.entries(EVENT_TYPES).filter(([_, v]) => v.category === 'macro').map(([k]) => k),
      tenant: Object.entries(EVENT_TYPES).filter(([_, v]) => v.category === 'tenant').map(([k]) => k),
    },
    vs_costar: {
      speed: 'seconds vs weeks',
      cost: '$0.0002/object vs $30K/seat/year',
      access: 'API-first vs GUI-first',
      composable: '9 skills chain vs static search',
      open: 'MIT license vs walled garden',
      market_aware: '4-tier heat weighting vs uniform',
      infrastructure: 'ports/rail/power injected vs property-only',
    },
    flywheel: 'More agents consume objects → more signals detected → more objects created → more agents consume. CoStar can\'t build this because they charge $30K to touch the data.',
  };
}
