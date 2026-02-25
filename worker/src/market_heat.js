/**
 * Market Heat Index — Where the Deals Are
 * =========================================
 * 80% of the intelligence engine is knowing which markets matter.
 * This ranks every state by industrial deal volume, absorption,
 * rent growth, and institutional capital flow.
 *
 * Tier 1: HOT — institutional capital, sub-4% vacancy, double-digit rent growth
 * Tier 2: WARM — active deal flow, good fundamentals, institutional interest
 * Tier 3: STEADY — secondary markets, stable but not explosive
 * Tier 4: COLD — limited institutional activity, high cap rates, niche only
 *
 * Cook priority: Tier 1 gets 10x the feed cycles. Tier 4 gets 1x.
 */

export const MARKET_HEAT = {
  // ══════════════════════════════════════════
  // TIER 1 — HOT (cook priority: 10x)
  // These are the markets moving $B+ annually
  // ══════════════════════════════════════════
  TX: { tier: 1, heat: 98, label: 'HOT',    cook_weight: 10, reason: 'DFW #1 new construction, Houston Ship Channel, Austin tech boom, no income tax migration' },
  CA: { tier: 1, heat: 97, label: 'HOT',    cook_weight: 10, reason: 'Inland Empire #1 absorption market, LA/Long Beach port complex, sub-2% vacancy, Rexford territory' },
  FL: { tier: 1, heat: 95, label: 'HOT',    cook_weight: 10, reason: 'No income tax migration wave, Miami LatAm trade, JAXPORT growth, South FL sub-3% vacancy' },
  NJ: { tier: 1, heat: 94, label: 'HOT',    cook_weight: 10, reason: 'Exit 8A #1 East Coast submarket, Port Newark, last-mile to NYC, rents doubled since 2020' },
  GA: { tier: 1, heat: 93, label: 'HOT',    cook_weight: 10, reason: 'Atlanta top-5 market, Savannah fastest-growing port, mega spec in South Atlanta' },
  IL: { tier: 1, heat: 92, label: 'HOT',    cook_weight: 10, reason: 'Chicago #2 market, I-80 mega corridor, OHare air cargo, CenterPoint/Bridge territory' },
  PA: { tier: 1, heat: 91, label: 'HOT',    cook_weight: 10, reason: 'Lehigh Valley/I-78 mega distribution, KOZ tax abatements, NJ port spillover' },
  AZ: { tier: 1, heat: 90, label: 'HOT',    cook_weight: 10, reason: 'Phoenix top-3 market, TSMC fab, 330M+ SF inventory, massive spec pipeline' },
  OH: { tier: 1, heat: 89, label: 'HOT',    cook_weight: 10, reason: 'Columbus top-5 logistics, Intel $20B fab, 600mi reach to 60% US population' },
  NC: { tier: 1, heat: 88, label: 'HOT',    cook_weight: 10, reason: 'Charlotte I-85 boom, FedEx Triad hub, RTP life sciences, Toyota battery plant' },

  // ══════════════════════════════════════════
  // TIER 2 — WARM (cook priority: 5x)
  // Active institutional markets, solid fundamentals
  // ══════════════════════════════════════════
  TN: { tier: 2, heat: 85, label: 'WARM',   cook_weight: 5, reason: 'Memphis FedEx superhub #1 air cargo, Nashville growth, Ford Blue Oval, no income tax' },
  IN: { tier: 2, heat: 84, label: 'WARM',   cook_weight: 5, reason: 'Indianapolis I-65/I-70 crossroads, Duke Realty legacy, Scannell/Becknell active' },
  NV: { tier: 2, heat: 83, label: 'WARM',   cook_weight: 5, reason: 'N Las Vegas mega boom, TRIC largest industrial park in US, no income tax, Tesla Giga' },
  WA: { tier: 2, heat: 82, label: 'WARM',   cook_weight: 5, reason: 'Kent Valley mega distribution, Port of Tacoma, Amazon HQ proximity, no income tax' },
  SC: { tier: 2, heat: 81, label: 'WARM',   cook_weight: 5, reason: 'Charleston port growing fast, BMW/Volvo supply chain, lowest mfg assessment in US' },
  VA: { tier: 2, heat: 80, label: 'WARM',   cook_weight: 5, reason: 'Ashburn = data center capital of world, Port of Virginia deepest East Coast, 70% internet traffic' },
  CO: { tier: 2, heat: 78, label: 'WARM',   cook_weight: 5, reason: 'Denver I-70 corridor, DIA e-commerce hub, strong rent growth' },
  MD: { tier: 2, heat: 76, label: 'WARM',   cook_weight: 5, reason: 'Port of Baltimore auto imports, I-81 corridor, BWI distribution' },
  KY: { tier: 2, heat: 75, label: 'WARM',   cook_weight: 5, reason: 'UPS Worldport #3 global air cargo, Amazon Air Hub CVG, bourbon belt logistics' },
  MO: { tier: 2, heat: 74, label: 'WARM',   cook_weight: 5, reason: 'KC logistics crossroads, BNSF intermodal, NorthPoint mega spec, St. Louis legacy' },
  UT: { tier: 2, heat: 73, label: 'WARM',   cook_weight: 5, reason: 'SLC I-15/I-80 crossroads, NW Quadrant spec boom, Meta data center' },
  OR: { tier: 2, heat: 72, label: 'WARM',   cook_weight: 5, reason: 'Portland Columbia corridor, Intel Hillsboro, no sales tax advantage, Measure 50 tax cap' },
  MA: { tier: 2, heat: 71, label: 'WARM',   cook_weight: 5, reason: 'Route 128 lab/flex premium, ultra-tight vacancy, life sciences capital' },
  NY: { tier: 2, heat: 70, label: 'WARM',   cook_weight: 5, reason: 'South Bronx most expensive last-mile in US, Brooklyn industrial premium, NYC ultra-tight' },
  MN: { tier: 2, heat: 68, label: 'WARM',   cook_weight: 5, reason: 'Twin Cities I-494 corridor, cold storage demand (General Mills, Cargill, Target)' },
  KS: { tier: 2, heat: 67, label: 'WARM',   cook_weight: 5, reason: 'KC metro logistics hub, NorthPoint spec, Wichita aerospace (Spirit, Textron)' },

  // ══════════════════════════════════════════
  // TIER 3 — STEADY (cook priority: 2x)
  // Secondary markets, stable but limited deal flow
  // ══════════════════════════════════════════
  MI: { tier: 3, heat: 62, label: 'STEADY', cook_weight: 2, reason: 'Detroit auto legacy + EV transition, Grand Rapids mfg, Ashley Capital active' },
  WI: { tier: 3, heat: 60, label: 'STEADY', cook_weight: 2, reason: 'Milwaukee I-94, Kenosha Chicago spillover, legacy manufacturing' },
  AL: { tier: 3, heat: 58, label: 'STEADY', cook_weight: 2, reason: 'Huntsville aerospace boom (Blue Origin, FBI), Mobile port, low taxes' },
  CT: { tier: 3, heat: 55, label: 'STEADY', cook_weight: 2, reason: 'I-91/I-95 corridors, legacy mfg → last-mile conversion, highest taxes in US' },
  NM: { tier: 3, heat: 53, label: 'STEADY', cook_weight: 2, reason: 'Santa Teresa nearshoring/border logistics growing, Sandia defense' },
  IA: { tier: 3, heat: 52, label: 'STEADY', cook_weight: 2, reason: 'Des Moines data centers (Meta, Microsoft), ag processing, STAG territory' },
  NE: { tier: 3, heat: 50, label: 'STEADY', cook_weight: 2, reason: 'Omaha central US distribution, Meta data center, underrated logistics' },
  NH: { tier: 3, heat: 48, label: 'STEADY', cook_weight: 2, reason: 'Southern NH Boston spillover, no income/sales tax, limited supply' },
  LA: { tier: 3, heat: 47, label: 'STEADY', cook_weight: 2, reason: 'Chemical corridor, LNG terminals, port bulk cargo, low assessment' },
  OK: { tier: 3, heat: 45, label: 'STEADY', cook_weight: 2, reason: 'Energy-driven, AA/USPS maintenance Tulsa, low assessment ratio' },
  AR: { tier: 3, heat: 44, label: 'STEADY', cook_weight: 2, reason: 'Walmart ecosystem drives NW Arkansas, last-mile for Walmart vendors' },
  ID: { tier: 3, heat: 43, label: 'STEADY', cook_weight: 2, reason: 'Boise growth market, Micron presence, limited institutional supply' },
  DE: { tier: 3, heat: 42, label: 'STEADY', cook_weight: 2, reason: 'No sales tax, I-95 strategic location, small but active' },

  // ══════════════════════════════════════════
  // TIER 4 — COLD (cook priority: 1x)
  // Niche markets, limited institutional activity
  // ══════════════════════════════════════════
  RI: { tier: 4, heat: 35, label: 'COLD',   cook_weight: 1, reason: 'Very small market, Quonset only institutional area' },
  ME: { tier: 4, heat: 33, label: 'COLD',   cook_weight: 1, reason: 'Portland last-mile only, limited supply' },
  HI: { tier: 4, heat: 30, label: 'COLD',   cook_weight: 1, reason: 'Island constraints, near-zero new development, ultra-tight by default' },
  MS: { tier: 4, heat: 28, label: 'COLD',   cook_weight: 1, reason: 'Aerospace niche (Golden Triangle), port Gulfport, limited institutional' },
  WV: { tier: 4, heat: 25, label: 'COLD',   cook_weight: 1, reason: 'Very limited market, Eastern Panhandle DC spillover only' },
  ND: { tier: 4, heat: 23, label: 'COLD',   cook_weight: 1, reason: 'Oil field services western ND, Fargo small tech, very limited' },
  SD: { tier: 4, heat: 22, label: 'COLD',   cook_weight: 1, reason: 'Sioux Falls growing but small, no income tax advantage' },
  VT: { tier: 4, heat: 20, label: 'COLD',   cook_weight: 1, reason: 'GlobalFoundries only institutional asset, very limited' },
  MT: { tier: 4, heat: 18, label: 'COLD',   cook_weight: 1, reason: 'Energy/ag only, very limited institutional market' },
  WY: { tier: 4, heat: 15, label: 'COLD',   cook_weight: 1, reason: 'Cheyenne data centers emerging, otherwise very limited' },
  AK: { tier: 4, heat: 10, label: 'COLD',   cook_weight: 1, reason: 'Cold storage and resource extraction only, high construction costs' },
};

/**
 * Get states by tier.
 */
export function getStatesByTier(tier) {
  return Object.entries(MARKET_HEAT)
    .filter(([_, v]) => v.tier === tier)
    .sort((a, b) => b[1].heat - a[1].heat)
    .map(([code, data]) => ({ code, ...data }));
}

/**
 * Get cook schedule — how many cycles per state based on heat.
 * Hot markets get 10x cycles. Cold gets 1x.
 * Total weight: ~200 units across 50 states.
 */
export function getCookSchedule() {
  const schedule = [];
  for (const [code, data] of Object.entries(MARKET_HEAT)) {
    for (let i = 0; i < data.cook_weight; i++) {
      schedule.push(code);
    }
  }
  // Shuffle for distribution
  for (let i = schedule.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [schedule[i], schedule[j]] = [schedule[j], schedule[i]];
  }
  return schedule;
}

/**
 * Summary stats.
 */
export function heatSummary() {
  const tiers = { 1: [], 2: [], 3: [], 4: [] };
  let totalWeight = 0;
  for (const [code, data] of Object.entries(MARKET_HEAT)) {
    tiers[data.tier].push(code);
    totalWeight += data.cook_weight;
  }
  return {
    tier_1_hot: { count: tiers[1].length, states: tiers[1], cook_share: `${Math.round(tiers[1].length * 10 / totalWeight * 100)}%` },
    tier_2_warm: { count: tiers[2].length, states: tiers[2], cook_share: `${Math.round(tiers[2].length * 5 / totalWeight * 100)}%` },
    tier_3_steady: { count: tiers[3].length, states: tiers[3], cook_share: `${Math.round(tiers[3].length * 2 / totalWeight * 100)}%` },
    tier_4_cold: { count: tiers[4].length, states: tiers[4], cook_share: `${Math.round(tiers[4].length * 1 / totalWeight * 100)}%` },
    total_states: 50,
    total_cook_weight: totalWeight,
  };
}
