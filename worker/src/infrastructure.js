/**
 * Industrial Infrastructure Map — Ports, Power, Rail, Last-Mile
 * ==============================================================
 * Industrial lives on infrastructure. Not zip codes.
 * A warehouse is only worth what it can reach and how fast.
 *
 * 4 pillars:
 *   PORT   — container, bulk, auto, LNG, inland port
 *   POWER  — grid capacity, renewable, data center zones
 *   RAIL   — Class I, intermodal, shortline
 *   LAST   — population density reach, same-day delivery zones
 */

// ═══════════════════════════════════════════════════════
// PORTS — where goods enter the country
// ═══════════════════════════════════════════════════════

export const PORTS = {
  // CONTAINER PORTS (TEU volume)
  'PORT_LA_LB': {
    name: 'Port of Los Angeles / Long Beach',
    type: 'container',
    state: 'CA',
    city: 'Los Angeles',
    teu_rank: 1,
    annual_teu: 17400000,
    industrial_radius_mi: 80,
    key_corridors: ['I-710', 'I-110', 'Alameda Corridor (rail)'],
    submarkets: ['Inland Empire', 'LA South Bay', 'Carson/Compton', 'City of Industry'],
    notes: '#1 US container complex. 40% of all US imports. Alameda Corridor dedicated freight rail to DTLA rail yards.',
  },
  'PORT_NEWARK': {
    name: 'Port Newark / Elizabeth',
    type: 'container',
    state: 'NJ',
    city: 'Newark',
    teu_rank: 2,
    annual_teu: 9500000,
    industrial_radius_mi: 60,
    key_corridors: ['NJ Turnpike', 'I-78', 'I-95', 'Exit 8A corridor'],
    submarkets: ['Exit 8A', 'Meadowlands', 'Central NJ', 'Lehigh Valley (PA)'],
    notes: '#1 East Coast port. Bayonne Bridge raised to 215ft for mega-ships. Exit 8A is the #1 East Coast industrial submarket.',
  },
  'PORT_SAVANNAH': {
    name: 'Port of Savannah (GPA)',
    type: 'container',
    state: 'GA',
    city: 'Savannah',
    teu_rank: 3,
    annual_teu: 5900000,
    industrial_radius_mi: 100,
    key_corridors: ['I-16', 'I-95', 'CSX/Norfolk Southern rail'],
    submarkets: ['Savannah (Bryan/Effingham)', 'South Atlanta', 'Statesboro'],
    notes: 'Fastest-growing US port. Garden City Terminal is single largest container terminal in North America. Inland port in Cordele.',
  },
  'PORT_HOUSTON': {
    name: 'Port Houston',
    type: 'container',
    state: 'TX',
    city: 'Houston',
    teu_rank: 4,
    annual_teu: 4300000,
    industrial_radius_mi: 70,
    key_corridors: ['I-10', 'I-45', 'Ship Channel', 'BNSF/UP rail'],
    submarkets: ['Houston Ship Channel', 'Baytown', 'La Porte', 'Pasadena'],
    notes: '#1 US port by total tonnage. Largest petrochemical complex in Western Hemisphere. $1B expansion underway.',
  },
  'PORT_VIRGINIA': {
    name: 'Port of Virginia (Norfolk)',
    type: 'container',
    state: 'VA',
    city: 'Norfolk',
    teu_rank: 5,
    annual_teu: 3700000,
    industrial_radius_mi: 80,
    key_corridors: ['I-64', 'I-85 (via I-64)', 'CSX/Norfolk Southern'],
    submarkets: ['Hampton Roads', 'Suffolk', 'Chesapeake', 'Richmond'],
    notes: 'Deepest port on East Coast (55ft). Only East Coast port that can handle fully loaded mega-ships. No air draft restrictions.',
  },
  'PORT_CHARLESTON': {
    name: 'Port of Charleston (SCPA)',
    type: 'container',
    state: 'SC',
    city: 'Charleston',
    teu_rank: 6,
    annual_teu: 2900000,
    industrial_radius_mi: 80,
    key_corridors: ['I-26', 'I-95', 'CSX/Norfolk Southern', 'Inland Port Greer'],
    submarkets: ['North Charleston', 'Summerville', 'Upstate (via Inland Port Greer)'],
    notes: 'Fastest-growing East Coast port by %. Inland Port Greer extends reach 200mi to I-85 automotive corridor (BMW, Volvo).',
  },
  'PORT_JAXPORT': {
    name: 'JAXPORT (Jacksonville)',
    type: 'container',
    state: 'FL',
    city: 'Jacksonville',
    teu_rank: 7,
    annual_teu: 1600000,
    industrial_radius_mi: 60,
    key_corridors: ['I-95', 'I-10', 'CSX/Norfolk Southern rail'],
    submarkets: ['Westside Jacksonville', 'Imeson', 'Airport area'],
    notes: '#1 US port for auto imports. Strategic I-95/I-10 crossroads. Amazon Air Hub.',
  },
  'PORT_OAKLAND': {
    name: 'Port of Oakland',
    type: 'container',
    state: 'CA',
    city: 'Oakland',
    teu_rank: 8,
    annual_teu: 2500000,
    industrial_radius_mi: 50,
    key_corridors: ['I-880', 'I-580', 'BNSF/UP rail'],
    submarkets: ['East Bay', 'Hayward/Fremont', 'Central Valley (via I-580)'],
    notes: 'Primary NorCal container port. Direct rail to Central Valley ag/cold storage.',
  },
  'PORT_TACOMA': {
    name: 'Northwest Seaport Alliance (Seattle/Tacoma)',
    type: 'container',
    state: 'WA',
    city: 'Tacoma',
    teu_rank: 9,
    annual_teu: 3600000,
    industrial_radius_mi: 50,
    key_corridors: ['I-5', 'SR-167', 'BNSF/UP rail'],
    submarkets: ['Kent Valley', 'Sumner/Puyallup', 'SeaTac area'],
    notes: '#1 Pacific NW port. Trans-Pacific trade gateway. Kent Valley is the industrial engine.',
  },
  'PORT_MIAMI': {
    name: 'PortMiami',
    type: 'container',
    state: 'FL',
    city: 'Miami',
    teu_rank: 10,
    annual_teu: 1300000,
    industrial_radius_mi: 40,
    key_corridors: ['I-95', 'I-75 (via Alligator Alley)', 'FEC Railway'],
    submarkets: ['Miami-Dade (Medley/Hialeah)', 'Broward', 'Homestead'],
    notes: 'Gateway to Latin America and Caribbean. #1 cruise port. LatAm trade drives unique cargo mix.',
  },
  'PORT_MOBILE': {
    name: 'Port of Mobile',
    type: 'container',
    state: 'AL',
    city: 'Mobile',
    teu_rank: 15,
    annual_teu: 400000,
    industrial_radius_mi: 60,
    key_corridors: ['I-10', 'I-65', 'CSX/Norfolk Southern/BNSF'],
    submarkets: ['Mobile', 'Baldwin County', 'Theodore'],
    notes: 'Growing container port. Airbus assembly. Austal shipbuilding. 5 Class I railroads.',
  },

  // INLAND PORTS
  'PORT_ALLIANCE': {
    name: 'Alliance Global Logistics Hub',
    type: 'inland_port',
    state: 'TX',
    city: 'Fort Worth',
    annual_teu: null,
    industrial_radius_mi: 30,
    key_corridors: ['I-35W', 'SH-170', 'BNSF Alliance Intermodal'],
    submarkets: ['Alliance', 'Northlake', 'Haslet', 'Roanoke'],
    notes: 'Largest inland port master-planned development in US. 26K+ acres. Hillwood development. FedEx, Amazon, UPS mega-facilities. BNSF intermodal terminal.',
  },
  'PORT_INLAND_EMPIRE': {
    name: 'BNSF Inland Empire Intermodal',
    type: 'inland_port',
    state: 'CA',
    city: 'San Bernardino',
    annual_teu: null,
    industrial_radius_mi: 40,
    key_corridors: ['I-10', 'I-15', 'I-215', 'BNSF/UP rail'],
    submarkets: ['San Bernardino', 'Redlands', 'Riverside', 'Perris'],
    notes: 'Largest US industrial market by absorption. Direct rail from LA/LB ports. 600M+ SF inventory. Vacancy <3%.',
  },
  'PORT_JOLIET': {
    name: 'CenterPoint Intermodal (Joliet/Elwood)',
    type: 'inland_port',
    state: 'IL',
    city: 'Joliet',
    annual_teu: null,
    industrial_radius_mi: 40,
    key_corridors: ['I-80', 'I-55', 'BNSF/UP intermodal'],
    submarkets: ['Joliet', 'Elwood', 'Romeoville', 'Plainfield'],
    notes: 'Largest inland port in North America. BNSF Logistics Park Chicago + UP Global IV. I-80 mega-warehouse corridor.',
  },
  'PORT_KC': {
    name: 'Kansas City Inland Port',
    type: 'inland_port',
    state: 'MO',
    city: 'Kansas City',
    annual_teu: null,
    industrial_radius_mi: 50,
    key_corridors: ['I-70', 'I-35', 'I-29', 'BNSF/UP/NS/KCS rail'],
    submarkets: ['KCI area', 'Gardner KS', 'Edwardsville KS', 'Lee\'s Summit'],
    notes: 'Geographic center of US. Only metro served by all Class I railroads. Foreign Trade Zone. NorthPoint mega-spec.',
  },
  'PORT_GREER': {
    name: 'SC Inland Port Greer',
    type: 'inland_port',
    state: 'SC',
    city: 'Greer',
    annual_teu: null,
    industrial_radius_mi: 40,
    key_corridors: ['I-85', 'Norfolk Southern rail'],
    submarkets: ['Greenville', 'Spartanburg', 'Duncan', 'Greer'],
    notes: 'Revolutionary concept — extends Port of Charleston 212 miles inland via NS rail. BMW supply chain anchor. Auto manufacturing corridor.',
  },
  'PORT_COLUMBUS': {
    name: 'Rickenbacker Inland Port',
    type: 'inland_port',
    state: 'OH',
    city: 'Columbus',
    annual_teu: null,
    industrial_radius_mi: 40,
    key_corridors: ['I-70', 'I-71', 'I-270', 'Norfolk Southern intermodal'],
    submarkets: ['Rickenbacker', 'Groveport', 'Obetz', 'Canal Winchester'],
    notes: 'Only inland port with adjacent cargo airport. Norfolk Southern intermodal. FTZ. Within 600mi of 60% of US/Canadian population.',
  },
};

// ═══════════════════════════════════════════════════════
// POWER — grid capacity, data center zones, renewables
// ═══════════════════════════════════════════════════════

export const POWER_ZONES = {
  'NOVA_DATA': {
    name: 'Northern Virginia Data Center Corridor',
    state: 'VA',
    type: 'data_center',
    capacity_mw: 4000,
    submarkets: ['Ashburn', 'Sterling', 'Manassas', 'Prince William County'],
    notes: '70%+ of global internet traffic. Dominion Energy dedicated substations. Largest data center concentration on Earth.',
  },
  'PHOENIX_POWER': {
    name: 'Phoenix West Valley Power Zone',
    state: 'AZ',
    type: 'grid_surplus',
    capacity_mw: 2000,
    submarkets: ['Goodyear', 'Buckeye', 'Surprise', 'West Phoenix'],
    notes: 'Abundant solar + grid capacity. TSMC fab + data centers driving massive power demand.',
  },
  'DALLAS_POWER': {
    name: 'Dallas Data Center & Industrial Power',
    state: 'TX',
    type: 'grid_surplus',
    capacity_mw: 3000,
    submarkets: ['Garland', 'Allen', 'Plano', 'Midlothian'],
    notes: 'ERCOT grid — deregulated, cheap power. Major data center market (QTS, CyrusOne, DataBank). Also cold storage power-heavy.',
  },
  'IOWA_WIND': {
    name: 'Iowa Wind Power Corridor',
    state: 'IA',
    type: 'renewable',
    capacity_mw: 12000,
    submarkets: ['Altoona', 'Des Moines', 'Council Bluffs'],
    notes: '#1 US state for wind power (60%+ of grid). Meta, Microsoft, Google data centers — drawn by cheap renewable power.',
  },
  'RENO_POWER': {
    name: 'Reno/TRIC Power Zone',
    state: 'NV',
    type: 'grid_surplus',
    capacity_mw: 1500,
    submarkets: ['TRIC', 'Sparks', 'Storey County'],
    notes: 'Tesla Gigafactory, Switch data center, Google. Low-cost geothermal + solar power.',
  },
  'QUINCY_WA': {
    name: 'Quincy/Moses Lake Data Center Zone',
    state: 'WA',
    type: 'hydropower',
    capacity_mw: 2000,
    submarkets: ['Quincy', 'Moses Lake', 'Wenatchee'],
    notes: 'Cheapest power in US (Grand Coulee Dam). Microsoft, Yahoo, Dell data centers. Hydropower = carbon neutral.',
  },
  'CHEYENNE_DATA': {
    name: 'Cheyenne Data Center Zone',
    state: 'WY',
    type: 'data_center',
    capacity_mw: 500,
    submarkets: ['Cheyenne'],
    notes: 'No income tax + cheap power + cool climate. Microsoft ECDC. Emerging market.',
  },
  'NUKE_CORRIDOR_IL': {
    name: 'Illinois Nuclear Power Corridor',
    state: 'IL',
    type: 'nuclear',
    capacity_mw: 11500,
    submarkets: ['Byron', 'Braidwood', 'LaSalle', 'Dresden'],
    notes: '#1 US state for nuclear generation. Reliable baseload for industrial. Constellation Energy fleet.',
  },
};

// ═══════════════════════════════════════════════════════
// RAIL — Class I intermodal, key junctions
// ═══════════════════════════════════════════════════════

export const RAIL_HUBS = {
  'RAIL_CHICAGO': {
    name: 'Chicago Rail Hub',
    state: 'IL',
    class_1_railroads: ['BNSF', 'UP', 'NS', 'CSX', 'CN', 'CP'],
    type: 'national_hub',
    intermodal_terminals: 7,
    notes: '#1 US rail hub. All 6 Class I railroads converge. 25% of all US rail traffic passes through Chicago. CREATE program upgrading bottlenecks.',
  },
  'RAIL_KC': {
    name: 'Kansas City Rail Hub',
    state: 'MO',
    class_1_railroads: ['BNSF', 'UP', 'NS', 'KCS/CPKC'],
    type: 'national_hub',
    intermodal_terminals: 4,
    notes: '#2 US rail hub. Only metro with all Class I railroads. KCS/CPKC connects to Mexico (nearshoring).',
  },
  'RAIL_MEMPHIS': {
    name: 'Memphis Rail Hub',
    state: 'TN',
    class_1_railroads: ['BNSF', 'UP', 'NS', 'CSX', 'CN'],
    type: 'national_hub',
    intermodal_terminals: 3,
    notes: '5 Class I railroads + FedEx superhub. Multimodal: air (#1 cargo) + rail + river (Mississippi). Most connected city in US.',
  },
  'RAIL_ATLANTA': {
    name: 'Atlanta Rail Hub',
    state: 'GA',
    class_1_railroads: ['NS', 'CSX'],
    type: 'regional_hub',
    intermodal_terminals: 3,
    notes: 'NS and CSX Southeast hub. Inman Yard is one of largest classification yards in US. Direct to Port of Savannah.',
  },
  'RAIL_DFW': {
    name: 'Dallas-Fort Worth Rail Hub',
    state: 'TX',
    class_1_railroads: ['BNSF', 'UP', 'KCS/CPKC'],
    type: 'regional_hub',
    intermodal_terminals: 3,
    notes: 'BNSF Alliance intermodal is anchor of Alliance Global Logistics Hub. UP Dallas intermodal. CPKC Mexico corridor.',
  },
  'RAIL_COLUMBUS': {
    name: 'Columbus Rail Hub',
    state: 'OH',
    class_1_railroads: ['NS', 'CSX'],
    type: 'regional_hub',
    intermodal_terminals: 2,
    notes: 'NS Rickenbacker intermodal + CSX. Geographic center of eastern US logistics.',
  },
  'RAIL_LA': {
    name: 'Los Angeles Rail Hub',
    state: 'CA',
    class_1_railroads: ['BNSF', 'UP'],
    type: 'port_rail',
    intermodal_terminals: 4,
    notes: 'Alameda Corridor = dedicated freight rail from LA/LB ports to downtown rail yards. BNSF San Bernardino intermodal feeds Inland Empire.',
  },
  'RAIL_CPKC_MEXICO': {
    name: 'CPKC Mexico Corridor',
    state: 'TX',
    class_1_railroads: ['CPKC'],
    type: 'cross_border',
    intermodal_terminals: 3,
    notes: 'Only single-line railroad connecting US-Mexico-Canada. Laredo, Eagle Pass, El Paso crossings. CRITICAL for nearshoring.',
  },
};

// ═══════════════════════════════════════════════════════
// LAST-MILE — population reach, same-day delivery zones
// ═══════════════════════════════════════════════════════

export const LAST_MILE_ZONES = {
  'LM_NYC_METRO': {
    name: 'NYC Metro Last-Mile',
    state: 'NJ',
    population_reach_1hr: 20000000,
    key_submarkets: ['Meadowlands', 'South Bronx', 'Brooklyn', 'Exit 8A (next-day)'],
    avg_rent_psf: 22,
    vacancy_pct: 1.8,
    notes: 'Most expensive last-mile market in US. South Bronx = closest industrial to Manhattan. Meadowlands = Amazon/UPS mega-sort.',
  },
  'LM_LA_METRO': {
    name: 'LA Metro Last-Mile',
    state: 'CA',
    population_reach_1hr: 13000000,
    key_submarkets: ['LA South Bay', 'City of Industry', 'Vernon/Commerce', 'Orange County'],
    avg_rent_psf: 18,
    vacancy_pct: 2.1,
    notes: 'Second-largest US metro. Infill industrial is irreplaceable — no new land. Vernon/Commerce = urban industrial core.',
  },
  'LM_CHICAGO': {
    name: 'Chicago Metro Last-Mile',
    state: 'IL',
    population_reach_1hr: 9500000,
    key_submarkets: ['OHare area', 'I-55 corridor', 'South suburbs', 'Kenosha (WI spillover)'],
    avg_rent_psf: 10,
    vacancy_pct: 4.2,
    notes: '#3 US metro. OHare area = air cargo + last-mile. I-55 = e-commerce fulfillment.',
  },
  'LM_DFW': {
    name: 'DFW Metro Last-Mile',
    state: 'TX',
    population_reach_1hr: 8000000,
    key_submarkets: ['South Dallas', 'Grand Prairie', 'Arlington', 'Mesquite'],
    avg_rent_psf: 8,
    vacancy_pct: 5.1,
    notes: '#4 US metro and growing fast. Population growth = last-mile demand growth. Rent below coastal but rising.',
  },
  'LM_HOUSTON': {
    name: 'Houston Metro Last-Mile',
    state: 'TX',
    population_reach_1hr: 7000000,
    key_submarkets: ['Northwest Houston', 'Katy', 'Missouri City', 'Baytown'],
    avg_rent_psf: 8,
    vacancy_pct: 5.8,
    notes: '#5 US metro. Energy economy drives unique tenant mix. Hurricane risk priced in.',
  },
  'LM_ATLANTA': {
    name: 'Atlanta Metro Last-Mile',
    state: 'GA',
    population_reach_1hr: 6500000,
    key_submarkets: ['South Atlanta/Fairburn', 'Midtown/West Midtown', 'Peachtree Corners'],
    avg_rent_psf: 8,
    vacancy_pct: 4.5,
    notes: 'Southeast logistics capital. Hartsfield-Jackson = #1 passenger airport (drives cargo too).',
  },
  'LM_MIAMI': {
    name: 'South Florida Last-Mile',
    state: 'FL',
    population_reach_1hr: 6500000,
    key_submarkets: ['Medley/Hialeah', 'Doral', 'Pompano Beach', 'Opa-Locka'],
    avg_rent_psf: 14,
    vacancy_pct: 2.8,
    notes: 'Unique market — LatAm trade + population growth + no income tax. Land-constrained = irreplaceable infill.',
  },
  'LM_PHILLY': {
    name: 'Philadelphia Metro Last-Mile',
    state: 'PA',
    population_reach_1hr: 6000000,
    key_submarkets: ['Southwest Philly', 'Camden (NJ)', 'I-76 corridor', 'Bucks County'],
    avg_rent_psf: 10,
    vacancy_pct: 3.5,
    notes: 'Spillover from NJ mega-market. Port of Philly for specialized cargo.',
  },
  'LM_SEATTLE': {
    name: 'Seattle/Tacoma Last-Mile',
    state: 'WA',
    population_reach_1hr: 4000000,
    key_submarkets: ['Kent Valley', 'Tukwila', 'Federal Way', 'Auburn'],
    avg_rent_psf: 16,
    vacancy_pct: 2.5,
    notes: 'Amazon HQ effect. Kent Valley = PNW industrial engine. Ultra-tight, no new land.',
  },
  'LM_PHOENIX': {
    name: 'Phoenix Metro Last-Mile',
    state: 'AZ',
    population_reach_1hr: 5000000,
    key_submarkets: ['West Phoenix', 'Chandler/Gilbert', 'Mesa', 'Tempe/Sky Harbor'],
    avg_rent_psf: 9,
    vacancy_pct: 5.5,
    notes: 'Fastest-growing large metro. Massive spec pipeline means higher vacancy but rent growth strong.',
  },
};

// ═══════════════════════════════════════════════════════
// AIR CARGO — the premium last-mile
// ═══════════════════════════════════════════════════════

export const AIR_CARGO = {
  'AIR_MEMPHIS': { name: 'Memphis (MEM)', state: 'TN', rank: 1, metric_tons: 4500000, anchor: 'FedEx World Hub', notes: '#1 cargo airport in world. FedEx global superhub.' },
  'AIR_ANCHORAGE': { name: 'Anchorage (ANC)', state: 'AK', rank: 2, metric_tons: 3200000, anchor: 'Trans-Pacific refueling', notes: 'Tech stop for Asia-US cargo flights. UPS, FedEx, Atlas Air.' },
  'AIR_LOUISVILLE': { name: 'Louisville (SDF)', state: 'KY', rank: 3, metric_tons: 2900000, anchor: 'UPS Worldport', notes: 'UPS global superhub. Worldport processes 2M packages/day.' },
  'AIR_MIAMI': { name: 'Miami (MIA)', state: 'FL', rank: 4, metric_tons: 2600000, anchor: 'LatAm gateway', notes: '#1 international freight airport in US. Perishables from LatAm.' },
  'AIR_LAX': { name: 'Los Angeles (LAX)', state: 'CA', rank: 5, metric_tons: 2500000, anchor: 'Trans-Pacific trade', notes: '#1 Pacific trade cargo airport. High-value goods.' },
  'AIR_OHARE': { name: 'Chicago O\'Hare (ORD)', state: 'IL', rank: 6, metric_tons: 1900000, anchor: 'Hub connectivity', notes: 'Central US air cargo hub. Amazon Air operations.' },
  'AIR_JFK': { name: 'New York JFK', state: 'NY', rank: 7, metric_tons: 1600000, anchor: 'Transatlantic trade', notes: '#1 transatlantic cargo. Pharma/high-value.' },
  'AIR_IND': { name: 'Indianapolis (IND)', state: 'IN', rank: 8, metric_tons: 1400000, anchor: 'FedEx #2 hub', notes: 'FedEx second-largest hub + Amazon Air Hub.' },
  'AIR_CVG': { name: 'Cincinnati/NKY (CVG)', state: 'KY', rank: 9, metric_tons: 1200000, anchor: 'Amazon Air Hub + DHL', notes: 'Amazon Air Hub (massive expansion). DHL Americas hub.' },
  'AIR_DFW': { name: 'Dallas/Fort Worth (DFW)', state: 'TX', rank: 10, metric_tons: 1100000, anchor: 'AA Cargo + Amazon', notes: 'American Airlines Cargo hub. Growing e-commerce air freight.' },
};

/**
 * Get all infrastructure for a state.
 */
export function getStateInfrastructure(stateCode) {
  const sc = stateCode.toUpperCase();
  return {
    ports: Object.entries(PORTS).filter(([_, p]) => p.state === sc).map(([id, p]) => ({ id, ...p })),
    power: Object.entries(POWER_ZONES).filter(([_, p]) => p.state === sc).map(([id, p]) => ({ id, ...p })),
    rail: Object.entries(RAIL_HUBS).filter(([_, r]) => r.state === sc).map(([id, r]) => ({ id, ...r })),
    last_mile: Object.entries(LAST_MILE_ZONES).filter(([_, l]) => l.state === sc).map(([id, l]) => ({ id, ...l })),
    air_cargo: Object.entries(AIR_CARGO).filter(([_, a]) => a.state === sc).map(([id, a]) => ({ id, ...a })),
  };
}

/**
 * Infrastructure-based system prompt addition.
 * Injects local infrastructure knowledge into the area expert.
 */
export function infrastructurePrompt(stateCode) {
  const infra = getStateInfrastructure(stateCode);
  const parts = [];

  if (infra.ports.length) {
    parts.push(`PORTS: ${infra.ports.map(p => `${p.name} (${p.type}, ${p.annual_teu ? p.annual_teu.toLocaleString() + ' TEU' : 'inland'})`).join('; ')}`);
  }
  if (infra.power.length) {
    parts.push(`POWER: ${infra.power.map(p => `${p.name} (${p.type}, ${p.capacity_mw}MW)`).join('; ')}`);
  }
  if (infra.rail.length) {
    parts.push(`RAIL: ${infra.rail.map(r => `${r.name} (${r.class_1_railroads.join('/')})`).join('; ')}`);
  }
  if (infra.last_mile.length) {
    parts.push(`LAST-MILE: ${infra.last_mile.map(l => `${l.name} (${(l.population_reach_1hr / 1000000).toFixed(1)}M pop, ${l.vacancy_pct}% vac, $${l.avg_rent_psf}/SF)`).join('; ')}`);
  }
  if (infra.air_cargo.length) {
    parts.push(`AIR CARGO: ${infra.air_cargo.map(a => `${a.name} (#${a.rank}, ${a.anchor})`).join('; ')}`);
  }

  return parts.length ? `\nINFRASTRUCTURE:\n${parts.join('\n')}` : '';
}
