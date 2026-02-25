"""
SwarmCRE Dataset Factory — Constants & Reference Data

All magic numbers, ranges, enums, and market data live here.
Everything the deal generator, templates, and quality gates consume.
"""

# ═══════════════════════════════════════════════════════════════════
# ASSET TYPES — 9 industrial-focused CRE types (incl. data centers)
# Weights mirror real CBRE industrial market composition
# ═══════════════════════════════════════════════════════════════════

ASSET_TYPES = {
    "infill_warehouse": {
        "weight": 0.30,
        "display": "infill warehouse",
        "sf_range": (15_000, 250_000),
        "clear_height_range": (24, 36),
        "office_pct_range": (0.03, 0.15),
        "dock_doors_per_10k_sf": (1.0, 3.0),
        "grade_doors_per_10k_sf": (0.0, 1.0),
        "truck_court_range": (110, 185),
        "trailer_spots_per_10k_sf": (0.5, 2.0),
        "column_spacing_options": ["40x40", "50x50", "50x52", "52x54"],
        "sprinkler_options": ["ESFR", "wet"],
        "typical_tenancy": ["single", "multi"],
        "cap_rate_range": (0.045, 0.070),
        "land_ratio_range": (2.5, 4.5),  # acres per 100k SF
        "year_built_range": (1975, 2025),
    },
    "small_bay_industrial": {
        "weight": 0.15,
        "display": "small-bay industrial",
        "sf_range": (8_000, 60_000),
        "clear_height_range": (16, 24),
        "office_pct_range": (0.10, 0.30),
        "dock_doors_per_10k_sf": (0.5, 2.0),
        "grade_doors_per_10k_sf": (1.0, 3.0),
        "truck_court_range": (60, 130),
        "trailer_spots_per_10k_sf": (0.0, 0.5),
        "column_spacing_options": ["30x30", "40x40", "30x40"],
        "sprinkler_options": ["wet", "dry"],
        "typical_tenancy": ["multi"],
        "cap_rate_range": (0.055, 0.085),
        "land_ratio_range": (2.0, 3.5),
        "year_built_range": (1970, 2020),
    },
    "light_industrial_flex": {
        "weight": 0.12,
        "display": "light industrial / flex",
        "sf_range": (10_000, 80_000),
        "clear_height_range": (14, 22),
        "office_pct_range": (0.20, 0.50),
        "dock_doors_per_10k_sf": (0.0, 1.5),
        "grade_doors_per_10k_sf": (1.0, 3.0),
        "truck_court_range": (40, 100),
        "trailer_spots_per_10k_sf": (0.0, 0.3),
        "column_spacing_options": ["30x30", "40x40"],
        "sprinkler_options": ["wet", "dry"],
        "typical_tenancy": ["multi"],
        "cap_rate_range": (0.060, 0.090),
        "land_ratio_range": (2.0, 3.0),
        "year_built_range": (1980, 2024),
    },
    "cross_dock": {
        "weight": 0.08,
        "display": "cross-dock distribution",
        "sf_range": (50_000, 300_000),
        "clear_height_range": (28, 40),
        "office_pct_range": (0.02, 0.08),
        "dock_doors_per_10k_sf": (2.0, 5.0),
        "grade_doors_per_10k_sf": (0.0, 0.5),
        "truck_court_range": (130, 195),
        "trailer_spots_per_10k_sf": (1.5, 4.0),
        "column_spacing_options": ["50x50", "52x54", "54x56", "60x60"],
        "sprinkler_options": ["ESFR"],
        "typical_tenancy": ["single"],
        "cap_rate_range": (0.040, 0.060),
        "land_ratio_range": (3.0, 6.0),
        "year_built_range": (1995, 2026),
    },
    "cold_storage": {
        "weight": 0.08,
        "display": "cold storage facility",
        "sf_range": (20_000, 150_000),
        "clear_height_range": (28, 40),
        "office_pct_range": (0.03, 0.10),
        "dock_doors_per_10k_sf": (1.5, 3.5),
        "grade_doors_per_10k_sf": (0.0, 0.5),
        "truck_court_range": (120, 180),
        "trailer_spots_per_10k_sf": (0.5, 2.0),
        "column_spacing_options": ["40x40", "50x50"],
        "sprinkler_options": ["wet", "dry"],
        "typical_tenancy": ["single", "multi"],
        "cap_rate_range": (0.055, 0.080),
        "land_ratio_range": (2.5, 4.0),
        "year_built_range": (1985, 2025),
        "cold_chain": True,
        "temp_zones": ["frozen", "cooler", "controlled_atmosphere"],
    },
    "ios_truck_yard": {
        "weight": 0.08,
        "display": "IOS / truck yard / outdoor storage",
        "sf_range": (2_000, 15_000),  # office/support building only
        "clear_height_range": (12, 18),
        "office_pct_range": (0.50, 1.00),  # mostly office since yard is outdoors
        "dock_doors_per_10k_sf": (0.0, 0.5),
        "grade_doors_per_10k_sf": (0.5, 2.0),
        "truck_court_range": (0, 0),  # N/A -- entire site is yard
        "trailer_spots_per_10k_sf": (0.0, 0.0),  # tracked as yard_spots instead
        "column_spacing_options": ["N/A"],
        "sprinkler_options": ["wet"],
        "typical_tenancy": ["single", "multi"],
        "cap_rate_range": (0.060, 0.095),
        "land_ratio_range": (8.0, 20.0),  # land-heavy
        "year_built_range": (1960, 2020),
        "yard_acres_range": (2.0, 25.0),
        "yard_spots_range": (20, 200),
    },
    "micro_fulfillment": {
        "weight": 0.07,
        "display": "micro-fulfillment / dark store conversion",
        "sf_range": (10_000, 50_000),
        "clear_height_range": (16, 28),
        "office_pct_range": (0.05, 0.20),
        "dock_doors_per_10k_sf": (0.5, 2.0),
        "grade_doors_per_10k_sf": (1.0, 4.0),
        "truck_court_range": (60, 120),
        "trailer_spots_per_10k_sf": (0.0, 0.5),
        "column_spacing_options": ["30x30", "40x40"],
        "sprinkler_options": ["ESFR", "wet"],
        "typical_tenancy": ["single"],
        "cap_rate_range": (0.050, 0.075),
        "land_ratio_range": (1.5, 3.0),
        "year_built_range": (1990, 2025),
    },
    "industrial_land": {
        "weight": 0.07,
        "display": "industrial land / entitled logistics site",
        "sf_range": (0, 0),  # no building yet
        "clear_height_range": (0, 0),
        "office_pct_range": (0.0, 0.0),
        "dock_doors_per_10k_sf": (0.0, 0.0),
        "grade_doors_per_10k_sf": (0.0, 0.0),
        "truck_court_range": (0, 0),
        "trailer_spots_per_10k_sf": (0.0, 0.0),
        "column_spacing_options": ["N/A"],
        "sprinkler_options": ["N/A"],
        "typical_tenancy": ["none"],
        "cap_rate_range": (0.0, 0.0),  # land doesn't have cap rates in the traditional sense
        "land_ratio_range": (0.0, 0.0),
        "year_built_range": (0, 0),
        "land_acres_range": (3.0, 50.0),
        "price_per_acre_range": (150_000, 2_500_000),
        "entitlement_options": ["fully_entitled", "partially_entitled", "raw"],
        "zoning_options": ["M1", "M2", "planned_industrial", "agricultural_rezone"],
    },
    "data_center": {
        "weight": 0.05,
        "display": "data center",
        "sf_range": (20_000, 500_000),
        "clear_height_range": (0, 0),  # not applicable — measured by MW / rack density
        "office_pct_range": (0.05, 0.15),
        "dock_doors_per_10k_sf": (0.0, 0.5),
        "grade_doors_per_10k_sf": (0.0, 0.5),
        "truck_court_range": (0, 60),
        "trailer_spots_per_10k_sf": (0.0, 0.0),
        "column_spacing_options": ["N/A"],
        "sprinkler_options": ["pre-action", "clean_agent", "dual_interlock"],
        "typical_tenancy": ["single", "multi"],
        "cap_rate_range": (0.040, 0.065),
        "land_ratio_range": (3.0, 8.0),
        "year_built_range": (2000, 2026),
        # --- Data-center-specific fields ---
        "power_mw_range": (1.0, 200.0),
        "pue_range": (1.10, 1.60),
        "rack_density_per_1000sf": (8, 30),
        "cooling_type_options": ["air_cooled", "liquid_cooled", "immersion", "hybrid", "rear_door_heat_exchanger"],
        "construction_cost_psf_range": (350, 1_200),
        "construction_cost_per_mw_range": (8_000_000, 15_000_000),
        "fiber_connectivity": True,
        "redundancy_level_options": ["N", "N+1", "2N", "2N+1"],
        "tier_classification_options": ["Tier_I", "Tier_II", "Tier_III", "Tier_IV"],
    },
}

# ═══════════════════════════════════════════════════════════════════
# DATA CENTER SPECIFICATIONS
# Detailed power, cooling, and technology parameters for data centers
# ═══════════════════════════════════════════════════════════════════

DATA_CENTER_SPECS = {
    "power": {
        "utility_voltage_options": ["12.47kV", "34.5kV", "69kV", "138kV"],
        "backup_power_options": ["diesel_generator", "natural_gas_generator", "battery_ups", "flywheel_ups"],
        "ups_runtime_minutes_range": (5, 30),
        "generator_runtime_hours_range": (24, 72),
        "power_density_watts_per_sf_range": (100, 350),
        "it_load_pct_of_total_range": (0.55, 0.75),
    },
    "cooling": {
        "air_cooled": {
            "pue_range": (1.30, 1.60),
            "capex_per_ton_range": (3_000, 6_000),
        },
        "liquid_cooled": {
            "pue_range": (1.10, 1.30),
            "capex_per_ton_range": (5_000, 10_000),
        },
        "immersion": {
            "pue_range": (1.02, 1.10),
            "capex_per_ton_range": (8_000, 15_000),
        },
        "hybrid": {
            "pue_range": (1.15, 1.35),
            "capex_per_ton_range": (4_500, 9_000),
        },
        "rear_door_heat_exchanger": {
            "pue_range": (1.15, 1.30),
            "capex_per_ton_range": (4_000, 8_000),
        },
    },
    "technology": {
        "gpu_density_per_rack_range": (4, 16),
        "rack_power_kw_range": (5, 100),
        "standard_rack_power_kw": 10,
        "high_density_rack_power_kw": 40,
        "ai_training_rack_power_kw": 100,
        "network_options": ["1GbE", "10GbE", "25GbE", "100GbE", "400GbE"],
        "fiber_strand_count_options": [48, 96, 144, 288, 576],
    },
    "certifications": {
        "tier_classification": {
            "Tier_I": {"uptime_pct": 99.671, "redundancy": "N", "annual_downtime_hours": 28.8},
            "Tier_II": {"uptime_pct": 99.741, "redundancy": "N+1", "annual_downtime_hours": 22.0},
            "Tier_III": {"uptime_pct": 99.982, "redundancy": "N+1", "annual_downtime_hours": 1.6},
            "Tier_IV": {"uptime_pct": 99.995, "redundancy": "2N+1", "annual_downtime_hours": 0.4},
        },
        "compliance_options": ["SOC2", "ISO27001", "HIPAA", "PCI-DSS", "FedRAMP"],
    },
    "financial": {
        "lease_term_years_range": (5, 20),
        "escalation_pct_range": (0.02, 0.04),
        "power_cost_per_kwh_range": (0.04, 0.12),
        "noi_per_mw_range": (800_000, 2_500_000),
    },
}

# ═══════════════════════════════════════════════════════════════════
# MARKET TIERS — rent, vacancy, cap rate ranges by tier
# ═══════════════════════════════════════════════════════════════════

MARKET_TIERS = {
    "gateway": {
        "weight": 0.25,
        "markets": [
            ("Los Angeles / Inland Empire", ["South Bay", "Mid-Counties", "Ontario/Rancho", "East IE"]),
            ("New York / New Jersey", ["Meadowlands", "Exit 8A", "Bronx/Queens", "Central NJ"]),
            ("Chicago", ["I-55 Corridor", "O'Hare", "I-80/Joliet", "South Suburbs"]),
            ("Dallas / Fort Worth", ["South Dallas", "DFW Airport", "Alliance", "I-35 Corridor"]),
            ("Atlanta", ["I-20 West", "I-85 South", "Hartsfield Area", "Northeast Atlanta"]),
        ],
        "rent_psf_range": (8.50, 18.00),
        "vacancy_range": (0.02, 0.06),
        "cap_rate_adjustment": -0.005,
    },
    "primary": {
        "weight": 0.30,
        "markets": [
            ("Phoenix", ["Sky Harbor", "West Valley", "Chandler/Gilbert", "Deer Valley"]),
            ("Nashville", ["I-24 Corridor", "Airport/Donelson", "Lebanon/Mt Juliet", "Cool Springs"]),
            ("Austin", ["East Austin", "Round Rock/Georgetown", "Buda/Kyle", "Airport/SE"]),
            ("Denver", ["I-70/Peña Blvd", "Aurora", "Centennial/DTC", "Brighton/Commerce City"]),
            ("Charlotte", ["Airport/West", "I-77 South", "I-85 North", "University/NoDa"]),
            ("Houston", ["Northwest", "I-10 East", "SW/Sugar Land", "Humble/IAH"]),
            ("Indianapolis", ["Plainfield/Airport", "I-70 West", "Whitestown", "Greenwood"]),
            ("Columbus", ["West Columbus", "Rickenbacker", "Etna/Pataskala", "Groveport"]),
        ],
        "rent_psf_range": (6.00, 12.00),
        "vacancy_range": (0.03, 0.08),
        "cap_rate_adjustment": 0.0,
    },
    "secondary": {
        "weight": 0.30,
        "markets": [
            ("Memphis", ["Airport/Lamar", "Olive Branch", "Southeast", "I-40 Corridor"]),
            ("Louisville", ["I-65 South", "Airport/UPS Hub", "Shepherdsville", "River Ridge"]),
            ("Kansas City", ["I-435/I-70", "Edwardsville KS", "KCI Airport", "Lee's Summit"]),
            ("El Paso", ["East El Paso", "Airport", "Santa Teresa NM", "West Side"]),
            ("San Antonio", ["I-35 South", "East Side", "Brooks City Base", "I-10 West"]),
            ("Reno / Sparks", ["Tahoe Reno Industrial", "North Valleys", "Spanish Springs", "USA Pkwy"]),
            ("Salt Lake City", ["I-15 Corridor", "West Valley", "Airport", "Ogden"]),
        ],
        "rent_psf_range": (4.00, 8.00),
        "vacancy_range": (0.05, 0.12),
        "cap_rate_adjustment": 0.010,
    },
    "tertiary": {
        "weight": 0.15,
        "markets": [
            ("Boise", ["Meridian", "Nampa/Caldwell", "Airport", "Eagle"]),
            ("Tucson", ["I-10 Corridor", "Airport", "South Tucson", "Marana"]),
            ("Greenville SC", ["I-85 Corridor", "GSP Airport", "Mauldin", "Easley"]),
            ("Savannah", ["Port Area", "I-16 Corridor", "West Chatham", "Garden City"]),
            ("Huntsville AL", ["Research Park", "I-565 Corridor", "Madison", "Decatur"]),
        ],
        "rent_psf_range": (3.00, 6.00),
        "vacancy_range": (0.06, 0.15),
        "cap_rate_adjustment": 0.020,
    },
}

# ═══════════════════════════════════════════════════════════════════
# TENANT CREDIT TIERS
# ═══════════════════════════════════════════════════════════════════

CREDIT_TIERS = {
    "investment_grade": {
        "weight": 0.20,
        "examples": [
            "Amazon", "FedEx", "Walmart", "UPS", "Home Depot",
            "Procter & Gamble", "PepsiCo", "Target", "Sysco",
            "XPO Logistics", "DHL", "Maersk", "Costco",
        ],
        "rent_premium": 1.00,
        "cap_discount": -0.005,
        "guarantee": "corporate",
        "ti_allowance_psf_range": (0, 5),
    },
    "national_credit": {
        "weight": 0.30,
        "examples": [
            "regional food distributor", "national 3PL operator",
            "multi-state manufacturing co", "national HVAC distributor",
            "automotive parts chain", "building materials supplier",
            "medical supply distributor", "tech hardware fulfillment",
        ],
        "rent_premium": 0.95,
        "cap_discount": 0.0,
        "guarantee": "corporate_limited",
        "ti_allowance_psf_range": (2, 10),
    },
    "local_credit": {
        "weight": 0.35,
        "examples": [
            "local manufacturer", "family-owned distributor",
            "regional food producer", "local e-commerce fulfillment",
            "construction materials yard", "cabinet/millwork shop",
            "local printing company", "HVAC contractor",
        ],
        "rent_premium": 0.85,
        "cap_discount": 0.010,
        "guarantee": "personal",
        "ti_allowance_psf_range": (3, 15),
    },
    "startup": {
        "weight": 0.15,
        "examples": [
            "e-commerce startup", "new logistics company",
            "cannabis cultivator (where legal)", "craft brewery/distillery",
            "EV fleet operator", "drone fulfillment startup",
            "meal kit assembly", "vertical farming operation",
        ],
        "rent_premium": 0.75,
        "cap_discount": 0.020,
        "guarantee": "personal_plus_deposit",
        "ti_allowance_psf_range": (5, 25),
    },
}

# ═══════════════════════════════════════════════════════════════════
# DEBT TEMPLATES
# ═══════════════════════════════════════════════════════════════════

DEBT_TEMPLATES = [
    {
        "name": "agency_perm",
        "display": "Agency Permanent",
        "ltv_range": (0.60, 0.70),
        "rate_range": (0.0525, 0.0700),
        "amort_years": 25,
        "io_years": 0,
        "term_years": 10,
        "min_dscr": 1.25,
    },
    {
        "name": "cmbs",
        "display": "CMBS",
        "ltv_range": (0.65, 0.75),
        "rate_range": (0.0575, 0.0750),
        "amort_years": 30,
        "io_years": 2,
        "term_years": 10,
        "min_dscr": 1.20,
    },
    {
        "name": "bridge",
        "display": "Bridge Loan",
        "ltv_range": (0.70, 0.80),
        "rate_range": (0.0750, 0.1000),
        "amort_years": 0,  # IO only
        "io_years": 3,
        "term_years": 3,
        "min_dscr": 1.10,
    },
    {
        "name": "bank_cre",
        "display": "Bank CRE",
        "ltv_range": (0.55, 0.65),
        "rate_range": (0.0600, 0.0775),
        "amort_years": 25,
        "io_years": 0,
        "term_years": 5,
        "min_dscr": 1.30,
    },
    {
        "name": "life_co",
        "display": "Life Company",
        "ltv_range": (0.50, 0.60),
        "rate_range": (0.0500, 0.0650),
        "amort_years": 30,
        "io_years": 0,
        "term_years": 15,
        "min_dscr": 1.35,
    },
    {
        "name": "sba_504",
        "display": "SBA 504",
        "ltv_range": (0.75, 0.85),
        "rate_range": (0.0575, 0.0725),
        "amort_years": 25,
        "io_years": 0,
        "term_years": 25,
        "min_dscr": 1.15,
    },
]

# ═══════════════════════════════════════════════════════════════════
# EXPENSE LINE ITEMS (per SF per year)
# ═══════════════════════════════════════════════════════════════════

EXPENSE_LINES = {
    "property_tax": {"range": (1.00, 3.50), "nnn_passthrough": True},
    "insurance": {"range": (0.25, 0.75), "nnn_passthrough": True},
    "cam_maintenance": {"range": (0.50, 2.00), "nnn_passthrough": True},
    "management_fee_pct": {"range": (0.03, 0.06), "nnn_passthrough": False},  # % of EGI
    "reserves": {"range": (0.10, 0.30), "nnn_passthrough": False},
    "utilities": {"range": (0.00, 1.50), "nnn_passthrough": True},
}

# Lease type determines which expenses are tenant-paid
LEASE_TYPES = {
    "NNN": {
        "weight": 0.55,
        "tenant_pays": ["property_tax", "insurance", "cam_maintenance", "utilities"],
        "display": "Triple Net (NNN)",
    },
    "modified_gross": {
        "weight": 0.30,
        "tenant_pays": ["utilities"],
        "display": "Modified Gross",
    },
    "gross": {
        "weight": 0.15,
        "tenant_pays": [],
        "display": "Full Service Gross",
    },
}

# ═══════════════════════════════════════════════════════════════════
# ESCALATION TYPES
# ═══════════════════════════════════════════════════════════════════

ESCALATION_TYPES = {
    "fixed": {"weight": 0.50, "range": (0.02, 0.035)},  # 2-3.5% annual
    "cpi": {"weight": 0.30, "range": (0.0, 0.0)},  # CPI-based (conceptual)
    "flat": {"weight": 0.20, "range": (0.0, 0.0)},  # no escalation
}

# ═══════════════════════════════════════════════════════════════════
# ECONOMIC INCENTIVES — grants, abatements, and development programs
# Used across all asset types; especially relevant for data centers,
# large industrial BTS, and deals in opportunity/enterprise zones.
# ═══════════════════════════════════════════════════════════════════

ECONOMIC_INCENTIVES = {
    "tif_tid": {
        "display": "Tax Increment Financing / Tax Increment District (TIF/TID)",
        "typical_value_range": (500_000, 50_000_000),
        "term_years": (10, 25),
        "common_requirements": [
            "but-for test (project would not proceed without incentive)",
            "public benefit demonstration",
            "job creation commitment",
            "minimum capital investment threshold",
            "blight or redevelopment area designation",
        ],
    },
    "property_tax_abatement": {
        "display": "Property Tax Abatement",
        "typical_value_range": (50_000, 10_000_000),
        "term_years": (5, 15),
        "common_requirements": [
            "minimum capital investment",
            "job creation or retention targets",
            "community benefit agreement",
            "clawback provisions if targets unmet",
        ],
    },
    "job_creation_grant": {
        "display": "Job Creation / Workforce Development Grant",
        "typical_value_range": (100_000, 20_000_000),
        "term_years": (3, 10),
        "common_requirements": [
            "minimum number of full-time jobs created",
            "minimum average wage threshold",
            "benefits package requirement",
            "retention period (typically 3-5 years)",
            "quarterly or annual reporting",
        ],
    },
    "infrastructure_grant": {
        "display": "Infrastructure / Public Improvement Grant",
        "typical_value_range": (250_000, 100_000_000),
        "term_years": (1, 10),
        "common_requirements": [
            "public infrastructure improvement (roads, utilities, water/sewer)",
            "environmental remediation needs",
            "matching funds from developer",
            "project completion timeline",
            "public access or benefit component",
        ],
    },
    "enterprise_zone_credit": {
        "display": "Enterprise Zone Tax Credit",
        "typical_value_range": (25_000, 5_000_000),
        "term_years": (5, 15),
        "common_requirements": [
            "location within designated enterprise zone",
            "eligible business activity",
            "job creation for zone residents",
            "investment in zone property",
            "annual certification and compliance",
        ],
    },
    "foreign_trade_zone": {
        "display": "Foreign Trade Zone (FTZ) Benefits",
        "typical_value_range": (100_000, 15_000_000),
        "term_years": (1, 99),  # ongoing as long as FTZ status maintained
        "common_requirements": [
            "FTZ Board approval and activation",
            "U.S. Customs and Border Protection oversight",
            "eligible merchandise processing",
            "annual reporting and compliance",
            "security and access control requirements",
        ],
        "benefit_types": [
            "duty deferral",
            "duty elimination on re-exports",
            "inverted tariff relief",
            "state/local ad valorem tax relief",
        ],
    },
    "opportunity_zone": {
        "display": "Qualified Opportunity Zone (QOZ) Benefits",
        "typical_value_range": (0, 0),  # value is capital gains tax benefit, not a direct grant
        "term_years": (10, 10),
        "common_requirements": [
            "investment through Qualified Opportunity Fund (QOF)",
            "location within designated Opportunity Zone census tract",
            "substantial improvement test (investment >= adjusted basis within 30 months)",
            "90% asset test compliance",
            "annual Form 8996 filing",
        ],
        "benefit_types": [
            "capital gains tax deferral",
            "10% basis step-up at 5 years (for investments made before 12/31/2026)",
            "permanent exclusion of gains on QOZ investment held 10+ years",
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════
# 1031 EXCHANGE & CRE TAXATION
# ═══════════════════════════════════════════════════════════════════

EXCHANGE_1031 = {
    "identification_period_days": 45,
    "closing_deadline_days": 180,
    "rules": {
        "three_property_rule": "Identify up to 3 replacement properties regardless of value",
        "200_percent_rule": "Identify any number of properties if total FMV <= 200% of relinquished",
        "95_percent_rule": "Identify any number if you acquire 95% of identified value",
    },
    "boot_types": [
        "cash_boot",          # cash received at closing
        "mortgage_boot",      # debt relief exceeding replacement debt
        "non_like_kind",      # personal property, inventory, etc.
    ],
    "exchange_types": {
        "simultaneous": {
            "display": "Simultaneous Exchange",
            "description": "Both properties close on the same day",
        },
        "delayed": {
            "display": "Delayed (Starker) Exchange",
            "description": "Standard 45/180-day timeline with Qualified Intermediary",
        },
        "reverse": {
            "display": "Reverse Exchange",
            "description": "Acquire replacement before selling relinquished; EAT holds title",
            "holding_period_days": 180,
        },
        "improvement": {
            "display": "Improvement (Build-to-Suit) Exchange",
            "description": "Improvements made to replacement property during exchange period",
        },
    },
    "qi_requirements": [
        "Must use Qualified Intermediary (QI) — cannot touch proceeds",
        "QI holds funds in escrow during exchange period",
        "Related parties cannot serve as QI",
        "Written exchange agreement required before closing of relinquished property",
    ],
    "disqualifying_events": [
        "Actual or constructive receipt of sale proceeds",
        "Exchange of non-like-kind property (e.g. primary residence, inventory, stock)",
        "Failure to identify within 45 days",
        "Failure to close within 180 days",
        "Related party sale with subsequent disposition within 2 years",
    ],
}

CRE_TAXATION = {
    "depreciation": {
        "commercial_years": 39,        # straight-line, non-residential
        "residential_years": 27.5,     # straight-line, residential rental
        "land_improvement_years": 15,  # parking lots, fencing, landscaping
        "personal_property_years": 5,  # tenant improvements, appliances
        "bonus_depreciation": {
            "2024": 0.60,              # 60% bonus (phasing down)
            "2025": 0.40,              # 40% bonus
            "2026": 0.20,              # 20% bonus
            "2027": 0.00,              # no bonus (unless extended)
        },
    },
    "capital_gains": {
        "federal_long_term_rate": 0.20,          # top rate on LTCG
        "federal_short_term_rate": 0.37,         # ordinary income rate
        "net_investment_income_tax": 0.038,      # NIIT / Medicare surtax
        "depreciation_recapture_rate": 0.25,     # Section 1250 unrecaptured gain
    },
    "cost_segregation": {
        "description": "Engineering study to reclassify building components into shorter-life categories",
        "typical_acceleration_pct": (0.15, 0.40),  # 15-40% of basis reclassified
        "categories": [
            {"name": "5-year property", "examples": "carpeting, appliances, signage, special lighting"},
            {"name": "7-year property", "examples": "office furniture, security systems, telecom"},
            {"name": "15-year property", "examples": "parking lots, fencing, landscaping, sidewalks"},
            {"name": "39-year property", "examples": "structural walls, roof, HVAC, plumbing, electrical"},
        ],
    },
    "entity_structures": {
        "llc": "Pass-through; flexible allocation; limited liability",
        "lp": "GP/LP structure; LP passive loss limits; common in syndications",
        "dst": "Delaware Statutory Trust; passive 1031-eligible fractional ownership",
        "tic": "Tenants-in-Common; each owner holds undivided interest; 1031-eligible",
        "reit": "REIT election; 90% distribution requirement; corporate-level tax exemption",
    },
    "passive_activity_rules": {
        "real_estate_professional_hours": 750,     # hours/year to qualify as RE professional
        "material_participation_hours": 500,       # hours for material participation
        "passive_loss_offset": "Passive losses only offset passive income unless RE professional",
        "at_risk_rules": "Losses limited to amount at risk (basis + recourse debt)",
    },
}

# 1031 exchange scenario types for task generation
EXCHANGE_SCENARIOS = [
    {
        "type": "standard_delayed",
        "description": "Sell relinquished industrial property, identify 3 replacement properties within 45 days",
        "complexity": "medium",
    },
    {
        "type": "boot_calculation",
        "description": "Exchange with mortgage boot — replacement property has less debt",
        "complexity": "high",
    },
    {
        "type": "partial_exchange",
        "description": "Partial 1031 exchange — some cash boot received, partial deferral",
        "complexity": "high",
    },
    {
        "type": "reverse_exchange",
        "description": "Acquire replacement via EAT before selling relinquished property",
        "complexity": "high",
    },
    {
        "type": "dst_replacement",
        "description": "Sell active property, exchange into DST for passive income",
        "complexity": "medium",
    },
    {
        "type": "cost_seg_analysis",
        "description": "Cost segregation study on acquired property to accelerate depreciation",
        "complexity": "medium",
    },
    {
        "type": "gain_deferral_calc",
        "description": "Calculate deferred gain, realized gain, recognized gain, and new basis",
        "complexity": "high",
    },
    {
        "type": "depreciation_recapture",
        "description": "Section 1250 depreciation recapture on sale without 1031 exchange",
        "complexity": "high",
    },
]

# Concepts that should appear in grant / incentive-related training tasks
INCENTIVE_CONCEPTS = [
    "TIF/TID financing structure", "property tax abatement negotiation",
    "job creation grant compliance", "infrastructure grant application",
    "enterprise zone eligibility", "foreign trade zone activation",
    "opportunity zone fund structuring", "clawback provision analysis",
    "incentive stacking strategies", "public-private partnership terms",
    "PILOT agreements", "sales tax exemption for construction materials",
    "utility rate negotiation for data centers", "state incentive comparison",
]

# ═══════════════════════════════════════════════════════════════════
# TENANT NAME COMPONENTS (for procedural generation)
# ═══════════════════════════════════════════════════════════════════

TENANT_PREFIXES = [
    "Pacific", "Atlantic", "Midwest", "Southern", "Western", "Eastern",
    "National", "American", "Premier", "Delta", "Apex", "Summit",
    "Eagle", "Falcon", "Horizon", "Pinnacle", "Compass", "Atlas",
    "Sterling", "Vanguard", "Global", "Metro", "Central", "Allied",
    "Pioneer", "Liberty", "Patriot", "Continental", "Mountain", "Coastal",
]

TENANT_SUFFIXES = [
    "Logistics", "Distribution", "Supply Chain", "Fulfillment", "Warehouse",
    "Manufacturing", "Industries", "Enterprises", "Group", "Solutions",
    "Services", "Systems", "Partners", "Corp", "Holdings", "Materials",
    "Products", "Packaging", "Assembly", "Transport", "Freight", "Cold Chain",
    "Data Systems", "Cloud Services", "Digital Infrastructure",
]

# ═══════════════════════════════════════════════════════════════════
# PROPERTY NAME COMPONENTS
# ═══════════════════════════════════════════════════════════════════

PROPERTY_PREFIXES = [
    "Commerce", "Industrial", "Logistics", "Distribution", "Gateway",
    "Crossroads", "Interchange", "Airport", "Metro", "Parkway",
    "Enterprise", "Business", "Technology", "Innovation", "Heritage",
    "Centennial", "Summit", "Valley", "Lakeside", "Riverside",
    "Digital", "Hyperscale", "Data",
]

PROPERTY_SUFFIXES = [
    "Center", "Park", "Plaza", "Hub", "Campus", "Commons",
    "Point", "Crossing", "Junction", "Yards", "Landing", "Terminal",
]

# ═══════════════════════════════════════════════════════════════════
# TASK TYPE DISTRIBUTION — matches the 1M target
# ═══════════════════════════════════════════════════════════════════

TASK_DISTRIBUTION = {
    "underwriting_calc": 0.27,
    "rent_roll_extraction": 0.060,
    "t12_normalization": 0.060,
    "lease_abstract_extraction": 0.060,
    "ic_memo": 0.18,
    "lease_reasoning": 0.13,
    "market_comp_narrative": 0.09,
    "risk_triage": 0.05,
    "exchange_1031": 0.04,
    "tax_analysis": 0.04,
    "loi_deliverable": 0.005,
    "structured_agent_output": 0.005,
}

# Difficulty mix
DIFFICULTY_WEIGHTS = {"low": 0.20, "medium": 0.55, "high": 0.25}

# ═══════════════════════════════════════════════════════════════════
# ROUNDING RULES
# ═══════════════════════════════════════════════════════════════════

ROUND_MONEY = 0        # round to nearest $1
ROUND_RATIO = 2        # round ratios to 2 decimal places
ROUND_PSF = 2          # round per-SF to 2 decimal places
ROUND_PCT = 3          # round percentages to 3 decimal places (e.g. 0.065)

# ═══════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════

SWARMCRE_SYSTEM_PROMPT = (
    "You are SwarmCRE, an expert commercial real estate AI assistant "
    "specializing in last-mile industrial, infill logistics, data centers, "
    "and adjacent CRE asset classes. "
    "You have the knowledge of a senior broker with 30+ years of experience and $8B+ in closed transactions. "
    "You understand data center power/cooling economics, economic development incentives "
    "(TIF/TID, tax abatements, opportunity zones, enterprise zones, FTZ), "
    "1031 exchanges (delayed, reverse, improvement, DST/TIC), CRE taxation "
    "(depreciation, cost segregation, capital gains, Section 1250 recapture), "
    "and public-private partnership structures. "
    "Be precise with numbers. Show your work on calculations. "
    "Use structured outputs when asked. Flag risks clearly. "
    "Assumptions must be explicit. Recommendations must be decision-ready."
)

TASK_SYSTEM_PROMPTS = {
    "underwriting_calc": (
        "You are SwarmCRE. Perform the requested underwriting calculation. "
        "Show all steps. Round money to the nearest dollar, ratios to 2 decimal places. "
        "For data center deals, account for power costs ($/kWh), PUE, and per-MW economics. "
        "When economic incentives apply (TIF, abatements, OZ), model their impact on returns. "
        "For 1031 exchange calculations, track adjusted basis, deferred gain, boot, and new basis. "
        "For tax analysis, apply correct depreciation schedules (39-year commercial, cost segregation), "
        "capital gains rates (20% LTCG + 3.8% NIIT + 25% Section 1250 recapture), and entity-level impacts. "
        "State assumptions explicitly."
    ),
    "rent_roll_extraction": (
        "You are SwarmCRE. Extract the rent roll from the provided information into structured JSON. "
        "Include all tenants, square footages, rents, lease dates, and escalation terms. "
        "Output valid JSON matching the RentRollJSON schema."
    ),
    "t12_normalization": (
        "You are SwarmCRE. Normalize the trailing 12-month operating statement. "
        "Identify non-recurring items, adjust for market rents if below-market, "
        "and present the stabilized T-12 in structured JSON matching the T12JSON schema."
    ),
    "lease_abstract_extraction": (
        "You are SwarmCRE. Extract key lease terms from the provided information into structured JSON. "
        "Include base rent, escalations, options, guarantees, restrictions, and key provisions. "
        "Output valid JSON matching the LeaseAbstractJSON schema."
    ),
    "ic_memo": (
        "You are SwarmCRE. Prepare an Investment Committee memo for the described deal. "
        "Include: Executive Summary, Property Overview, Market Analysis, Tenant Analysis, "
        "Financial Summary, Risk Factors, and Recommendation (Proceed/Caution/Kill). "
        "For data centers, include power/cooling infrastructure and technology risk sections. "
        "Where economic incentives are present, include an Incentives & Public Funding section "
        "detailing TIF/abatement/grant terms, clawback risk, and net impact on returns. "
        "Be concise but thorough. Assumptions explicit. Decision-ready."
    ),
    "lease_reasoning": (
        "You are SwarmCRE. Analyze the lease scenario described and provide expert reasoning. "
        "Consider tenant leverage, market conditions, comparable terms, and negotiation strategy. "
        "Be specific about dollar amounts and percentages."
    ),
    "market_comp_narrative": (
        "You are SwarmCRE. Provide market analysis and comparable transaction analysis. "
        "Include comp selection rationale, adjustments, and a submarket narrative. "
        "For data centers, compare on a per-MW and per-kW basis in addition to per-SF. "
        "Note availability of economic incentives in the submarket when relevant. "
        "Use structured JSON for comp data, narrative for analysis."
    ),
    "risk_triage": (
        "You are SwarmCRE. Perform a risk triage on the described deal. "
        "Identify deal killers, risk factors by category, severity, and mitigation strategies. "
        "For data centers, evaluate power supply reliability, cooling redundancy, obsolescence risk, "
        "and tenant concentration. For incentive-dependent deals, assess clawback exposure, "
        "compliance risk, and political/legislative risk to ongoing benefits. "
        "Provide a clear Proceed/Caution/Kill recommendation with reasoning."
    ),
    "loi_deliverable": (
        "You are SwarmCRE. Draft an LOI / term sheet for the described transaction. "
        "Include purchase price, earnest money, DD period, closing timeline, contingencies, "
        "and special conditions. For data centers, include power/connectivity contingencies. "
        "Where incentives are part of the deal, include incentive assignment/transfer provisions "
        "and compliance continuation requirements. Output structured JSON matching the LOITermSheet schema."
    ),
    "exchange_1031": (
        "You are SwarmCRE. Analyze the 1031 exchange scenario described. "
        "Calculate gain, boot, recognized gain, deferred gain, and new basis. "
        "Track the 45-day identification and 180-day closing deadlines. "
        "Apply the 3-property rule, 200% rule, or 95% rule as applicable. "
        "For reverse exchanges, note EAT holding requirements. "
        "For DST replacements, analyze passive income and fractional ownership implications. "
        "Show all calculations. Round money to nearest dollar."
    ),
    "tax_analysis": (
        "You are SwarmCRE. Perform the requested CRE tax analysis. "
        "Apply correct depreciation schedules: 39-year straight-line for commercial, "
        "27.5-year for residential, 15-year for land improvements, 5/7-year for personal property. "
        "For cost segregation, show the reclassification breakdown and bonus depreciation impact. "
        "For sale analysis, calculate Section 1250 depreciation recapture (25%), "
        "long-term capital gains (20%), and Net Investment Income Tax (3.8%). "
        "Consider entity structure implications (LLC, LP, DST, REIT). "
        "Show all steps. State assumptions explicitly."
    ),
    "structured_agent_output": (
        "You are SwarmCRE operating in agent mode. Process the request and return "
        "a structured JSON response with task_type, output, confidence score, "
        "reasoning chain, and follow-up actions. Suitable for downstream agent consumption."
    ),
}
