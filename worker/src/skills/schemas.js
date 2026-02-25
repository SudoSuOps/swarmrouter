/**
 * Skill Validation Schemas
 * ========================
 * One validate function per skill. Returns { valid, errors[] }.
 * Checks: required fields, types, enums, numeric ranges.
 */

// ── Helpers ─────────────────────────────────────────────

function checkType(obj, field, type) {
  if (obj[field] === null || obj[field] === undefined) return null;
  if (type === 'array') return Array.isArray(obj[field]) ? null : `${field} must be array`;
  if (type === 'object') return (typeof obj[field] === 'object' && !Array.isArray(obj[field])) ? null : `${field} must be object`;
  return typeof obj[field] === type ? null : `${field} must be ${type}`;
}

function checkEnum(obj, field, values) {
  if (!obj[field]) return null;
  return values.includes(obj[field]) ? null : `${field} must be one of: ${values.join(', ')}`;
}

function checkRange(obj, field, min, max) {
  if (obj[field] === null || obj[field] === undefined) return null;
  if (typeof obj[field] !== 'number') return `${field} must be number`;
  if (obj[field] < min || obj[field] > max) return `${field} must be ${min}-${max}`;
  return null;
}

function checkRequired(obj, fields) {
  return fields
    .filter(f => obj[f] === undefined)
    .map(f => `missing required field: ${f}`);
}

function collect(...checks) {
  return checks.filter(Boolean);
}

// ── Broker Senior ───────────────────────────────────────

export function validateBrokerSenior(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'deal_verdict']),
    checkEnum(output, 'deal_verdict', ['pursue', 'pass', 'watch']),
    checkType(output, 'pricing', 'object'),
    checkType(output, 'risk_assessment', 'object'),
    checkType(output, 'deal_structure', 'object'),
    checkType(output, 'market_context', 'object'),
    checkType(output, 'ic_notes', 'string'),
  ].filter(Boolean);

  if (output.pricing) {
    errors.push(...collect(
      checkType(output.pricing, 'value_range', 'array'),
      checkRange(output.pricing, 'cap_rate', 0, 0.30),
      checkType(output.pricing, 'price_psf', 'number'),
      checkType(output.pricing, 'basis', 'string'),
    ));
  }

  if (output.risk_assessment) {
    errors.push(...collect(
      checkRange(output.risk_assessment, 'score', 0, 100),
      checkType(output.risk_assessment, 'flags', 'array'),
      checkType(output.risk_assessment, 'mitigants', 'array'),
    ));
  }

  if (output.deal_structure) {
    errors.push(...collect(
      checkType(output.deal_structure, 'hold_period', 'number'),
      checkRange(output.deal_structure, 'target_irr', 0, 1),
      checkRange(output.deal_structure, 'exit_cap', 0, 0.30),
    ));
  }

  return { valid: errors.length === 0, errors };
}

// ── Broker Junior ───────────────────────────────────────

export function validateBrokerJunior(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'task_type']),
    checkEnum(output, 'task_type', ['tour_prep', 'comp_pull', 'stacking_plan', 'prospect_list']),
    checkType(output, 'deliverable', 'object'),
    checkType(output, 'next_steps', 'array'),
    checkType(output, 'time_estimate', 'string'),
  ].filter(Boolean);

  return { valid: errors.length === 0, errors };
}

// ── Intelligence Query ──────────────────────────────────

export function validateIntelligenceQuery(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'query_interpreted']),
    checkType(output, 'query_interpreted', 'string'),
    checkType(output, 'objects_found', 'number'),
    checkType(output, 'results', 'array'),
    checkType(output, 'reasoning', 'string'),
    checkType(output, 'suggested_queries', 'array'),
  ].filter(Boolean);

  if (Array.isArray(output.results)) {
    for (let i = 0; i < Math.min(output.results.length, 5); i++) {
      const r = output.results[i];
      if (!r.object_id) errors.push(`results[${i}] missing object_id`);
      if (r.relevance !== undefined && (typeof r.relevance !== 'number' || r.relevance < 0 || r.relevance > 1)) {
        errors.push(`results[${i}].relevance must be 0-1`);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

// ── Bookmaker OM ────────────────────────────────────────

export function validateBookmaker(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'reason_layer', 'ui_layer']),
    checkType(output, 'reason_layer', 'object'),
    checkType(output, 'ui_layer', 'object'),
  ].filter(Boolean);

  if (output.reason_layer) {
    errors.push(...collect(
      checkType(output.reason_layer, 'investment_thesis', 'string'),
      checkType(output.reason_layer, 'financial_summary', 'object'),
      checkType(output.reason_layer, 'market_analysis', 'string'),
      checkType(output.reason_layer, 'risk_factors', 'array'),
      checkType(output.reason_layer, 'comparable_sales', 'array'),
    ));

    if (output.reason_layer.financial_summary) {
      const fs = output.reason_layer.financial_summary;
      errors.push(...collect(
        checkRange(fs, 'cap_rate', 0, 0.30),
      ));
    }
  }

  if (output.ui_layer) {
    errors.push(...collect(
      checkType(output.ui_layer, 'om_title', 'string'),
      checkType(output.ui_layer, 'executive_summary', 'string'),
      checkType(output.ui_layer, 'sections', 'array'),
      checkEnum(output.ui_layer, 'format', ['markdown', 'html', 'text']),
    ));

    if (Array.isArray(output.ui_layer.sections)) {
      for (let i = 0; i < output.ui_layer.sections.length; i++) {
        const s = output.ui_layer.sections[i];
        if (!s.heading) errors.push(`ui_layer.sections[${i}] missing heading`);
        if (!s.content) errors.push(`ui_layer.sections[${i}] missing content`);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

// ── Deal Tracker ────────────────────────────────────────

const DEAL_STAGES = ['prospect', 'loi', 'due_diligence', 'under_contract', 'closed', 'dead'];
const TRACKER_ACTIONS = ['create', 'update', 'query', 'alert'];
const MILESTONE_STATUSES = ['pending', 'done', 'overdue'];

export function validateDealTracker(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'action']),
    checkEnum(output, 'action', TRACKER_ACTIONS),
    checkType(output, 'deal', 'object'),
  ].filter(Boolean);

  if (output.deal) {
    errors.push(...collect(
      checkType(output.deal, 'deal_id', 'string'),
      checkType(output.deal, 'property', 'string'),
      checkEnum(output.deal, 'stage', DEAL_STAGES),
      checkType(output.deal, 'milestones', 'array'),
      checkType(output.deal, 'key_dates', 'object'),
      checkType(output.deal, 'next_action', 'string'),
      checkType(output.deal, 'days_in_stage', 'number'),
    ));

    if (Array.isArray(output.deal.milestones)) {
      for (let i = 0; i < output.deal.milestones.length; i++) {
        const m = output.deal.milestones[i];
        if (!m.name) errors.push(`milestones[${i}] missing name`);
        if (m.status && !MILESTONE_STATUSES.includes(m.status)) {
          errors.push(`milestones[${i}].status must be: ${MILESTONE_STATUSES.join(', ')}`);
        }
      }
    }
  }

  if (output.pipeline_summary) {
    errors.push(...collect(
      checkType(output.pipeline_summary, 'active_deals', 'number'),
      checkType(output.pipeline_summary, 'total_value', 'number'),
      checkType(output.pipeline_summary, 'overdue_items', 'array'),
    ));
  }

  return { valid: errors.length === 0, errors };
}

// ── Developer ───────────────────────────────────────────

const DEV_TASK_TYPES = ['api_integration', 'code_gen', 'skill_compose', 'debug', 'deploy'];

export function validateDeveloper(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'task_type']),
    checkEnum(output, 'task_type', DEV_TASK_TYPES),
    checkType(output, 'deliverable', 'object'),
    checkType(output, 'notes', 'string'),
  ].filter(Boolean);

  if (output.deliverable) {
    errors.push(...collect(
      checkType(output.deliverable, 'code', 'string'),
      checkType(output.deliverable, 'language', 'string'),
      checkType(output.deliverable, 'files_affected', 'array'),
      checkType(output.deliverable, 'dependencies', 'array'),
    ));
  }

  if (output.validation) {
    errors.push(...collect(
      checkType(output.validation, 'syntax_valid', 'boolean'),
      checkType(output.validation, 'tests_suggested', 'array'),
    ));
  }

  return { valid: errors.length === 0, errors };
}

// ── Signal Scraper ──────────────────────────────────────

const SIGNAL_EVENT_TYPES = ['just_listed', 'just_sold', 'under_contract', 'price_reduction', 'new_development', 'lease_signed', 'top_producer'];

export function validateSignalScraper(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'events_detected']),
    checkType(output, 'events_detected', 'number'),
    checkType(output, 'events', 'array'),
    checkType(output, 'raw_signals_processed', 'number'),
    checkType(output, 'signal_to_event_ratio', 'number'),
    checkType(output, 'sources_scanned', 'array'),
  ].filter(Boolean);

  if (Array.isArray(output.events)) {
    for (let i = 0; i < output.events.length; i++) {
      const e = output.events[i];
      errors.push(...collect(
        checkEnum(e, 'event_type', SIGNAL_EVENT_TYPES),
        checkRange(e, 'confidence', 0, 1),
        checkType(e, 'property', 'string'),
        checkType(e, 'source', 'string'),
      ));
    }
  }

  if (output.signal_to_event_ratio !== undefined) {
    errors.push(...collect(checkRange(output, 'signal_to_event_ratio', 0, 1)));
  }

  return { valid: errors.length === 0, errors };
}

// ── Investor ────────────────────────────────────────────

const INVESTOR_MODES = ['match', 'profile', 'deploy'];
const BUYER_TYPES = ['institutional', 'private_equity', 'reit', 'exchange_1031', 'private_capital', 'foreign_capital', 'developer'];
const POOL_SIZES = ['deep', 'moderate', 'thin'];
const TENSION_LEVELS = ['high', 'moderate', 'low'];

export function validateInvestor(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'mode']),
    checkEnum(output, 'mode', INVESTOR_MODES),
    checkType(output, 'matches', 'array'),
    checkType(output, 'recommendation', 'string'),
  ].filter(Boolean);

  if (Array.isArray(output.matches)) {
    for (let i = 0; i < output.matches.length; i++) {
      const m = output.matches[i];
      errors.push(...collect(
        checkEnum(m, 'buyer_type', BUYER_TYPES),
        checkRange(m, 'probability', 0, 1),
        checkType(m, 'rationale', 'string'),
        checkType(m, 'example_buyers', 'array'),
      ));
      if (m.pricing_expectation) {
        errors.push(...collect(
          checkRange(m.pricing_expectation, 'cap_rate', 0, 0.30),
        ));
      }
    }
  }

  if (output.market_depth) {
    errors.push(...collect(
      checkEnum(output.market_depth, 'buyer_pool_size', POOL_SIZES),
      checkType(output.market_depth, 'expected_offers', 'number'),
      checkEnum(output.market_depth, 'competitive_tension', TENSION_LEVELS),
    ));
  }

  return { valid: errors.length === 0, errors };
}

// ── 1031 Exchange ───────────────────────────────────────

const EXCHANGE_MODES = ['qualify', 'timeline', 'boot_calc', 'match', 'reverse'];
const URGENCY_LEVELS = ['green', 'yellow', 'red'];
const ID_RULES = ['3_property', '200_pct', '95_pct'];

export function validateExchange1031(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'mode']),
    checkEnum(output, 'mode', EXCHANGE_MODES),
    checkType(output, 'recommendation', 'string'),
  ].filter(Boolean);

  if (output.qualification) {
    errors.push(...collect(
      checkType(output.qualification, 'qualifies', 'boolean'),
      checkType(output.qualification, 'like_kind', 'boolean'),
      checkType(output.qualification, 'held_for_investment', 'boolean'),
      checkType(output.qualification, 'issues', 'array'),
    ));
  }

  if (output.relinquished) {
    errors.push(...collect(
      checkType(output.relinquished, 'property', 'string'),
    ));
  }

  if (output.timeline) {
    errors.push(...collect(
      checkEnum(output.timeline, 'urgency', URGENCY_LEVELS),
    ));
  }

  if (output.boot_analysis) {
    errors.push(...collect(
      checkType(output.boot_analysis, 'cash_boot', 'number'),
      checkType(output.boot_analysis, 'mortgage_boot', 'number'),
      checkType(output.boot_analysis, 'total_boot', 'number'),
      checkType(output.boot_analysis, 'estimated_tax_liability', 'number'),
      checkType(output.boot_analysis, 'tax_savings_if_exchanged', 'number'),
    ));
  }

  if (output.replacement_targets) {
    errors.push(...collect(
      checkEnum(output.replacement_targets, 'rule_used', ID_RULES),
      checkType(output.replacement_targets, 'targets', 'array'),
    ));
  }

  if (output.risks) {
    errors.push(...collect(checkType(output, 'risks', 'array')));
  }

  return { valid: errors.length === 0, errors };
}

// ── Market Report ───────────────────────────────────────

export function validateMarketReport(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'submarket', 'snapshot']),
    checkType(output, 'submarket', 'string'),
    checkType(output, 'metro', 'string'),
    checkType(output, 'report_period', 'string'),
    checkType(output, 'snapshot', 'object'),
    checkType(output, 'trends', 'object'),
    checkType(output, 'notable_transactions', 'array'),
    checkType(output, 'forecast', 'object'),
    checkType(output, 'sources', 'array'),
  ].filter(Boolean);

  if (output.snapshot) {
    errors.push(...collect(
      checkRange(output.snapshot, 'vacancy_rate', 0, 1),
      checkRange(output.snapshot, 'avg_cap_rate', 0, 0.30),
      checkType(output.snapshot, 'avg_asking_rent_psf', 'number'),
      checkType(output.snapshot, 'rent_growth_yoy', 'number'),
      checkType(output.snapshot, 'inventory_sf', 'number'),
      checkType(output.snapshot, 'under_construction_sf', 'number'),
      checkType(output.snapshot, 'net_absorption_sf', 'number'),
      checkType(output.snapshot, 'deliveries_sf', 'number'),
    ));
  }

  return { valid: errors.length === 0, errors };
}

// ── Lead Scorer ─────────────────────────────────────────

const LEAD_TIERS = ['hot', 'warm', 'cold', 'dead'];
const FOLLOW_UP_PRIORITIES = ['immediate', 'high', 'medium', 'low', 'none'];

export function validateLeadScorer(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'score', 'tier', 'signals']),
    checkRange(output, 'score', 0, 100),
    checkEnum(output, 'tier', LEAD_TIERS),
    checkType(output, 'signals', 'object'),
    checkType(output, 'recommended_action', 'object'),
  ].filter(Boolean);

  if (output.signals && typeof output.signals === 'object') {
    for (const [key, val] of Object.entries(output.signals)) {
      if (typeof val !== 'object' || val === null) {
        errors.push(`signals.${key} must be object with score and detail`);
      } else {
        if (typeof val.score !== 'number' || val.score < 0 || val.score > 100) {
          errors.push(`signals.${key}.score must be number 0-100`);
        }
        if (val.detail !== undefined && typeof val.detail !== 'string') {
          errors.push(`signals.${key}.detail must be string`);
        }
      }
    }
  }

  if (output.recommended_action) {
    errors.push(...collect(
      checkEnum(output.recommended_action, 'priority', FOLLOW_UP_PRIORITIES),
      checkType(output.recommended_action, 'action', 'string'),
    ));
  }

  return { valid: errors.length === 0, errors };
}

// ── Email Composer ──────────────────────────────────────

const EMAIL_TYPES = ['cold_outreach', 'follow_up', 'listing_update', 'offer_letter', 'introduction', 'closing_notice'];
const EMAIL_TONES = ['formal', 'casual', 'urgent', 'friendly', 'executive', 'professional'];

export function validateEmailComposer(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'email_type', 'subject_line', 'body', 'tone']),
    checkEnum(output, 'email_type', EMAIL_TYPES),
    checkType(output, 'subject_line', 'string'),
    checkType(output, 'body', 'string'),
    checkEnum(output, 'tone', EMAIL_TONES),
    checkType(output, 'follow_up_sequence', 'array'),
  ].filter(Boolean);

  return { valid: errors.length === 0, errors };
}

// ── Comp Analyzer ───────────────────────────────────────

export function validateCompAnalyzer(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'comparables', 'analysis']),
    checkType(output, 'comparables', 'array'),
    checkType(output, 'analysis', 'object'),
  ].filter(Boolean);

  if (output.analysis) {
    errors.push(...collect(
      checkType(output.analysis, 'indicated_value_range', 'array'),
      checkType(output.analysis, 'weighted_avg_psf', 'number'),
      checkType(output.analysis, 'weighted_avg_value', 'number'),
      checkRange(output.analysis, 'implied_cap_rate', 0, 0.30),
    ));
  }

  return { valid: errors.length === 0, errors };
}

// ── Rent Roll Analyzer ──────────────────────────────────

export function validateRentRollAnalyzer(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'tenants', 'summary']),
    checkType(output, 'tenants', 'array'),
    checkType(output, 'summary', 'object'),
    checkType(output, 'vacancy', 'object'),
    checkType(output, 'rollover_analysis', 'object'),
    checkType(output, 'risk_flags', 'array'),
    checkType(output, 'recommendations', 'array'),
  ].filter(Boolean);

  if (output.summary) {
    errors.push(...collect(
      checkType(output.summary, 'walt_years', 'number'),
      checkRange(output.summary, 'occupancy_rate', 0, 1),
      checkType(output.summary, 'total_annual_rent', 'number'),
      checkType(output.summary, 'weighted_avg_rent_psf', 'number'),
    ));
  }

  return { valid: errors.length === 0, errors };
}

// ── Debt Analyzer ───────────────────────────────────────

export function validateDebtAnalyzer(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'sizing', 'lender_matches']),
    checkType(output, 'sizing', 'object'),
    checkType(output, 'lender_matches', 'array'),
    checkType(output, 'rate_sensitivity', 'array'),
    checkType(output, 'cash_flow_impact', 'object'),
  ].filter(Boolean);

  if (output.sizing) {
    errors.push(...collect(
      checkRange(output.sizing, 'max_ltv', 0, 1),
      checkType(output.sizing, 'max_loan_amount', 'number'),
      checkType(output.sizing, 'recommended_loan', 'number'),
      checkType(output.sizing, 'equity_required', 'number'),
    ));
    if (output.sizing.dscr !== undefined) {
      if (typeof output.sizing.dscr !== 'number' || output.sizing.dscr <= 0) {
        errors.push('sizing.dscr must be > 0');
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

// ── Tax Assessor ────────────────────────────────────────

export function validateTaxAssessor(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'current_assessment', 'appeal_recommendation']),
    checkType(output, 'current_assessment', 'object'),
    checkType(output, 'appeal_recommendation', 'object'),
    checkType(output, 'market_analysis', 'object'),
    checkType(output, 'supporting_evidence', 'object'),
    checkType(output, 'process', 'object'),
  ].filter(Boolean);

  if (output.appeal_recommendation) {
    errors.push(...collect(
      checkType(output.appeal_recommendation, 'should_appeal', 'boolean'),
      checkRange(output.appeal_recommendation, 'confidence', 0, 1),
      checkType(output.appeal_recommendation, 'potential_tax_savings', 'number'),
    ));
  }

  return { valid: errors.length === 0, errors };
}

// ── Site Selector ───────────────────────────────────────

export function validateSiteSelector(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'recommended_markets', 'recommendation']),
    checkType(output, 'recommended_markets', 'array'),
    checkType(output, 'recommendation', 'string'),
    checkType(output, 'requirements', 'object'),
  ].filter(Boolean);

  if (Array.isArray(output.recommended_markets)) {
    for (let i = 0; i < output.recommended_markets.length; i++) {
      const m = output.recommended_markets[i];
      errors.push(...collect(
        checkRange(m, 'overall_score', 0, 100),
        checkType(m, 'market', 'string'),
        checkType(m, 'highlights', 'array'),
        checkType(m, 'risks', 'array'),
      ));
    }
  }

  return { valid: errors.length === 0, errors };
}

// ── Portfolio Optimizer ─────────────────────────────────

export function validatePortfolioOptimizer(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'concentration_analysis', 'risk_flags', 'recommendations']),
    checkType(output, 'concentration_analysis', 'object'),
    checkType(output, 'risk_flags', 'array'),
    checkType(output, 'recommendations', 'array'),
    checkType(output, 'holdings', 'array'),
    checkType(output, 'portfolio', 'object'),
    checkType(output, 'optimized_metrics', 'object'),
  ].filter(Boolean);

  if (output.concentration_analysis) {
    errors.push(...collect(
      checkRange(output.concentration_analysis, 'concentration_score', 0, 100),
      checkType(output.concentration_analysis, 'by_market', 'object'),
      checkType(output.concentration_analysis, 'by_tenant', 'object'),
    ));
  }

  return { valid: errors.length === 0, errors };
}

// ── News Digest ─────────────────────────────────────────

const MARKET_SENTIMENTS = ['bullish', 'bearish', 'neutral', 'mixed'];

export function validateNewsDigest(output) {
  const errors = [
    ...checkRequired(output, ['skill', 'top_stories', 'sentiment']),
    checkType(output, 'top_stories', 'array'),
    checkEnum(output, 'sentiment', MARKET_SENTIMENTS),
    checkType(output, 'sentiment_score', 'number'),
    checkType(output, 'market_pulse', 'object'),
    checkType(output, 'action_items', 'array'),
    checkType(output, 'sources', 'array'),
  ].filter(Boolean);

  if (output.sentiment_score !== undefined) {
    errors.push(...collect(
      checkRange(output, 'sentiment_score', -1, 1),
    ));
  }

  if (Array.isArray(output.top_stories)) {
    for (let i = 0; i < output.top_stories.length; i++) {
      const s = output.top_stories[i];
      errors.push(...collect(
        checkType(s, 'headline', 'string'),
        checkType(s, 'source', 'string'),
        checkEnum(s, 'impact', MARKET_SENTIMENTS),
      ));
    }
  }

  return { valid: errors.length === 0, errors };
}

// ── Schema Registry ─────────────────────────────────────

export const VALIDATORS = {
  broker_senior: validateBrokerSenior,
  broker_junior: validateBrokerJunior,
  intelligence_query: validateIntelligenceQuery,
  bookmaker: validateBookmaker,
  deal_tracker: validateDealTracker,
  developer: validateDeveloper,
  signal_scraper: validateSignalScraper,
  investor: validateInvestor,
  exchange_1031: validateExchange1031,
  market_report: validateMarketReport,
  lead_scorer: validateLeadScorer,
  email_composer: validateEmailComposer,
  comp_analyzer: validateCompAnalyzer,
  rent_roll_analyzer: validateRentRollAnalyzer,
  debt_analyzer: validateDebtAnalyzer,
  tax_assessor: validateTaxAssessor,
  site_selector: validateSiteSelector,
  portfolio_optimizer: validatePortfolioOptimizer,
  news_digest: validateNewsDigest,
};
