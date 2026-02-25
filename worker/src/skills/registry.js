/**
 * Skill Registry — Core Execution Engine
 * ========================================
 * The brain. Resolves skill name → system prompt → AI → validate → R2.
 *
 * Functions:
 *   executeSkill(name, input, env)  — Full execution: AI → validate → R2
 *   listSkills()                     — All skills with specs
 *   getSkillSpec(name)               — Full spec for one skill
 *   testSkill(name, input)           — Mock + validate (no AI)
 *   evalSkill(name, input, env)      — Real execution + metrics
 */

import { BROKER_SENIOR } from './broker_senior.js';
import { BROKER_JUNIOR } from './broker_junior.js';
import { INTELLIGENCE_QUERY } from './intelligence_query.js';
import { BOOKMAKER } from './bookmaker.js';
import { DEAL_TRACKER } from './deal_tracker.js';
import { DEVELOPER } from './developer.js';
import { SIGNAL_SCRAPER } from './signal_scraper.js';
import { INVESTOR } from './investor.js';
import { EXCHANGE_1031 } from './exchange_1031.js';
import { MARKET_REPORT } from './market_report.js';
import { LEAD_SCORER } from './lead_scorer.js';
import { EMAIL_COMPOSER } from './email_composer.js';
import { COMP_ANALYZER } from './comp_analyzer.js';
import { RENT_ROLL_ANALYZER } from './rent_roll_analyzer.js';
import { DEBT_ANALYZER } from './debt_analyzer.js';
import { TAX_ASSESSOR } from './tax_assessor.js';
import { SITE_SELECTOR } from './site_selector.js';
import { PORTFOLIO_OPTIMIZER } from './portfolio_optimizer.js';
import { NEWS_DIGEST } from './news_digest.js';
import { VALIDATORS } from './schemas.js';
import { MOCKS } from './mocks.js';

// ── Edge Model Config ───────────────────────────────────

const EDGE_MODEL = '@cf/qwen/qwen3-30b-a3b-fp8';
const FALLBACK_MODEL = '@cf/meta/llama-3.2-3b-instruct';
const EDGE_TEMP = 0.2;  // Slightly higher than cook — skills need some creativity
const MAX_TOKENS = 4096;

// ── Skill Registry ──────────────────────────────────────

export const SKILL_REGISTRY = {
  broker_senior: BROKER_SENIOR,
  broker_junior: BROKER_JUNIOR,
  intelligence_query: INTELLIGENCE_QUERY,
  bookmaker: BOOKMAKER,
  deal_tracker: DEAL_TRACKER,
  developer: DEVELOPER,
  signal_scraper: SIGNAL_SCRAPER,
  investor: INVESTOR,
  exchange_1031: EXCHANGE_1031,
  market_report: MARKET_REPORT,
  lead_scorer: LEAD_SCORER,
  email_composer: EMAIL_COMPOSER,
  comp_analyzer: COMP_ANALYZER,
  rent_roll_analyzer: RENT_ROLL_ANALYZER,
  debt_analyzer: DEBT_ANALYZER,
  tax_assessor: TAX_ASSESSOR,
  site_selector: SITE_SELECTOR,
  portfolio_optimizer: PORTFOLIO_OPTIMIZER,
  news_digest: NEWS_DIGEST,
};

// ── Execute Skill ───────────────────────────────────────

export async function executeSkill(name, input, env) {
  const skill = SKILL_REGISTRY[name];
  if (!skill) return { success: false, error: `Unknown skill: ${name}`, available: Object.keys(SKILL_REGISTRY) };

  const validate = VALIDATORS[name];
  const startTime = Date.now();

  // Build prompt
  const userContent = typeof input === 'string' ? input
    : `${input.context || 'Analyze this data'}\n\nDATA:\n${JSON.stringify(input.data || input, null, 2).substring(0, 6000)}`;

  // Call edge AI
  let result;
  try {
    result = await env.AI.run(EDGE_MODEL, {
      messages: [
        { role: 'system', content: skill.systemPrompt },
        { role: 'user', content: userContent },
      ],
      max_tokens: MAX_TOKENS,
      temperature: EDGE_TEMP,
    });
  } catch (primaryErr) {
    try {
      result = await env.AI.run(FALLBACK_MODEL, {
        messages: [
          { role: 'system', content: skill.systemPrompt },
          { role: 'user', content: userContent },
        ],
        max_tokens: MAX_TOKENS,
        temperature: EDGE_TEMP,
      });
    } catch (fallbackErr) {
      return { success: false, error: `AI error: ${primaryErr.message}`, fallback_error: fallbackErr.message };
    }
  }

  const latencyMs = Date.now() - startTime;

  // Extract JSON from AI response
  let output;
  try {
    output = extractJSON(result);
  } catch (e) {
    const rawContent = extractRawContent(result);
    return {
      success: false,
      error: 'Model returned non-JSON',
      raw_response: rawContent.substring(0, 1000),
      latency_ms: latencyMs,
    };
  }

  // Ensure skill field
  if (!output.skill) output.skill = name;

  // Validate
  const validation = validate ? validate(output) : { valid: true, errors: [] };

  // Store in R2
  const objectId = `skill_${name}_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`;
  const stored = {
    object_id: objectId,
    object_type: 'skill_output',
    skill: name,
    output,
    validation,
    metadata: {
      model: EDGE_MODEL,
      latency_ms: latencyMs,
      input_chars: userContent.length,
      output_chars: JSON.stringify(output).length,
      token_estimate: Math.ceil(userContent.length / 4) + Math.ceil(JSON.stringify(output).length / 4),
      created_at: new Date().toISOString(),
    },
  };

  try {
    await env.INTELLIGENCE.put(`pio/skills/${name}/${objectId}`, JSON.stringify(stored), {
      customMetadata: {
        skill: name,
        valid: String(validation.valid),
        created_at: stored.metadata.created_at,
      },
    });
  } catch (storeErr) {
    // Don't fail the skill if storage fails — return the output anyway
  }

  return {
    success: true,
    skill: name,
    output,
    validation,
    metadata: stored.metadata,
    object_id: objectId,
    stored: `pio/skills/${name}/${objectId}`,
  };
}

// ── List Skills ─────────────────────────────────────────

export function listSkills() {
  return Object.entries(SKILL_REGISTRY).map(([name, skill]) => ({
    name,
    version: skill.version,
    description: skill.description,
    role: skill.role,
    endpoints: {
      execute: `POST /skill/${name}`,
      spec: `GET /skill/${name}/spec`,
      mock: `GET /skill/${name}/mock`,
      test: `POST /skill/${name}/test`,
      eval: `POST /skill/${name}/eval`,
    },
    examples: (skill.examples || []).length,
  }));
}

// ── Get Skill Spec ──────────────────────────────────────

export function getSkillSpec(name) {
  const skill = SKILL_REGISTRY[name];
  if (!skill) return null;

  return {
    name: skill.name,
    version: skill.version,
    description: skill.description,
    role: skill.role,
    system_prompt: skill.systemPrompt,
    examples: skill.examples || [],
    has_validator: !!VALIDATORS[name],
    has_mock: !!MOCKS[name],
    endpoints: {
      execute: `POST /skill/${name}`,
      spec: `GET /skill/${name}/spec`,
      mock: `GET /skill/${name}/mock`,
      test: `POST /skill/${name}/test`,
      eval: `POST /skill/${name}/eval`,
    },
  };
}

// ── Test Skill (mock + validate) ────────────────────────

export function testSkill(name) {
  const skill = SKILL_REGISTRY[name];
  if (!skill) return { success: false, error: `Unknown skill: ${name}` };

  const mock = MOCKS[name];
  if (!mock) return { success: false, error: `No mock for skill: ${name}` };

  const validate = VALIDATORS[name];
  if (!validate) return { success: false, error: `No validator for skill: ${name}` };

  const validation = validate(mock);

  return {
    success: validation.valid,
    skill: name,
    test_type: 'mock_validation',
    mock_output: mock,
    validation,
    message: validation.valid
      ? `Mock response passes all ${name} schema checks`
      : `Mock response FAILS ${name} schema: ${validation.errors.join(', ')}`,
  };
}

// ── Eval Skill (real execution + metrics) ───────────────

export async function evalSkill(name, input, env) {
  const startTime = Date.now();
  const result = await executeSkill(name, input, env);
  const totalTime = Date.now() - startTime;

  const inputStr = typeof input === 'string' ? input : JSON.stringify(input);
  const outputStr = result.output ? JSON.stringify(result.output) : '';

  return {
    skill: name,
    success: result.success,
    metrics: {
      total_latency_ms: totalTime,
      ai_latency_ms: result.metadata?.latency_ms || 0,
      input_tokens_estimate: Math.ceil(inputStr.length / 4),
      output_tokens_estimate: Math.ceil(outputStr.length / 4),
      total_tokens_estimate: Math.ceil(inputStr.length / 4) + Math.ceil(outputStr.length / 4),
      schema_valid: result.validation?.valid || false,
      schema_errors: result.validation?.errors?.length || 0,
      field_completeness: result.output ? fieldCompleteness(result.output) : 0,
    },
    output: result.output,
    validation: result.validation,
    object_id: result.object_id,
  };
}

// ── Helpers ─────────────────────────────────────────────

function extractJSON(result) {
  let content = '';

  if (result.choices && result.choices[0] && result.choices[0].message) {
    content = result.choices[0].message.content || '';
  } else if (typeof result.response === 'string') {
    content = result.response;
  } else if (typeof result === 'string') {
    content = result;
  } else {
    content = JSON.stringify(result);
  }

  // Strip Qwen3 thinking blocks
  content = content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
  // Strip markdown fences
  content = content.replace(/^```(?:json)?\s*/m, '').replace(/\s*```$/m, '').trim();
  // Find JSON object
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (jsonMatch) return JSON.parse(jsonMatch[0]);
  return JSON.parse(content);
}

function extractRawContent(result) {
  if (result.choices && result.choices[0] && result.choices[0].message) {
    return result.choices[0].message.content || '';
  }
  if (typeof result.response === 'string') return result.response;
  if (typeof result === 'string') return result;
  return JSON.stringify(result);
}

function fieldCompleteness(obj) {
  if (!obj || typeof obj !== 'object') return 0;
  const fields = Object.keys(obj);
  if (fields.length === 0) return 0;
  const nonNull = fields.filter(k => obj[k] !== null && obj[k] !== undefined && obj[k] !== '');
  return Math.round((nonNull.length / fields.length) * 100);
}
