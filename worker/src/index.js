/**
 * SwarmRouter — Intelligence Objects API + Edge Cooker
 * ====================================================
 * router.swarmandbee.com
 *
 * Open source Intelligence Objects. The ocean.
 * 100 edge workers → AI Gateway → Qwen3-30B-A3B (3B active) → R2 finality.
 *
 * READ endpoints (open):
 *   GET /                              → API info + stats
 *   GET /pio/{object_id}               → Single Intelligence Object by ID
 *   GET /pio/address/{normalized_addr}  → Lookup by address
 *   GET /pio/search?q=...              → Search objects
 *   GET /pio/feed?type=...&limit=      → Latest objects feed
 *   GET /objects/stats                  → Intelligence bucket stats
 *   GET /vaults                         → All vault stats
 *   GET /health                         → Health check
 *
 * COOK endpoints (edge inference → R2):
 *   POST /cook                         → Cook raw data into Intelligence Object
 *   POST /cook/batch                   → Batch cook (up to 25)
 *   POST /cook/edgar                   → Cook from SEC EDGAR data
 *   POST /cook/market                  → Cook market Intelligence Object
 *
 * WRITE endpoints:
 *   POST /pio                          → Direct ingest (pre-cooked)
 *   POST /pio/batch                    → Batch ingest
 *
 * SKILL endpoints (composable intelligence operations):
 *   GET  /skills                       → List all skills with specs
 *   GET  /skill/{name}/spec            → Full skill spec
 *   GET  /skill/{name}/mock            → Mock response (no AI)
 *   POST /skill/{name}/test            → Test harness (mock + validate)
 *   POST /skill/{name}/eval            → Structured eval (tokens, latency)
 *   POST /skill/{name}                 → Execute skill (AI → validate → R2)
 */

import { STATES, stateExpertPrompt, STATE_CODES } from './states.js';
import { MARKET_HEAT, getStatesByTier, heatSummary } from './market_heat.js';
import { PORTS, POWER_ZONES, RAIL_HUBS, LAST_MILE_ZONES, AIR_CARGO, getStateInfrastructure, infrastructurePrompt } from './infrastructure.js';
import { executeSkill, listSkills, getSkillSpec, testSkill, evalSkill } from './skills/registry.js';
import { MOCKS } from './skills/mocks.js';
import { testAllSkills, failureSimulation, batchEval } from './skills/eval.js';
import { EVENT_TYPES, createEventObject, processEvent, getFeedSchedule, eventMachineStats } from './event_machine.js';
import { initSchema, insertEvent, queryEvents, getEvent, getEventStats, upsertEntity, linkEventEntity, extractEntities, queryEntities, getEntity, getEntityTimeline } from './db.js';
import { embedAndStore, memorySearch, memoryContext, memoryStats } from './memory.js';
import { walletMiddleware, authenticateRequest, deductCredits, getBalance, getUsage, topupWallet, createWallet, resolveEndpointKey, PUBLIC_ENDPOINTS, ENDPOINT_COSTS } from './wallet.js';
import { apiDocsResponse, API_VERSION } from './api_docs.js';

// ═══════════════════════════════════════════════════════
// EDGE MODEL CONFIG
// ═══════════════════════════════════════════════════════
const EDGE_MODEL = '@cf/qwen/qwen3-30b-a3b-fp8';  // 30B MoE, 3B active. The cooker.
const EDGE_TEMP = 0.1;  // Low temp — structuring, not creating.

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key, X-Admin-Key',
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { 'Content-Type': 'application/json', ...CORS },
  });
}

function err(message, status = 400) {
  return json({ error: message, status }, status);
}

function objectId(seed) {
  // Simple hash for deterministic IDs
  let hash = 0;
  const str = seed + Date.now().toString();
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return 'pio_' + Math.abs(hash).toString(16).padStart(12, '0');
}

// ═══════════════════════════════════════════════════════
// ROUTER
// ═══════════════════════════════════════════════════════

export default {
  async fetch(request, env) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }

    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;
    const host = url.hostname;
    const isApiDomain = host === 'api.router.swarmandbee.com';

    // Init D1 schema on first request (idempotent)
    if (env.DB && !env._dbReady) {
      try { await initSchema(env.DB); env._dbReady = true; } catch (_) {}
    }

    try {
      // ── API domain root → docs ─────────────────────
      if (isApiDomain && (path === '/' || path === '')) return json(apiDocsResponse());

      // ── Wallet routes (require auth, no credit cost) ──
      if (path === '/wallet/balance' && method === 'GET') return handleWalletBalance(request, env);
      if (path === '/wallet/usage' && method === 'GET') return handleWalletUsage(request, url, env);
      if (path === '/wallet/create' && method === 'POST') return handleWalletCreate(request, env);
      if (path === '/wallet/topup' && method === 'POST') return handleWalletTopup(request, env);

      // ── Memory routes ──────────────────────────────
      if (path === '/memory/search' && method === 'POST') return handleMemorySearch(request, env, isApiDomain);
      if (path === '/memory/context' && method === 'POST') return handleMemoryContext(request, env, isApiDomain);
      if (path === '/memory/stats' && method === 'GET') return handleMemoryStats(env);

      // ── Event DB query routes ──────────────────────
      if (path === '/events/query' && method === 'GET') return handleEventsQuery(url, env, isApiDomain, request);
      if (path === '/events/stats/db' && method === 'GET') return handleEventsStatsDB(env);

      // ── Entity routes ──────────────────────────────
      if (path === '/entities' && method === 'GET') return handleEntitiesQuery(url, env, isApiDomain, request);

      const entityTimelineMatch = path.match(/^\/entities\/([a-f0-9-]+)\/timeline$/);
      if (entityTimelineMatch && method === 'GET') return handleEntityTimeline(entityTimelineMatch[1], env, isApiDomain, request);

      const entityGetMatch = path.match(/^\/entities\/([a-f0-9-]+)$/);
      if (entityGetMatch && method === 'GET') return handleEntityGet(entityGetMatch[1], env, isApiDomain, request);

      const eventGetMatch = path.match(/^\/events\/([a-f0-9-]+)$/);
      if (eventGetMatch && method === 'GET') return handleEventGet(eventGetMatch[1], env, isApiDomain, request);

      // ── Static routes ───────────────────────────────
      if (path === '/' || path === '') return handleRoot(env);
      if (path === '/health') return json({ status: 'ok', ts: new Date().toISOString(), engine: 'swarmrouter', model: EDGE_MODEL });
      if (path === '/objects/stats') return handleStats(env);
      if (path === '/vaults') return handleVaults(env);

      // ── Read routes ─────────────────────────────────
      if (path === '/pio/search' && method === 'GET') return handleSearch(url, env);
      if (path === '/pio/feed' && method === 'GET') return handleFeed(url, env);

      // ── Cook routes (AI inference → R2) ─────────────
      if (path === '/cook' && method === 'POST') return handleCook(request, env);
      if (path === '/cook/batch' && method === 'POST') return handleCookBatch(request, env);
      if (path === '/cook/edgar' && method === 'POST') return handleCookEdgar(request, env);
      if (path === '/cook/market' && method === 'POST') return handleCookMarket(request, env);

      // ── State routes (50 area experts) ──────────────
      const stateMatch = path.match(/^\/cook\/([A-Za-z]{2})$/);
      if (stateMatch && method === 'POST') return handleStateCook(stateMatch[1].toUpperCase(), request, env);

      if (path === '/states' && method === 'GET') return handleStates(env);

      const stateStatsMatch = path.match(/^\/states\/([A-Za-z]{2})$/);
      if (stateStatsMatch && method === 'GET') return handleStateStats(stateStatsMatch[1].toUpperCase(), env);

      const stateFeedMatch = path.match(/^\/states\/([A-Za-z]{2})\/feed$/);
      if (stateFeedMatch && method === 'GET') return handleStateFeed(stateFeedMatch[1].toUpperCase(), url, env);

      // ── Market heat routes ──────────────────────────
      if (path === '/markets' && method === 'GET') return handleMarkets();
      if (path === '/markets/hot' && method === 'GET') return handleMarketsHot();
      if (path === '/markets/schedule' && method === 'GET') return handleCookSchedule();

      // ── Infrastructure routes ───────────────────────
      if (path === '/infrastructure' && method === 'GET') return handleInfrastructure();
      if (path === '/infrastructure/ports' && method === 'GET') return json({ ports: PORTS });
      if (path === '/infrastructure/rail' && method === 'GET') return json({ rail: RAIL_HUBS });
      if (path === '/infrastructure/power' && method === 'GET') return json({ power: POWER_ZONES });
      if (path === '/infrastructure/lastmile' && method === 'GET') return json({ last_mile: LAST_MILE_ZONES });
      if (path === '/infrastructure/aircargo' && method === 'GET') return json({ air_cargo: AIR_CARGO });

      const infraStateMatch = path.match(/^\/infrastructure\/([A-Za-z]{2})$/);
      if (infraStateMatch && method === 'GET') return json(getStateInfrastructure(infraStateMatch[1].toUpperCase()));

      // ── Skill routes (composable intelligence) ─────
      if (path === '/skills' && method === 'GET') return handleSkillsList();
      if (path === '/skills/test' && method === 'POST') return handleSkillsTestAll();

      const skillSpecMatch = path.match(/^\/skill\/([a-z0-9_]+)\/spec$/);
      if (skillSpecMatch && method === 'GET') return handleSkillSpec(skillSpecMatch[1]);

      const skillMockMatch = path.match(/^\/skill\/([a-z0-9_]+)\/mock$/);
      if (skillMockMatch && method === 'GET') return handleSkillMock(skillMockMatch[1]);

      const skillTestMatch = path.match(/^\/skill\/([a-z0-9_]+)\/test$/);
      if (skillTestMatch && method === 'POST') return handleSkillTest(skillTestMatch[1]);

      const skillEvalMatch = path.match(/^\/skill\/([a-z0-9_]+)\/eval$/);
      if (skillEvalMatch && method === 'POST') return handleSkillEval(skillEvalMatch[1], request, env);

      const skillFailMatch = path.match(/^\/skill\/([a-z0-9_]+)\/fail$/);
      if (skillFailMatch && method === 'POST') return handleSkillFailSim(skillFailMatch[1], env);

      const skillExecMatch = path.match(/^\/skill\/([a-z0-9_]+)$/);
      if (skillExecMatch && method === 'POST') return handleSkillExec(skillExecMatch[1], request, env);

      // ── Event Machine routes ──────────────────────
      if (path === '/events' && method === 'GET') return json(eventMachineStats());
      if (path === '/events/types' && method === 'GET') return json({ event_types: EVENT_TYPES });
      if (path === '/events/schedule' && method === 'GET') return json(getFeedSchedule());
      if (path === '/events/process' && method === 'POST') return handleEventProcess(request, env);

      // ── Write routes (direct ingest) ────────────────
      if (path === '/pio' && method === 'POST') return handleIngest(request, env);
      if (path === '/pio/batch' && method === 'POST') return handleBatchIngest(request, env);

      // ── Dynamic PIO lookup ──────────────────────────
      const addrMatch = path.match(/^\/pio\/address\/(.+)$/);
      if (addrMatch && method === 'GET') return handleGetByAddress(decodeURIComponent(addrMatch[1]), env);

      const pioMatch = path.match(/^\/pio\/(.+)$/);
      if (pioMatch && method === 'GET') return handleGetPIO(pioMatch[1], env);

      return err('Not found — try GET / for endpoints', 404);

    } catch (e) {
      return err(`Internal error: ${e.message}`, 500);
    }
  },
};


// ═══════════════════════════════════════════════════════
// READ HANDLERS
// ═══════════════════════════════════════════════════════

async function handleRoot(env) {
  const objCount = await countObjects(env.INTELLIGENCE);
  return json({
    name: 'SwarmRouter — Intelligence Objects API',
    engine: 'swarmrouter',
    model: EDGE_MODEL,
    tagline: 'One call. One object. Everything an agent needs.',
    intelligence_objects: objCount,
    endpoints: {
      read: {
        'GET /': 'API info',
        'GET /pio/{id}': 'Intelligence Object by ID',
        'GET /pio/address/{addr}': 'Intelligence Object by address',
        'GET /pio/search?q={query}&type={type}': 'Search objects',
        'GET /pio/feed?type={type}&limit={n}': 'Latest feed',
        'GET /objects/stats': 'Bucket stats',
        'GET /vaults': 'All vault stats',
        'GET /health': 'Health check',
      },
      cook: {
        'POST /cook': 'Cook raw data → Intelligence Object',
        'POST /cook/{STATE}': 'Cook with area expert (50 states: TX, CA, FL...)',
        'POST /cook/batch': 'Batch cook (max 25)',
        'POST /cook/edgar': 'Cook from SEC EDGAR data',
        'POST /cook/market': 'Cook market Intelligence Object',
      },
      states: {
        'GET /states': 'All 50 state profiles + object counts',
        'GET /states/{ST}': 'State profile + stats',
        'GET /states/{ST}/feed': 'State sandbox feed',
      },
      write: {
        'POST /pio': 'Direct ingest',
        'POST /pio/batch': 'Batch ingest (max 500)',
      },
      skills: {
        'GET /skills': 'List all skills with specs',
        'GET /skill/{name}/spec': 'Full skill spec (system prompt, schema, examples)',
        'GET /skill/{name}/mock': 'Mock response (no AI call)',
        'POST /skill/{name}': 'Execute skill (AI → validate → R2)',
        'POST /skill/{name}/test': 'Test harness (mock + validate)',
        'POST /skill/{name}/eval': 'Structured eval (tokens, latency, schema)',
        'POST /skill/{name}/fail': 'Failure simulation (bad inputs)',
        'POST /skills/test': 'Test all skills (mock validation suite)',
      },
      events: {
        'GET /events': 'Event Machine architecture + stats',
        'GET /events/types': 'All event types (deal, supply, ownership, macro, tenant)',
        'GET /events/schedule': 'Feed schedule (what to cook and when)',
        'POST /events/process': 'Process raw events → Intelligence Objects',
      },
    },
    skills: listSkills().map(s => s.name),
    vaults: ['sb-intelligence', 'sb-medical', 'sb-aviation', 'sb-cre', 'sb-core'],
    memory: { endpoint: '/memory/search', model: '@cf/baai/bge-base-en-v1.5', dimensions: 768 },
    event_db: { query: '/events/query', entities: '/entities', stats: '/events/stats/db' },
    wallets: { balance: '/wallet/balance', usage: '/wallet/usage', docs: 'https://api.router.swarmandbee.com/' },
    api_service: 'https://api.router.swarmandbee.com',
    version: API_VERSION,
    source: 'https://github.com/swarmandbee/swarmrouter',
    license: 'MIT',
  });
}

async function handleGetPIO(id, env) {
  for (const suffix of ['', '.json']) {
    const r2obj = await env.INTELLIGENCE.get(`pio/${id}${suffix}`);
    if (r2obj) {
      const obj = await r2obj.json();
      return json({ object: obj, source: 'sb-intelligence', retrieved_at: new Date().toISOString() });
    }
  }
  return err(`Intelligence Object '${id}' not found`, 404);
}

async function handleGetByAddress(addr, env) {
  const normalized = normalizeAddress(addr);
  const r2obj = await env.INTELLIGENCE.get(`pio/by-addr/${normalized}`);
  if (r2obj) {
    const obj = await r2obj.json();
    return json({ object: obj, source: 'sb-intelligence', retrieved_at: new Date().toISOString() });
  }
  return err(`No Intelligence Object for address: ${addr}`, 404);
}

async function handleSearch(url, env) {
  const q = (url.searchParams.get('q') || '').toLowerCase();
  const assetType = url.searchParams.get('type') || '';
  const market = url.searchParams.get('market') || '';
  const limit = Math.min(parseInt(url.searchParams.get('limit') || '50'), 200);

  if (!q && !assetType && !market) return err('Provide at least one: q, type, or market');

  const prefix = assetType ? `pio/by-type/${assetType}/` : 'pio/';
  const listed = await env.INTELLIGENCE.list({ prefix, limit: 1000 });
  const results = [];

  for (const item of listed.objects) {
    if (results.length >= limit) break;
    const r2item = await env.INTELLIGENCE.get(item.key);
    const obj = r2item ? await r2item.json() : null;
    if (!obj) continue;
    const blob = JSON.stringify(obj).toLowerCase();
    if ((!q || blob.includes(q)) && (!market || (obj.market_tier || '').toLowerCase().includes(market.toLowerCase()))) {
      results.push(obj);
    }
  }

  return json({ query: { q, type: assetType, market }, count: results.length, objects: results });
}

async function handleFeed(url, env) {
  const assetType = url.searchParams.get('type') || '';
  const limit = Math.min(parseInt(url.searchParams.get('limit') || '25'), 100);
  const prefix = assetType ? `pio/by-type/${assetType}/` : 'pio/';
  const listed = await env.INTELLIGENCE.list({ prefix, limit });

  const objects = [];
  for (const item of listed.objects) {
    const r2item = await env.INTELLIGENCE.get(item.key);
    const obj = r2item ? await r2item.json() : null;
    if (obj) objects.push(obj);
  }

  return json({ feed: assetType || 'all', count: objects.length, objects, as_of: new Date().toISOString() });
}

async function handleStats(env) {
  const listed = await env.INTELLIGENCE.list({ limit: 1000 });
  const byType = {};
  let total = 0;

  for (const item of listed.objects) {
    total++;
    const parts = item.key.split('/');
    if (parts[1] === 'by-type' && parts[2]) byType[parts[2]] = (byType[parts[2]] || 0) + 1;
  }

  return json({ bucket: 'sb-intelligence', total_objects: total, by_asset_type: byType, truncated: listed.truncated, as_of: new Date().toISOString() });
}

async function handleVaults(env) {
  const vaults = {};
  for (const [name, bucket] of Object.entries({ 'sb-intelligence': env.INTELLIGENCE, 'sb-medical': env.MEDICAL, 'sb-aviation': env.AVIATION, 'sb-cre': env.CRE, 'sb-core': env.CORE })) {
    try {
      await bucket.list({ limit: 1 });
      vaults[name] = { status: 'online' };
    } catch (e) {
      vaults[name] = { status: 'error', error: e.message };
    }
  }
  return json({ vaults, as_of: new Date().toISOString() });
}


// ═══════════════════════════════════════════════════════
// COOK HANDLERS — THE EDGE COOKER
// ═══════════════════════════════════════════════════════

const COOK_SYSTEM = `You are an Intelligence Object structuring engine for commercial real estate.
Your job: take raw property/financial data and structure it into a precise JSON Intelligence Object.
Rules:
- ONLY use data provided. Never hallucinate or invent values.
- If a field cannot be determined, set it to null.
- All financial values in USD (no formatting, raw numbers).
- Cap rates as decimals (0.055 not 5.5%).
- Square footage as integers.
- Return ONLY valid JSON. No explanation. No markdown fences.`;

async function handleCook(request, env) {
  const body = await request.json();
  const rawData = body.data || body.raw_data || body;
  const context = body.context || 'Structure this CRE data into an Intelligence Object';
  const schema = body.schema || 'property';  // 'property' | 'portfolio' | 'market'

  const schemaTemplate = schema === 'market' ? MARKET_SCHEMA : schema === 'portfolio' ? PORTFOLIO_SCHEMA : PROPERTY_SCHEMA;

  const prompt = `${context}

RAW DATA:
${JSON.stringify(rawData).substring(0, 6000)}

OUTPUT SCHEMA (fill from raw data, null if unknown):
${schemaTemplate}`;

  let result;
  try {
    result = await env.AI.run(EDGE_MODEL, {
      messages: [
        { role: 'system', content: COOK_SYSTEM },
        { role: 'user', content: prompt },
      ],
      max_tokens: 4096,
      temperature: EDGE_TEMP,
    });
  } catch (aiErr) {
    // Fallback to Llama 3.2 3B if Qwen3 fails
    try {
      result = await env.AI.run('@cf/meta/llama-3.2-3b-instruct', {
        messages: [
          { role: 'system', content: COOK_SYSTEM },
          { role: 'user', content: prompt },
        ],
        max_tokens: 4096,
        temperature: EDGE_TEMP,
      });
    } catch (fallbackErr) {
      return json({ cooked: false, error: `AI error: ${aiErr.message}`, fallback_error: fallbackErr.message }, 500);
    }
  }

  let obj;
  try {
    obj = extractJSON(result);
  } catch (e) {
    return json({ cooked: false, error: 'Model returned non-JSON', raw_response: (result.response || JSON.stringify(result)).substring(0, 1000) }, 422);
  }

  // Ensure required fields
  if (!obj.object_id) obj.object_id = objectId(JSON.stringify(rawData).substring(0, 100));
  if (!obj.object_type) obj.object_type = 'property_intelligence_object';
  if (!obj.created_at) obj.created_at = new Date().toISOString();
  obj.cooked_by = EDGE_MODEL;
  obj.version = '1.0';

  // Store in R2
  await storeObject(env.INTELLIGENCE, obj, env);

  return json({ cooked: true, object: obj }, 201);
}

async function handleCookBatch(request, env) {
  const body = await request.json();
  const items = body.items || [];
  if (!Array.isArray(items) || items.length === 0) return err('Provide items[] array');
  if (items.length > 25) return err('Max 25 items per batch');

  const results = [];
  for (const item of items) {
    try {
      const prompt = `Structure this into an Intelligence Object:

RAW DATA:
${JSON.stringify(item.data || item).substring(0, 4000)}

CONTEXT: ${item.context || 'CRE property data'}

${PROPERTY_SCHEMA}`;

      let result;
      try {
        result = await env.AI.run(EDGE_MODEL, {
          messages: [{ role: 'system', content: COOK_SYSTEM }, { role: 'user', content: prompt }],
          max_tokens: 4096, temperature: EDGE_TEMP,
        });
      } catch {
        result = await env.AI.run('@cf/meta/llama-3.2-3b-instruct', {
          messages: [{ role: 'system', content: COOK_SYSTEM }, { role: 'user', content: prompt }],
          max_tokens: 4096, temperature: EDGE_TEMP,
        });
      }

      const obj = extractJSON(result);
      if (!obj.object_id) obj.object_id = objectId(JSON.stringify(item).substring(0, 100));
      obj.object_type = obj.object_type || 'property_intelligence_object';
      obj.created_at = new Date().toISOString();
      obj.cooked_by = EDGE_MODEL;
      await storeObject(env.INTELLIGENCE, obj, env);
      results.push({ cooked: true, object_id: obj.object_id });
    } catch (e) {
      results.push({ cooked: false, error: e.message });
    }
  }

  return json({ batch_size: items.length, cooked: results.filter(r => r.cooked).length, failed: results.filter(r => !r.cooked).length, results }, 201);
}

async function handleCookEdgar(request, env) {
  const body = await request.json();
  const { ticker, financials, filings_text } = body;

  if (!ticker) return err('Provide ticker');

  const prompt = `Structure this REIT data into a portfolio Intelligence Object:

TICKER: ${ticker}
FINANCIALS: ${JSON.stringify(financials || {}).substring(0, 3000)}
FILING EXCERPT: ${(filings_text || '').substring(0, 5000)}

${PORTFOLIO_SCHEMA}`;

  let result;
  try {
    result = await env.AI.run(EDGE_MODEL, {
      messages: [{ role: 'system', content: COOK_SYSTEM }, { role: 'user', content: prompt }],
      max_tokens: 4096, temperature: EDGE_TEMP,
    });
  } catch {
    result = await env.AI.run('@cf/meta/llama-3.2-3b-instruct', {
      messages: [{ role: 'system', content: COOK_SYSTEM }, { role: 'user', content: prompt }],
      max_tokens: 4096, temperature: EDGE_TEMP,
    });
  }

  let obj;
  try {
    obj = extractJSON(result);
  } catch (e) {
    return json({ cooked: false, error: 'Model returned non-JSON', raw_response: (result.response || '').substring(0, 500) }, 422);
  }

  obj.object_id = obj.object_id || objectId(ticker);
  obj.object_type = 'portfolio_intelligence_object';
  obj.created_at = new Date().toISOString();
  obj.cooked_by = EDGE_MODEL;
  obj.source_type = 'edgar';
  await storeObject(env.INTELLIGENCE, obj, env);

  return json({ cooked: true, object: obj }, 201);
}

async function handleCookMarket(request, env) {
  const body = await request.json();
  const { market, indicators } = body;

  const prompt = `Structure this market data into a market Intelligence Object:

MARKET: ${market || 'national'}
INDICATORS: ${JSON.stringify(indicators || {}).substring(0, 4000)}

${MARKET_SCHEMA}`;

  let result;
  try {
    result = await env.AI.run(EDGE_MODEL, {
      messages: [{ role: 'system', content: COOK_SYSTEM }, { role: 'user', content: prompt }],
      max_tokens: 4096, temperature: EDGE_TEMP,
    });
  } catch {
    result = await env.AI.run('@cf/meta/llama-3.2-3b-instruct', {
      messages: [{ role: 'system', content: COOK_SYSTEM }, { role: 'user', content: prompt }],
      max_tokens: 4096, temperature: EDGE_TEMP,
    });
  }

  let obj;
  try {
    obj = extractJSON(result);
  } catch (e) {
    return json({ cooked: false, error: 'Model returned non-JSON', raw_response: (result.response || '').substring(0, 500) }, 422);
  }

  obj.object_id = obj.object_id || objectId(market || 'national');
  obj.object_type = 'market_intelligence_object';
  obj.created_at = new Date().toISOString();
  obj.cooked_by = EDGE_MODEL;
  await storeObject(env.INTELLIGENCE, obj, env);

  return json({ cooked: true, object: obj }, 201);
}


// ═══════════════════════════════════════════════════════
// STATE HANDLERS — 50 AREA EXPERTS
// ═══════════════════════════════════════════════════════

async function handleStateCook(stateCode, request, env) {
  if (!STATES[stateCode]) return err(`Unknown state: ${stateCode}. Use 2-letter code (TX, CA, etc.)`, 400);

  const body = await request.json();
  const rawData = body.data || body.raw_data || body;
  const context = body.context || `Property in ${STATES[stateCode].name}`;
  const schema = body.schema || 'property';
  const schemaTemplate = schema === 'market' ? MARKET_SCHEMA : schema === 'portfolio' ? PORTFOLIO_SCHEMA : PROPERTY_SCHEMA;

  // Inject area expert knowledge + infrastructure context
  const expertPrompt = stateExpertPrompt(stateCode) + infrastructurePrompt(stateCode);

  const prompt = `${context}

RAW DATA:
${JSON.stringify(rawData).substring(0, 6000)}

OUTPUT SCHEMA (fill from raw data, null if unknown):
${schemaTemplate}`;

  let result;
  try {
    result = await env.AI.run(EDGE_MODEL, {
      messages: [
        { role: 'system', content: expertPrompt },
        { role: 'user', content: prompt },
      ],
      max_tokens: 4096,
      temperature: EDGE_TEMP,
    });
  } catch (aiErr) {
    try {
      result = await env.AI.run('@cf/meta/llama-3.2-3b-instruct', {
        messages: [
          { role: 'system', content: expertPrompt },
          { role: 'user', content: prompt },
        ],
        max_tokens: 4096,
        temperature: EDGE_TEMP,
      });
    } catch (fallbackErr) {
      return json({ cooked: false, error: `AI error: ${aiErr.message}` }, 500);
    }
  }

  let obj;
  try {
    obj = extractJSON(result);
  } catch (e) {
    return json({ cooked: false, error: 'Model returned non-JSON', raw_response: (result.response || JSON.stringify(result)).substring(0, 1000) }, 422);
  }

  // Stamp state metadata
  if (!obj.object_id) obj.object_id = objectId(stateCode + JSON.stringify(rawData).substring(0, 100));
  obj.object_type = obj.object_type || 'property_intelligence_object';
  obj.state = stateCode;
  obj.created_at = new Date().toISOString();
  obj.cooked_by = EDGE_MODEL;
  obj.area_expert = stateCode;
  obj.version = '1.0';

  // Store in state sandbox: pio/states/{ST}/{id}
  const stateKey = `pio/states/${stateCode}/${obj.object_id}`;
  await env.INTELLIGENCE.put(stateKey, JSON.stringify(obj), {
    customMetadata: {
      state: stateCode,
      asset_type: obj.asset_type || 'unknown',
      address: obj.address || '',
      created_at: obj.created_at,
    },
  });

  // Also store in main index
  await storeObject(env.INTELLIGENCE, obj, env);

  return json({
    cooked: true,
    state: stateCode,
    area_expert: STATES[stateCode].name,
    sandbox: `pio/states/${stateCode}/`,
    object: obj,
  }, 201);
}

async function handleStates(env) {
  // Return all 50 states with their profiles and object counts
  const stateStats = {};

  for (const code of STATE_CODES) {
    const s = STATES[code];
    const prefix = `pio/states/${code}/`;
    let count = 0;
    try {
      const listed = await env.INTELLIGENCE.list({ prefix, limit: 1 });
      count = listed.objects.length + (listed.truncated ? '+' : 0);
    } catch {}

    stateStats[code] = {
      name: s.name,
      objects: count,
      cap_rate_range: s.cap_rate_range,
      tax_rate: s.property_tax_rate,
      income_tax: s.income_tax,
      markets: s.markets,
      sandbox: `pio/states/${code}/`,
    };
  }

  return json({
    total_states: STATE_CODES.length,
    states: stateStats,
    cook_endpoint: 'POST /cook/{STATE_CODE}',
    feed_endpoint: 'GET /states/{STATE_CODE}/feed',
  });
}

async function handleStateStats(stateCode, env) {
  if (!STATES[stateCode]) return err(`Unknown state: ${stateCode}`, 400);

  const s = STATES[stateCode];
  const prefix = `pio/states/${stateCode}/`;
  const listed = await env.INTELLIGENCE.list({ prefix, limit: 1000 });

  const heat = MARKET_HEAT[stateCode] || { tier: 4, heat: 0, label: 'UNKNOWN', cook_weight: 1 };

  return json({
    state: stateCode,
    name: s.name,
    heat: { tier: heat.tier, score: heat.heat, label: heat.label, cook_weight: heat.cook_weight, reason: heat.reason },
    profile: s,
    sandbox: prefix,
    objects: listed.objects.length,
    truncated: listed.truncated,
    cook_endpoint: `POST /cook/${stateCode}`,
    feed_endpoint: `GET /states/${stateCode}/feed`,
  });
}

async function handleStateFeed(stateCode, url, env) {
  if (!STATES[stateCode]) return err(`Unknown state: ${stateCode}`, 400);

  const limit = Math.min(parseInt(url.searchParams.get('limit') || '25'), 100);
  const prefix = `pio/states/${stateCode}/`;
  const listed = await env.INTELLIGENCE.list({ prefix, limit });

  const objects = [];
  for (const item of listed.objects) {
    const r2item = await env.INTELLIGENCE.get(item.key);
    const obj = r2item ? await r2item.json() : null;
    if (obj) objects.push(obj);
  }

  return json({
    state: stateCode,
    name: STATES[stateCode].name,
    count: objects.length,
    objects,
    as_of: new Date().toISOString(),
  });
}


// ═══════════════════════════════════════════════════════
// MARKET HEAT HANDLERS
// ═══════════════════════════════════════════════════════

function handleMarkets() {
  const summary = heatSummary();
  const ranked = Object.entries(MARKET_HEAT)
    .sort((a, b) => b[1].heat - a[1].heat)
    .map(([code, data]) => ({
      state: code,
      name: STATES[code]?.name || code,
      heat: data.heat,
      tier: data.tier,
      label: data.label,
      cook_weight: data.cook_weight,
      reason: data.reason,
      markets: STATES[code]?.markets || [],
      cap_rate_range: STATES[code]?.cap_rate_range || [],
    }));

  return json({
    title: 'Market Heat Index — Where the Deals Are',
    summary,
    ranked,
  });
}

function handleMarketsHot() {
  const hot = getStatesByTier(1);
  const warm = getStatesByTier(2);

  return json({
    title: 'Hot & Warm Markets — 80% of Deal Flow',
    hot: hot.map(s => ({
      ...s,
      name: STATES[s.code]?.name,
      markets: STATES[s.code]?.markets,
      industrial_hubs: STATES[s.code]?.industrial_hubs,
      cap_rate_range: STATES[s.code]?.cap_rate_range,
    })),
    warm: warm.map(s => ({
      ...s,
      name: STATES[s.code]?.name,
      markets: STATES[s.code]?.markets,
      cap_rate_range: STATES[s.code]?.cap_rate_range,
    })),
    message: 'These markets represent ~80% of US industrial deal volume. Cook priority weighted accordingly.',
  });
}

function handleCookSchedule() {
  const summary = heatSummary();
  return json({
    title: 'Cook Schedule — Weighted by Market Heat',
    description: 'Hot markets (Tier 1) get 10x cook cycles. Cold (Tier 4) get 1x.',
    ...summary,
    example: 'In a 200-cycle cook run: TX gets ~10 cycles, VT gets ~1 cycle.',
  });
}


// ═══════════════════════════════════════════════════════
// INFRASTRUCTURE HANDLERS
// ═══════════════════════════════════════════════════════

function handleInfrastructure() {
  return json({
    title: 'Industrial Infrastructure Map — Ports, Power, Rail, Last-Mile',
    pillars: {
      ports: { count: Object.keys(PORTS).length, description: 'Container + inland ports', endpoint: '/infrastructure/ports' },
      power: { count: Object.keys(POWER_ZONES).length, description: 'Grid capacity, data center zones, renewables', endpoint: '/infrastructure/power' },
      rail: { count: Object.keys(RAIL_HUBS).length, description: 'Class I hubs, intermodal terminals', endpoint: '/infrastructure/rail' },
      last_mile: { count: Object.keys(LAST_MILE_ZONES).length, description: 'Population reach, same-day delivery zones', endpoint: '/infrastructure/lastmile' },
      air_cargo: { count: Object.keys(AIR_CARGO).length, description: 'Air cargo hubs, premium last-mile', endpoint: '/infrastructure/aircargo' },
    },
    by_state: '/infrastructure/{ST}',
    principle: 'Industrial lives on infrastructure. Not zip codes. A warehouse is only worth what it can reach and how fast.',
  });
}


// ═══════════════════════════════════════════════════════
// SKILL HANDLERS — COMPOSABLE INTELLIGENCE
// ═══════════════════════════════════════════════════════

function handleSkillsList() {
  const skills = listSkills();
  return json({
    title: 'Swarm Skills — Composable Intelligence Operations',
    description: 'Each skill: input → AI (Qwen3-30B-A3B) → schema validate → R2 finality',
    total_skills: skills.length,
    skills,
    endpoints: {
      list: 'GET /skills',
      spec: 'GET /skill/{name}/spec',
      mock: 'GET /skill/{name}/mock',
      execute: 'POST /skill/{name}',
      test: 'POST /skill/{name}/test',
      eval: 'POST /skill/{name}/eval',
      fail: 'POST /skill/{name}/fail',
      test_all: 'POST /skills/test',
    },
  });
}

function handleSkillSpec(name) {
  const spec = getSkillSpec(name);
  if (!spec) return err(`Unknown skill: ${name}. GET /skills for available skills.`, 404);
  return json({ spec });
}

function handleSkillMock(name) {
  const mock = MOCKS[name];
  if (!mock) return err(`No mock for skill: ${name}`, 404);
  return json({ skill: name, mock, note: 'This is a mock response — no AI call was made' });
}

function handleSkillTest(name) {
  const result = testSkill(name);
  return json(result, result.success ? 200 : 422);
}

function handleSkillsTestAll() {
  const results = testAllSkills();
  return json(results, results.failed === 0 ? 200 : 422);
}

async function handleSkillExec(name, request, env) {
  let input;
  try {
    input = await request.json();
  } catch {
    return err('Invalid JSON body');
  }
  const result = await executeSkill(name, input, env);
  return json(result, result.success ? 201 : 422);
}

async function handleSkillEval(name, request, env) {
  let input;
  try {
    input = await request.json();
  } catch {
    return err('Invalid JSON body');
  }
  const result = await evalSkill(name, input, env);
  return json(result);
}

async function handleSkillFailSim(name, env) {
  const result = await failureSimulation(name, env);
  return json(result);
}


// ═══════════════════════════════════════════════════════
// EVENT MACHINE HANDLERS
// ═══════════════════════════════════════════════════════

async function handleEventProcess(request, env) {
  let body;
  try { body = await request.json(); } catch { return err('Invalid JSON body'); }

  const events = body.events || [body];
  if (!Array.isArray(events)) return err('Provide event or events[] array');
  if (events.length > 100) return err('Max 100 events per batch');

  const results = [];
  for (const event of events) {
    try {
      const result = await processEvent(event, env);

      // D1: insert event + extract entities
      if (env.DB) {
        try {
          const eventObj = result.event || event;
          const dbEventId = await insertEvent(env.DB, eventObj);

          // Entity extraction + resolution
          const entities = extractEntities(eventObj);
          for (const entity of entities) {
            const entityId = await upsertEntity(env.DB, entity);
            await linkEventEntity(env.DB, dbEventId, entityId, entity.entity_type);
          }
          result.db_event_id = dbEventId;
          result.entities_extracted = entities.length;
        } catch (_) { /* D1 non-critical */ }
      }

      // Memory: embed the event
      if (env.VECTORIZE) {
        try {
          const eventObj = result.event || event;
          await embedAndStore(env, eventObj, `pio/events/${eventObj.event_type}/${eventObj.object_id}`);
        } catch (_) { /* Vectorize non-critical */ }
      }

      results.push({ success: true, ...result });
    } catch (e) {
      results.push({ success: false, error: e.message });
    }
  }

  const processed = results.filter(r => r.success).length;
  return json({
    processed,
    failed: results.length - processed,
    total: results.length,
    results,
  }, 201);
}


// ═══════════════════════════════════════════════════════
// WALLET HANDLERS
// ═══════════════════════════════════════════════════════

async function handleWalletBalance(request, env) {
  if (!env.DB) return err('Database not available', 503);
  const wallet = await authenticateRequest(env.DB, request);
  if (!wallet) return err('API key required', 401);
  const balance = await getBalance(env.DB, wallet.id);
  return json(balance);
}

async function handleWalletUsage(request, url, env) {
  if (!env.DB) return err('Database not available', 503);
  const wallet = await authenticateRequest(env.DB, request);
  if (!wallet) return err('API key required', 401);
  const days = url.searchParams.get('days') || '7';
  const usage = await getUsage(env.DB, wallet.id, { days });
  return json(usage);
}

async function handleWalletCreate(request, env) {
  if (!env.DB) return err('Database not available', 503);
  const adminKey = request.headers.get('X-Admin-Key');
  if (!adminKey || adminKey !== env.ADMIN_KEY) return err('Admin access required', 403);
  const body = await request.json();
  const result = await createWallet(env.DB, body.name || 'unnamed', body.tier || 'free');
  return json(result, 201);
}

async function handleWalletTopup(request, env) {
  if (!env.DB) return err('Database not available', 503);
  const adminKey = request.headers.get('X-Admin-Key');
  if (!adminKey || adminKey !== env.ADMIN_KEY) return err('Admin access required', 403);
  const body = await request.json();
  if (!body.wallet_id || !body.amount) return err('Provide wallet_id and amount');
  const result = await topupWallet(env.DB, body.wallet_id, body.amount, body.reference);
  return json(result);
}


// ═══════════════════════════════════════════════════════
// MEMORY HANDLERS
// ═══════════════════════════════════════════════════════

async function handleMemorySearch(request, env, isApiDomain) {
  if (!env.VECTORIZE) return err('Vectorize not available', 503);
  if (isApiDomain) {
    const mw = await walletMiddleware(env, request, 'memory_search');
    if (mw.error) return mw.response;
    const body = await request.json();
    const result = await memorySearch(env, body.query, { limit: body.limit });
    await deductCredits(env.DB, mw.wallet.id, mw.cost, '/memory/search', 'POST', 200, 0);
    return json(result);
  }
  const body = await request.json();
  return json(await memorySearch(env, body.query, { limit: body.limit }));
}

async function handleMemoryContext(request, env, isApiDomain) {
  if (!env.VECTORIZE) return err('Vectorize not available', 503);
  if (isApiDomain) {
    const mw = await walletMiddleware(env, request, 'memory_context');
    if (mw.error) return mw.response;
    const body = await request.json();
    const result = await memoryContext(env, body.query, { timeframe: body.timeframe, limit: body.limit });
    await deductCredits(env.DB, mw.wallet.id, mw.cost, '/memory/context', 'POST', 200, 0);
    return json(result);
  }
  const body = await request.json();
  return json(await memoryContext(env, body.query, { timeframe: body.timeframe, limit: body.limit }));
}

async function handleMemoryStats(env) {
  const stats = await memoryStats(env);
  return json(stats);
}


// ═══════════════════════════════════════════════════════
// EVENT DB + ENTITY HANDLERS
// ═══════════════════════════════════════════════════════

async function handleEventsQuery(url, env, isApiDomain, request) {
  if (!env.DB) return err('Database not available', 503);
  if (isApiDomain) {
    const mw = await walletMiddleware(env, request, 'events_query');
    if (mw.error) return mw.response;
    const result = await queryEvents(env.DB, Object.fromEntries(url.searchParams));
    await deductCredits(env.DB, mw.wallet.id, mw.cost, '/events/query', 'GET', 200, 0);
    return json(result);
  }
  return json(await queryEvents(env.DB, Object.fromEntries(url.searchParams)));
}

async function handleEventsStatsDB(env) {
  if (!env.DB) return err('Database not available', 503);
  return json(await getEventStats(env.DB));
}

async function handleEventGet(id, env, isApiDomain, request) {
  if (!env.DB) return err('Database not available', 503);
  if (isApiDomain) {
    const mw = await walletMiddleware(env, request, 'event_get');
    if (mw.error) return mw.response;
    const event = await getEvent(env.DB, id);
    if (!event) return err('Event not found', 404);
    await deductCredits(env.DB, mw.wallet.id, mw.cost, `/events/${id}`, 'GET', 200, 0);
    return json(event);
  }
  const event = await getEvent(env.DB, id);
  if (!event) return err('Event not found', 404);
  return json(event);
}

async function handleEntitiesQuery(url, env, isApiDomain, request) {
  if (!env.DB) return err('Database not available', 503);
  if (isApiDomain) {
    const mw = await walletMiddleware(env, request, 'entities_query');
    if (mw.error) return mw.response;
    const entities = await queryEntities(env.DB, Object.fromEntries(url.searchParams));
    await deductCredits(env.DB, mw.wallet.id, mw.cost, '/entities', 'GET', 200, 0);
    return json({ entities });
  }
  return json({ entities: await queryEntities(env.DB, Object.fromEntries(url.searchParams)) });
}

async function handleEntityGet(id, env, isApiDomain, request) {
  if (!env.DB) return err('Database not available', 503);
  if (isApiDomain) {
    const mw = await walletMiddleware(env, request, 'entity_get');
    if (mw.error) return mw.response;
    const entity = await getEntity(env.DB, id);
    if (!entity) return err('Entity not found', 404);
    await deductCredits(env.DB, mw.wallet.id, mw.cost, `/entities/${id}`, 'GET', 200, 0);
    return json(entity);
  }
  const entity = await getEntity(env.DB, id);
  if (!entity) return err('Entity not found', 404);
  return json(entity);
}

async function handleEntityTimeline(id, env, isApiDomain, request) {
  if (!env.DB) return err('Database not available', 503);
  if (isApiDomain) {
    const mw = await walletMiddleware(env, request, 'entity_timeline');
    if (mw.error) return mw.response;
    const timeline = await getEntityTimeline(env.DB, id);
    await deductCredits(env.DB, mw.wallet.id, mw.cost, `/entities/${id}/timeline`, 'GET', 200, 0);
    return json({ entity_id: id, events: timeline });
  }
  return json({ entity_id: id, events: await getEntityTimeline(env.DB, id) });
}


// ═══════════════════════════════════════════════════════
// WRITE HANDLERS (DIRECT INGEST)
// ═══════════════════════════════════════════════════════

async function handleIngest(request, env) {
  const body = await request.json();
  if (!body.object_id || !body.object_type) return err('Requires object_id and object_type');
  await storeObject(env.INTELLIGENCE, body, env);
  return json({ ingested: true, object_id: body.object_id }, 201);
}

async function handleBatchIngest(request, env) {
  const body = await request.json();
  const objects = body.objects || [];
  if (!Array.isArray(objects) || objects.length === 0) return err('Provide objects[] array');
  if (objects.length > 500) return err('Max 500 per batch');

  let ingested = 0, failed = 0;
  for (const obj of objects) {
    try {
      if (!obj.object_id || !obj.object_type) { failed++; continue; }
      await storeObject(env.INTELLIGENCE, obj, env);
      ingested++;
    } catch { failed++; }
  }

  return json({ ingested, failed, total: objects.length }, 201);
}


// ═══════════════════════════════════════════════════════
// SCHEMAS — WHAT GETS MINTED
// ═══════════════════════════════════════════════════════

const PROPERTY_SCHEMA = `{
  "object_id": "pio_{hash}",
  "object_type": "property_intelligence_object",
  "address": "full street address",
  "city": "", "state": "", "zip": "",
  "asset_type": "industrial|warehouse|flex|cold_storage|data_center|logistics",
  "building_sf": null,
  "land_acres": null,
  "year_built": null,
  "clear_height_ft": null,
  "dock_doors": null,
  "estimated_value": null,
  "cap_rate": null,
  "noi": null,
  "rent_per_sf": null,
  "occupancy": null,
  "owner": "",
  "tenant": "",
  "risk_factors": [],
  "confidence_score": 0.0,
  "sources": []
}`;

const PORTFOLIO_SCHEMA = `{
  "object_id": "pio_{hash}",
  "object_type": "portfolio_intelligence_object",
  "entity": {"name": "", "ticker": "", "entity_type": "reit"},
  "portfolio": {"total_properties": null, "total_sf": null, "occupancy_rate": null, "markets": []},
  "financials": {
    "revenue": null, "noi": null, "total_assets": null, "total_debt": null,
    "equity": null, "interest_expense": null, "rental_revenue": null,
    "dividends_per_share": null, "implied_cap_rate": null,
    "debt_to_assets": null, "interest_coverage": null
  },
  "risk_factors": [],
  "confidence_score": 0.0,
  "sources": []
}`;

const MARKET_SCHEMA = `{
  "object_id": "mio_{hash}",
  "object_type": "market_intelligence_object",
  "market": "",
  "indicators": {},
  "treasury_10yr": null,
  "cap_rate_spread": null,
  "unemployment": null,
  "cpi": null,
  "risk_factors": [],
  "confidence_score": 0.0,
  "sources": []
}`;


// ═══════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════

function extractJSON(result) {
  // Workers AI can return multiple formats:
  // 1. { response: "json string" } — standard text gen
  // 2. { choices: [{ message: { content: "..." } }] } — chat completion format
  // 3. Direct object

  let content = '';

  // Try chat completion format first (Qwen3 returns this)
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
  // Find the last JSON object (in case there's preamble text)
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    return JSON.parse(jsonMatch[0]);
  }
  return JSON.parse(content);
}

function normalizeAddress(addr) {
  return addr.toLowerCase().replace(/[^a-z0-9]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
}

async function countObjects(bucket) {
  try {
    const listed = await bucket.list({ prefix: 'pio/', limit: 1000 });
    return listed.objects.length + (listed.truncated ? '+' : '');
  } catch { return 0; }
}

async function storeObject(bucket, obj, env) {
  const id = obj.object_id;
  const key = `pio/${id}`;

  await bucket.put(key, JSON.stringify(obj), {
    customMetadata: {
      asset_type: obj.asset_type || obj.source_type || 'unknown',
      address: obj.address || '',
      created_at: obj.created_at || new Date().toISOString(),
      object_type: obj.object_type || '',
    },
  });

  // Type index
  const typeVal = obj.asset_type || obj.source_type;
  if (typeVal) {
    await bucket.put(`pio/by-type/${typeVal}/${id}`, JSON.stringify(obj));
  }

  // Address index
  if (obj.address) {
    await bucket.put(`pio/by-addr/${normalizeAddress(obj.address)}`, JSON.stringify(obj));
  }

  // Memory: embed into Vectorize (non-blocking)
  if (env && env.VECTORIZE) {
    try { await embedAndStore(env, obj, key); } catch (_) { /* non-critical */ }
  }
}
