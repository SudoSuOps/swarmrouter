/**
 * Memory Layer — Vectorize Embeddings + Semantic Search
 * =====================================================
 * Every Intelligence Object gets embedded → Vectorize index.
 * Semantic search across the entire knowledge base.
 * Temporal context reasoning over time.
 *
 * Model: @cf/baai/bge-base-en-v1.5 (768 dims)
 */

import { insertMemoryIndex } from './db.js';

const EMBED_MODEL = '@cf/baai/bge-base-en-v1.5';

// ── Embed Text ──────────────────────────────────────

export async function embedText(env, text) {
  const input = text.substring(0, 2000); // BGE context window
  const result = await env.AI.run(EMBED_MODEL, { text: [input] });
  if (result.data && result.data[0]) return result.data[0];
  throw new Error('Embedding model returned no data');
}

// ── Build Embedding Text ────────────────────────────

export function buildEmbeddingText(obj) {
  if (!obj) return '';

  const parts = [];

  // Object type context
  if (obj.object_type) parts.push(obj.object_type.replace(/_/g, ' '));

  // Property fields
  if (obj.address) parts.push(obj.address);
  if (obj.city) parts.push(obj.city);
  if (obj.state) parts.push(obj.state);
  if (obj.asset_type) parts.push(obj.asset_type.replace(/_/g, ' '));
  if (obj.tenant) parts.push(`tenant: ${obj.tenant}`);
  if (obj.sf) parts.push(`${obj.sf} SF`);

  // Financial signals
  if (obj.cap_rate) parts.push(`cap rate ${obj.cap_rate}`);
  if (obj.price) parts.push(`price $${obj.price}`);
  if (obj.noi) parts.push(`NOI $${obj.noi}`);

  // Event fields
  if (obj.event_type) parts.push(obj.event_type.replace(/_/g, ' '));
  if (obj.description) parts.push(obj.description);
  if (obj.summary) parts.push(obj.summary);
  if (obj.company) parts.push(obj.company);
  if (obj.ticker) parts.push(obj.ticker);

  // Skill output
  if (obj.skill) parts.push(`skill: ${obj.skill}`);
  if (obj.deal_verdict) parts.push(`verdict: ${obj.deal_verdict}`);
  if (obj.investment_thesis) parts.push(obj.investment_thesis);
  if (obj.ic_notes) parts.push(obj.ic_notes);

  // Market context
  if (obj.market) parts.push(obj.market);
  if (obj.submarket) parts.push(obj.submarket);
  if (obj.market_context) {
    if (typeof obj.market_context === 'string') parts.push(obj.market_context);
    else if (obj.market_context.submarket) parts.push(obj.market_context.submarket);
  }

  // Risk factors
  if (Array.isArray(obj.risk_factors)) parts.push(obj.risk_factors.join(', '));

  // Generic fallback
  if (parts.length === 0 && typeof obj === 'object') {
    const str = JSON.stringify(obj);
    return str.substring(0, 1000);
  }

  return parts.join(' | ').substring(0, 2000);
}

// ── Embed and Store ─────────────────────────────────

export async function embedAndStore(env, object, r2Key) {
  const text = buildEmbeddingText(object);
  if (!text || text.length < 10) return null;

  const objectId = object.object_id || object.id || r2Key;
  const vectorId = `mem_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`;

  // Generate embedding
  const vector = await embedText(env, text);

  // Store in Vectorize
  await env.VECTORIZE.upsert([{
    id: vectorId,
    values: vector,
    metadata: {
      object_id: objectId,
      object_type: object.object_type || 'unknown',
      state: object.state || '',
      asset_type: object.asset_type || '',
    },
  }]);

  // Store metadata in D1
  if (env.DB) {
    await insertMemoryIndex(env.DB, {
      id: vectorId,
      object_id: objectId,
      object_type: object.object_type || 'unknown',
      r2_key: r2Key || '',
      summary: text.substring(0, 500),
      state: object.state || null,
      asset_type: object.asset_type || null,
    });
  }

  return vectorId;
}

// ── Memory Search ───────────────────────────────────

export async function memorySearch(env, query, options = {}) {
  const limit = Math.min(options.limit || 10, 50);

  // Embed the query
  const queryVector = await embedText(env, query);

  // Search Vectorize
  const results = await env.VECTORIZE.query(queryVector, {
    topK: limit,
    returnMetadata: 'all',
  });

  if (!results || !results.matches || results.matches.length === 0) {
    return { query, results: [], total: 0 };
  }

  // Hydrate from R2 if possible
  const hydrated = [];
  for (const match of results.matches) {
    const item = {
      vector_id: match.id,
      score: match.score,
      object_id: match.metadata?.object_id || '',
      object_type: match.metadata?.object_type || '',
      state: match.metadata?.state || '',
      asset_type: match.metadata?.asset_type || '',
    };

    // Try to get summary from D1 memory_index
    if (env.DB) {
      try {
        const memRow = await env.DB.prepare(
          'SELECT summary, r2_key FROM memory_index WHERE id = ?'
        ).bind(match.id).first();
        if (memRow) {
          item.summary = memRow.summary;
          item.r2_key = memRow.r2_key;
        }
      } catch (_) { /* non-critical */ }
    }

    hydrated.push(item);
  }

  return {
    query,
    results: hydrated,
    total: hydrated.length,
    model: EMBED_MODEL,
    dimensions: 768,
  };
}

// ── Memory Context (Temporal Reasoning) ─────────────

export async function memoryContext(env, query, options = {}) {
  const timeframe = options.timeframe || '6m';
  const limit = Math.min(options.limit || 10, 30);

  // Step 1: Semantic search for relevant objects
  const searchResults = await memorySearch(env, query, { limit });

  // Step 2: Get temporal data from D1 if available
  let timeline = [];
  if (env.DB && searchResults.results.length > 0) {
    // Find events related to the same state/market
    const states = [...new Set(searchResults.results.map(r => r.state).filter(Boolean))];
    if (states.length > 0) {
      const placeholders = states.map(() => '?').join(',');
      const timeLimit = timeframeToDate(timeframe);
      try {
        const events = await env.DB.prepare(`
          SELECT event_type, state, ticker, confidence, created_at, data_json
          FROM events WHERE state IN (${placeholders}) AND created_at >= ?
          ORDER BY created_at DESC LIMIT 20
        `).bind(...states, timeLimit).all();
        timeline = events.results;
      } catch (_) { /* non-critical */ }
    }
  }

  return {
    query,
    timeframe,
    relevant_objects: searchResults.results,
    timeline: timeline.map(e => ({
      event_type: e.event_type,
      state: e.state,
      ticker: e.ticker,
      confidence: e.confidence,
      date: e.created_at,
    })),
    context_depth: searchResults.total + timeline.length,
  };
}

// ── Memory Stats ────────────────────────────────────

export async function memoryStats(env) {
  const stats = { vectors: 0, indexed: 0, model: EMBED_MODEL, dimensions: 768 };

  if (env.DB) {
    try {
      const { getMemoryStats } = await import('./db.js');
      const dbStats = await getMemoryStats(env.DB);
      stats.indexed = dbStats.total_indexed;
      stats.by_type = dbStats.by_type;
    } catch (_) { /* D1 not ready */ }
  }

  return stats;
}

// ── Helpers ──────────────────────────────────────────

function timeframeToDate(tf) {
  const now = new Date();
  const match = tf.match(/^(\d+)([dmyh])$/);
  if (!match) return new Date(now.getTime() - 180 * 24 * 60 * 60 * 1000).toISOString(); // default 6m

  const [, num, unit] = match;
  const n = parseInt(num);
  switch (unit) {
    case 'h': now.setHours(now.getHours() - n); break;
    case 'd': now.setDate(now.getDate() - n); break;
    case 'm': now.setMonth(now.getMonth() - n); break;
    case 'y': now.setFullYear(now.getFullYear() - n); break;
  }
  return now.toISOString();
}
