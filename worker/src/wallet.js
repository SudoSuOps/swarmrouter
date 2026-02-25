/**
 * Wallet Layer — API Keys, Credits, Usage Metering
 * =================================================
 * Every metered API call: authenticate → check balance → execute → deduct → log
 * API keys: sb_live_ + 32-char hex, SHA-256 hashed in D1.
 */

// ── Endpoint Costs (credits) ────────────────────────

export const ENDPOINT_COSTS = {
  // Cook endpoints: 1 credit (AI inference + R2 storage)
  'cook':           1,
  'cook_batch':     1,   // per item
  'cook_edgar':     1,
  'cook_market':    1,
  'cook_state':     1,

  // Skill execution: 2 credits (AI + validation + R2)
  'skill_exec':     2,
  'skill_eval':     2,
  'skill_fail':     1,

  // Read queries: 0.5 credits (D1 only)
  'events_query':   0.5,
  'event_get':      0.5,
  'entities_query': 0.5,
  'entity_get':     0.5,
  'entity_timeline': 0.5,
  'pio_search':     0.5,
  'pio_feed':       0.5,
  'pio_get':        0.5,

  // Memory: 1 credit (embedding + Vectorize)
  'memory_search':  1,
  'memory_context': 1,

  // Event processing: 1 credit per event
  'events_process': 1,

  // Write: 0.5 credits
  'pio_ingest':     0.5,
};

// ── Public Endpoints (no auth required) ─────────────

export const PUBLIC_ENDPOINTS = new Set([
  '/',
  '/health',
  '/skills',
  '/events',
  '/events/types',
  '/events/schedule',
  '/events/stats/db',
  '/memory/stats',
  '/states',
  '/markets',
  '/markets/hot',
  '/markets/schedule',
  '/infrastructure',
  '/vaults',
  '/objects/stats',
]);

// ── SHA-256 Hash ────────────────────────────────────

async function sha256(text) {
  const encoded = new TextEncoder().encode(text);
  const hash = await crypto.subtle.digest('SHA-256', encoded);
  return Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2, '0')).join('');
}

// ── Generate API Key ────────────────────────────────

function generateApiKey() {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  const hex = Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
  return `sb_live_${hex}`;
}

// ── Create Wallet ───────────────────────────────────

export async function createWallet(db, name, tier = 'free') {
  const id = crypto.randomUUID();
  const apiKey = generateApiKey();
  const apiKeyHash = await sha256(apiKey);
  const prefix = apiKey.substring(0, 12) + '...';

  const initialBalance = tier === 'pro' ? 1000 : tier === 'starter' ? 100 : 10;

  await db.prepare(`
    INSERT INTO wallets (id, api_key, api_key_prefix, name, balance, tier, rate_limit)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `).bind(id, apiKeyHash, prefix, name, initialBalance, tier, tier === 'pro' ? 300 : tier === 'starter' ? 120 : 60).run();

  return {
    wallet_id: id,
    api_key: apiKey,  // Only shown once!
    api_key_prefix: prefix,
    name,
    balance: initialBalance,
    tier,
  };
}

// ── Authenticate Request ────────────────────────────

export async function authenticateRequest(db, request) {
  // Check X-API-Key header first, then Authorization: Bearer
  let apiKey = request.headers.get('X-API-Key');
  if (!apiKey) {
    const auth = request.headers.get('Authorization');
    if (auth && auth.startsWith('Bearer ')) {
      apiKey = auth.substring(7).trim();
    }
  }
  if (!apiKey) return null;

  const keyHash = await sha256(apiKey);
  const wallet = await db.prepare(
    'SELECT * FROM wallets WHERE api_key = ? AND active = 1'
  ).bind(keyHash).first();

  return wallet || null;
}

// ── Check Balance ───────────────────────────────────

export function checkBalance(wallet, cost) {
  return wallet && wallet.balance >= cost;
}

// ── Deduct Credits ──────────────────────────────────

export async function deductCredits(db, walletId, cost, endpoint, method, status, latencyMs) {
  const logId = crypto.randomUUID();
  await db.batch([
    db.prepare('UPDATE wallets SET balance = balance - ?, updated_at = datetime(\'now\') WHERE id = ?')
      .bind(cost, walletId),
    db.prepare(`
      INSERT INTO usage_log (id, wallet_id, endpoint, credits_used, method, status, latency_ms)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(logId, walletId, endpoint, cost, method, status, latencyMs),
  ]);
}

// ── Get Balance ─────────────────────────────────────

export async function getBalance(db, walletId) {
  const wallet = await db.prepare(
    'SELECT id, api_key_prefix, name, balance, tier, rate_limit, created_at, updated_at FROM wallets WHERE id = ?'
  ).bind(walletId).first();
  return wallet;
}

// ── Get Usage ───────────────────────────────────────

export async function getUsage(db, walletId, options = {}) {
  const days = parseInt(options.days) || 7;
  const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();

  const usage = await db.prepare(`
    SELECT endpoint, COUNT(*) as calls, SUM(credits_used) as total_credits,
           AVG(latency_ms) as avg_latency_ms
    FROM usage_log WHERE wallet_id = ? AND created_at >= ?
    GROUP BY endpoint ORDER BY total_credits DESC
  `).bind(walletId, since).all();

  const dailyUsage = await db.prepare(`
    SELECT DATE(created_at) as day, COUNT(*) as calls, SUM(credits_used) as credits
    FROM usage_log WHERE wallet_id = ? AND created_at >= ?
    GROUP BY DATE(created_at) ORDER BY day DESC
  `).bind(walletId, since).all();

  const totalCredits = await db.prepare(`
    SELECT SUM(credits_used) as total FROM usage_log WHERE wallet_id = ? AND created_at >= ?
  `).bind(walletId, since).first();

  return {
    wallet_id: walletId,
    period_days: days,
    since,
    total_credits_used: totalCredits?.total || 0,
    by_endpoint: usage.results,
    by_day: dailyUsage.results,
  };
}

// ── Topup Wallet ────────────────────────────────────

export async function topupWallet(db, walletId, amount, reference) {
  const logId = crypto.randomUUID();
  await db.batch([
    db.prepare('UPDATE wallets SET balance = balance + ?, updated_at = datetime(\'now\') WHERE id = ?')
      .bind(amount, walletId),
    db.prepare('INSERT INTO topup_log (id, wallet_id, amount, reference) VALUES (?, ?, ?, ?)')
      .bind(logId, walletId, amount, reference || null),
  ]);

  const wallet = await getBalance(db, walletId);
  return { wallet_id: walletId, added: amount, new_balance: wallet?.balance || 0, reference };
}

// ── Wallet Middleware ────────────────────────────────

export async function walletMiddleware(env, request, endpointKey) {
  const cost = ENDPOINT_COSTS[endpointKey] || 1;

  const wallet = await authenticateRequest(env.DB, request);
  if (!wallet) {
    return {
      error: true,
      response: new Response(JSON.stringify({
        error: 'Authentication required',
        message: 'Provide API key via X-API-Key header or Authorization: Bearer',
        docs: 'https://api.router.swarmandbee.com/',
      }), { status: 401, headers: { 'Content-Type': 'application/json' } }),
    };
  }

  if (!checkBalance(wallet, cost)) {
    return {
      error: true,
      response: new Response(JSON.stringify({
        error: 'Insufficient credits',
        balance: wallet.balance,
        required: cost,
        endpoint: endpointKey,
        topup: 'POST /wallet/topup',
      }), { status: 402, headers: { 'Content-Type': 'application/json' } }),
    };
  }

  return { error: false, wallet, cost };
}

// ── Resolve Endpoint Key ────────────────────────────

export function resolveEndpointKey(method, path) {
  if (path.startsWith('/cook/batch')) return 'cook_batch';
  if (path.startsWith('/cook/edgar')) return 'cook_edgar';
  if (path.startsWith('/cook/market')) return 'cook_market';
  if (path.match(/^\/cook\/[A-Z]{2}$/)) return 'cook_state';
  if (path.startsWith('/cook')) return 'cook';

  if (path.match(/^\/skill\/[a-z0-9_]+\/eval$/)) return 'skill_eval';
  if (path.match(/^\/skill\/[a-z0-9_]+\/fail$/)) return 'skill_fail';
  if (path.match(/^\/skill\/[a-z0-9_]+$/)) return 'skill_exec';

  if (path === '/events/query') return 'events_query';
  if (path === '/events/process') return 'events_process';
  if (path.match(/^\/events\/[a-f0-9-]+$/)) return 'event_get';

  if (path === '/entities') return 'entities_query';
  if (path.match(/^\/entities\/[a-f0-9-]+\/timeline$/)) return 'entity_timeline';
  if (path.match(/^\/entities\/[a-f0-9-]+$/)) return 'entity_get';

  if (path === '/memory/search') return 'memory_search';
  if (path === '/memory/context') return 'memory_context';

  if (path === '/pio/search') return 'pio_search';
  if (path === '/pio/feed') return 'pio_feed';
  if (path.startsWith('/pio/batch')) return 'pio_ingest';
  if (method === 'POST' && path === '/pio') return 'pio_ingest';
  if (path.startsWith('/pio/')) return 'pio_get';

  return 'cook'; // default fallback
}
