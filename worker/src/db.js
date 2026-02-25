/**
 * D1 Database Layer — Event Database + Entity Resolution + Wallets
 * ================================================================
 * 8 tables: events, entities, event_entities, memory_index,
 *           market_snapshots, wallets, usage_log, topup_log
 */

// ── Schema ───────────────────────────────────────────

const SCHEMA = `
CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    object_id       TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    category        TEXT NOT NULL,
    source          TEXT,
    ticker          TEXT,
    state           TEXT,
    market          TEXT,
    confidence      REAL DEFAULT 0,
    priority        INTEGER DEFAULT 5,
    price           REAL,
    cap_rate        REAL,
    sf              INTEGER,
    data_json       TEXT,
    processed       INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_category ON events(category);
CREATE INDEX IF NOT EXISTS idx_events_state ON events(state);
CREATE INDEX IF NOT EXISTS idx_events_ticker ON events(ticker);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_events_confidence ON events(confidence);

CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    entity_type     TEXT NOT NULL,
    name            TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    ticker          TEXT,
    state           TEXT,
    address         TEXT,
    metadata_json   TEXT,
    first_seen      TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen       TEXT NOT NULL DEFAULT (datetime('now')),
    event_count     INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized_name);
CREATE INDEX IF NOT EXISTS idx_entities_state ON entities(state);
CREATE INDEX IF NOT EXISTS idx_entities_ticker ON entities(ticker);

CREATE TABLE IF NOT EXISTS event_entities (
    event_id        TEXT NOT NULL,
    entity_id       TEXT NOT NULL,
    role            TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (event_id, entity_id, role)
);
CREATE INDEX IF NOT EXISTS idx_ee_event ON event_entities(event_id);
CREATE INDEX IF NOT EXISTS idx_ee_entity ON event_entities(entity_id);

CREATE TABLE IF NOT EXISTS memory_index (
    id              TEXT PRIMARY KEY,
    object_id       TEXT NOT NULL,
    object_type     TEXT NOT NULL,
    r2_key          TEXT NOT NULL,
    summary         TEXT,
    state           TEXT,
    asset_type      TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_memory_object ON memory_index(object_id);
CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_index(object_type);
CREATE INDEX IF NOT EXISTS idx_memory_state ON memory_index(state);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id              TEXT PRIMARY KEY,
    state           TEXT NOT NULL,
    market          TEXT,
    asset_type      TEXT,
    vacancy_rate    REAL,
    avg_cap_rate    REAL,
    avg_rent_psf    REAL,
    deal_count      INTEGER,
    snapshot_json   TEXT,
    period          TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_snapshots_state ON market_snapshots(state);
CREATE INDEX IF NOT EXISTS idx_snapshots_period ON market_snapshots(period);

CREATE TABLE IF NOT EXISTS wallets (
    id              TEXT PRIMARY KEY,
    api_key         TEXT NOT NULL UNIQUE,
    api_key_prefix  TEXT NOT NULL,
    name            TEXT NOT NULL,
    balance         REAL NOT NULL DEFAULT 0,
    tier            TEXT NOT NULL DEFAULT 'free',
    rate_limit      INTEGER DEFAULT 60,
    active          INTEGER DEFAULT 1,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS usage_log (
    id              TEXT PRIMARY KEY,
    wallet_id       TEXT NOT NULL,
    endpoint        TEXT NOT NULL,
    credits_used    REAL NOT NULL,
    method          TEXT NOT NULL,
    status          INTEGER NOT NULL,
    latency_ms      INTEGER,
    metadata_json   TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_usage_wallet ON usage_log(wallet_id);
CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_log(created_at);

CREATE TABLE IF NOT EXISTS topup_log (
    id              TEXT PRIMARY KEY,
    wallet_id       TEXT NOT NULL,
    amount          REAL NOT NULL,
    reference       TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_topup_wallet ON topup_log(wallet_id);
`;

// ── Init Schema ──────────────────────────────────────

export async function initSchema(db) {
  const statements = SCHEMA.split(';').map(s => s.trim()).filter(s => s.length > 0);
  const batch = statements.map(sql => db.prepare(sql + ';'));
  await db.batch(batch);
  return { tables: 8, indexes: statements.filter(s => s.startsWith('CREATE INDEX')).length };
}

// ── UUID ─────────────────────────────────────────────

function uuid() {
  return crypto.randomUUID();
}

function normalize(name) {
  return (name || '').toLowerCase().replace(/[^a-z0-9\s]/g, '').replace(/\s+/g, ' ').trim();
}

// ── Events ───────────────────────────────────────────

export async function insertEvent(db, event) {
  const id = uuid();
  await db.prepare(`
    INSERT INTO events (id, object_id, event_type, category, source, ticker, state, market,
                        confidence, priority, price, cap_rate, sf, data_json, processed)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).bind(
    id,
    event.object_id || '',
    event.event_type || 'unknown',
    event.category || 'unknown',
    event.source || null,
    event.ticker || null,
    event.state || null,
    event.market || null,
    event.confidence || 0,
    event.priority || 5,
    event.price || null,
    event.cap_rate || null,
    event.sf || null,
    JSON.stringify(event),
    event.processed ? 1 : 0
  ).run();
  return id;
}

export async function queryEvents(db, filters = {}) {
  const clauses = [];
  const params = [];

  if (filters.type) { clauses.push('event_type = ?'); params.push(filters.type); }
  if (filters.category) { clauses.push('category = ?'); params.push(filters.category); }
  if (filters.state) { clauses.push('state = ?'); params.push(filters.state); }
  if (filters.ticker) { clauses.push('ticker = ?'); params.push(filters.ticker); }
  if (filters.source) { clauses.push('source = ?'); params.push(filters.source); }
  if (filters.after) { clauses.push('created_at >= ?'); params.push(filters.after); }
  if (filters.before) { clauses.push('created_at <= ?'); params.push(filters.before); }
  if (filters.min_confidence) { clauses.push('confidence >= ?'); params.push(parseFloat(filters.min_confidence)); }

  const where = clauses.length > 0 ? `WHERE ${clauses.join(' AND ')}` : '';
  const limit = Math.min(parseInt(filters.limit) || 50, 200);
  const offset = parseInt(filters.offset) || 0;

  const sql = `SELECT * FROM events ${where} ORDER BY created_at DESC LIMIT ? OFFSET ?`;
  params.push(limit, offset);

  const result = await db.prepare(sql).bind(...params).all();

  const countSql = `SELECT COUNT(*) as total FROM events ${where}`;
  const countParams = params.slice(0, -2);
  const countResult = await db.prepare(countSql).bind(...countParams).first();

  return {
    events: result.results.map(row => ({
      ...row,
      data: row.data_json ? JSON.parse(row.data_json) : null,
    })),
    total: countResult?.total || 0,
    limit,
    offset,
  };
}

export async function getEvent(db, id) {
  const event = await db.prepare('SELECT * FROM events WHERE id = ?').bind(id).first();
  if (!event) return null;

  const entities = await db.prepare(`
    SELECT e.*, ee.role FROM entities e
    JOIN event_entities ee ON ee.entity_id = e.id
    WHERE ee.event_id = ?
  `).bind(id).all();

  return {
    ...event,
    data: event.data_json ? JSON.parse(event.data_json) : null,
    entities: entities.results,
  };
}

export async function getEventStats(db) {
  const byType = await db.prepare(`
    SELECT event_type, COUNT(*) as count FROM events GROUP BY event_type ORDER BY count DESC
  `).all();
  const byCategory = await db.prepare(`
    SELECT category, COUNT(*) as count FROM events GROUP BY category ORDER BY count DESC
  `).all();
  const byState = await db.prepare(`
    SELECT state, COUNT(*) as count FROM events WHERE state IS NOT NULL GROUP BY state ORDER BY count DESC LIMIT 20
  `).all();
  const total = await db.prepare('SELECT COUNT(*) as total FROM events').first();
  const latest = await db.prepare('SELECT created_at FROM events ORDER BY created_at DESC LIMIT 1').first();

  return {
    total_events: total?.total || 0,
    latest_event: latest?.created_at || null,
    by_type: byType.results,
    by_category: byCategory.results,
    by_state: byState.results,
  };
}

// ── Entities ─────────────────────────────────────────

export async function upsertEntity(db, entity) {
  const normalizedName = normalize(entity.name);
  const existing = await db.prepare(
    'SELECT id, event_count FROM entities WHERE entity_type = ? AND normalized_name = ?'
  ).bind(entity.entity_type, normalizedName).first();

  if (existing) {
    await db.prepare(`
      UPDATE entities SET last_seen = datetime('now'), event_count = event_count + 1,
        metadata_json = COALESCE(?, metadata_json)
      WHERE id = ?
    `).bind(
      entity.metadata_json ? JSON.stringify(entity.metadata_json) : null,
      existing.id
    ).run();
    return existing.id;
  }

  const id = uuid();
  await db.prepare(`
    INSERT INTO entities (id, entity_type, name, normalized_name, ticker, state, address, metadata_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `).bind(
    id,
    entity.entity_type,
    entity.name,
    normalizedName,
    entity.ticker || null,
    entity.state || null,
    entity.address || null,
    entity.metadata_json ? JSON.stringify(entity.metadata_json) : null
  ).run();
  return id;
}

export async function linkEventEntity(db, eventId, entityId, role) {
  await db.prepare(`
    INSERT OR IGNORE INTO event_entities (event_id, entity_id, role) VALUES (?, ?, ?)
  `).bind(eventId, entityId, role).run();
}

export async function queryEntities(db, filters = {}) {
  const clauses = [];
  const params = [];

  if (filters.type) { clauses.push('entity_type = ?'); params.push(filters.type); }
  if (filters.state) { clauses.push('state = ?'); params.push(filters.state); }
  if (filters.ticker) { clauses.push('ticker = ?'); params.push(filters.ticker); }
  if (filters.q) { clauses.push('normalized_name LIKE ?'); params.push(`%${normalize(filters.q)}%`); }

  const where = clauses.length > 0 ? `WHERE ${clauses.join(' AND ')}` : '';
  const limit = Math.min(parseInt(filters.limit) || 50, 200);

  const sql = `SELECT * FROM entities ${where} ORDER BY event_count DESC, last_seen DESC LIMIT ?`;
  params.push(limit);

  const result = await db.prepare(sql).bind(...params).all();
  return result.results;
}

export async function getEntity(db, id) {
  const entity = await db.prepare('SELECT * FROM entities WHERE id = ?').bind(id).first();
  if (!entity) return null;
  return entity;
}

export async function getEntityTimeline(db, entityId, limit = 50) {
  const events = await db.prepare(`
    SELECT e.*, ee.role FROM events e
    JOIN event_entities ee ON ee.event_id = e.id
    WHERE ee.entity_id = ?
    ORDER BY e.created_at DESC LIMIT ?
  `).bind(entityId, limit).all();

  return events.results.map(row => ({
    ...row,
    data: row.data_json ? JSON.parse(row.data_json) : null,
  }));
}

// ── Entity Extraction from Events ────────────────────

export function extractEntities(event) {
  const entities = [];
  const data = typeof event === 'string' ? {} : event;

  // REIT / company
  if (data.ticker || data.company) {
    entities.push({
      entity_type: 'reit',
      name: data.company || data.ticker,
      ticker: data.ticker,
      state: data.state,
    });
  }

  // Property
  if (data.property || data.address) {
    entities.push({
      entity_type: 'property',
      name: data.property || data.address,
      state: data.state,
      address: data.address,
    });
  }

  // Buyer / seller
  if (data.buyer) entities.push({ entity_type: 'investor', name: data.buyer, state: data.state });
  if (data.seller) entities.push({ entity_type: 'investor', name: data.seller, state: data.state });

  // Tenant
  if (data.tenant) entities.push({ entity_type: 'tenant', name: data.tenant, state: data.state });

  // Broker
  if (data.broker) entities.push({ entity_type: 'broker', name: data.broker, state: data.state });

  return entities;
}

// ── Memory Index ─────────────────────────────────────

export async function insertMemoryIndex(db, record) {
  await db.prepare(`
    INSERT OR REPLACE INTO memory_index (id, object_id, object_type, r2_key, summary, state, asset_type)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `).bind(
    record.id,
    record.object_id,
    record.object_type || 'unknown',
    record.r2_key || '',
    record.summary || null,
    record.state || null,
    record.asset_type || null
  ).run();
}

export async function getMemoryStats(db) {
  const total = await db.prepare('SELECT COUNT(*) as total FROM memory_index').first();
  const byType = await db.prepare(
    'SELECT object_type, COUNT(*) as count FROM memory_index GROUP BY object_type ORDER BY count DESC'
  ).all();
  return { total_indexed: total?.total || 0, by_type: byType.results };
}

// ── Market Snapshots ─────────────────────────────────

export async function insertSnapshot(db, snapshot) {
  const id = uuid();
  await db.prepare(`
    INSERT INTO market_snapshots (id, state, market, asset_type, vacancy_rate, avg_cap_rate, avg_rent_psf, deal_count, snapshot_json, period)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).bind(
    id, snapshot.state, snapshot.market || null, snapshot.asset_type || null,
    snapshot.vacancy_rate || null, snapshot.avg_cap_rate || null, snapshot.avg_rent_psf || null,
    snapshot.deal_count || null, JSON.stringify(snapshot), snapshot.period
  ).run();
  return id;
}

export async function querySnapshots(db, state, periods = 4) {
  const result = await db.prepare(
    'SELECT * FROM market_snapshots WHERE state = ? ORDER BY period DESC LIMIT ?'
  ).bind(state, periods).all();
  return result.results;
}
