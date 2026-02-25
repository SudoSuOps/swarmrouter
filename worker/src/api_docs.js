/**
 * API Documentation — api.router.swarmandbee.com
 * ===============================================
 * Full API docs rendered as JSON at the root endpoint.
 */

export const API_VERSION = '2.0.0';

export function apiDocsResponse() {
  return {
    name: 'Swarm & Bee CRE Intelligence API',
    version: API_VERSION,
    description: 'Edge-native commercial real estate intelligence platform. 19 composable skills, event machine, semantic memory, 50 state area experts. Powered by Qwen3-30B-A3B on Cloudflare Workers.',
    base_url: 'https://api.router.swarmandbee.com',
    authentication: {
      method: 'API Key',
      header: 'X-API-Key: sb_live_...',
      alternative: 'Authorization: Bearer sb_live_...',
      get_key: 'Contact admin or POST /wallet/create (admin)',
    },
    credit_system: {
      description: 'All metered endpoints deduct credits from your wallet balance.',
      costs: {
        cook: '1 credit — AI inference + R2 storage',
        skill_execution: '2 credits — AI + schema validation + R2',
        query: '0.5 credits — D1 read operations',
        memory_search: '1 credit — embedding + Vectorize search',
        event_processing: '1 credit per event',
      },
      response_headers: {
        'X-Credits-Used': 'Credits deducted for this request',
        'X-Credits-Remaining': 'Current wallet balance',
      },
    },
    endpoints: {
      public: {
        description: 'No API key required',
        routes: [
          { method: 'GET', path: '/', description: 'This documentation' },
          { method: 'GET', path: '/health', description: 'Health check' },
          { method: 'GET', path: '/skills', description: 'List all 19 skills' },
          { method: 'GET', path: '/events', description: 'Event Machine architecture' },
          { method: 'GET', path: '/events/types', description: 'All 22 event types' },
          { method: 'GET', path: '/events/stats/db', description: 'Event database statistics' },
          { method: 'GET', path: '/memory/stats', description: 'Memory index statistics' },
          { method: 'GET', path: '/states', description: '50 state area experts' },
          { method: 'GET', path: '/markets', description: 'Market heat index (Tier 1-4)' },
          { method: 'GET', path: '/infrastructure', description: 'Infrastructure map (ports, rail, power)' },
        ],
      },
      cook: {
        description: 'Edge AI inference — raw data → Intelligence Objects',
        cost: '1 credit',
        routes: [
          { method: 'POST', path: '/cook', description: 'Cook raw CRE data', body: '{ "data": {...}, "context": "..." }' },
          { method: 'POST', path: '/cook/batch', description: 'Batch cook (max 25)', body: '{ "items": [{...}] }' },
          { method: 'POST', path: '/cook/edgar', description: 'Cook EDGAR REIT filing', body: '{ "ticker": "PLD", "financials": {...} }' },
          { method: 'POST', path: '/cook/{STATE}', description: 'State expert cook (e.g., /cook/TX)', body: '{ "data": {...} }' },
        ],
      },
      skills: {
        description: 'Composable, schema-validated intelligence operations',
        routes: [
          { method: 'POST', path: '/skill/{name}', description: 'Execute skill (2 credits)', body: '{ "data": {...}, "context": "..." }' },
          { method: 'GET', path: '/skill/{name}/spec', description: 'Skill spec (system prompt, schema)' },
          { method: 'GET', path: '/skill/{name}/mock', description: 'Mock response (no AI call)' },
          { method: 'POST', path: '/skill/{name}/test', description: 'Test harness (mock + validate)' },
          { method: 'POST', path: '/skill/{name}/eval', description: 'Eval (real AI + metrics)' },
        ],
        available_skills: [
          'broker_senior', 'broker_junior', 'intelligence_query', 'bookmaker',
          'deal_tracker', 'developer', 'signal_scraper', 'investor', 'exchange_1031',
          'market_report', 'lead_scorer', 'email_composer', 'comp_analyzer',
          'rent_roll_analyzer', 'debt_analyzer', 'tax_assessor', 'site_selector',
          'portfolio_optimizer', 'news_digest',
        ],
      },
      events: {
        description: 'Event Machine — signals to Intelligence Objects',
        routes: [
          { method: 'POST', path: '/events/process', description: 'Process events (1 credit/event, max 100)', body: '{ "events": [{...}] }' },
          { method: 'GET', path: '/events/query', description: 'Query events (0.5 credits)', params: 'type, category, state, ticker, after, before, limit' },
          { method: 'GET', path: '/events/{id}', description: 'Single event + linked entities' },
          { method: 'GET', path: '/events/types', description: 'All 22 event types (public)' },
          { method: 'GET', path: '/events/schedule', description: 'Feed schedule (public)' },
        ],
      },
      entities: {
        description: 'Entity resolution — properties, investors, tenants, brokers',
        routes: [
          { method: 'GET', path: '/entities', description: 'Search entities (0.5 credits)', params: 'type, q, state, ticker, limit' },
          { method: 'GET', path: '/entities/{id}', description: 'Entity detail' },
          { method: 'GET', path: '/entities/{id}/timeline', description: 'Entity event timeline' },
        ],
      },
      memory: {
        description: 'Semantic memory — embedding search + temporal reasoning',
        routes: [
          { method: 'POST', path: '/memory/search', description: 'Semantic search (1 credit)', body: '{ "query": "cold storage in Florida", "limit": 10 }' },
          { method: 'POST', path: '/memory/context', description: 'Temporal context (1 credit)', body: '{ "query": "Savannah logistics", "timeframe": "6m" }' },
          { method: 'GET', path: '/memory/stats', description: 'Index statistics (public)' },
        ],
      },
      wallet: {
        description: 'API key management and credit balance',
        routes: [
          { method: 'GET', path: '/wallet/balance', description: 'Current balance (requires API key)' },
          { method: 'GET', path: '/wallet/usage', description: 'Usage history', params: 'days (default 7)' },
          { method: 'POST', path: '/wallet/create', description: 'Create wallet (admin only)', body: '{ "name": "ACME Corp", "tier": "starter" }' },
          { method: 'POST', path: '/wallet/topup', description: 'Add credits (admin only)', body: '{ "wallet_id": "...", "amount": 100 }' },
        ],
      },
    },
    examples: {
      cook_property: {
        curl: 'curl -X POST https://api.router.swarmandbee.com/cook -H "X-API-Key: sb_live_..." -H "Content-Type: application/json" -d \'{"data":{"address":"1200 Commerce Dr, Dallas, TX","sf":85000,"asset_type":"warehouse"}}\'',
      },
      skill_execution: {
        curl: 'curl -X POST https://api.router.swarmandbee.com/skill/broker_senior -H "X-API-Key: sb_live_..." -H "Content-Type: application/json" -d \'{"data":{"address":"Alliance TX warehouse","sf":85000,"tenant":"FedEx","cap_rate":0.055}}\'',
      },
      memory_search: {
        curl: 'curl -X POST https://api.router.swarmandbee.com/memory/search -H "X-API-Key: sb_live_..." -H "Content-Type: application/json" -d \'{"query":"cold storage properties near major ports","limit":5}\'',
      },
    },
    infrastructure: {
      runtime: 'Cloudflare Workers (edge, 300+ cities)',
      ai_model: 'Qwen3-30B-A3B-FP8 (30B MoE, 3B active)',
      storage: 'R2 (objects) + D1 (structured queries) + Vectorize (semantic search)',
      cost_per_object: '~$0.0002',
    },
  };
}
