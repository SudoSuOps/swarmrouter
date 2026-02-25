# SwarmRouter — Intelligence Objects Platform

## What This Is

Edge-native commercial real estate intelligence platform. 19 composable AI skills, event machine, semantic memory, 50 state area experts, wallet-metered API service. Live at `router.swarmandbee.com` and `api.router.swarmandbee.com`.

## Architecture

```
Cloudflare Workers (edge, 300+ cities)
├── AI:        Qwen3-30B-A3B-FP8 primary, Llama-3.2-3B fallback
├── R2:        sb-intelligence (objects), sb-medical, sb-aviation, sb-cre, sb-core
├── D1:        swarm-intelligence-db (events, entities, wallets, memory index)
├── Vectorize: swarm-memory (768-dim BGE-Base embeddings, cosine)
└── Worker:    worker/src/index.js (main router)
```

## Key Directories

- `worker/` — Cloudflare Worker (the live API)
  - `worker/src/index.js` — Main router, all endpoints
  - `worker/src/skills/` — 19 skill modules + registry + schemas + mocks + eval
  - `worker/src/db.js` — D1 schema, event CRUD, entity resolution
  - `worker/src/memory.js` — Vectorize embeddings, semantic search, temporal context
  - `worker/src/wallet.js` — API keys, credits, metering
  - `worker/src/api_docs.js` — API documentation endpoint
  - `worker/src/event_machine.js` — 22 event types, signal → Intelligence Object pipeline
  - `worker/src/states.js` — 50 state area expert prompts
  - `worker/src/market_heat.js` — Market tier classification
  - `worker/src/infrastructure.js` — Ports, rail, power, last-mile, air cargo
- `data/` — Training data factory
  - `data/swarmcre_dataset/` — SwarmCRE 1M-pair dataset factory
  - `data/swarmcre_dataset/output/` — Generated training data (gitignored)
- `edgar_pump.py` — SEC EDGAR → Event Machine feed (15 REIT tickers)
- `intelligence_cooker.py` — Batch cooker for R2 objects
- `pio.py` — Intelligence Object CLI

## Worker Commands

```bash
# Deploy
cd worker && npx wrangler deploy --remote

# Tail logs
cd worker && npx wrangler tail --remote

# D1 queries
npx wrangler d1 execute swarm-intelligence-db --remote --command "SELECT COUNT(*) FROM events"

# Secrets
npx wrangler secret put ADMIN_KEY --remote
```

## Skills (19)

broker_senior, broker_junior, intelligence_query, bookmaker, deal_tracker, developer, signal_scraper, investor, exchange_1031, market_report, lead_scorer, email_composer, comp_analyzer, rent_roll_analyzer, debt_analyzer, tax_assessor, site_selector, portfolio_optimizer, news_digest

Each skill: system prompt → edge AI → JSON validation → R2 storage. Test with `POST /skills/test` (all 19 mock validation).

## Dataset Factory

```bash
# Build 1M training pairs (100K deals × 10 tasks)
cd /home/swarm/Desktop/swarmrouter
python3 -m data.swarmcre_dataset.make_swarmcre --deals 100000 --shards 8

# Enrich via Together.ai (ic_memo + lease_reasoning)
TOGETHER_API_KEY=... python3 data/swarmcre_dataset/enrich_pass.py

# Validate
python3 -m data.swarmcre_dataset.make_swarmcre --validate
```

## Conventions

- **Edge model**: Always `@cf/qwen/qwen3-30b-a3b-fp8` primary, `@cf/meta/llama-3.2-3b-instruct` fallback
- **Embedding model**: `@cf/baai/bge-base-en-v1.5` (768 dimensions)
- **Skill pattern**: Each skill is a separate file in `worker/src/skills/` exporting `{ name, version, role, description, systemPrompt, examples }`
- **API keys**: `sb_live_` prefix + 32-char hex, SHA-256 hashed in D1
- **Credit costs**: cook=1, skill=2, query=0.5, memory=1, event=1
- **Object IDs**: `pio_` prefix for Intelligence Objects, `evt_` for events, `mem_` for memory vectors
- **R2 paths**: `pio/{object_id}` for objects, `pio/skills/{name}/{id}` for skill outputs
- **Dual domain**: `router.swarmandbee.com` (free/internal), `api.router.swarmandbee.com` (metered)
- **No breaking changes** to existing endpoints — additive only
- **Wrangler**: Always use `npx wrangler` with `--remote` flag
- **Cloudflare Account**: 6abec5e82728df0610a98be9364918e4

## Testing

```bash
# Health check
curl https://router.swarmandbee.com/health

# All 19 skills pass mock validation
curl -X POST https://router.swarmandbee.com/skills/test

# Execute a skill
curl -X POST https://router.swarmandbee.com/skill/broker_senior \
  -H "Content-Type: application/json" \
  -d '{"data":{"address":"Alliance TX warehouse","sf":85000}}'

# Semantic memory search
curl -X POST https://router.swarmandbee.com/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query":"cold storage Florida","limit":5}'

# Event database query
curl "https://router.swarmandbee.com/events/query?type=reit_filing&state=TX"

# API docs (metered domain)
curl https://api.router.swarmandbee.com/
```
