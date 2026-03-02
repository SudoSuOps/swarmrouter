-- model_builds: isolated endcap build registry
-- Each row = one sealed, versioned training dataset deliverable
-- NOT mixed with any other data tables

CREATE TABLE IF NOT EXISTS model_builds (
    id              BIGSERIAL PRIMARY KEY,
    build_id        TEXT NOT NULL UNIQUE,   -- e.g. swarmrouter-4b0-v1-20260302_174500
    model_name      TEXT NOT NULL,          -- e.g. swarmrouter-4b0
    version         TEXT NOT NULL,          -- e.g. v1
    sealed_at       TIMESTAMPTZ NOT NULL,
    train_pairs     INTEGER NOT NULL,
    eval_pairs      INTEGER NOT NULL,
    sha256_train    TEXT NOT NULL,
    sha256_eval     TEXT NOT NULL,
    manifest        JSONB NOT NULL,         -- full manifest including domain/model dist
    domain_dist     JSONB,
    model_dist      JSONB,
    status          TEXT NOT NULL DEFAULT 'sealed',  -- sealed | training | trained | deployed
    r2_bucket       TEXT NOT NULL DEFAULT 'sb-builds',
    r2_prefix       TEXT NOT NULL,          -- path prefix in R2
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_builds_model_name ON model_builds(model_name);
CREATE INDEX IF NOT EXISTS idx_model_builds_status ON model_builds(status);

ALTER TABLE model_builds ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_role_only" ON model_builds
    USING (auth.role() = 'service_role');
