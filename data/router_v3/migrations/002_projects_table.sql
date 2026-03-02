-- projects: master project registry
-- Each project = a named, versioned ML build program
-- Every project gets a unique project_id (the "property tax ID")
-- All builds, datasets, and artifacts reference this ID

CREATE TABLE IF NOT EXISTS projects (
    id                BIGSERIAL PRIMARY KEY,
    project_id        TEXT NOT NULL UNIQUE,    -- e.g. PRJ-001-SWARMROUTER
    project_name      TEXT NOT NULL,           -- e.g. Project:SwarmRouter
    status            TEXT NOT NULL DEFAULT 'draft',
                                               -- draft | active | completed | archived
    phase             TEXT,                    -- current active phase
    description       TEXT,
    om_r2_path        TEXT,                    -- R2 path to signed OM document
    om_signed_at      TIMESTAMPTZ,
    om_signed_by      TEXT,
    phases            JSONB,                   -- full phase plan
    deliverables      JSONB,                   -- deliverables per phase
    model_lineup      JSONB,                   -- all models in this project
    infrastructure    JSONB,                   -- hardware, buckets, tables
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- build_phases: individual phase records under a project
CREATE TABLE IF NOT EXISTS build_phases (
    id                BIGSERIAL PRIMARY KEY,
    project_id        TEXT NOT NULL REFERENCES projects(project_id),
    phase_id          TEXT NOT NULL UNIQUE,    -- e.g. PRJ-001-PHASE-1
    phase_name        TEXT NOT NULL,           -- e.g. SwarmRouter-4B-0 Block Zero
    phase_number      INTEGER NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending',
                                               -- pending | active | completed | failed
    proposal_path     TEXT,                    -- R2 path to build proposal
    build_id          TEXT,                    -- references model_builds.build_id
    started_at        TIMESTAMPTZ,
    completed_at      TIMESTAMPTZ,
    acceptance_results JSONB,
    notes             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_build_phases_project ON build_phases(project_id);
CREATE INDEX IF NOT EXISTS idx_build_phases_status ON build_phases(status);

ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE build_phases ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_role_only" ON projects
    USING (auth.role() = 'service_role');
CREATE POLICY "service_role_only" ON build_phases
    USING (auth.role() = 'service_role');
