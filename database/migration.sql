-- ============================================================
-- AI Pipeline — Supabase Database Migration
-- Run this in your Supabase SQL editor:
--   supabase.com → your project → SQL Editor → New Query → paste → Run
-- ============================================================


-- ── Main pipeline runs table ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    filename        TEXT NOT NULL,
    size_bytes      BIGINT DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending',
    quality_score   FLOAT,
    error_message   TEXT,

    -- Full result stored as JSONB so we can query inside it later
    result_json     JSONB,

    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);


-- ── Indexes for fast lookups ──────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_runs_run_id    ON pipeline_runs (run_id);
CREATE INDEX IF NOT EXISTS idx_runs_status    ON pipeline_runs (status);
CREATE INDEX IF NOT EXISTS idx_runs_created   ON pipeline_runs (created_at DESC);


-- ── Auto-update updated_at on every row change ────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_pipeline_runs_updated ON pipeline_runs;
CREATE TRIGGER trg_pipeline_runs_updated
    BEFORE UPDATE ON pipeline_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();


-- ── Row Level Security (RLS) ──────────────────────────────────────────────────
-- This allows your backend (using the service role key) full access
-- while the anon key can only READ rows.
ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;

-- Allow backend (authenticated / service role) full access
CREATE POLICY "service_full_access" ON pipeline_runs
    FOR ALL USING (true);

-- Allow anonymous read (for frontend status polling)
CREATE POLICY "anon_read" ON pipeline_runs
    FOR SELECT USING (true);


-- ── Verification query — run after migration ──────────────────────────────────
-- SELECT column_name, data_type FROM information_schema.columns
-- WHERE table_name = 'pipeline_runs'
-- ORDER BY ordinal_position;
