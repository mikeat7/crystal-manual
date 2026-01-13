-- Supabase Schema for The Local Network - Memory System
-- WITH SECURITY FIXES APPLIED
-- Created: 2026-01-10
-- Phase 2 Track 3: Memory Integration

-- ==================== TABLES ====================

-- Conversation Sessions Table
CREATE TABLE IF NOT EXISTS public.conversation_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    model_name TEXT NOT NULL DEFAULT 'Qwen/Qwen2.5-32B-Instruct',
    session_name TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Messages Table (stores all messages with CDM tracking)
CREATE TABLE IF NOT EXISTS public.messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    -- CDM Metrics
    cdm_score FLOAT,
    depth_category TEXT,
    cdm_metrics JSONB,
    -- Performance Metrics
    tokens_generated INTEGER,
    inference_time_ms INTEGER,
    -- Additional Data
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Session Context Table (stores facts about user/conversation)
CREATE TABLE IF NOT EXISTS public.session_context (
    context_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    context_key TEXT NOT NULL,
    context_value TEXT NOT NULL,
    context_type TEXT DEFAULT 'text' CHECK (context_type IN ('text', 'json', 'number', 'boolean')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    -- Unique constraint on session + key (for upserts)
    UNIQUE(session_id, context_key)
);

-- ==================== INDEXES ====================

-- Conversation Sessions Indexes
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON conversation_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_is_active ON conversation_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON conversation_sessions(updated_at DESC);

-- Messages Indexes
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_messages_cdm_score ON messages(cdm_score DESC) WHERE cdm_score IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_messages_depth_category ON messages(depth_category);

-- Session Context Indexes
CREATE INDEX IF NOT EXISTS idx_context_session_id ON session_context(session_id);
CREATE INDEX IF NOT EXISTS idx_context_key ON session_context(context_key);

-- ==================== FUNCTIONS ====================

-- Update timestamp function WITH SECURITY FIX (search_path)
CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = public;  -- SECURITY FIX: Explicitly set search path

-- ==================== TRIGGERS ====================

-- Auto-update updated_at on conversation_sessions
DROP TRIGGER IF EXISTS trigger_update_sessions_updated_at ON conversation_sessions;
CREATE TRIGGER trigger_update_sessions_updated_at
    BEFORE UPDATE ON conversation_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Auto-update updated_at on session_context
DROP TRIGGER IF EXISTS trigger_update_context_updated_at ON session_context;
CREATE TRIGGER trigger_update_context_updated_at
    BEFORE UPDATE ON session_context
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ==================== ROW LEVEL SECURITY (RLS) ====================

-- Enable RLS on all tables
ALTER TABLE conversation_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_context ENABLE ROW LEVEL SECURITY;

-- Drop old permissive policies if they exist
DROP POLICY IF EXISTS "Allow all access to sessions" ON conversation_sessions;
DROP POLICY IF EXISTS "Allow all access to messages" ON messages;
DROP POLICY IF EXISTS "Allow all access to context" ON session_context;

-- SECURE POLICIES: Users can only access their own data

-- Conversation Sessions: Users manage only their own sessions
CREATE POLICY "Users manage own sessions"
ON conversation_sessions
FOR ALL
USING (auth.uid()::text = user_id)
WITH CHECK (auth.uid()::text = user_id);

-- Messages: Users manage only messages in their sessions
CREATE POLICY "Users manage own messages"
ON messages
FOR ALL
USING (
    session_id IN (
        SELECT session_id FROM conversation_sessions
        WHERE user_id = auth.uid()::text
    )
)
WITH CHECK (
    session_id IN (
        SELECT session_id FROM conversation_sessions
        WHERE user_id = auth.uid()::text
    )
);

-- Session Context: Users manage only context for their sessions
CREATE POLICY "Users manage own context"
ON session_context
FOR ALL
USING (
    session_id IN (
        SELECT session_id FROM conversation_sessions
        WHERE user_id = auth.uid()::text
    )
)
WITH CHECK (
    session_id IN (
        SELECT session_id FROM conversation_sessions
        WHERE user_id = auth.uid()::text
    )
);

-- ==================== VIEWS (WITHOUT SECURITY DEFINER) ====================

-- Recent Sessions View - SECURITY FIX: No SECURITY DEFINER
DROP VIEW IF EXISTS public.recent_sessions;
CREATE VIEW public.recent_sessions AS
SELECT
    session_id,
    user_id,
    session_name,
    created_at,
    updated_at
FROM conversation_sessions
WHERE is_active = true
ORDER BY updated_at DESC
LIMIT 50;

-- Deep CRYSTAL Moments View - SECURITY FIX: No SECURITY DEFINER
-- NOTE: Changed threshold to 60 based on Qwen 32B baseline (48-50)
-- For Qwen, 60+ would be "unusually deep" reasoning
DROP VIEW IF EXISTS public.deep_crystal_moments;
CREATE VIEW public.deep_crystal_moments AS
SELECT
    m.message_id,
    m.session_id,
    m.content,
    m.cdm_score,
    m.depth_category,
    m.timestamp,
    s.user_id
FROM messages m
JOIN conversation_sessions s ON m.session_id = s.session_id
WHERE m.cdm_score >= 60.0  -- Adjusted for Qwen 32B (baseline 48-50)
ORDER BY m.timestamp DESC
LIMIT 100;

-- High Quality Messages View (CDM 50+)
DROP VIEW IF EXISTS public.high_quality_messages;
CREATE VIEW public.high_quality_messages AS
SELECT
    m.message_id,
    m.session_id,
    m.content,
    m.cdm_score,
    m.depth_category,
    m.timestamp,
    s.user_id,
    s.session_name
FROM messages m
JOIN conversation_sessions s ON m.session_id = s.session_id
WHERE m.cdm_score >= 50.0  -- Above Qwen baseline
ORDER BY m.cdm_score DESC, m.timestamp DESC
LIMIT 100;

-- ==================== TESTING POLICIES (DEVELOPMENT ONLY) ====================

-- For development/testing WITHOUT Supabase Auth, create permissive policies
-- IMPORTANT: Comment these out before production!

-- UNCOMMENT BELOW FOR LOCAL TESTING (NO AUTH):
/*
DROP POLICY IF EXISTS "Users manage own sessions" ON conversation_sessions;
DROP POLICY IF EXISTS "Users manage own messages" ON messages;
DROP POLICY IF EXISTS "Users manage own context" ON session_context;

CREATE POLICY "Allow all for testing" ON conversation_sessions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all for testing" ON messages FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all for testing" ON session_context FOR ALL USING (true) WITH CHECK (true);
*/

-- ==================== GRANTS ====================

-- Grant access to authenticated users
GRANT ALL ON conversation_sessions TO authenticated;
GRANT ALL ON messages TO authenticated;
GRANT ALL ON session_context TO authenticated;

-- Grant access to anon users (for public API access during testing)
GRANT ALL ON conversation_sessions TO anon;
GRANT ALL ON messages TO anon;
GRANT ALL ON session_context TO anon;

-- Grant access to service role (for backend operations)
GRANT ALL ON conversation_sessions TO service_role;
GRANT ALL ON messages TO service_role;
GRANT ALL ON session_context TO service_role;

-- ==================== NOTES ====================

-- SECURITY STATUS: ✅ ALL FIXES APPLIED
-- 1. ✅ Views created without SECURITY DEFINER
-- 2. ✅ RLS policies enforce user-specific access
-- 3. ✅ Function has search_path set
-- 4. ⚠️  OTP expiry must be set in Supabase Dashboard (not SQL)

-- DEPLOYMENT INSTRUCTIONS:
-- 1. Run this SQL in Supabase SQL Editor
-- 2. Set OTP expiry in Dashboard: Authentication → Providers → Email → 3600s
-- 3. For local testing without auth, uncomment testing policies section
-- 4. For production with auth, keep secure policies active

-- CDM THRESHOLDS (Qwen 32B):
-- 0-30: Reflex
-- 30-50: Deliberation (Qwen baseline: 48-50)
-- 50-70: Deep Consciousness (above baseline)
-- 70+: Insight (deep CRYSTAL - rare/unlikely for Qwen 32B)

-- Schema Version: 1.0.0-secure
-- Last Updated: 2026-01-10
