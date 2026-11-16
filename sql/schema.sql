-- ============================================================================
-- Supabase PostgreSQL Schema for UES Family Rental Real Estate Agent
-- Run this inside the Supabase SQL Editor (100% Free Tier)
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 1. Evaluated Listings Table (Deduplication + Score Audit Trail)
CREATE TABLE IF NOT EXISTS public.listings (
    listing_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    address TEXT NOT NULL,
    unit TEXT DEFAULT '',
    neighborhood TEXT NOT NULL,
    price INTEGER NOT NULL,
    bedrooms NUMERIC(3, 1) NOT NULL,
    bathrooms NUMERIC(3, 1) DEFAULT 1.0,
    sqft INTEGER,
    floor_info TEXT DEFAULT '',
    description TEXT DEFAULT '',
    amenities JSONB DEFAULT '[]'::jsonb,
    image_urls JSONB DEFAULT '[]'::jsonb,
    floorplan_urls JSONB DEFAULT '[]'::jsonb,
    overall_score INTEGER NOT NULL DEFAULT 0,
    light_score INTEGER NOT NULL DEFAULT 0,
    kitchen_layout_score INTEGER NOT NULL DEFAULT 0,
    has_open_island_or_breakfast_bar BOOLEAN NOT NULL DEFAULT FALSE,
    has_high_natural_light BOOLEAN NOT NULL DEFAULT FALSE,
    disqualifiers_detected JSONB DEFAULT '[]'::jsonb,
    evaluation_json JSONB DEFAULT '{}'::jsonb,
    alerted BOOLEAN NOT NULL DEFAULT FALSE,
    telegram_message_id BIGINT,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_listings_overall_score ON public.listings (overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_listings_alerted ON public.listings (alerted);
CREATE INDEX IF NOT EXISTS idx_listings_neighborhood ON public.listings (neighborhood);

-- 2. User Feedback & Active Training Memory Log
CREATE TABLE IF NOT EXISTS public.feedback_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    listing_id TEXT NOT NULL REFERENCES public.listings(listing_id) ON DELETE CASCADE,
    vote TEXT NOT NULL CHECK (vote IN ('YES', 'NO')),
    reason TEXT DEFAULT '',
    feedback_source TEXT NOT NULL DEFAULT 'inline_button', -- 'inline_button', 'text_reply', 'voice_note'
    listing_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_memory_created_at ON public.feedback_memory (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_memory_vote ON public.feedback_memory (vote);
CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_memory_unique_listing ON public.feedback_memory (listing_id);

-- 3. Pending Feedback State (Tracks when user clicks [ NO 👎 ] awaiting text/voice explanation)
CREATE TABLE IF NOT EXISTS public.pending_feedback_prompts (
    chat_id BIGINT PRIMARY KEY,
    listing_id TEXT NOT NULL REFERENCES public.listings(listing_id) ON DELETE CASCADE,
    prompt_message_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Enable Row Level Security (RLS) so only service role / authorized key can access
ALTER TABLE public.listings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.feedback_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.pending_feedback_prompts ENABLE ROW LEVEL SECURITY;
