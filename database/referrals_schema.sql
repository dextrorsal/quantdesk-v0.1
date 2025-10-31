-- Users, referrals, auth nonces, and chat tables (Supabase/Postgres)

create table if not exists public.users (
	id uuid primary key default gen_random_uuid(),
	wallet_pubkey text unique not null,
	email text,
	username text,
	avatar_url text,
	role text not null default 'tester',
	referrer_pubkey text,
	created_at timestamptz not null default now()
);

create table if not exists public.referrals (
	id uuid primary key default gen_random_uuid(),
	referrer_pubkey text not null,
	referee_pubkey text not null unique,
	activated boolean not null default false,
	activated_at timestamptz,
	created_at timestamptz not null default now()
);

create table if not exists public.auth_nonces (
	wallet_pubkey text primary key,
	nonce text not null,
	created_at timestamptz not null default now()
);

create table if not exists public.chat_messages (
	id bigserial primary key,
	channel text not null default 'global',
	author_pubkey text not null,
	message text not null,
	created_at timestamptz not null default now()
);

-- Simple indexes
create index if not exists idx_users_wallet on public.users (wallet_pubkey);
create index if not exists idx_referrals_referrer on public.referrals (referrer_pubkey);
create index if not exists idx_chat_channel_time on public.chat_messages (channel, created_at desc);

-- RLS (enable in Supabase UI or here)

-- Enable RLS (run these once in Supabase SQL editor)
-- ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.referrals ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

-- Policies for users table
-- Allow all users to read other user profiles (e.g., for chat display)
CREATE POLICY IF NOT EXISTS "Users are viewable by everyone." ON public.users FOR SELECT USING (TRUE);
-- Allow authenticated users to insert their own profile on signup
CREATE POLICY IF NOT EXISTS "Authenticated users can insert their own profile." ON public.users FOR INSERT WITH CHECK (auth.uid() = id);
-- Allow authenticated users to update their own profile
CREATE POLICY IF NOT EXISTS "Authenticated users can update their own profile." ON public.users FOR UPDATE USING (auth.uid() = id);

-- Policies for referrals table
-- Allow users to read referrals they are part of (referrer or referee)
CREATE POLICY IF NOT EXISTS "Referrals are viewable by participants." ON public.referrals FOR SELECT USING (
    auth.uid() = (SELECT id FROM public.users WHERE wallet_pubkey = referrer_pubkey) OR
    auth.uid() = (SELECT id FROM public.users WHERE wallet_pubkey = referee_pubkey)
);
-- Allow authenticated users to insert new referrals (when a new referee signs up)
CREATE POLICY IF NOT EXISTS "Authenticated users can insert new referrals." ON public.referrals FOR INSERT WITH CHECK (
    -- Ensure the referee is the one signing up
    auth.uid() = (SELECT id FROM public.users WHERE wallet_pubkey = referee_pubkey)
);
-- Policy to allow updates to the 'activated' status by service role only (backend process)
CREATE POLICY IF NOT EXISTS "Service role can update referral activation status." ON public.referrals FOR UPDATE USING (true) WITH CHECK (
    -- This policy needs to be carefully managed. For full backend control without RLS, use the service role key.
    -- If RLS is enabled, ensure only authorized backend processes (e.g., via functions) can trigger this.
    pg_has_role(auth.role(), 'supabase_admin', true) -- Example for service role
);

-- Policies for chat_messages table
-- Allow all authenticated users to view chat history
CREATE POLICY IF NOT EXISTS "Chat messages are viewable by authenticated users." ON public.chat_messages FOR SELECT USING (auth.uid() IS NOT NULL);
-- Allow authenticated users to insert their own chat messages
CREATE POLICY IF NOT EXISTS "Authenticated users can insert their own chat messages." ON public.chat_messages FOR INSERT WITH CHECK (
    auth.uid() = (SELECT id FROM public.users WHERE wallet_pubkey = author_pubkey)
);

-- Add activated_at timestamp to referrals table
ALTER TABLE public.referrals
ADD COLUMN IF NOT EXISTS activated_at timestamptz;

-- Update activate_referral function to consider a minimum volume threshold
CREATE OR REPLACE FUNCTION public.activate_referral(
    p_referee text,
    p_min_volume numeric DEFAULT 0 -- Placeholder for minimum trading volume
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- Only activate if the referee exists and hasn't been activated yet
    -- and if the (mock) minimum volume threshold is met.
    -- In a real scenario, p_min_volume would be compared against actual trading volume.
    UPDATE public.referrals
    SET activated_at = NOW(), activated = TRUE
    WHERE referee_pubkey = p_referee
      AND activated_at IS NULL
      AND p_min_volume > 0; -- Simple mock condition for activation

    -- Also mark the user as activated if they are the referee
    UPDATE public.users
    SET is_activated = TRUE
    WHERE wallet_pubkey = p_referee
      AND is_activated IS FALSE
      AND p_min_volume > 0;
END;
$$;

-- Add is_activated column to users table
ALTER TABLE public.users
ADD COLUMN IF NOT EXISTS is_activated boolean NOT NULL DEFAULT FALSE;


