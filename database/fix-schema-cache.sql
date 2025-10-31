-- Fix Supabase Schema Cache Issue
-- This script ensures the database schema is properly applied and cached

-- First, let's check what's actually in the users table
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'users' 
ORDER BY ordinal_position;

-- If wallet_address column is missing, add it
DO $$ 
BEGIN
    -- Check if wallet_address column exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'wallet_address'
    ) THEN
        -- Add the missing column
        ALTER TABLE users ADD COLUMN wallet_address TEXT UNIQUE;
        
        -- If there are existing records, we need to populate them
        -- This is a one-time migration for existing data
        UPDATE users SET wallet_address = COALESCE(wallet_pubkey, 'unknown-' || id::text) 
        WHERE wallet_address IS NULL;
        
        -- Make it NOT NULL after populating
        ALTER TABLE users ALTER COLUMN wallet_address SET NOT NULL;
        
        RAISE NOTICE 'Added wallet_address column to users table';
    ELSE
        RAISE NOTICE 'wallet_address column already exists';
    END IF;
END $$;

-- Ensure the markets table has required columns
DO $$ 
BEGIN
    -- Check if program_id column exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'markets' AND column_name = 'program_id'
    ) THEN
        -- Add the missing column with a default value
        ALTER TABLE markets ADD COLUMN program_id TEXT DEFAULT 'default-program-id';
        
        RAISE NOTICE 'Added program_id column to markets table';
    ELSE
        RAISE NOTICE 'program_id column already exists';
    END IF;
END $$;

-- Refresh the schema cache
-- This forces Supabase to reload the schema information
NOTIFY pgrst, 'reload schema';

-- Verify the schema is correct
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name IN ('users', 'markets')
ORDER BY table_name, ordinal_position;
