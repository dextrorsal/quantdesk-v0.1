-- QuantDesk Database Migration Validation Script
-- This script helps you validate which Supabase project you're connected to
-- and ensures the migration goes to the correct database

-- =============================================
-- STEP 1: VALIDATE CURRENT CONNECTION
-- =============================================

-- Check PostgreSQL version
SELECT version() as postgresql_version;

-- Check if TimescaleDB is available
SELECT 
    CASE 
        WHEN EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') 
        THEN 'TimescaleDB is AVAILABLE'
        ELSE 'TimescaleDB is NOT AVAILABLE'
    END as timescaledb_status;

-- Check current database name
SELECT current_database() as current_database_name;

-- Check current user
SELECT current_user as current_user;

-- Check if we're connected to Supabase
SELECT 
    CASE 
        WHEN current_database() LIKE '%supabase%' OR current_user = 'postgres'
        THEN 'Connected to Supabase'
        ELSE 'Connected to local PostgreSQL'
    END as connection_type;

-- =============================================
-- STEP 2: CHECK EXISTING SCHEMA
-- =============================================

-- List all existing tables
SELECT 
    schemaname,
    tablename,
    tableowner
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY tablename;

-- Check if this looks like QuantDesk database
SELECT 
    CASE 
        WHEN EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'users' AND schemaname = 'public')
        AND EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'markets' AND schemaname = 'public')
        THEN 'This appears to be QuantDesk database'
        ELSE 'This does NOT appear to be QuantDesk database'
    END as database_identification;

-- Check existing extensions
SELECT 
    extname,
    extversion,
    extrelocatable
FROM pg_extension
ORDER BY extname;

-- =============================================
-- STEP 3: VALIDATE PROJECT SETTINGS
-- =============================================

-- Check if we can create extensions (admin privileges)
DO $$
BEGIN
    -- Try to create a test extension to check privileges
    BEGIN
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        RAISE NOTICE 'SUCCESS: Can create extensions - you have admin privileges';
    EXCEPTION
        WHEN insufficient_privilege THEN
            RAISE NOTICE 'WARNING: Cannot create extensions - limited privileges';
        WHEN OTHERS THEN
            RAISE NOTICE 'INFO: Extension already exists or other issue';
    END;
END $$;

-- =============================================
-- STEP 4: RECOMMENDATIONS BASED ON VERSION
-- =============================================

DO $$
DECLARE
    pg_version TEXT;
    has_timescale BOOLEAN;
    db_name TEXT;
BEGIN
    -- Get PostgreSQL version
    SELECT split_part(version(), ' ', 2) INTO pg_version;
    
    -- Check TimescaleDB availability
    SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') INTO has_timescale;
    
    -- Get database name
    SELECT current_database() INTO db_name;
    
    RAISE NOTICE '=============================================';
    RAISE NOTICE 'DATABASE VALIDATION RESULTS';
    RAISE NOTICE '=============================================';
    RAISE NOTICE 'PostgreSQL Version: %', pg_version;
    RAISE NOTICE 'Database Name: %', db_name;
    RAISE NOTICE 'TimescaleDB Available: %', has_timescale;
    RAISE NOTICE '=============================================';
    
    -- Provide recommendations
    IF pg_version LIKE '15%' THEN
        RAISE NOTICE 'PostgreSQL 15 detected - TimescaleDB is supported';
        IF has_timescale THEN
            RAISE NOTICE 'RECOMMENDATION: Use full migration with TimescaleDB features';
        ELSE
            RAISE NOTICE 'RECOMMENDATION: Install TimescaleDB extension first';
        END IF;
    ELSIF pg_version LIKE '17%' THEN
        RAISE NOTICE 'PostgreSQL 17 detected - TimescaleDB is NOT supported';
        RAISE NOTICE 'RECOMMENDATION: Use PostgreSQL 17 compatible migration (without TimescaleDB)';
    ELSE
        RAISE NOTICE 'Unknown PostgreSQL version - proceed with caution';
    END IF;
    
    -- Check if this is the right database
    IF db_name ILIKE '%quantdesk%' THEN
        RAISE NOTICE 'SUCCESS: Connected to QuantDesk database';
    ELSIF db_name ILIKE '%dextrosal%' THEN
        RAISE NOTICE 'WARNING: Connected to Dextrosal database - this may be wrong!';
    ELSE
        RAISE NOTICE 'INFO: Database name does not match expected names';
    END IF;
    
    RAISE NOTICE '=============================================';
END $$;

-- =============================================
-- STEP 5: SAFETY CHECKS
-- =============================================

-- Check if there's existing data that could be affected
DO $$
DECLARE
    user_count INTEGER;
    market_count INTEGER;
    position_count INTEGER;
BEGIN
    -- Count existing data
    SELECT COUNT(*) INTO user_count FROM information_schema.tables WHERE table_name = 'users';
    IF user_count > 0 THEN
        SELECT COUNT(*) INTO user_count FROM users;
    ELSE
        user_count := 0;
    END IF;
    
    SELECT COUNT(*) INTO market_count FROM information_schema.tables WHERE table_name = 'markets';
    IF market_count > 0 THEN
        SELECT COUNT(*) INTO market_count FROM markets;
    ELSE
        market_count := 0;
    END IF;
    
    SELECT COUNT(*) INTO position_count FROM information_schema.tables WHERE table_name = 'positions';
    IF position_count > 0 THEN
        SELECT COUNT(*) INTO position_count FROM positions;
    ELSE
        position_count := 0;
    END IF;
    
    RAISE NOTICE 'EXISTING DATA CHECK:';
    RAISE NOTICE 'Users: %', user_count;
    RAISE NOTICE 'Markets: %', market_count;
    RAISE NOTICE 'Positions: %', position_count;
    
    IF user_count > 0 OR market_count > 0 OR position_count > 0 THEN
        RAISE NOTICE 'WARNING: Existing data found - migration will preserve data';
        RAISE NOTICE 'RECOMMENDATION: Backup your data before proceeding';
    ELSE
        RAISE NOTICE 'INFO: No existing data found - safe to proceed';
    END IF;
END $$;

-- =============================================
-- STEP 6: FINAL RECOMMENDATIONS
-- =============================================

DO $$
DECLARE
    pg_version TEXT;
    has_timescale BOOLEAN;
    db_name TEXT;
BEGIN
    SELECT split_part(version(), ' ', 2) INTO pg_version;
    SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') INTO has_timescale;
    SELECT current_database() INTO db_name;
    
    RAISE NOTICE '=============================================';
    RAISE NOTICE 'FINAL RECOMMENDATIONS';
    RAISE NOTICE '=============================================';
    
    IF pg_version LIKE '17%' THEN
        RAISE NOTICE 'For PostgreSQL 17:';
        RAISE NOTICE '1. Use migration-to-production-pg17.sql (without TimescaleDB)';
        RAISE NOTICE '2. TimescaleDB features will be disabled';
        RAISE NOTICE '3. Performance may be slightly lower for time-series data';
        RAISE NOTICE '4. All other features will work normally';
    ELSIF pg_version LIKE '15%' THEN
        RAISE NOTICE 'For PostgreSQL 15:';
        RAISE NOTICE '1. Use migration-to-production.sql (with TimescaleDB)';
        RAISE NOTICE '2. TimescaleDB will provide better performance';
        RAISE NOTICE '3. All features will be available';
    END IF;
    
    RAISE NOTICE '';
    RAISE NOTICE 'NEXT STEPS:';
    RAISE NOTICE '1. Verify you are connected to the correct Supabase project';
    RAISE NOTICE '2. Run the appropriate migration script';
    RAISE NOTICE '3. Test with test-schema.sql';
    RAISE NOTICE '=============================================';
END $$;
