#!/bin/bash

# Fix Supabase Schema Cache Issue
# This script applies the database schema fix to resolve the wallet_address column issue

echo "🔧 Fixing Supabase Schema Cache Issue..."

# Check if we have Supabase credentials
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_SERVICE_ROLE_KEY" ]; then
    echo "❌ Missing Supabase credentials!"
    echo "Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables"
    echo ""
    echo "You can get these from your Supabase project dashboard:"
    echo "1. Go to https://supabase.com/dashboard"
    echo "2. Select your project"
    echo "3. Go to Settings > API"
    echo "4. Copy the Project URL and service_role key"
    echo ""
    echo "Then run:"
    echo "export SUPABASE_URL='your-project-url'"
    echo "export SUPABASE_SERVICE_ROLE_KEY='your-service-role-key'"
    echo "bash database/fix-schema-cache.sh"
    exit 1
fi

# Create connection string
CONNECTION_STRING="postgresql://postgres:${SUPABASE_SERVICE_ROLE_KEY}@${SUPABASE_URL#https://}"

echo "📊 Applying schema fix to Supabase database..."

# Apply the schema fix
psql "$CONNECTION_STRING" -f database/fix-schema-cache.sql

if [ $? -eq 0 ]; then
    echo "✅ Schema fix applied successfully!"
    echo ""
    echo "🧪 Now run the tests to verify the fix:"
    echo "cd backend && pnpm run test -- --testPathPattern='hackathon-core'"
else
    echo "❌ Schema fix failed!"
    echo "Please check your Supabase credentials and try again."
    exit 1
fi
