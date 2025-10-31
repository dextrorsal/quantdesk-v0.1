#!/usr/bin/env node

/**
 * QuantDesk Security Checker
 * This script helps you verify your Supabase security setup
 */

const { createClient } = require('@supabase/supabase-js');
require('dotenv').config();

// Load environment variables
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('❌ Missing Supabase credentials in .env file');
  console.log('Please add SUPABASE_URL and SUPABASE_ANON_KEY to your .env file');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

async function checkSecurity() {
  console.log('🔒 Checking QuantDesk Security Setup...\n');

  try {
    // Check if we can connect
    console.log('1. Testing database connection...');
    const { data, error } = await supabase.from('markets').select('count').limit(1);
    
    if (error) {
      console.log('❌ Database connection failed:', error.message);
      return;
    }
    console.log('✅ Database connection successful');

    // Check RLS status on key tables
    console.log('\n2. Checking Row Level Security (RLS) status...');
    
    const tables = ['markets', 'users', 'positions', 'orders', 'trades', 'oracle_prices'];
    
    for (const table of tables) {
      try {
        // Try to query without authentication (should fail if RLS is enabled)
        const { data, error } = await supabase.from(table).select('*').limit(1);
        
        if (error && error.message.includes('RLS')) {
          console.log(`✅ RLS enabled on ${table}`);
        } else if (error) {
          console.log(`⚠️  ${table}: ${error.message}`);
        } else {
          console.log(`❌ RLS NOT enabled on ${table} - This is a security risk!`);
        }
      } catch (err) {
        console.log(`⚠️  Could not check ${table}: ${err.message}`);
      }
    }

    // Check if we can access data without auth (security test)
    console.log('\n3. Testing unauthorized access...');
    try {
      const { data, error } = await supabase.from('markets').select('*');
      if (data && data.length > 0) {
        console.log('❌ SECURITY RISK: Can access data without authentication!');
        console.log('   RLS is not properly configured or policies are too permissive');
      } else {
        console.log('✅ Unauthorized access blocked');
      }
    } catch (err) {
      console.log('✅ Unauthorized access blocked');
    }

    // Check rate limiting
    console.log('\n4. Testing rate limiting...');
    const promises = Array(10).fill().map(() => 
      supabase.from('markets').select('count').limit(1)
    );
    
    try {
      await Promise.all(promises);
      console.log('⚠️  No rate limiting detected - consider implementing');
    } catch (err) {
      console.log('✅ Rate limiting appears to be working');
    }

    console.log('\n🔒 Security Check Complete!');
    console.log('\n📋 Next Steps:');
    console.log('1. Enable RLS on all tables in Supabase dashboard');
    console.log('2. Create authentication policies');
    console.log('3. Add auth middleware to all API endpoints');
    console.log('4. Monitor database usage');

  } catch (error) {
    console.error('❌ Security check failed:', error.message);
  }
}

// Run the security check
checkSecurity();
