#!/usr/bin/env node

// Setup script for QuantDesk admin users
// This script creates the initial admin users with proper password hashes

const bcrypt = require('bcrypt');
const { createClient } = require('@supabase/supabase-js');

// Supabase configuration
const supabaseUrl = process.env.SUPABASE_URL || 'your-supabase-url';
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || 'your-service-role-key';
const supabase = createClient(supabaseUrl, supabaseKey);

async function setupAdminUsers() {
  console.log('🚀 Setting up QuantDesk admin users...');

  try {
    // Hash passwords
    const saltRounds = 10;
    const dexPasswordHash = await bcrypt.hash('quantdex47', saltRounds);
    const gorniPasswordHash = await bcrypt.hash('quantgorni31', saltRounds);

    // Create admin users
    const adminUsers = [
      {
        username: 'dex',
        password_hash: dexPasswordHash,
        role: 'founding-dev',
        permissions: ['read', 'write', 'admin', 'super-admin', 'founding-dev'],
        is_active: true
      },
      {
        username: 'gorni',
        password_hash: gorniPasswordHash,
        role: 'admin',
        permissions: ['read', 'write', 'admin'],
        is_active: true
      }
    ];

    // Insert users into database
    for (const user of adminUsers) {
      const { data, error } = await supabase
        .from('admin_users')
        .upsert(user, { onConflict: 'username' });

      if (error) {
        console.error(`❌ Error creating user ${user.username}:`, error);
      } else {
        console.log(`✅ Created/updated user: ${user.username} (${user.role})`);
      }
    }

    console.log('🎉 Admin users setup complete!');
    console.log('');
    console.log('📋 Login credentials:');
    console.log('   Founding Dev: dex / quantdex47');
    console.log('   Admin: gorni / quantgorni31');
    console.log('');
    console.log('🔐 Roles and permissions:');
    console.log('   - founding-dev: Full access to everything');
    console.log('   - admin: Standard admin access');
    console.log('   - super-admin: Advanced admin access (requires 2FA)');

  } catch (error) {
    console.error('❌ Setup failed:', error);
    process.exit(1);
  }
}

// Run setup
setupAdminUsers();
