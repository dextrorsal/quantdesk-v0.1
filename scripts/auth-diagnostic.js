#!/usr/bin/env node

/**
 * Authentication Diagnostic Script
 * Quickly identifies JWT authentication issues
 */

const fs = require('fs');
const path = require('path');

console.log('🔍 QuantDesk Authentication Diagnostic Tool\n');

// 1. Check all environment files
console.log('📁 Checking Environment Files:');
const envFiles = [
  'backend/.env',
  'backend/.env.backup', 
  'backend/.env.example',
  '.env',
  '.env.local'
];

envFiles.forEach(file => {
  if (fs.existsSync(file)) {
    console.log(`✅ Found: ${file}`);
    const content = fs.readFileSync(file, 'utf8');
    const jwtLines = content.split('\n').filter(line => line.includes('JWT_SECRET'));
    jwtLines.forEach(line => {
      console.log(`   ${line}`);
    });
  } else {
    console.log(`❌ Missing: ${file}`);
  }
});

console.log('\n🔧 Checking JWT Usage in Code:');

// 2. Find all JWT_SECRET usage in backend
const backendDir = 'backend/src';
const files = getAllFiles(backendDir);

let jwtUsages = [];

files.forEach(file => {
  if (file.endsWith('.ts') || file.endsWith('.js')) {
    const content = fs.readFileSync(file, 'utf8');
    const lines = content.split('\n');
    
    lines.forEach((line, index) => {
      if (line.includes('JWT_SECRET')) {
        jwtUsages.push({
          file: file.replace(process.cwd() + '/', ''),
          line: index + 1,
          content: line.trim(),
          hasFallback: line.includes('||'),
          fallbackValue: line.includes('||') ? line.split('||')[1]?.trim() : null
        });
      }
    });
  }
});

console.log(`Found ${jwtUsages.length} JWT_SECRET usages:\n`);

jwtUsages.forEach(usage => {
  console.log(`📄 ${usage.file}:${usage.line}`);
  console.log(`   ${usage.content}`);
  if (usage.hasFallback) {
    console.log(`   ⚠️  HAS FALLBACK: ${usage.fallbackValue}`);
  } else {
    console.log(`   ✅ No fallback`);
  }
  console.log('');
});

// 3. Check environment loading
console.log('🌍 Environment Loading Check:');
console.log('Current process.env.JWT_SECRET:', process.env.JWT_SECRET || 'UNDEFINED');

// 4. Test JWT token creation and verification
console.log('\n🧪 JWT Token Test:');
try {
  const jwt = require('jsonwebtoken');
  const testSecret = 'test-jwt-secret';
  const testPayload = { wallet_pubkey: 'test-wallet', iat: Math.floor(Date.now() / 1000), exp: Math.floor(Date.now() / 1000) + 3600 };
  
  const token = jwt.sign(testPayload, testSecret);
  console.log('✅ Token created successfully');
  console.log(`   Token: ${token.substring(0, 50)}...`);
  
  const decoded = jwt.verify(token, testSecret);
  console.log('✅ Token verified successfully');
  console.log(`   Decoded: ${JSON.stringify(decoded)}`);
  
} catch (error) {
  console.log('❌ JWT test failed:', error.message);
}

// 5. Check for multiple JWT libraries
console.log('\n📦 JWT Library Check:');
try {
  const packageJson = JSON.parse(fs.readFileSync('backend/package.json', 'utf8'));
  const jwtDeps = Object.keys(packageJson.dependencies || {}).filter(dep => dep.includes('jwt'));
  console.log('JWT dependencies:', jwtDeps.length > 0 ? jwtDeps : 'None found');
} catch (error) {
  console.log('❌ Could not read package.json');
}

// 6. Quick fix suggestions
console.log('\n💡 Quick Fix Suggestions:');
console.log('1. Ensure JWT_SECRET is set in backend/.env');
console.log('2. Remove all hardcoded fallbacks (|| "secret")');
console.log('3. Restart backend with explicit JWT_SECRET export');
console.log('4. Check for conflicting environment files');

console.log('\n🚀 Recommended Fix Command:');
console.log('cd backend && export JWT_SECRET="test-jwt-secret" && npx ts-node src/server.ts');

function getAllFiles(dir) {
  let results = [];
  const list = fs.readdirSync(dir);
  
  list.forEach(file => {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    
    if (stat && stat.isDirectory()) {
      results = results.concat(getAllFiles(fullPath));
    } else {
      results.push(fullPath);
    }
  });
  
  return results;
}
