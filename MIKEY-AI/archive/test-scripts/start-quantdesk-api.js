#!/usr/bin/env node

/**
 * Check and Start QuantDesk API Server
 */

const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔍 Checking QuantDesk API Server Status');
console.log('');

// Check if backend directory exists
const backendPath = path.join(__dirname, '..', 'backend');
if (!fs.existsSync(backendPath)) {
  console.log('❌ Backend directory not found at:', backendPath);
  process.exit(1);
}

console.log('✅ Backend directory found:', backendPath);

// Check if package.json exists
const packageJsonPath = path.join(backendPath, 'package.json');
if (!fs.existsSync(packageJsonPath)) {
  console.log('❌ package.json not found in backend directory');
  process.exit(1);
}

console.log('✅ package.json found');

// Check if start script exists
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
if (!packageJson.scripts || !packageJson.scripts.start) {
  console.log('❌ No start script found in package.json');
  process.exit(1);
}

console.log('✅ Start script found:', packageJson.scripts.start);

// Check if port 3002 is in use
exec('lsof -i :3002', (error, stdout, stderr) => {
  if (stdout) {
    console.log('⚠️  Port 3002 is already in use:');
    console.log(stdout);
    console.log('');
    console.log('🔧 To kill existing process:');
    console.log('   pkill -f "node.*3002"');
    console.log('   or');
    console.log('   kill -9 <PID>');
  } else {
    console.log('✅ Port 3002 is available');
    console.log('');
    console.log('🚀 Starting QuantDesk API Server...');
    console.log('');
    
    // Start the backend server
    const startProcess = exec('cd ../backend && npm start', (error, stdout, stderr) => {
      if (error) {
        console.log('❌ Error starting server:', error.message);
        return;
      }
      console.log('✅ Server started successfully');
    });
    
    startProcess.stdout.on('data', (data) => {
      console.log(data.toString());
    });
    
    startProcess.stderr.on('data', (data) => {
      console.log('Error:', data.toString());
    });
  }
});

console.log('');
console.log('📋 Manual Commands:');
console.log('   cd ../backend && npm start');
console.log('   or');
console.log('   cd ../backend && PORT=3002 npm start');
console.log('');
console.log('🔍 Check if running:');
console.log('   curl http://localhost:3002/health');
console.log('   or');
console.log('   curl http://localhost:3002/api/health');
