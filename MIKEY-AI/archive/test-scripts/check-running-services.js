// check-running-services.js
const { exec } = require('child_process');

function checkRunningServices() {
  console.log('🔍 Checking Running Services...\n');
  
  // Check what's running on common ports
  const ports = [3000, 3001, 3002, 3003, 5432, 6379];
  
  ports.forEach(port => {
    exec(`lsof -i :${port}`, (error, stdout, stderr) => {
      if (stdout) {
        console.log(`Port ${port}: ${stdout.trim()}`);
      } else {
        console.log(`Port ${port}: Not in use`);
      }
    });
  });
  
  console.log('\n📋 Expected Services:');
  console.log('- Port 3001: Frontend (React/Vite)');
  console.log('- Port 3002: Backend (QuantDesk API)');
  console.log('- Port 3003: Mikey AI');
  console.log('- Port 5432: PostgreSQL');
  console.log('- Port 6379: Redis');
  
  console.log('\n🚀 To start services:');
  console.log('Frontend: cd frontend && npm start');
  console.log('Backend: cd backend && npm start');
  console.log('Mikey AI: cd MIKEY-AI && PORT=3003 npm start');
}

checkRunningServices();
