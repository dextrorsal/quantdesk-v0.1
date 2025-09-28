#!/bin/bash

echo "🔍 QuantDesk Debug Test Suite"
echo "============================="
echo ""

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed or not in PATH"
    exit 1
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed or not in PATH"
    exit 1
fi

echo "✅ Node.js and npm are available"
echo ""

# Install required packages if not already installed
echo "📦 Installing required packages..."
cd /home/dex/Desktop/quantdesk

if [ ! -d "node_modules" ]; then
    echo "Installing axios and socket.io-client..."
    npm install axios socket.io-client ws
fi

echo ""
echo "🚀 Running Debug Tests..."
echo "========================"

# Test 1: Pyth Network Connection
echo ""
echo "1️⃣ Testing Pyth Network Connection..."
echo "------------------------------------"
node debug-pyth-connection.js

echo ""
echo "2️⃣ Testing Backend WebSocket..."
echo "-------------------------------"
node test-backend-websocket.js

echo ""
echo "3️⃣ Testing Frontend Price System..."
echo "-----------------------------------"
node test-frontend-price-system.js

echo ""
echo "🎯 Debug Test Suite Complete!"
echo "============================="
echo ""
echo "📋 Summary:"
echo "   - Check the output above for any errors"
echo "   - Common issues to look for:"
echo "     • Backend not running on port 3002"
echo "     • Frontend not running on port 3001"
echo "     • Pyth Network API format issues"
echo "     • WebSocket connection problems"
echo "     • Network/firewall blocking connections"
echo ""
echo "💡 Next steps:"
echo "   1. Fix any errors identified above"
echo "   2. Start backend: cd backend && npm run dev"
echo "   3. Start frontend: cd frontend && npm run dev"
echo "   4. Test the live price system in browser"
