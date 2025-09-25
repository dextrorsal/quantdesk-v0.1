#!/bin/bash

# QuantDesk Platform Launcher
# ===========================
# This script starts the complete QuantDesk platform (frontend + backend services)

echo "🚀 QuantDesk Platform Launcher"
echo "=============================="
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down QuantDesk platform..."
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "✅ Frontend server stopped"
    fi
    if [ ! -z "$SOLANA_PID" ]; then
        kill $SOLANA_PID 2>/dev/null
        echo "✅ Solana validator stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if we're in the right directory
if [ ! -d "frontend" ] || [ ! -d "quantdesk-perp-dex" ]; then
    echo "❌ Error: Required directories not found!"
    echo "Please run this script from the QuantDesk root directory"
    exit 1
fi

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed!"
    echo "Please install Node.js 18+ and try again"
    exit 1
fi

# Check Solana CLI
if ! command -v solana &> /dev/null; then
    echo "❌ Error: Solana CLI is not installed!"
    echo "Please install Solana CLI and try again"
    exit 1
fi

# Check Anchor
if ! command -v anchor &> /dev/null; then
    echo "❌ Error: Anchor CLI is not installed!"
    echo "Please install Anchor CLI and try again"
    exit 1
fi

echo "✅ All prerequisites found!"

# Start Solana validator (if not already running)
echo ""
echo "🔧 Starting Solana validator..."
if pgrep -f "solana-test-validator" > /dev/null; then
    echo "✅ Solana validator already running"
else
    echo "🚀 Starting new Solana validator..."
    solana-test-validator --reset --quiet &
    SOLANA_PID=$!
    sleep 3
    echo "✅ Solana validator started (PID: $SOLANA_PID)"
fi

# Configure Solana CLI for local development
echo "⚙️  Configuring Solana CLI for local development..."
solana config set --url localhost
solana config set --keypair ~/.config/solana/id.json 2>/dev/null || {
    echo "🔑 Creating new Solana keypair..."
    solana-keygen new --no-bip39-passphrase --silent
}

# Install frontend dependencies if needed
echo ""
echo "📦 Checking frontend dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to install frontend dependencies!"
        exit 1
    fi
    echo "✅ Frontend dependencies installed!"
else
    echo "✅ Frontend dependencies already installed"
fi

# Build smart contracts
echo ""
echo "🔨 Building smart contracts..."
cd ../quantdesk-perp-dex
anchor build
if [ $? -ne 0 ]; then
    echo "❌ Error: Smart contract build failed!"
    exit 1
fi
echo "✅ Smart contracts built successfully!"

# Deploy smart contracts
echo "🚀 Deploying smart contracts..."
anchor deploy
if [ $? -ne 0 ]; then
    echo "❌ Error: Smart contract deployment failed!"
    exit 1
fi
echo "✅ Smart contracts deployed successfully!"

# Start frontend development server
echo ""
echo "🎯 Starting QuantDesk Frontend..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

# Wait a moment for the server to start
sleep 3

echo ""
echo "🎉 QuantDesk Platform is now running!"
echo "====================================="
echo ""
echo "📱 Frontend Application:"
echo "   🌐 http://localhost:5173"
echo ""
echo "🔧 Solana Validator:"
echo "   🌐 http://localhost:8899"
echo ""
echo "🎮 Features Available:"
echo "   📊 Landing Page - Professional marketing page"
echo "   📈 Trading Page - Professional trading interface with charts"
echo "   💼 Portfolio Page - Comprehensive P&L analytics and risk metrics"
echo "   📋 Markets Page - Market overview and selection"
echo ""
echo "🔄 Services will automatically reload when you make changes"
echo "⏹️  Press Ctrl+C to stop all services"
echo ""

# Wait for user to stop the services
wait
