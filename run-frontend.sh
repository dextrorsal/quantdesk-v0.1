#!/bin/bash

# QuantDesk Frontend Development Server
# =====================================
# This script starts the frontend development server so you can see QuantDesk in your browser

echo "🚀 QuantDesk Frontend Development Server"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    echo "❌ Error: frontend directory not found!"
    echo "Please run this script from the QuantDesk root directory"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed!"
    echo "Please install Node.js 18+ and try again"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Error: Node.js version 18+ required!"
    echo "Current version: $(node --version)"
    echo "Please upgrade Node.js and try again"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"

# Navigate to frontend directory
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to install dependencies!"
        exit 1
    fi
    echo "✅ Dependencies installed successfully!"
else
    echo "✅ Dependencies already installed"
fi

# Optional TypeScript check (non-blocking for vendor code during dev)
echo "🔍 Checking TypeScript compilation (non-blocking)..."
if npm run type-check; then
    echo "✅ TypeScript compilation successful!"
else
    echo "⚠️  TypeScript errors detected (likely in vendored UI). Proceeding with dev server..."
fi

echo ""
echo "🎯 Starting QuantDesk Frontend Development Server..."
echo "=================================================="
echo ""
echo "📱 The application will be available at:"
echo "   🌐 http://localhost:5173"
echo ""
echo "🎮 Features you can explore:"
echo "   📊 Landing Page - Professional marketing page"
echo "   📈 Trading Page - Professional trading interface with charts"
echo "   💼 Portfolio Page - Comprehensive P&L analytics and risk metrics"
echo "   📋 Markets Page - Market overview and selection"
echo ""
echo "🔄 The server will automatically reload when you make changes"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Start the development server
npm run dev
