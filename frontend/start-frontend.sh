#!/bin/bash

# Quantify Trading System - Frontend Startup Script

echo "🔥 Quantify Trading System - Frontend"
echo "======================================"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Navigate to frontend directory
cd "$(dirname "$0")" || exit 1

echo "📦 Installing dependencies..."
if [ ! -d "node_modules" ]; then
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
else
    echo "✅ Dependencies already installed"
fi

echo "🚀 Starting development server..."
echo "📍 Frontend will be available at: http://localhost:3001"
echo "📍 Make sure backend is running at: http://localhost:3002"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the development server
npm run dev
