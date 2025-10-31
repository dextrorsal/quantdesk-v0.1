#!/bin/bash

# MIKEY-AI to QuantDesk Bridge Startup Script

echo "🚀 Starting MIKEY-AI to QuantDesk Bridge Service..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18+ is required. Current version: $(node -v)"
    exit 1
fi

# Navigate to bridge directory
cd "$(dirname "$0")"

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found. Please run this script from the bridge directory."
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration before starting the service."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Set default port if not provided to avoid conflict with frontend (3001)
export PORT=${PORT:-3000}

# Start the service
echo "🎯 Starting bridge service on port $PORT..."
echo "📊 Health check: http://localhost:$PORT/health"
echo "📚 API docs: See README.md for endpoint documentation"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

npm run dev
