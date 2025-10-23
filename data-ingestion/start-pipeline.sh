#!/bin/bash

# QuantDesk Data Ingestion Pipeline Startup Script

echo "🚀 Starting QuantDesk Data Ingestion Pipeline..."

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   sudo systemctl start redis-server"
    echo "   or"
    echo "   redis-server"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please copy env.example to .env and configure it."
    exit 1
fi

# Create required directories
mkdir -p logs
mkdir -p .pids

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the pipeline
echo "🏃 Starting collectors and workers..."

# Start collectors in background
echo "📡 Starting data collectors..."
node src/collectors/price-collector.js &
PRICE_COLLECTOR_PID=$!

node src/collectors/whale-monitor.js &
WHALE_MONITOR_PID=$!

node src/collectors/news-scraper.js &
NEWS_SCRAPER_PID=$!

# Start workers in background
echo "⚙️ Starting data workers..."
node src/workers/price-writer.js &
PRICE_WRITER_PID=$!

node src/workers/analytics-writer.js &
ANALYTICS_WRITER_PID=$!

# Start monitoring dashboard
echo "📊 Starting monitoring dashboard..."
node src/monitoring/dashboard.js &
DASHBOARD_PID=$!

# Save PIDs for cleanup
echo $PRICE_COLLECTOR_PID > .pids/price-collector.pid
echo $WHALE_MONITOR_PID > .pids/whale-monitor.pid
echo $NEWS_SCRAPER_PID > .pids/news-scraper.pid
echo $PRICE_WRITER_PID > .pids/price-writer.pid
echo $ANALYTICS_WRITER_PID > .pids/analytics-writer.pid
echo $DASHBOARD_PID > .pids/dashboard.pid

echo "✅ Data ingestion pipeline started!"
echo ""
echo "📊 Monitoring Dashboard: http://localhost:3003"
echo "📝 Logs: tail -f logs/ingestion.log"
echo ""
echo "🛑 To stop the pipeline: ./stop-pipeline.sh"
echo ""

# Wait for user to stop
echo "Press Ctrl+C to stop the pipeline..."

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping data ingestion pipeline..."
    
    # Kill all processes
    kill $PRICE_COLLECTOR_PID 2>/dev/null
    kill $WHALE_MONITOR_PID 2>/dev/null
    kill $NEWS_SCRAPER_PID 2>/dev/null
    kill $PRICE_WRITER_PID 2>/dev/null
    kill $ANALYTICS_WRITER_PID 2>/dev/null
    kill $DASHBOARD_PID 2>/dev/null
    
    # Remove PID files
    rm -f .pids/*.pid
    
    echo "✅ Pipeline stopped."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for processes
wait
