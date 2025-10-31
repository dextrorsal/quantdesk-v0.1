#!/bin/bash

# QuantDesk Data Ingestion Pipeline Stop Script

echo "🛑 Stopping QuantDesk Data Ingestion Pipeline..."

# Create pids directory if it doesn't exist
mkdir -p .pids

# Stop all processes
for pid_file in .pids/*.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        process_name=$(basename "$pid_file" .pid)
        
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $process_name (PID: $pid)..."
            kill "$pid"
        else
            echo "$process_name (PID: $pid) is not running"
        fi
        
        rm -f "$pid_file"
    fi
done

# Kill any remaining node processes related to our pipeline
pkill -f "price-collector.js" 2>/dev/null
pkill -f "whale-monitor.js" 2>/dev/null
pkill -f "news-scraper.js" 2>/dev/null
pkill -f "price-writer.js" 2>/dev/null
pkill -f "analytics-writer.js" 2>/dev/null
pkill -f "dashboard.js" 2>/dev/null

echo "✅ Data ingestion pipeline stopped."
