#!/bin/bash

# kill-docs.sh
# Kill the QuantDesk documentation site server

echo "🛑 Stopping QuantDesk Documentation Site..."

# Run the kill script from docs-site directory
if [ -f "docs-site/kill-docs-site.sh" ]; then
    ./docs-site/kill-docs-site.sh
else
    echo "❌ kill-docs-site.sh not found in docs-site directory"
    echo "🔍 Looking for processes on port 8080..."
    
    PIDS=$(lsof -ti:8080)
    
    if [ -z "$PIDS" ]; then
        echo "✅ No processes found on port 8080"
    else
        echo "🎯 Found processes: $PIDS"
        echo "💀 Killing processes..."
        
        for PID in $PIDS; do
            echo "   Killing PID: $PID"
            kill -9 $PID
        done
        
        echo "✅ Documentation site stopped"
    fi
fi
