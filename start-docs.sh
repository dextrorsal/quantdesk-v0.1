#!/bin/bash

# start-docs.sh
# Start the QuantDesk documentation site server

echo "🚀 Starting QuantDesk Documentation Site..."

# Check if docs-site directory exists
if [ ! -d "docs-site" ]; then
    echo "❌ docs-site directory not found!"
    echo "📁 Please run this script from the QuantDesk project root"
    exit 1
fi

# Check if serve.py exists
if [ ! -f "docs-site/serve.py" ]; then
    echo "❌ serve.py not found in docs-site directory!"
    exit 1
fi

# Change to docs-site directory and start server
cd docs-site
echo "📁 Starting server from: $(pwd)"
echo "🌐 Documentation will be available at: http://localhost:8080"
echo "⏹️  Press Ctrl+C to stop the server"
echo "🛑 Or run './kill-docs.sh' from project root to stop"
echo ""

python serve.py
