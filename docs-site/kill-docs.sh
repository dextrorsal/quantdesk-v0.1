#!/bin/bash

# kill-docs-site.sh
# Kill the QuantDesk documentation site server

echo "🛑 Stopping QuantDesk Documentation Site..."

# Kill processes on port 8080 (default docs port)
echo "🔍 Looking for processes on port 8080..."
PIDS=$(lsof -ti:8080)

if [ -z "$PIDS" ]; then
    echo "✅ No processes found on port 8080"
    echo "📚 Documentation site is not running"
else
    echo "🎯 Found processes: $PIDS"
    echo "💀 Killing processes..."
    
    for PID in $PIDS; do
        echo "   Killing PID: $PID"
        kill -9 $PID
    done
    
    echo "✅ Documentation site stopped"
fi

# Also check for any Python processes running serve.py
echo "🔍 Looking for Python serve.py processes..."
PYTHON_PIDS=$(pgrep -f "python.*serve.py")

if [ -z "$PYTHON_PIDS" ]; then
    echo "✅ No Python serve.py processes found"
else
    echo "🎯 Found Python serve.py processes: $PYTHON_PIDS"
    echo "💀 Killing Python serve.py processes..."
    
    for PID in $PYTHON_PIDS; do
        echo "   Killing PID: $PID"
        kill -9 $PID
    done
    
    echo "✅ Python serve.py processes stopped"
fi

echo ""
echo "🎉 Documentation site cleanup complete!"
echo "📝 You can now start the docs site again with:"
echo "   cd docs-site && python serve.py"
echo ""
