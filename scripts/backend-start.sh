#!/bin/bash

# QuantDesk Backend + Redis Startup Wrapper
# This script starts the backend with Redis automatically

echo "🚀 Starting QuantDesk Backend with Redis..."
echo "============================================="

# Change to project root
cd "$(dirname "$0")"

# Start backend with Redis flag
./backend/start-backend.sh --redis
