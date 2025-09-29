#!/bin/bash

# RPC Performance Testing Script
# Usage: ./test-rpc.sh [test-type] [options]

echo "🚀 QuantDesk RPC Load Balancer Testing"
echo "======================================"

# Check if backend is running
if ! curl -s http://localhost:3002/api/rpc/health > /dev/null; then
    echo "❌ Backend not running. Please start it first:"
    echo "   cd backend && ./start-backend.sh"
    exit 1
fi

echo "✅ Backend is running"

# Check if we have environment variables set
if [ -z "$HELIUS_RPC_1_URL" ] && [ -z "$QUICKNODE_1_RPC_URL" ]; then
    echo "⚠️  Warning: No RPC URLs found in environment variables"
    echo "   Make sure to set your RPC URLs in .env file"
    echo "   Example: HELIUS_RPC_1_URL=https://your-helius-url.com"
fi

# Change to project root
cd "$(dirname "$0")/.."

# Parse command line arguments
TEST_TYPE=${1:-"all"}
REQUESTS=${2:-""}

echo "🧪 Running RPC performance tests..."

# Run from backend directory where dependencies are installed
cd backend

case $TEST_TYPE in
    "all")
        echo "Running comprehensive test suite..."
        node ../scripts/simple-rpc-test.js
        ;;
    "api")
        echo "Running API endpoint test..."
        node ../scripts/simple-rpc-test.js --api --requests ${REQUESTS:-20}
        ;;
    "high-freq")
        echo "Running high-frequency test..."
        node ../scripts/simple-rpc-test.js --high-freq --requests ${REQUESTS:-50}
        ;;
    "stress")
        echo "Running stress test..."
        node ../scripts/simple-rpc-test.js --stress --requests ${REQUESTS:-100} --duration 10
        ;;
    "rate-limit")
        echo "Running rate limit test..."
        node ../scripts/simple-rpc-test.js --rate-limit
        ;;
    "speeds")
        echo "Running RPC speed test..."
        echo "⚠️  Note: Speed test requires Solana dependencies. Using API test instead."
        node ../scripts/simple-rpc-test.js --api --requests ${REQUESTS:-20}
        ;;
    "load-balancer")
        echo "Running load balancer test..."
        echo "⚠️  Note: Load balancer test requires Solana dependencies. Using API test instead."
        node ../scripts/simple-rpc-test.js --api --requests ${REQUESTS:-20}
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [test-type] [requests]"
        echo ""
        echo "Test Types:"
        echo "  all          - Run all tests (default)"
        echo "  api          - Test API endpoints (20 requests)"
        echo "  high-freq    - High-frequency requests (50 requests)"
        echo "  stress       - Ultimate stress test (100 requests)"
        echo "  rate-limit   - Rate limit detection test"
        echo "  speeds       - RPC response time test"
        echo "  load-balancer- Load balancer distribution test"
        echo ""
        echo "Examples:"
        echo "  $0                    # Run all tests"
        echo "  $0 api 50            # Test API with 50 requests"
        echo "  $0 stress 200        # Stress test with 200 requests"
        echo "  $0 high-freq 100     # High-frequency test with 100 requests"
        exit 0
        ;;
    *)
        echo "❌ Unknown test type: $TEST_TYPE"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac

echo ""
echo "📊 Test completed! Check the results above."
echo ""
echo "💡 Available Test Types:"
echo "   ./test-rpc.sh all          # Comprehensive test suite"
echo "   ./test-rpc.sh api 50       # API test with 50 requests"
echo "   ./test-rpc.sh high-freq 100# High-frequency test with 100 requests"
echo "   ./test-rpc.sh stress 200   # Stress test with 200 requests"
echo "   ./test-rpc.sh rate-limit   # Rate limit detection test"
echo "   ./test-rpc.sh speeds       # RPC response time test"
echo "   ./test-rpc.sh load-balancer# Load balancer distribution test"
echo ""
echo "📈 Monitor endpoints:"
echo "   curl http://localhost:3002/api/rpc/stats    # Real-time stats"
echo "   curl http://localhost:3002/api/rpc/health   # Provider status"
