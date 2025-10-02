#!/bin/bash

# 🧪 QuantDesk Test Runner
# Runs all available tests in the correct order

echo "🧪 QuantDesk Test Runner"
echo "========================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to run a test
run_test() {
    local test_name=$1
    local test_file=$2
    local description=$3
    
    echo -e "\n${BLUE}🔍 $test_name${NC}"
    echo -e "${YELLOW}📋 $description${NC}"
    echo -e "${YELLOW}📁 Running: $test_file${NC}"
    echo "──────────────────────────────────────────────────"
    
    if [ -f "$test_file" ]; then
        if node "$test_file" 2>/dev/null; then
            echo -e "${GREEN}✅ $test_name completed${NC}"
            return 0
        else
            echo -e "${RED}❌ $test_name failed${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Test file not found: $test_file${NC}"
        return 1
    fi
}

# Check if services are running
echo -e "${YELLOW}🔍 Checking service status...${NC}"
backend_status=$(curl -s http://localhost:3002/health | grep -o '"status":"healthy"' || echo "not_running")
mikey_status=$(curl -s http://localhost:3000/health | head -1 || echo "not_running")

if [ "$backend_status" = '"status":"healthy"' ]; then
    echo -e "${GREEN}✅ Backend is running${NC}"
else
    echo -e "${RED}❌ Backend is not running${NC}"
fi

if [ "$mikey_status" != "not_running" ]; then
    echo -e "${GREEN}✅ MIKEY-AI is running${NC}"
else
    echo -e "${RED}❌ MIKEY-AI is not running${NC}"
fi

echo ""

# Run comprehensive test suite
echo -e "${BLUE}🚀 Running Comprehensive Test Suite...${NC}"
run_test "Comprehensive Test Suite" "tests/comprehensive-test-suite.js" "Tests all major platform components"

# Run individual test suites
echo -e "\n${BLUE}🔧 Running Individual Test Suites...${NC}"

# Oracle/Pyth tests
run_test "Pyth Oracle Tests" "tests/integration/test-oracle.js" "Tests Pyth price feed integration"

# API tests
run_test "API Improvements" "tests/integration/test-api-improvements.js" "Tests API endpoints and functionality"

# MIKEY-AI tests
run_test "MIKEY-AI Integration" "MIKEY-AI/archive/test-scripts/test-mikey-ai.js" "Tests AI trading assistant"

# Data pipeline tests
run_test "Data Pipeline" "data-ingestion/test/pipeline-test.js" "Tests data ingestion pipeline"

# Trading demo
run_test "Trading Demo" "scripts/standalone-trading-demo.js" "Tests standalone trading simulation"

echo -e "\n${GREEN}🎉 Test suite completed!${NC}"
echo -e "\n${YELLOW}📋 Available individual tests:${NC}"
echo -e "${WHITE}  • node tests/integration/test-oracle.js${NC}"
echo -e "${WHITE}  • node tests/integration/test-api-improvements.js${NC}"
echo -e "${WHITE}  • node tests/integration/test-new-markets.js${NC}"
echo -e "${WHITE}  • node tests/integration/test-advanced-orders.js${NC}"
echo -e "${WHITE}  • node MIKEY-AI/archive/test-scripts/test-mikey-ai.js${NC}"
echo -e "${WHITE}  • node data-ingestion/test/pipeline-test.js${NC}"
echo -e "${WHITE}  • node scripts/standalone-trading-demo.js${NC}"

echo -e "\n${BLUE}🚀 Your QuantDesk Platform is ready for testing!${NC}"
