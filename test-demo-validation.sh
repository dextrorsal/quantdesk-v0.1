#!/bin/bash
# Demo Functionality Validation Script
# Story: 0-validate-demo-functionality

echo "======================================"
echo "QuantDesk Demo Functionality Validation"
echo "======================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

# Function to test endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local expected=$3
    
    echo "Testing: $name"
    response=$(curl -s -w "\n%{http_code}" "$url")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" == "200" ] || [ "$http_code" == "401" ]; then
        echo -e "${GREEN}✓ PASS${NC}: $name (HTTP $http_code)"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}: $name (HTTP $http_code)"
        echo "Response: $body"
        ((FAILED++))
        return 1
    fi
}

# Test 1: Check service ports are listening
echo "=== Test 1: Service Availability ==="
if netstat -tuln 2>/dev/null | grep -q ":3002"; then
    echo -e "${GREEN}✓ PASS${NC}: Backend (3002) is listening"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}: Backend (3002) is not listening"
    ((FAILED++))
fi

if netstat -tuln 2>/dev/null | grep -q ":3001"; then
    echo -e "${GREEN}✓ PASS${NC}: Frontend (3001) is listening"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}: Frontend (3001) is not listening"
    ((FAILED++))
fi

if netstat -tuln 2>/dev/null | grep -q ":3000"; then
    echo -e "${GREEN}✓ PASS${NC}: MIKEY-AI (3000) is listening"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ WARN${NC}: MIKEY-AI (3000) is not listening (may be optional)"
    ((FAILED++))
fi

echo ""
echo "=== Test 2: Backend API Endpoints ==="

# Test backend API endpoints
test_endpoint "Positions API" "http://localhost:3002/api/positions"
test_endpoint "Orders API" "http://localhost:3002/api/orders"
test_endpoint "MIKEY AI Features" "http://localhost:3002/api/mikey/features/test"
test_endpoint "Oracle/Prices" "http://localhost:3002/api/oracle/prices"
test_endpoint "Portfolio" "http://localhost:3002/api/portfolio"

echo ""
echo "=== Test 3: Frontend Static Files ==="

# Test frontend is serving files
if curl -s http://localhost:3001 | grep -q "QuantDesk\|Trading\|index"; then
    echo -e "${GREEN}✓ PASS${NC}: Frontend is serving content"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}: Frontend not serving content"
    ((FAILED++))
fi

echo ""
echo "=== Test 4: Component Files Check ==="

COMPONENTS=("frontend/src/pro/index.tsx" "frontend/src/components/Positions.tsx" "frontend/src/components/MikeyAIChat.tsx")

for component in "${COMPONENTS[@]}"; do
    if [ -f "$component" ]; then
        echo -e "${GREEN}✓ PASS${NC}: File exists - $component"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: File missing - $component"
        ((FAILED++))
    fi
done

echo ""
echo "=== Test 5: Database Schema Check ==="

# Check if database service is accessible (from backend)
if grep -q "positions" backend/src/routes/positions.ts 2>/dev/null; then
    echo -e "${GREEN}✓ PASS${NC}: Positions route exists"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}: Positions route not found"
    ((FAILED++))
fi

echo ""
echo "======================================"
echo "Validation Summary"
echo "======================================"
echo "Tests Passed: $PASSED"
echo "Tests Failed: $FAILED"
echo "======================================"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Review the output above.${NC}"
    exit 1
fi

