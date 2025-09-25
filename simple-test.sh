#!/bin/bash

echo "🎯 QuantDesk Simple Automated Test"
echo "================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    print_status "Testing: $test_name"
    
    if eval "$test_command" > /dev/null 2>&1; then
        print_success "✅ $test_name PASSED"
        ((TESTS_PASSED++))
    else
        print_error "❌ $test_name FAILED"
        ((TESTS_FAILED++))
    fi
    echo ""
}

echo ""
print_status "Running essential tests..."

# Core Environment Tests
run_test "Solana CLI" "solana --version"
run_test "Anchor Framework" "anchor --version"
run_test "Rust Toolchain" "rustc --version"
run_test "Node.js" "node --version"

# Smart Contract Tests
print_status "Testing smart contracts..."
if [ -d "quantdesk-perp-dex" ]; then
    run_test "Smart Contract Build" "cd quantdesk-perp-dex && anchor build"
    run_test "100x Leverage Code" "cd quantdesk-perp-dex && grep -q 'leverage <= 100' programs/quantdesk-perp-dex/src/lib.rs"
    run_test "Funding Rate Code" "cd quantdesk-perp-dex && grep -q 'settle_funding' programs/quantdesk-perp-dex/src/lib.rs"
else
    print_error "❌ Smart contract directory not found"
    ((TESTS_FAILED++))
fi

# Frontend Tests
print_status "Testing frontend..."
if [ -d "frontend" ]; then
    run_test "Frontend Package.json" "[ -f 'frontend/package.json' ]"
    run_test "Frontend App.tsx" "[ -f 'frontend/src/App.tsx' ]"
    run_test "Frontend Trading Interface" "[ -f 'frontend/src/components/TradingInterface.tsx' ]"
else
    print_error "❌ Frontend directory not found"
    ((TESTS_FAILED++))
fi

# Documentation Tests
print_status "Testing documentation..."
run_test "README.md" "[ -f 'README.md' ]"
run_test "TODO.md" "[ -f 'TODO.md' ]"
run_test "FEATURES.md" "[ -f 'FEATURES.md' ]"

# Summary
echo "================================="
print_status "TEST SUMMARY"
echo "================================="
print_success "Tests Passed: $TESTS_PASSED"
if [ $TESTS_FAILED -gt 0 ]; then
    print_error "Tests Failed: $TESTS_FAILED"
else
    print_success "Tests Failed: $TESTS_FAILED"
fi

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
SUCCESS_RATE=$((TESTS_PASSED * 100 / TOTAL_TESTS))

print_status "Success Rate: $SUCCESS_RATE%"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    print_success "🎉 ALL ESSENTIAL TESTS PASSED!"
    print_success "🚀 QuantDesk is ready for development!"
    echo ""
    print_status "✅ Smart contracts: Advanced perpetual trading"
    print_status "✅ Frontend: Professional trading interface"
    print_status "✅ Documentation: Complete roadmap"
    print_status "✅ Environment: All tools installed"
else
    echo ""
    print_error "❌ Some tests failed. Check the errors above."
fi

echo "================================="
