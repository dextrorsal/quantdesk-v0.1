#!/bin/bash

echo "🎯 QuantDesk Comprehensive Test Suite"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    print_status "Running: $test_name"
    
    if eval "$test_command"; then
        print_success "$test_name PASSED"
        ((TESTS_PASSED++))
    else
        print_error "$test_name FAILED"
        ((TESTS_FAILED++))
    fi
    echo ""
}

echo ""
print_status "Starting comprehensive test suite..."
echo ""

# 1. Environment Setup Tests
print_status "=== ENVIRONMENT SETUP TESTS ==="
run_test "Solana CLI Installation" "solana --version"
run_test "Anchor Framework Installation" "anchor --version"
run_test "Rust Toolchain" "rustc --version"
run_test "Node.js Installation" "node --version"

# 2. Project Build Tests
print_status "=== PROJECT BUILD TESTS ==="
run_test "Smart Contract Compilation" "anchor build"
run_test "TypeScript Compilation" "cd frontend && npm run build"

# 3. Smart Contract Tests
print_status "=== SMART CONTRACT TESTS ==="
run_test "Market Initialization" "./setup-test.sh"
run_test "Position Management" "anchor test --skip-local-validator"
run_test "Funding Settlement" "anchor test --skip-local-validator"
run_test "Liquidation System" "anchor test --skip-local-validator"

# 4. Frontend Tests
print_status "=== FRONTEND TESTS ==="
run_test "Frontend Dependencies" "cd frontend && npm install"
run_test "Frontend Build" "cd frontend && npm run build"
run_test "Frontend Linting" "cd frontend && npm run lint"

# 5. Integration Tests
print_status "=== INTEGRATION TESTS ==="
run_test "Wallet Integration" "cd frontend && npm run test:wallet"
run_test "Trading Interface" "cd frontend && npm run test:trading"

# 6. Performance Tests
print_status "=== PERFORMANCE TESTS ==="
run_test "Smart Contract Gas Usage" "anchor test --skip-local-validator --gas-report"
run_test "Frontend Bundle Size" "cd frontend && npm run analyze"

# Summary
echo ""
echo "====================================="
print_status "TEST SUITE SUMMARY"
echo "====================================="
print_success "Tests Passed: $TESTS_PASSED"
if [ $TESTS_FAILED -gt 0 ]; then
    print_error "Tests Failed: $TESTS_FAILED"
else
    print_success "Tests Failed: $TESTS_FAILED"
fi

# Overall result
if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    print_success "🎉 ALL TESTS PASSED! QuantDesk is ready for deployment!"
    echo ""
    print_status "Next steps:"
    echo "  1. Deploy to devnet: anchor deploy --provider.cluster devnet"
    echo "  2. Deploy to mainnet: anchor deploy --provider.cluster mainnet"
    echo "  3. Start frontend: cd frontend && npm run dev"
    echo ""
else
    echo ""
    print_error "❌ Some tests failed. Please review the errors above."
    echo ""
    print_status "Troubleshooting:"
    echo "  1. Check Solana CLI: solana --version"
    echo "  2. Check Anchor: anchor --version"
    echo "  3. Restart validator: pkill -f solana-test-validator"
    echo "  4. Run setup: ./setup-test.sh"
    echo ""
fi

echo "====================================="
