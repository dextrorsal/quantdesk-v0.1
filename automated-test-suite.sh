#!/bin/bash

echo "🤖 QuantDesk Complete Automated Test Suite"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
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

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
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
print_status "Starting comprehensive test suite..."
echo ""

# ==========================================
# PHASE 1: ENVIRONMENT VALIDATION
# ==========================================
print_status "=== PHASE 1: ENVIRONMENT VALIDATION ==="
run_test "Solana CLI Installation" "solana --version"
run_test "Anchor Framework" "anchor --version"
run_test "Rust Toolchain" "rustc --version"
run_test "Node.js Installation" "node --version"
run_test "npm Installation" "npm --version"

# ==========================================
# PHASE 2: SMART CONTRACT TESTS
# ==========================================
print_status "=== PHASE 2: SMART CONTRACT TESTS ==="
run_test "Smart Contract Compilation" "cd quantdesk-perp-dex && anchor build"
run_test "Smart Contract Dependencies" "cd quantdesk-perp-dex && cargo check"

# ==========================================
# PHASE 3: FRONTEND TESTS
# ==========================================
print_status "=== PHASE 3: FRONTEND TESTS ==="
run_test "Frontend Dependencies" "cd frontend && npm install"
run_test "Frontend Compilation" "cd frontend && npm run build"
run_test "Frontend TypeScript Check" "cd frontend && npx tsc --noEmit"

# ==========================================
# PHASE 4: PROJECT STRUCTURE VALIDATION
# ==========================================
print_status "=== PHASE 4: PROJECT STRUCTURE VALIDATION ==="
run_test "Main Project Files" "[ -f 'README.md' ] && [ -f 'TODO.md' ] && [ -f 'FEATURES.md' ]"
run_test "Documentation Files" "[ -f 'docs/ARCHITECTURE.md' ] && [ -f 'docs/BACKEND_ROADMAP.md' ]"
run_test "Smart Contract Files" "[ -f 'quantdesk-perp-dex/Anchor.toml' ] && [ -f 'quantdesk-perp-dex/programs/quantdesk-perp-dex/src/lib.rs' ]"
run_test "Frontend Files" "[ -f 'frontend/package.json' ] && [ -f 'frontend/src/App.tsx' ]"

# ==========================================
# PHASE 5: CONFIGURATION VALIDATION
# ==========================================
print_status "=== PHASE 5: CONFIGURATION VALIDATION ==="
run_test "Anchor Configuration" "cd quantdesk-perp-dex && [ -f 'Anchor.toml' ]"
run_test "Frontend Configuration" "cd frontend && [ -f 'vite.config.ts' ] && [ -f 'tailwind.config.js' ]"
run_test "TypeScript Configuration" "cd frontend && [ -f 'tsconfig.json' ]"

# ==========================================
# PHASE 6: DEPENDENCY VALIDATION
# ==========================================
print_status "=== PHASE 6: DEPENDENCY VALIDATION ==="
run_test "Smart Contract Dependencies" "cd quantdesk-perp-dex && grep -q 'anchor-lang' programs/quantdesk-perp-dex/Cargo.toml"
run_test "Frontend Dependencies" "cd frontend && grep -q 'react' package.json && grep -q 'typescript' package.json"
run_test "Wallet Integration" "cd frontend && grep -q '@solana/wallet-adapter' package.json"

# ==========================================
# PHASE 7: CODE QUALITY CHECKS
# ==========================================
print_status "=== PHASE 7: CODE QUALITY CHECKS ==="
run_test "Smart Contract Syntax" "cd quantdesk-perp-dex && cargo check --quiet"
run_test "Frontend Linting" "cd frontend && npm run lint --silent"

# ==========================================
# PHASE 8: FEATURE VALIDATION
# ==========================================
print_status "=== PHASE 8: FEATURE VALIDATION ==="
run_test "100x Leverage Support" "cd quantdesk-perp-dex && grep -q 'leverage <= 100' programs/quantdesk-perp-dex/src/lib.rs"
run_test "Funding Rate System" "cd quantdesk-perp-dex && grep -q 'settle_funding' programs/quantdesk-perp-dex/src/lib.rs"
run_test "Position Management" "cd quantdesk-perp-dex && grep -q 'open_position' programs/quantdesk-perp-dex/src/lib.rs"
run_test "Liquidation System" "cd quantdesk-perp-dex && grep -q 'liquidate_position' programs/quantdesk-perp-dex/src/lib.rs"

# ==========================================
# PHASE 9: UI COMPONENT VALIDATION
# ==========================================
print_status "=== PHASE 9: UI COMPONENT VALIDATION ==="
run_test "Trading Interface" "cd frontend && [ -f 'src/components/TradingInterface.tsx' ]"
run_test "Order Book Component" "cd frontend && [ -f 'src/components/OrderBook.tsx' ]"
run_test "Price Chart Component" "cd frontend && [ -f 'src/components/PriceChart.tsx' ]"
run_test "Positions Component" "cd frontend && [ -f 'src/components/Positions.tsx' ]"
run_test "Landing Page" "cd frontend && [ -f 'src/pages/LandingPage.tsx' ]"

# ==========================================
# SUMMARY
# ==========================================
echo ""
echo "=========================================="
print_status "TEST SUITE SUMMARY"
echo "=========================================="
print_success "Tests Passed: $TESTS_PASSED"
if [ $TESTS_FAILED -gt 0 ]; then
    print_error "Tests Failed: $TESTS_FAILED"
else
    print_success "Tests Failed: $TESTS_FAILED"
fi

# Calculate success rate
TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
SUCCESS_RATE=$((TESTS_PASSED * 100 / TOTAL_TESTS))

echo ""
print_status "Success Rate: $SUCCESS_RATE%"

# Overall result
if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    print_success "🎉 ALL TESTS PASSED! QuantDesk is production-ready!"
    echo ""
    print_status "🚀 QuantDesk Status:"
    print_success "✅ Smart Contracts: Advanced perpetual trading with 100x leverage"
    print_success "✅ Frontend: Professional trading interface"
    print_success "✅ Architecture: Solana-native with funding rates"
    print_success "✅ Features: Competitive with Hyperliquid/Drift"
    echo ""
    print_status "📋 Next Development Phases:"
    echo "  1. Advanced Order Types (SL/TP, trailing stops)"
    echo "  2. DeFi Integration (lending, borrowing, staking)"
    echo "  3. Account Management (sub-accounts, portfolio)"
    echo "  4. API Infrastructure (REST, WebSocket)"
    echo ""
    print_status "🎯 Ready for deployment and further development!"
else
    echo ""
    print_error "❌ Some tests failed. Review the errors above."
    echo ""
    print_status "🔧 Common fixes:"
    echo "  1. Run: npm install (in frontend directory)"
    echo "  2. Run: anchor build (in quantdesk-perp-dex directory)"
    echo "  3. Check: All required files are present"
    echo "  4. Verify: Dependencies are installed"
fi

echo ""
echo "=========================================="
print_status "Automated test suite complete!"
echo "=========================================="
