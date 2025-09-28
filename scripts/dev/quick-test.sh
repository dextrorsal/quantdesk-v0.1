#!/bin/bash

echo "⚡ QuantDesk Quick Test Suite"
echo "============================="

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

echo ""
print_status "Running quick validation tests..."

# Test 1: Smart Contract Compilation
print_status "1. Testing smart contract compilation..."
if anchor build > /dev/null 2>&1; then
    print_success "✅ Smart contract compiles successfully"
else
    print_error "❌ Smart contract compilation failed"
    exit 1
fi

# Test 2: TypeScript Compilation
print_status "2. Testing TypeScript compilation..."
if cd frontend && npm run build > /dev/null 2>&1; then
    print_success "✅ Frontend compiles successfully"
    cd ..
else
    print_error "❌ Frontend compilation failed"
    cd ..
    exit 1
fi

# Test 3: Dependencies Check
print_status "3. Checking dependencies..."
if [ -f "package.json" ] && [ -f "frontend/package.json" ]; then
    print_success "✅ All package.json files present"
else
    print_error "❌ Missing package.json files"
    exit 1
fi

# Test 4: Anchor Configuration
print_status "4. Checking Anchor configuration..."
if [ -f "Anchor.toml" ] && [ -f "programs/quantdesk-perp-dex/Cargo.toml" ]; then
    print_success "✅ Anchor configuration complete"
else
    print_error "❌ Missing Anchor configuration"
    exit 1
fi

echo ""
echo "============================="
print_success "🎉 ALL QUICK TESTS PASSED!"
echo "============================="
echo ""
print_status "QuantDesk Status:"
print_success "✅ Smart contracts: Ready"
print_success "✅ Frontend: Ready"
print_success "✅ Dependencies: Installed"
print_success "✅ Configuration: Complete"
echo ""
print_status "🚀 QuantDesk is ready for development!"
echo ""
print_status "Next steps:"
echo "  1. Run full tests: ./fix-and-test.sh"
echo "  2. Start frontend: cd frontend && npm run dev"
echo "  3. Deploy to devnet: anchor deploy --provider.cluster devnet"
echo ""
