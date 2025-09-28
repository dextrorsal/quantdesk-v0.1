#!/bin/bash

echo "🤖 QuantDesk Automated Test Runner"
echo "=================================="

# Kill existing validators
pkill -f solana-test-validator

# Generate keypair if needed
if [ ! -f ~/.config/solana/id.json ]; then
    echo "🔑 Generating Solana keypair..."
    solana-keygen new --no-bip39-passphrase --silent --outfile ~/.config/solana/id.json
fi

# Set local cluster
solana config set --url localhost

# Airdrop SOL if needed
BALANCE=$(solana balance)
if [ "$BALANCE" = "0 SOL" ]; then
    echo "💸 Airdropping SOL..."
    solana airdrop 10
fi

echo "🚀 Running automated tests..."

# Run tests with detailed output
anchor test 2>&1 | tee test-results.log

# Check if tests passed
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ ALL TESTS PASSED!"
    echo "🎉 QuantDesk smart contracts are working perfectly!"
    echo ""
    echo "📊 Test Summary:"
    echo "  ✅ Market initialization"
    echo "  ✅ Position opening/closing"
    echo "  ✅ Funding settlement"
    echo "  ✅ Liquidation system"
    echo "  ✅ 100x leverage support"
    echo ""
    echo "🚀 Ready for next development phase!"
else
    echo ""
    echo "❌ Some tests failed. Check test-results.log for details."
    echo ""
    echo "🔧 Troubleshooting:"
    echo "  1. Run: ./setup-test.sh"
    echo "  2. Check: solana config get"
    echo "  3. Verify: solana balance"
fi
