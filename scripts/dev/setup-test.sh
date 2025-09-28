#!/bin/bash

echo "🚀 Setting up QuantDesk Perpetual DEX Test Environment..."

# Kill any existing validators
echo "📋 Cleaning up existing validators..."
pkill -f solana-test-validator || true

# Generate a new keypair if it doesn't exist
if [ ! -f ~/.config/solana/id.json ]; then
    echo "🔑 Generating new Solana keypair..."
    solana-keygen new --no-bip39-passphrase --silent --outfile ~/.config/solana/id.json
fi

# Set Solana to use local cluster
echo "⚙️  Configuring Solana cluster..."
solana config set --url localhost

# Check if we have SOL for testing
echo "💰 Checking SOL balance..."
BALANCE=$(solana balance)
if [ "$BALANCE" = "0 SOL" ]; then
    echo "💸 Airdropping SOL for testing..."
    solana airdrop 10
fi

echo "✅ Test environment setup complete!"
echo "📊 Current balance: $(solana balance)"
echo "🔗 Cluster: $(solana config get | grep 'RPC URL')"

# Run the tests
echo "🧪 Running automated tests..."
anchor test

echo "🎉 Test automation complete!"
