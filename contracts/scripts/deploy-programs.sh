#!/bin/bash

# QuantDesk Program Deployment Script
# This script deploys all specialized QuantDesk programs

set -e

echo "🚀 Starting QuantDesk Program Deployment"
echo "========================================"

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

# Check if Solana CLI is installed
if ! command -v solana &> /dev/null; then
    print_error "Solana CLI is not installed. Please install it first."
    exit 1
fi

# Check if Anchor is installed
if ! command -v anchor &> /dev/null; then
    print_error "Anchor CLI is not installed. Please install it first."
    exit 1
fi

# Set cluster (default to devnet)
CLUSTER=${1:-devnet}
print_status "Deploying to cluster: $CLUSTER"

# Set Solana cluster
solana config set --url $CLUSTER

# Check wallet balance
WALLET_BALANCE=$(solana balance)
print_status "Wallet balance: $WALLET_BALANCE SOL"

# Check if wallet has enough SOL for deployment
if (( $(echo "$WALLET_BALANCE < 5.0" | bc -l) )); then
    print_warning "Wallet balance is low. You may need more SOL for deployment."
    print_status "Requesting airdrop..."
    solana airdrop 2
fi

# Function to deploy a program
deploy_program() {
    local program_name=$1
    local keypair_file=$2
    
    print_status "Deploying $program_name..."
    
    if [ ! -f "$keypair_file" ]; then
        print_error "Keypair file $keypair_file not found!"
        return 1
    fi
    
    # Build the program
    print_status "Building $program_name..."
    anchor build --program-name $program_name
    
    if [ $? -eq 0 ]; then
        print_success "$program_name built successfully"
    else
        print_error "Failed to build $program_name"
        return 1
    fi
    
    # Deploy the program
    print_status "Deploying $program_name to $CLUSTER..."
    anchor deploy --program-name $program_name --program-keypair $keypair_file
    
    if [ $? -eq 0 ]; then
        print_success "$program_name deployed successfully"
    else
        print_error "Failed to deploy $program_name"
        return 1
    fi
}

# Deploy all programs
print_status "Starting deployment of all QuantDesk programs..."

# Deploy Core Program
deploy_program "quantdesk_core" "quantdesk-core-keypair.json"

# Deploy Trading Program
deploy_program "quantdesk_trading" "quantdesk-trading-keypair.json"

# Deploy Collateral Program
deploy_program "quantdesk_collateral" "quantdesk-collateral-keypair.json"

# Deploy Security Program
deploy_program "quantdesk_security" "quantdesk-security-keypair.json"

# Deploy Oracle Program
deploy_program "quantdesk_oracle" "quantdesk-oracle-keypair.json"

# Deploy original monolithic program (for comparison)
deploy_program "quantdesk_perp_dex" "target/deploy/quantdesk_perp_dex-keypair.json"

print_success "All QuantDesk programs deployed successfully!"
print_status "Deployment completed on cluster: $CLUSTER"

# Display program IDs
echo ""
echo "📋 Program IDs:"
echo "==============="
echo "Core Program: CNfhSBoMkRbDEQ2EC3RkfJ2S39Up6WJLr4U31ZL49LrU"
echo "Trading Program: AvxWXu25yWhDXJBy1V5GYcn2eVws4F2QWK5G3zV4t8sZ"
echo "Collateral Program: GPrakftrbBUUiir2MpQZv6G7UB5Jq8yNGHV5YTVYPQ5i"
echo "Security Program: 84b7Khx4uj7mHDvn2V63kNSwkcgpagrBgZSdTJ7kTxWW"
echo "Oracle Program: 8gjwta4tMQshM7HbnEMsdFUMqjRe7XgVnxJVbcmf3cAC"
echo "Original Program: C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw"

echo ""
print_success "🎉 QuantDesk Program Deployment Complete!"
