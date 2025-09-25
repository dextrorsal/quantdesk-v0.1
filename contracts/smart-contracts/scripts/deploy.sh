#!/bin/bash

# QuantDesk Smart Contract Deployment Script
# Supports localnet, devnet, and testnet deployment

set -e

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists solana; then
        print_error "Solana CLI not found. Please install it first."
        exit 1
    fi
    
    if ! command_exists anchor; then
        print_error "Anchor CLI not found. Please install it first."
        exit 1
    fi
    
    if ! command_exists cargo; then
        print_error "Cargo not found. Please install Rust first."
        exit 1
    fi
    
    print_success "All prerequisites found"
}

# Function to set up network
setup_network() {
    local network=$1
    
    print_status "Setting up Solana network: $network"
    
    case $network in
        "localnet")
            solana config set --url localhost
            ;;
        "devnet")
            solana config set --url https://api.devnet.solana.com
            ;;
        "testnet")
            solana config set --url https://api.testnet.solana.com
            ;;
        *)
            print_error "Invalid network: $network. Use localnet, devnet, or testnet"
            exit 1
            ;;
    esac
    
    print_success "Network set to $network"
}

# Function to check wallet and SOL balance
check_wallet() {
    local network=$1
    
    print_status "Checking wallet configuration..."
    
    # Check if wallet exists
    if [ ! -f ~/.config/solana/id.json ]; then
        print_warning "No wallet found. Creating new wallet..."
        solana-keygen new --no-bip39-passphrase --silent
    fi
    
    # Get wallet address
    local wallet_address=$(solana address)
    print_status "Wallet address: $wallet_address"
    
    # Check SOL balance
    local balance=$(solana balance)
    print_status "Current SOL balance: $balance"
    
    # Request airdrop if balance is low
    if [ "$network" != "localnet" ] && [ "$balance" -lt 1 ]; then
        print_status "Requesting SOL airdrop..."
        solana airdrop 2
        local new_balance=$(solana balance)
        print_success "New SOL balance: $new_balance"
    fi
}

# Function to build the program
build_program() {
    print_status "Building smart contract..."
    
    # Clean previous build
    anchor clean
    
    # Build the program
    anchor build
    
    if [ $? -eq 0 ]; then
        print_success "Smart contract built successfully"
    else
        print_error "Build failed"
        exit 1
    fi
}

# Function to deploy the program
deploy_program() {
    local network=$1
    
    print_status "Deploying smart contract to $network..."
    
    # Deploy the program
    anchor deploy --provider.cluster $network
    
    if [ $? -eq 0 ]; then
        print_success "Smart contract deployed successfully to $network"
        
        # Get program ID
        local program_id=$(solana address --keypair target/deploy/quantdesk_perp_dex-keypair.json)
        print_status "Program ID: $program_id"
        
        # Verify deployment
        solana program show $program_id
        
    else
        print_error "Deployment failed"
        exit 1
    fi
}

# Function to run tests
run_tests() {
    local network=$1
    
    print_status "Running tests on $network..."
    
    # Run Anchor tests
    anchor test --provider.cluster $network
    
    if [ $? -eq 0 ]; then
        print_success "All tests passed"
    else
        print_error "Tests failed"
        exit 1
    fi
}

# Function to initialize markets
initialize_markets() {
    local network=$1
    
    print_status "Initializing markets on $network..."
    
    # This would run a script to initialize markets
    # For now, just log the action
    print_status "Market initialization would happen here"
    print_warning "Manual market initialization required"
}

# Function to update environment files
update_env_files() {
    local network=$1
    local program_id=$1
    
    print_status "Updating environment files..."
    
    # Update backend environment
    if [ -f "../../backend/src/config/environment.ts" ]; then
        sed -i "s/SOLANA_NETWORK=.*/SOLANA_NETWORK=$network/" "../../backend/src/config/environment.ts"
        sed -i "s/PROGRAM_ID=.*/PROGRAM_ID=$program_id/" "../../backend/src/config/environment.ts"
        
        case $network in
            "localnet")
                sed -i "s/RPC_URL=.*/RPC_URL=http:\/\/localhost:8899/" "../../backend/src/config/environment.ts"
                ;;
            "devnet")
                sed -i "s/RPC_URL=.*/RPC_URL=https:\/\/api.devnet.solana.com/" "../../backend/src/config/environment.ts"
                ;;
            "testnet")
                sed -i "s/RPC_URL=.*/RPC_URL=https:\/\/api.testnet.solana.com/" "../../backend/src/config/environment.ts"
                ;;
        esac
        
        print_success "Environment files updated"
    fi
}

# Main deployment function
deploy() {
    local network=$1
    local run_tests_flag=$2
    
    print_status "Starting QuantDesk deployment to $network"
    
    # Check prerequisites
    check_prerequisites
    
    # Set up network
    setup_network $network
    
    # Check wallet
    check_wallet $network
    
    # Build program
    build_program
    
    # Deploy program
    deploy_program $network
    
    # Run tests if requested
    if [ "$run_tests_flag" = "--test" ]; then
        run_tests $network
    fi
    
    # Initialize markets
    initialize_markets $network
    
    # Update environment files
    update_env_files $network
    
    print_success "Deployment to $network completed successfully!"
    print_status "Next steps:"
    print_status "1. Update your frontend configuration with the new program ID"
    print_status "2. Initialize markets using the admin interface"
    print_status "3. Test trading functionality"
}

# Function to show help
show_help() {
    echo "QuantDesk Smart Contract Deployment Script"
    echo ""
    echo "Usage: $0 [NETWORK] [OPTIONS]"
    echo ""
    echo "Networks:"
    echo "  localnet    Deploy to local Solana validator"
    echo "  devnet      Deploy to Solana devnet"
    echo "  testnet     Deploy to Solana testnet"
    echo ""
    echo "Options:"
    echo "  --test      Run tests after deployment"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 devnet --test"
    echo "  $0 localnet"
    echo "  $0 testnet"
}

# Main script logic
main() {
    if [ $# -eq 0 ] || [ "$1" = "--help" ]; then
        show_help
        exit 0
    fi
    
    local network=$1
    local test_flag=$2
    
    # Validate network
    if [[ ! "$network" =~ ^(localnet|devnet|testnet)$ ]]; then
        print_error "Invalid network: $network"
        show_help
        exit 1
    fi
    
    # Deploy
    deploy $network $test_flag
}

# Run main function with all arguments
main "$@"
