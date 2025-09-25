#!/bin/bash

# QuantDesk Quick Deployment Script
# Deploys the complete DEX to Solana devnet/testnet

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v solana &> /dev/null; then
        print_error "Solana CLI not found. Please install it first."
        exit 1
    fi
    
    if ! command -v anchor &> /dev/null; then
        print_error "Anchor CLI not found. Please install it first."
        exit 1
    fi
    
    if ! command -v node &> /dev/null; then
        print_error "Node.js not found. Please install it first."
        exit 1
    fi
    
    if ! command -v npm &> /dev/null; then
        print_error "npm not found. Please install it first."
        exit 1
    fi
    
    print_success "All prerequisites found"
}

# Function to deploy smart contracts
deploy_contracts() {
    local network=$1
    
    print_status "Deploying smart contracts to $network..."
    
    cd contracts/smart-contracts
    
    # Build and deploy
    anchor clean
    anchor build
    
    if [ "$network" = "devnet" ]; then
        solana config set --url https://api.devnet.solana.com
        solana airdrop 2
        anchor deploy --provider.cluster devnet
    elif [ "$network" = "testnet" ]; then
        solana config set --url https://api.testnet.solana.com
        solana airdrop 2
        anchor deploy --provider.cluster testnet
    fi
    
    # Get program ID
    local program_id=$(solana address --keypair target/deploy/quantdesk_perp_dex-keypair.json)
    print_success "Smart contracts deployed! Program ID: $program_id"
    
    cd ../..
}

# Function to setup backend
setup_backend() {
    print_status "Setting up backend..."
    
    cd backend
    
    # Install dependencies
    npm install
    
    # Create environment file
    cat > .env << EOF
NODE_ENV=development
PORT=3001
SUPABASE_URL=https://vabqtnsrmvccgegzvztv.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZhYnF0bnNybXZjY2dlZ3p2enR2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg2ODIxMTMsImV4cCI6MjA3NDI1ODExM30.Eof_X57BnJh5JnxUNjRBSkFa22GbQajN01aFACbwZW4
DATABASE_URL=postgresql://postgres:[YOUR_PASSWORD]@db.vabqtnsrmvccgegzvztv.supabase.co:5432/postgres
SOLANA_NETWORK=devnet
RPC_URL=https://api.devnet.solana.com
WS_URL=wss://api.devnet.solana.com
PROGRAM_ID=G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_EXPIRES_IN=7d
FRONTEND_URL=http://localhost:3000
EOF
    
    print_success "Backend setup complete"
    print_warning "Please update DATABASE_URL with your Supabase password"
    
    cd ..
}

# Function to setup frontend
setup_frontend() {
    print_status "Setting up frontend..."
    
    cd frontend
    
    # Install dependencies
    npm install
    
    # Create environment file
    cat > .env.local << EOF
VITE_SOLANA_NETWORK=devnet
VITE_RPC_URL=https://api.devnet.solana.com
VITE_WS_URL=wss://api.devnet.solana.com
VITE_PROGRAM_ID=G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J
VITE_BACKEND_URL=http://localhost:3001
EOF
    
    print_success "Frontend setup complete"
    
    cd ..
}

# Function to start services
start_services() {
    print_status "Starting QuantDesk services..."
    
    # Start backend in background
    cd backend
    npm run dev &
    BACKEND_PID=$!
    cd ..
    
    # Wait for backend to start
    sleep 5
    
    # Start frontend in background
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..
    
    print_success "Services started!"
    print_status "Backend PID: $BACKEND_PID"
    print_status "Frontend PID: $FRONTEND_PID"
    print_status "Backend: http://localhost:3001"
    print_status "Frontend: http://localhost:3000"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    cd contracts/smart-contracts
    anchor test --provider.cluster devnet
    cd ../..
    
    print_success "Tests completed"
}

# Function to show status
show_status() {
    print_status "QuantDesk Deployment Status"
    echo ""
    echo "✅ Smart Contracts: Deployed to devnet"
    echo "✅ Database: Supabase configured"
    echo "✅ Backend: Node.js API server"
    echo "✅ Frontend: React trading interface"
    echo "✅ Oracle: Pyth Network integration"
    echo "✅ WebSocket: Real-time data streaming"
    echo ""
    print_status "Next steps:"
    echo "1. Update DATABASE_URL in backend/.env"
    echo "2. Connect wallet to devnet"
    echo "3. Start trading!"
    echo ""
    print_status "Access your DEX:"
    echo "Frontend: http://localhost:3000"
    echo "Backend API: http://localhost:3001"
    echo "Health Check: http://localhost:3001/health"
}

# Function to show help
show_help() {
    echo "QuantDesk Quick Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [NETWORK]"
    echo ""
    echo "Commands:"
    echo "  deploy     Deploy complete DEX"
    echo "  contracts  Deploy only smart contracts"
    echo "  backend    Setup backend only"
    echo "  frontend   Setup frontend only"
    echo "  start      Start all services"
    echo "  test       Run tests"
    echo "  status     Show deployment status"
    echo "  help       Show this help"
    echo ""
    echo "Networks:"
    echo "  devnet     Solana devnet (default)"
    echo "  testnet    Solana testnet"
    echo ""
    echo "Examples:"
    echo "  $0 deploy devnet"
    echo "  $0 contracts testnet"
    echo "  $0 start"
}

# Main function
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    local command=$1
    local network=${2:-devnet}
    
    case $command in
        "deploy")
            check_prerequisites
            deploy_contracts $network
            setup_backend
            setup_frontend
            show_status
            ;;
        "contracts")
            check_prerequisites
            deploy_contracts $network
            ;;
        "backend")
            setup_backend
            ;;
        "frontend")
            setup_frontend
            ;;
        "start")
            start_services
            ;;
        "test")
            run_tests
            ;;
        "status")
            show_status
            ;;
        "help")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
