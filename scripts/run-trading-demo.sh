#!/bin/bash

# 🚀 QuantDesk Trading Demo Runner
# This script runs the complete trading demo with smart contract simulation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

print_demo() {
    echo -e "${PURPLE}[DEMO]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to start backend if not running
start_backend() {
    if check_port 3002; then
        print_success "Backend already running on port 3002"
    else
        print_status "Starting backend server..."
        cd backend
        npm run dev > ../logs/backend.log 2>&1 &
        BACKEND_PID=$!
        cd ..
        
        # Wait for backend to start
        print_status "Waiting for backend to start..."
        sleep 5
        
        if check_port 3002; then
            print_success "Backend started successfully (PID: $BACKEND_PID)"
        else
            print_error "Failed to start backend"
            exit 1
        fi
    fi
}

# Function to start smart contract simulator
start_smart_contract_simulator() {
    print_status "Starting smart contract simulator..."
    node scripts/smart-contract-simulator.js > logs/smart-contracts.log 2>&1 &
    SIMULATOR_PID=$!
    
    sleep 2
    print_success "Smart contract simulator started (PID: $SIMULATOR_PID)"
}

# Function to run the trading demo
run_trading_demo() {
    print_demo "🚀 Starting QuantDesk Trading Demo!"
    print_demo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_demo ""
    print_demo "This demo simulates:"
    print_demo "  ✅ Wallet connection and authentication"
    print_demo "  ✅ User account creation"
    print_demo "  ✅ Token deposits (SOL, USDC, USDT, BTC, ETH)"
    print_demo "  ✅ Trading account management"
    print_demo "  ✅ Real-time order execution"
    print_demo "  ✅ Position management with PnL"
    print_demo "  ✅ Smart contract interactions"
    print_demo "  ✅ Liquidation events"
    print_demo "  ✅ Cross-collateral trading"
    print_demo ""
    print_demo "Press Ctrl+C to exit the demo"
    print_demo ""
    
    # Install demo dependencies if needed
    if [ ! -d "node_modules" ]; then
        print_status "Installing demo dependencies..."
        npm install chalk ora axios readline
    fi
    
    # Run the demo
    node scripts/trading-demo.js
}

# Function to show demo features
show_demo_features() {
    echo -e "${CYAN}🎯 QuantDesk Trading Demo Features${NC}"
    echo ""
    echo -e "${YELLOW}📱 User Experience:${NC}"
    echo "  • Interactive CLI interface with real-time updates"
    echo "  • Wallet connection simulation"
    echo "  • Account creation and management"
    echo "  • Token deposit/withdrawal flows"
    echo ""
    echo -e "${YELLOW}💰 Trading Features:${NC}"
    echo "  • Market orders with instant execution"
    echo "  • Limit orders with price validation"
    echo "  • Multiple trading accounts per user"
    echo "  • Cross-collateral support"
    echo "  • Real-time PnL calculation"
    echo ""
    echo -e "${YELLOW}🔗 Smart Contract Simulation:${NC}"
    echo "  • On-chain account creation"
    echo "  • Token transfers and deposits"
    echo "  • Order matching engine"
    echo "  • Position liquidation events"
    echo "  • Oracle price updates"
    echo ""
    echo -e "${YELLOW}📊 Advanced Features:${NC}"
    echo "  • Portfolio analytics"
    echo "  • Risk management"
    echo "  • Order history tracking"
    echo "  • Multi-account support"
    echo "  • Delegated trading"
    echo ""
}

# Function to clean up processes
cleanup() {
    print_status "Cleaning up processes..."
    
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        print_status "Backend stopped"
    fi
    
    if [ ! -z "$SIMULATOR_PID" ]; then
        kill $SIMULATOR_PID 2>/dev/null || true
        print_status "Smart contract simulator stopped"
    fi
    
    print_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "QuantDesk Trading Demo Runner"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  demo        Run the interactive trading demo (default)"
    echo "  features    Show demo features"
    echo "  backend     Start only the backend server"
    echo "  simulator   Start only the smart contract simulator"
    echo "  clean       Clean up running processes"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 demo"
    echo "  $0 features"
    echo "  $0 backend"
    echo ""
}

# Main function
main() {
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Set up signal handlers for cleanup
    trap cleanup EXIT INT TERM
    
    local command=${1:-demo}
    
    case $command in
        "demo")
            show_demo_features
            echo ""
            start_backend
            start_smart_contract_simulator
            echo ""
            run_trading_demo
            ;;
        "features")
            show_demo_features
            ;;
        "backend")
            start_backend
            print_success "Backend is running. Press Ctrl+C to stop."
            wait
            ;;
        "simulator")
            start_smart_contract_simulator
            print_success "Smart contract simulator is running. Press Ctrl+C to stop."
            wait
            ;;
        "clean")
            cleanup
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
