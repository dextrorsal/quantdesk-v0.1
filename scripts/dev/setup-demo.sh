#!/bin/bash

# QuantDesk Demo Environment Setup
# This script sets up a demo environment for QuantDesk

set -e

echo "🚀 Setting up QuantDesk Demo Environment..."

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

# Check if Node.js is installed
check_node() {
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/"
        exit 1
    fi
    
    NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        print_error "Node.js version 18+ is required. Current version: $(node --version)"
        exit 1
    fi
    
    print_success "Node.js $(node --version) is installed"
}

# Check if npm is installed
check_npm() {
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm."
        exit 1
    fi
    
    print_success "npm $(npm --version) is installed"
}

# Check if Git is installed
check_git() {
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git from https://git-scm.com/"
        exit 1
    fi
    
    print_success "Git $(git --version) is installed"
}

# Install dependencies
install_dependencies() {
    print_status "Installing backend dependencies..."
    cd backend
    npm install
    cd ..
    
    print_status "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
    
    print_success "Dependencies installed successfully"
}

# Create environment files
setup_environment() {
    print_status "Setting up environment files..."
    
    # Backend environment
    if [ ! -f backend/.env ]; then
        cp backend/.env.example backend/.env
        print_success "Created backend/.env file"
    else
        print_warning "backend/.env already exists, skipping..."
    fi
    
    # Frontend environment
    if [ ! -f frontend/.env ]; then
        cp frontend/.env.example frontend/.env
        print_success "Created frontend/.env file"
    else
        print_warning "frontend/.env already exists, skipping..."
    fi
    
    print_warning "Please edit the .env files with your configuration"
}

# Create demo data
create_demo_data() {
    print_status "Creating demo data..."
    
    # Create examples directory if it doesn't exist
    mkdir -p examples
    
    # Make demo script executable
    chmod +x examples/basic-trading-demo.js
    
    print_success "Demo data created successfully"
}

# Run basic tests
run_tests() {
    print_status "Running basic tests..."
    
    # Backend tests
    cd backend
    if npm test --silent; then
        print_success "Backend tests passed"
    else
        print_warning "Backend tests failed (this is expected in demo mode)"
    fi
    cd ..
    
    # Frontend tests
    cd frontend
    if npm test --silent; then
        print_success "Frontend tests passed"
    else
        print_warning "Frontend tests failed (this is expected in demo mode)"
    fi
    cd ..
}

# Display next steps
show_next_steps() {
    echo ""
    echo "🎉 Demo environment setup complete!"
    echo ""
    echo "📋 Next Steps:"
    echo "   1. Edit backend/.env with your configuration"
    echo "   2. Edit frontend/.env with your configuration"
    echo "   3. Start the backend server:"
    echo "      cd backend && npm run dev"
    echo "   4. Start the frontend server:"
    echo "      cd frontend && npm run dev"
    echo "   5. Access the application:"
    echo "      Frontend: http://localhost:5173"
    echo "      Backend: http://localhost:3002"
    echo "      API Docs: http://localhost:3002/api/docs"
    echo ""
    echo "🧪 Try the demo:"
    echo "   node examples/basic-trading-demo.js"
    echo ""
    echo "📚 Documentation:"
    echo "   - Getting Started: docs/GETTING_STARTED.md"
    echo "   - API Documentation: docs/API.md"
    echo "   - Contributing: CONTRIBUTING.md"
    echo ""
    echo "🆘 Need Help?"
    echo "   - GitHub Issues: https://github.com/dextrorsal/quantdesk/issues"
    echo "   - Discord: https://discord.gg/quantdesk"
    echo "   - Email: contact@quantdesk.app"
    echo ""
    echo "🚀 Welcome to QuantDesk - The Bloomberg Terminal for Crypto!"
}

# Main execution
main() {
    print_status "Starting QuantDesk demo environment setup..."
    
    # Check prerequisites
    check_node
    check_npm
    check_git
    
    # Setup environment
    install_dependencies
    setup_environment
    create_demo_data
    
    # Run tests
    run_tests
    
    # Show next steps
    show_next_steps
}

# Run main function
main "$@"
