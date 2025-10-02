#!/bin/bash

# QuantDesk WSL Setup Validation Script
# This script validates that the WSL development environment is properly configured

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

# Function to check if running in WSL
is_wsl() {
    if [ -f /proc/version ]; then
        grep -q Microsoft /proc/version
    else
        return 1
    fi
}

# Function to get WSL version
get_wsl_version() {
    if is_wsl; then
        wsl --version 2>/dev/null | grep -o "WSL version [0-9]" | cut -d' ' -f3 || echo "1"
    else
        echo "0"
    fi
}

# Function to check system requirements
check_system_requirements() {
    print_status "Checking system requirements..."
    
    # Check if running in WSL
    if ! is_wsl; then
        print_error "❌ Not running in WSL"
        return 1
    else
        print_success "✅ Running in WSL"
    fi
    
    # Check WSL version
    WSL_VERSION=$(get_wsl_version)
    if [ "$WSL_VERSION" -ge 2 ]; then
        print_success "✅ WSL2 detected"
    else
        print_warning "⚠️  WSL1 detected (WSL2 recommended)"
    fi
    
    # Check Ubuntu version
    if [ -f /etc/os-release ]; then
        UBUNTU_VERSION=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
        if [[ "$UBUNTU_VERSION" == "24.04" ]]; then
            print_success "✅ Ubuntu 24.04 detected"
        else
            print_warning "⚠️  Ubuntu $UBUNTU_VERSION detected (24.04 recommended)"
        fi
    fi
    
    # Check available memory
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$MEMORY_GB" -ge 4 ]; then
        print_success "✅ Sufficient memory: ${MEMORY_GB}GB"
    else
        print_warning "⚠️  Low memory: ${MEMORY_GB}GB (4GB+ recommended)"
    fi
    
    # Check disk space
    DISK_SPACE=$(df -h . | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "${DISK_SPACE%.*}" -ge 10 ]; then
        print_success "✅ Sufficient disk space: ${DISK_SPACE}GB free"
    else
        print_warning "⚠️  Low disk space: ${DISK_SPACE}GB free (10GB+ recommended)"
    fi
}

# Function to check Node.js installation
check_nodejs() {
    print_status "Checking Node.js installation..."
    
    if command_exists node; then
        NODE_VERSION=$(node --version)
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d'v' -f2 | cut -d'.' -f1)
        
        if [ "$NODE_MAJOR" -ge 20 ]; then
            print_success "✅ Node.js $NODE_VERSION (20.x+ required)"
        else
            print_error "❌ Node.js $NODE_VERSION (20.x+ required)"
            return 1
        fi
    else
        print_error "❌ Node.js not found"
        return 1
    fi
    
    if command_exists npm; then
        NPM_VERSION=$(npm --version)
        print_success "✅ npm $NPM_VERSION"
    else
        print_error "❌ npm not found"
        return 1
    fi
    
    if command_exists yarn; then
        YARN_VERSION=$(yarn --version)
        print_success "✅ Yarn $YARN_VERSION"
    else
        print_warning "⚠️  Yarn not found (optional but recommended)"
    fi
}

# Function to check Docker installation
check_docker() {
    print_status "Checking Docker installation..."
    
    if command_exists docker; then
        DOCKER_VERSION=$(docker --version)
        print_success "✅ Docker: $DOCKER_VERSION"
        
        # Test Docker
        if docker run --rm hello-world >/dev/null 2>&1; then
            print_success "✅ Docker is working correctly"
        else
            print_warning "⚠️  Docker is installed but not working"
            print_warning "   Check Docker Desktop WSL2 integration"
        fi
    else
        print_warning "⚠️  Docker not found"
        print_warning "   Install Docker Desktop for Windows with WSL2 integration"
    fi
}

# Function to check Git configuration
check_git() {
    print_status "Checking Git configuration..."
    
    if command_exists git; then
        GIT_VERSION=$(git --version)
        print_success "✅ Git: $GIT_VERSION"
        
        # Check Git configuration
        if [ -n "$(git config --global user.name)" ]; then
            print_success "✅ Git user.name configured"
        else
            print_warning "⚠️  Git user.name not configured"
        fi
        
        if [ -n "$(git config --global user.email)" ]; then
            print_success "✅ Git user.email configured"
        else
            print_warning "⚠️  Git user.email not configured"
        fi
        
        # Check WSL-specific Git settings
        if [ "$(git config --global core.autocrlf)" = "input" ]; then
            print_success "✅ Git core.autocrlf configured for WSL"
        else
            print_warning "⚠️  Git core.autocrlf not configured for WSL"
        fi
    else
        print_error "❌ Git not found"
        return 1
    fi
}

# Function to check QuantDesk project
check_quantdesk_project() {
    print_status "Checking QuantDesk project..."
    
    if [ -d "$HOME/quantdesk" ]; then
        print_success "✅ QuantDesk project directory exists"
        
        cd "$HOME/quantdesk"
        
        # Check if it's a Git repository
        if [ -d ".git" ]; then
            print_success "✅ QuantDesk is a Git repository"
        else
            print_warning "⚠️  QuantDesk directory is not a Git repository"
        fi
        
        # Check package.json
        if [ -f "package.json" ]; then
            print_success "✅ package.json exists"
        else
            print_error "❌ package.json not found"
            return 1
        fi
        
        # Check environment file
        if [ -f ".env" ]; then
            print_success "✅ .env file exists"
        else
            print_warning "⚠️  .env file not found (copy from env.example)"
        fi
        
        # Check if node_modules exists
        if [ -d "node_modules" ]; then
            print_success "✅ Node.js dependencies installed"
        else
            print_warning "⚠️  Node.js dependencies not installed (run npm install)"
        fi
        
        # Check startup scripts
        if [ -f "start-all-services.sh" ]; then
            print_success "✅ start-all-services.sh exists"
        else
            print_warning "⚠️  start-all-services.sh not found"
        fi
        
        # Check individual service directories
        [ -d "backend" ] && print_success "✅ Backend directory exists"
        [ -d "frontend" ] && print_success "✅ Frontend directory exists"
        [ -d "admin-dashboard" ] && print_success "✅ Admin dashboard directory exists"
        
    else
        print_error "❌ QuantDesk project directory not found"
        print_error "   Clone the repository: git clone https://github.com/dextrorsal/quantdesk.git"
        return 1
    fi
}

# Function to check services
check_services() {
    print_status "Checking QuantDesk services..."
    
    # Check if services are running
    if pgrep -f "node.*backend" >/dev/null; then
        print_success "✅ Backend service is running"
    else
        print_warning "⚠️  Backend service is not running"
    fi
    
    if pgrep -f "node.*frontend" >/dev/null; then
        print_success "✅ Frontend service is running"
    else
        print_warning "⚠️  Frontend service is not running"
    fi
    
    if pgrep -f "node.*admin" >/dev/null; then
        print_success "✅ Admin dashboard service is running"
    else
        print_warning "⚠️  Admin dashboard service is not running"
    fi
    
    # Check service health endpoints
    if curl -s http://localhost:3001/health >/dev/null 2>&1; then
        print_success "✅ Backend health endpoint responding"
    else
        print_warning "⚠️  Backend health endpoint not responding"
    fi
    
    if curl -s http://localhost:5173 >/dev/null 2>&1; then
        print_success "✅ Frontend service responding"
    else
        print_warning "⚠️  Frontend service not responding"
    fi
    
    if curl -s http://localhost:3000 >/dev/null 2>&1; then
        print_success "✅ Admin dashboard responding"
    else
        print_warning "⚠️  Admin dashboard not responding"
    fi
}

# Function to check Cursor IDE
check_cursor_ide() {
    print_status "Checking Cursor IDE configuration..."
    
    # Check if Cursor is installed on Windows
    if command_exists cursor; then
        print_success "✅ Cursor CLI is available"
    else
        print_warning "⚠️  Cursor CLI not found"
        print_warning "   Install Cursor IDE and WSL extension"
    fi
    
    # Check if we can open the project
    if [ -d "$HOME/quantdesk" ]; then
        print_success "✅ Project directory accessible for Cursor"
    else
        print_warning "⚠️  Project directory not accessible"
    fi
}

# Function to generate summary report
generate_summary() {
    echo ""
    echo "📊 Validation Summary"
    echo "===================="
    echo ""
    
    # Count successes and warnings
    SUCCESS_COUNT=$(grep -c "✅" <<< "$OUTPUT" || echo "0")
    WARNING_COUNT=$(grep -c "⚠️" <<< "$OUTPUT" || echo "0")
    ERROR_COUNT=$(grep -c "❌" <<< "$OUTPUT" || echo "0")
    
    echo "✅ Successes: $SUCCESS_COUNT"
    echo "⚠️  Warnings: $WARNING_COUNT"
    echo "❌ Errors: $ERROR_COUNT"
    echo ""
    
    if [ "$ERROR_COUNT" -eq 0 ]; then
        if [ "$WARNING_COUNT" -eq 0 ]; then
            print_success "🎉 All checks passed! Your WSL setup is perfect."
        else
            print_warning "⚠️  Setup is functional but has some warnings."
            print_warning "   Review the warnings above for potential improvements."
        fi
    else
        print_error "❌ Setup has errors that need to be fixed."
        print_error "   Review the errors above and run the setup script again."
    fi
    
    echo ""
    echo "📋 Next Steps:"
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "1. Fix the errors listed above"
        echo "2. Run the setup script: ./setup-wsl-dev.sh"
        echo "3. Re-run this validation: ./test-wsl-setup.sh"
    else
        echo "1. Start services: cd ~/quantdesk && ./quick-start.sh"
        echo "2. Check health: cd ~/quantdesk && ./health-check.sh"
        echo "3. Open in Cursor: cd ~/quantdesk && cursor ."
    fi
    echo ""
}

# Main execution function
main() {
    echo "🔍 QuantDesk WSL Setup Validation"
    echo "================================="
    echo ""
    
    # Capture output for summary
    OUTPUT=""
    
    # Run all checks
    OUTPUT+=$(check_system_requirements 2>&1)
    OUTPUT+=$(check_nodejs 2>&1)
    OUTPUT+=$(check_docker 2>&1)
    OUTPUT+=$(check_git 2>&1)
    OUTPUT+=$(check_quantdesk_project 2>&1)
    OUTPUT+=$(check_services 2>&1)
    OUTPUT+=$(check_cursor_ide 2>&1)
    
    # Generate summary
    generate_summary
}

# Run main function
main "$@"
