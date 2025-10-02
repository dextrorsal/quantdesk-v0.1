#!/bin/bash

# QuantDesk WSL Development Environment Setup Script
# Optimized for WSL2 Ubuntu 24.04 with best practices
# This script automates the setup of the development environment

set -e  # Exit on any error

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
        print_error "This script must be run in WSL Ubuntu 24.04"
        exit 1
    fi
    
    # Check WSL version
    WSL_VERSION=$(get_wsl_version)
    if [ "$WSL_VERSION" -lt 2 ]; then
        print_warning "WSL2 is recommended for better performance"
        print_warning "Consider upgrading to WSL2: wsl --set-default-version 2"
    fi
    
    # Check available memory
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$MEMORY_GB" -lt 4 ]; then
        print_warning "Low memory detected (${MEMORY_GB}GB). Consider allocating more RAM to WSL."
    fi
    
    # Check disk space
    DISK_SPACE=$(df -h . | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "${DISK_SPACE%.*}" -lt 10 ]; then
        print_warning "Low disk space detected. Ensure at least 10GB free space."
    fi
    
    print_success "System requirements check complete"
}

# Function to update system packages
update_system() {
    print_status "Updating system packages..."
    
    sudo apt update
    sudo apt upgrade -y
    
    print_success "System packages updated"
}

# Function to install essential packages
install_essential_packages() {
    print_status "Installing essential packages..."
    
    sudo apt install -y \
        curl \
        wget \
        git \
        build-essential \
        python3 \
        python3-pip \
        lsof \
        libssl-dev \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        jq \
        unzip \
        zip \
        rsync \
        vim \
        nano \
        htop \
        tree
    
    print_success "Essential packages installed"
}

# Function to configure Git
configure_git() {
    print_status "Configuring Git..."
    
    # Set Git configuration for WSL
    git config --global core.autocrlf input
    git config --global init.defaultBranch main
    
    # Configure user if not already set
    if [ -z "$(git config --global user.name)" ]; then
        read -p "Enter your Git username: " GIT_USERNAME
        git config --global user.name "$GIT_USERNAME"
    fi
    
    if [ -z "$(git config --global user.email)" ]; then
        read -p "Enter your Git email: " GIT_EMAIL
        git config --global user.email "$GIT_EMAIL"
    fi
    
    print_success "Git configured"
}

# Function to install Node.js 20.x
install_nodejs() {
    print_status "Installing Node.js 20.x..."
    
    if ! command_exists node; then
        # Install Node.js 20.x using NodeSource repository
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
    else
        NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
        if [ "$NODE_VERSION" -lt 20 ]; then
            print_warning "Node.js version is $NODE_VERSION, upgrading to 20.x..."
            curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
            sudo apt-get install -y nodejs
        fi
    fi
    
    # Verify Node.js installation
    NODE_VERSION=$(node --version)
    NPM_VERSION=$(npm --version)
    print_success "Node.js $NODE_VERSION and npm $NPM_VERSION installed"
}

# Function to install Yarn
install_yarn() {
    print_status "Installing Yarn..."
    
    if ! command_exists yarn; then
        npm install -g yarn
    fi
    
    YARN_VERSION=$(yarn --version)
    print_success "Yarn $YARN_VERSION installed"
}

# Function to check Docker installation
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command_exists docker; then
        print_warning "Docker not found. Please install Docker Desktop for Windows with WSL2 integration."
        print_warning "Visit: https://www.docker.com/products/docker-desktop/"
        print_warning "After installation, enable WSL2 integration in Docker Desktop settings."
        return 1
    else
        DOCKER_VERSION=$(docker --version)
        print_success "Docker found: $DOCKER_VERSION"
        
        # Test Docker
        if docker run --rm hello-world >/dev/null 2>&1; then
            print_success "Docker is working correctly"
        else
            print_warning "Docker is installed but not working. Check Docker Desktop WSL2 integration."
            return 1
        fi
    fi
}

# Function to set up shell environment
setup_shell_environment() {
    print_status "Setting up shell environment..."
    
    # Add Node.js to PATH (if not already there)
    if ! grep -q "node_modules/.bin" ~/.bashrc; then
        echo 'export PATH="$PATH:./node_modules/.bin"' >> ~/.bashrc
    fi
    
    # Add useful aliases
    if ! grep -q "QuantDesk aliases" ~/.bashrc; then
        cat >> ~/.bashrc << 'EOF'

# QuantDesk aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias qd='cd ~/quantdesk'
alias qd-start='cd ~/quantdesk && ./start-all-services.sh'
alias qd-stop='cd ~/quantdesk && pkill -f "node.*backend" && pkill -f "node.*frontend" && pkill -f "node.*admin"'
alias qd-logs='cd ~/quantdesk && tail -f logs/*.log'
alias qd-health='curl http://localhost:3001/health'
EOF
    fi
    
    print_success "Shell environment configured"
}

# Function to clone QuantDesk repository
clone_quantdesk() {
    print_status "Setting up QuantDesk project..."
    
    if [ ! -d "$HOME/quantdesk" ]; then
        print_status "Cloning QuantDesk repository..."
        cd "$HOME"
        git clone https://github.com/dextrorsal/quantdesk.git
        print_success "QuantDesk repository cloned"
    else
        print_status "QuantDesk repository already exists, updating..."
        cd "$HOME/quantdesk"
        git pull origin main
        print_success "QuantDesk repository updated"
    fi
}

# Function to install project dependencies
install_project_dependencies() {
    print_status "Installing QuantDesk dependencies..."
    cd "$HOME/quantdesk"
    
    # Make scripts executable
    chmod +x *.sh 2>/dev/null || true
    chmod +x scripts/*.sh 2>/dev/null || true
    
    # Install dependencies
    if [ -f "package.json" ]; then
        npm install
        
        # Install all service dependencies
        if [ -f "package.json" ] && grep -q "install:all" package.json; then
            npm run install:all
        else
            # Install dependencies for each service
            [ -d "backend" ] && cd backend && npm install && cd ..
            [ -d "frontend" ] && cd frontend && npm install && cd ..
            [ -d "admin-dashboard" ] && cd admin-dashboard && npm install && cd ..
            [ -d "data-ingestion" ] && cd data-ingestion && npm install && cd ..
        fi
    fi
    
    print_success "QuantDesk dependencies installed"
}

# Function to set up environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    cd "$HOME/quantdesk"
    
    if [ ! -f ".env" ]; then
        if [ -f "env.example" ]; then
            cp env.example .env
            print_warning "Environment file created from template"
            print_warning "Please edit .env file with your configuration"
        else
            print_warning "No env.example found, creating basic .env file"
            cat > .env << 'EOF'
# QuantDesk Environment Configuration
NODE_ENV=development

# Database
DATABASE_URL=postgresql://username:password@localhost:5432/quantdesk

# Redis
REDIS_URL=redis://localhost:6379

# JWT Secret
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production-minimum-32-chars

# Solana Network
SOLANA_NETWORK=devnet
RPC_URL=https://api.devnet.solana.com

# Pyth Network
PYTH_NETWORK_URL=https://hermes.pyth.network/v2/updates/price/latest

# Ports
BACKEND_PORT=3001
FRONTEND_PORT=5173
ADMIN_PORT=3000
EOF
        fi
    else
        print_status "Environment file already exists"
    fi
    
    print_success "Environment variables configured"
}

# Function to create useful scripts
create_useful_scripts() {
    print_status "Creating useful scripts..."
    
    cd "$HOME/quantdesk"
    
    # Create quick start script
    cat > "quick-start.sh" << 'EOF'
#!/bin/bash
echo "🚀 Starting QuantDesk services..."
cd ~/quantdesk
./start-all-services.sh
EOF
    chmod +x "quick-start.sh"
    
    # Create stop script
    cat > "quick-stop.sh" << 'EOF'
#!/bin/bash
echo "🛑 Stopping QuantDesk services..."
pkill -f "node.*backend" 2>/dev/null || true
pkill -f "node.*frontend" 2>/dev/null || true
pkill -f "node.*admin" 2>/dev/null || true
echo "✅ Services stopped"
EOF
    chmod +x "quick-stop.sh"
    
    # Create health check script
    cat > "health-check.sh" << 'EOF'
#!/bin/bash
echo "🔍 Checking QuantDesk services health..."
echo ""

echo "Backend (port 3001):"
curl -s http://localhost:3001/health && echo "✅ Backend is healthy" || echo "❌ Backend is not responding"

echo ""
echo "Frontend (port 5173):"
curl -s http://localhost:5173 >/dev/null && echo "✅ Frontend is healthy" || echo "❌ Frontend is not responding"

echo ""
echo "Admin Dashboard (port 3000):"
curl -s http://localhost:3000 >/dev/null && echo "✅ Admin Dashboard is healthy" || echo "❌ Admin Dashboard is not responding"

echo ""
echo "Running Node.js processes:"
ps aux | grep node | grep -v grep
EOF
    chmod +x "health-check.sh"
    
    print_success "Useful scripts created"
}

# Function to run final verification
verify_installation() {
    print_status "Running final verification..."
    
    echo ""
    echo "🔍 Installation Verification:"
    echo "=============================="
    
    # Check Node.js
    if command_exists node; then
        print_success "✅ Node.js: $(node --version)"
    else
        print_error "❌ Node.js not found"
    fi
    
    # Check npm
    if command_exists npm; then
        print_success "✅ npm: $(npm --version)"
    else
        print_error "❌ npm not found"
    fi
    
    # Check Yarn
    if command_exists yarn; then
        print_success "✅ Yarn: $(yarn --version)"
    else
        print_error "❌ Yarn not found"
    fi
    
    # Check Git
    if command_exists git; then
        print_success "✅ Git: $(git --version)"
    else
        print_error "❌ Git not found"
    fi
    
    # Check Docker
    if command_exists docker; then
        print_success "✅ Docker: $(docker --version)"
    else
        print_warning "⚠️  Docker not found (install Docker Desktop)"
    fi
    
    # Check project structure
    if [ -d "$HOME/quantdesk" ]; then
        print_success "✅ QuantDesk project: $HOME/quantdesk"
    else
        print_error "❌ QuantDesk project not found"
    fi
    
    echo ""
}

# Main execution function
main() {
    echo "🚀 QuantDesk WSL Development Environment Setup"
    echo "=============================================="
    echo ""
    
    # Run all setup functions
    check_system_requirements
    update_system
    install_essential_packages
    configure_git
    install_nodejs
    install_yarn
    check_docker
    setup_shell_environment
    clone_quantdesk
    install_project_dependencies
    setup_environment
    create_useful_scripts
    verify_installation
    
    echo ""
    echo "🎉 Setup Complete!"
    echo "================"
    echo ""
    echo "📋 Next steps:"
    echo "1. Edit ~/quantdesk/.env with your configuration"
    echo "2. Install Docker Desktop for Windows with WSL2 integration (if not done)"
    echo "3. Configure Cursor IDE to connect to WSL"
    echo "4. Start services: cd ~/quantdesk && ./quick-start.sh"
    echo ""
    echo "🔗 Useful commands:"
    echo "- Start services: cd ~/quantdesk && ./quick-start.sh"
    echo "- Stop services: cd ~/quantdesk && ./quick-stop.sh"
    echo "- Check health: cd ~/quantdesk && ./health-check.sh"
    echo "- View logs: cd ~/quantdesk && ./qd-logs"
    echo "- Open project: cd ~/quantdesk && cursor ."
    echo ""
    echo "📖 For detailed instructions, see: ~/quantdesk/WSL/WSL_SETUP_GUIDE.md"
    echo ""
    print_success "WSL development environment setup complete! 🚀"
}

# Run main function
main "$@"