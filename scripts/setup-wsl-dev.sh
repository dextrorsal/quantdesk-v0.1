#!/bin/bash

# WSL Development Environment Setup Script for QuantDesk
# This script automates the setup process for Windows collaborators

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 QuantDesk WSL Development Environment Setup${NC}"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install packages
install_packages() {
    echo -e "\n${BLUE}📦 Installing base packages...${NC}"
    sudo apt update
    sudo apt install -y \
        build-essential \
        g++ \
        make \
        python3 \
        python3-pip \
        python-is-python3 \
        git \
        curl \
        unzip \
        pkg-config \
        libssl-dev \
        lsof \
        procps \
        ca-certificates \
        gnupg \
        lsb-release \
        jq
    echo -e "${GREEN}✅ Base packages installed${NC}"
}

# Function to install Node.js via NVM
install_nodejs() {
    echo -e "\n${BLUE}📦 Installing Node.js via NVM...${NC}"
    
    if ! command_exists nvm; then
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
        source ~/.bashrc
    fi
    
    nvm install 20
    nvm use 20
    nvm alias default 20
    
    echo -e "${GREEN}✅ Node.js $(node -v) installed${NC}"
}

# Function to setup Docker
setup_docker() {
    echo -e "\n${BLUE}🐳 Setting up Docker...${NC}"
    
    if ! command_exists docker; then
        echo -e "${YELLOW}⚠️  Docker not found. Please install Docker Desktop for Windows and enable WSL integration.${NC}"
        echo -e "${YELLOW}   Then run: docker --version${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Docker is available${NC}"
}

# Function to setup PostgreSQL
setup_postgresql() {
    echo -e "\n${BLUE}🐘 Setting up PostgreSQL...${NC}"
    
    if ! command_exists psql; then
        sudo apt install -y postgresql postgresql-contrib
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
    fi
    
    # Create database and user
    sudo -u postgres psql -c "CREATE DATABASE quantdesk;" 2>/dev/null || true
    sudo -u postgres psql -c "CREATE USER quantdesk WITH PASSWORD 'quantdesk123';" 2>/dev/null || true
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE quantdesk TO quantdesk;" 2>/dev/null || true
    
    echo -e "${GREEN}✅ PostgreSQL configured${NC}"
}

# Function to setup Redis
setup_redis() {
    echo -e "\n${BLUE}🔴 Setting up Redis...${NC}"
    
    if ! command_exists redis-server; then
        sudo apt install -y redis-server
        sudo systemctl start redis-server
        sudo systemctl enable redis-server
    fi
    
    echo -e "${GREEN}✅ Redis configured${NC}"
}

# Function to configure Git
configure_git() {
    echo -e "\n${BLUE}🔧 Configuring Git...${NC}"
    
    git config --global core.autocrlf input
    
    if [ -z "$(git config --global user.name)" ]; then
        echo -e "${YELLOW}Please configure Git user:${NC}"
        read -p "Enter your name: " git_name
        read -p "Enter your email: " git_email
        git config --global user.name "$git_name"
        git config --global user.email "$git_email"
    fi
    
    echo -e "${GREEN}✅ Git configured${NC}"
}

# Function to install project dependencies
install_dependencies() {
    echo -e "\n${BLUE}📦 Installing project dependencies...${NC}"
    
    # Install root dependencies
    npm run install:all
    
    # Install data-ingestion dependencies
    cd data-ingestion
    npm install
    cd ..
    
    echo -e "${GREEN}✅ Dependencies installed${NC}"
}

# Function to setup environment files
setup_environment() {
    echo -e "\n${BLUE}⚙️  Setting up environment files...${NC}"
    
    # Copy environment templates
    cp env.example .env
    cp data-ingestion/env.example data-ingestion/.env
    
    echo -e "${GREEN}✅ Environment files created${NC}"
    echo -e "${YELLOW}⚠️  Please edit .env and data-ingestion/.env with your actual values${NC}"
}

# Function to make scripts executable
make_scripts_executable() {
    echo -e "\n${BLUE}🔧 Making scripts executable...${NC}"
    
    chmod +x start-all-services.sh
    chmod +x backend/start-backend.sh
    chmod +x admin-dashboard/start-admin.sh
    chmod +x frontend/start-frontend.sh
    chmod +x data-ingestion/start-pipeline.sh
    chmod +x scripts/*.sh
    
    echo -e "${GREEN}✅ Scripts made executable${NC}"
}

# Function to verify installation
verify_installation() {
    echo -e "\n${BLUE}🔍 Verifying installation...${NC}"
    
    echo -e "Node.js: $(node -v)"
    echo -e "npm: $(npm -v)"
    echo -e "Git: $(git --version)"
    
    if command_exists docker; then
        echo -e "Docker: $(docker --version)"
    fi
    
    if command_exists psql; then
        echo -e "PostgreSQL: $(psql --version)"
    fi
    
    if command_exists redis-server; then
        echo -e "Redis: $(redis-server --version)"
    fi
    
    echo -e "${GREEN}✅ Installation verification complete${NC}"
}

# Main execution
main() {
    echo -e "${GREEN}Starting setup process...${NC}"
    
    install_packages
    install_nodejs
    setup_docker
    setup_postgresql
    setup_redis
    configure_git
    install_dependencies
    setup_environment
    make_scripts_executable
    verify_installation
    
    echo -e "\n${GREEN}🎉 Setup complete!${NC}"
    echo -e "\n${BLUE}📋 Next steps:${NC}"
    echo -e "1. Edit .env files with your actual values"
    echo -e "2. Start services: ./start-all-services.sh"
    echo -e "3. Open Cursor: cursor ."
    echo -e "4. Run dependency audit: ./scripts/socket-audit.sh"
    
    echo -e "\n${BLUE}🔗 Useful commands:${NC}"
    echo -e "  Start all services: ./start-all-services.sh"
    echo -e "  Check service health: curl http://localhost:3002/health"
    echo -e "  View logs: tail -f logs/*.log"
    echo -e "  Stop services: pkill -f 'node.*backend'"
    
    echo -e "\n${GREEN}✨ Happy coding!${NC}"
}

# Run main function
main "$@"
