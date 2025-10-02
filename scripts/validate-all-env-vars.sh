#!/bin/bash

# QuantDesk Environment Variable Validation Script
# Validates environment variables for all services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to extract environment variables from code
extract_env_vars() {
    local dir="$1"
    
    if [ -d "$dir" ]; then
        find "$dir" -name "*.js" -o -name "*.ts" -o -name "*.tsx" -o -name "*.jsx" 2>/dev/null | \
        xargs grep -h "process\.env\|import\.meta\.env" 2>/dev/null | \
        grep -o "process\.env\.[A-Z_][A-Z0-9_]*\|process\.env\['[A-Z_][A-Z0-9_]*'\]\|import\.meta\.env\.[A-Z_][A-Z0-9_]*" | \
        sed 's/process\.env\.//g' | \
        sed "s/process\.env\['//g" | \
        sed "s/'\]//g" | \
        sed 's/import\.meta\.env\.//g' | \
        sort | uniq
    fi
}

# Function to extract variables from .env file
extract_env_file_vars() {
    local env_file="$1"
    if [ -f "$env_file" ]; then
        grep "^[A-Z_][A-Z0-9_]*=" "$env_file" 2>/dev/null | cut -d'=' -f1 | sort | uniq
    fi
}

# Function to validate service
validate_service() {
    local service_name="$1"
    local service_dir="$2"
    local env_file="$3"
    
    echo ""
    echo "🔍 Validating $service_name..."
    echo "================================"
    
    if [ ! -d "$service_dir" ]; then
        print_error "$service_dir directory not found"
        return 1
    fi
    
    if [ ! -f "$env_file" ]; then
        print_error "$env_file not found"
        return 1
    fi
    
    print_status "Extracting environment variables from $service_name code..."
    local code_vars=$(extract_env_vars "$service_dir")
    
    print_status "Extracting variables from $env_file..."
    local env_vars=$(extract_env_file_vars "$env_file")
    
    echo ""
    echo "📊 $service_name Results:"
    echo "• Variables used in code: $(echo "$code_vars" | wc -l)"
    echo "• Variables defined in .env: $(echo "$env_vars" | wc -l)"
    
    # Show first few variables as examples
    if [ -n "$code_vars" ]; then
        echo "• Code variables (first 5): $(echo "$code_vars" | head -5 | tr '\n' ' ')"
    fi
    
    if [ -n "$env_vars" ]; then
        echo "• Env variables (first 5): $(echo "$env_vars" | head -5 | tr '\n' ' ')"
    fi
}

# Main function
main() {
    echo "🔍 QuantDesk Environment Variable Validation"
    echo "============================================="
    
    validate_service "Backend" "backend/src" "backend/.env"
    validate_service "Data Ingestion" "data-ingestion/src" "data-ingestion/.env"
    validate_service "Frontend" "frontend/src" "frontend/.env"
    validate_service "MIKEY-AI" "MIKEY-AI/src" "MIKEY-AI/.env"
    validate_service "Admin Dashboard" "admin-dashboard/src" "admin-dashboard/.env"
    validate_service "Root" "." ".env"
    
    echo ""
    print_success "Environment variable validation complete!"
}

main "$@"
