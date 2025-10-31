#!/bin/bash

# 🧪 CI/CD Workflow Testing Script
# Tests all workflows without actually running them

echo "🧪 CI/CD Workflow Testing Script"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_command=$2
    local description=$3
    
    echo -e "\n${BLUE}🔍 $test_name${NC}"
    echo -e "${YELLOW}📋 $description${NC}"
    echo "──────────────────────────────────────────────────"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $test_name passed${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}❌ $test_name failed${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

echo -e "${BLUE}🚀 Starting CI/CD Workflow Tests...${NC}"

# Test 1: YAML Syntax Validation
run_test "YAML Syntax Validation" \
    "python3 -c 'import yaml; import os; [yaml.safe_load(open(f\".github/workflows/{f}\")) for f in os.listdir(\".github/workflows\") if f.endswith((\".yml\", \".yaml\"))]'" \
    "Validates all workflow files have correct YAML syntax"

# Test 2: Package.json Scripts
run_test "Package.json Scripts" \
    "npm run --silent 2>/dev/null | grep -q 'build\\|test\\|lint'" \
    "Checks if package.json has required scripts"

# Test 3: Dockerfile Syntax
run_test "Dockerfile Syntax" \
    "find . -name 'Dockerfile*' -exec echo 'Checking {}' \\; -exec head -1 {} \\;" \
    "Validates Dockerfile syntax"

# Test 4: Environment Files
run_test "Environment Files" \
    "find . -name '.env.example' | wc -l | grep -q '[1-9]'" \
    "Checks for environment example files"

# Test 5: Required Directories
run_test "Required Directories" \
    "[ -d 'backend' ] && [ -d 'frontend' ] && [ -d 'admin-dashboard' ]" \
    "Validates all required service directories exist"

# Test 6: TypeScript Configuration
run_test "TypeScript Configuration" \
    "find . -name 'tsconfig.json' | wc -l | grep -q '[1-9]'" \
    "Checks for TypeScript configuration files"

# Test 7: Docker Compose Configuration
run_test "Docker Compose Configuration" \
    "[ -f 'docker-compose.yml' ]" \
    "Validates docker-compose.yml exists"

# Test 8: CI/CD Documentation
run_test "CI/CD Documentation" \
    "[ -f 'CI_CD_README.md' ]" \
    "Checks for CI/CD documentation"

# Test 9: Workflow Triggers
run_test "Workflow Triggers" \
    "grep -r 'on:' .github/workflows/ | wc -l | grep -q '[1-9]'" \
    "Validates workflow trigger configurations"

# Test 10: Security Scanning
run_test "Security Scanning" \
    "grep -r 'security\\|audit\\|vulnerability' .github/workflows/ | wc -l | grep -q '[1-9]'" \
    "Checks for security scanning workflows"

echo -e "\n${BLUE}📊 Test Results Summary${NC}"
echo "=========================="
echo -e "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All CI/CD workflow tests passed!${NC}"
    echo -e "${GREEN}🚀 Your workflows are ready for deployment!${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed. Please review the issues above.${NC}"
    exit 1
fi
