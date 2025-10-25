#!/bin/bash

# QuantDesk Perpetual DEX - Comprehensive Test Execution Script
# This script runs all tests with proper setup, reporting, and cleanup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_DIR="contracts"
COVERAGE_DIR="coverage"
RESULTS_DIR="test-results"
LOG_DIR="logs"

# Create necessary directories
mkdir -p $COVERAGE_DIR
mkdir -p $RESULTS_DIR
mkdir -p $LOG_DIR

echo -e "${BLUE}🚀 QuantDesk Perpetual DEX - Comprehensive Test Suite${NC}"
echo -e "${BLUE}=================================================${NC}"

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}📋 $1${NC}"
    echo -e "${YELLOW}$(printf '=%.0s' {1..50})${NC}"
}

# Function to print test results
print_results() {
    local test_type=$1
    local exit_code=$2
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ $test_type tests passed${NC}"
    else
        echo -e "${RED}❌ $test_type tests failed${NC}"
    fi
}

# Function to check if Solana test validator is running
check_validator() {
    print_section "Checking Solana Test Validator"
    
    if ! pgrep -f "solana-test-validator" > /dev/null; then
        echo -e "${YELLOW}⚠️  Solana test validator not running. Starting...${NC}"
        solana-test-validator --reset --quiet &
        VALIDATOR_PID=$!
        echo "Validator PID: $VALIDATOR_PID"
        sleep 5  # Wait for validator to start
    else
        echo -e "${GREEN}✅ Solana test validator is running${NC}"
    fi
}

# Function to setup test environment
setup_environment() {
    print_section "Setting Up Test Environment"
    
    cd $TEST_DIR
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}📦 Installing dependencies...${NC}"
        npm install
    fi
    
    # Build the program
    echo -e "${YELLOW}🔨 Building Solana program...${NC}"
    anchor build
    
    # Deploy the program
    echo -e "${YELLOW}🚀 Deploying program to test validator...${NC}"
    anchor deploy
    
    echo -e "${GREEN}✅ Environment setup complete${NC}"
}

# Function to run unit tests
run_unit_tests() {
    print_section "Running Unit Tests"
    
    echo -e "${BLUE}Testing individual smart contract functions...${NC}"
    npm run test:unit 2>&1 | tee $LOG_DIR/unit-tests.log
    UNIT_EXIT_CODE=${PIPESTATUS[0]}
    print_results "Unit" $UNIT_EXIT_CODE
    return $UNIT_EXIT_CODE
}

# Function to run integration tests
run_integration_tests() {
    print_section "Running Integration Tests"
    
    echo -e "${BLUE}Testing complete workflows and contract interactions...${NC}"
    npm run test:integration 2>&1 | tee $LOG_DIR/integration-tests.log
    INTEGRATION_EXIT_CODE=${PIPESTATUS[0]}
    print_results "Integration" $INTEGRATION_EXIT_CODE
    return $INTEGRATION_EXIT_CODE
}

# Function to run security tests
run_security_tests() {
    print_section "Running Security Tests"
    
    echo -e "${BLUE}Testing for vulnerabilities and security issues...${NC}"
    npm run test:security 2>&1 | tee $LOG_DIR/security-tests.log
    SECURITY_EXIT_CODE=${PIPESTATUS[0]}
    print_results "Security" $SECURITY_EXIT_CODE
    return $SECURITY_EXIT_CODE
}

# Function to run performance tests
run_performance_tests() {
    print_section "Running Performance Tests"
    
    echo -e "${BLUE}Testing gas optimization and transaction throughput...${NC}"
    npm run test:performance 2>&1 | tee $LOG_DIR/performance-tests.log
    PERFORMANCE_EXIT_CODE=${PIPESTATUS[0]}
    print_results "Performance" $PERFORMANCE_EXIT_CODE
    return $PERFORMANCE_EXIT_CODE
}

# Function to run coverage analysis
run_coverage() {
    print_section "Running Coverage Analysis"
    
    echo -e "${BLUE}Analyzing test coverage...${NC}"
    npm run test:coverage 2>&1 | tee $LOG_DIR/coverage.log
    COVERAGE_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $COVERAGE_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ Coverage analysis complete${NC}"
        echo -e "${BLUE}📊 Coverage report generated in $COVERAGE_DIR/${NC}"
    else
        echo -e "${RED}❌ Coverage analysis failed${NC}"
    fi
    
    return $COVERAGE_EXIT_CODE
}

# Function to generate test report
generate_report() {
    print_section "Generating Test Report"
    
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    
    # Count test results
    if [ -f "$LOG_DIR/unit-tests.log" ]; then
        local unit_passed=$(grep -c "passing" $LOG_DIR/unit-tests.log || echo "0")
        local unit_failed=$(grep -c "failing" $LOG_DIR/unit-tests.log || echo "0")
        total_tests=$((total_tests + unit_passed + unit_failed))
        passed_tests=$((passed_tests + unit_passed))
        failed_tests=$((failed_tests + unit_failed))
    fi
    
    if [ -f "$LOG_DIR/integration-tests.log" ]; then
        local integration_passed=$(grep -c "passing" $LOG_DIR/integration-tests.log || echo "0")
        local integration_failed=$(grep -c "failing" $LOG_DIR/integration-tests.log || echo "0")
        total_tests=$((total_tests + integration_passed + integration_failed))
        passed_tests=$((passed_tests + integration_passed))
        failed_tests=$((failed_tests + integration_failed))
    fi
    
    if [ -f "$LOG_DIR/security-tests.log" ]; then
        local security_passed=$(grep -c "passing" $LOG_DIR/security-tests.log || echo "0")
        local security_failed=$(grep -c "failing" $LOG_DIR/security-tests.log || echo "0")
        total_tests=$((total_tests + security_passed + security_failed))
        passed_tests=$((passed_tests + security_passed))
        failed_tests=$((failed_tests + security_failed))
    fi
    
    if [ -f "$LOG_DIR/performance-tests.log" ]; then
        local performance_passed=$(grep -c "passing" $LOG_DIR/performance-tests.log || echo "0")
        local performance_failed=$(grep -c "failing" $LOG_DIR/performance-tests.log || echo "0")
        total_tests=$((total_tests + performance_passed + performance_failed))
        passed_tests=$((passed_tests + performance_passed))
        failed_tests=$((failed_tests + performance_failed))
    fi
    
    # Generate HTML report
    cat > $RESULTS_DIR/test-report.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>QuantDesk Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .passed { color: green; }
        .failed { color: red; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>QuantDesk Perpetual DEX - Test Report</h1>
        <p>Generated on: $(date)</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <p>Total Tests: $total_tests</p>
        <p class="passed">Passed: $passed_tests</p>
        <p class="failed">Failed: $failed_tests</p>
        <p>Success Rate: $((passed_tests * 100 / (total_tests + 1)))%</p>
    </div>
    
    <div class="section">
        <h3>Test Categories</h3>
        <ul>
            <li>Unit Tests - Individual function testing</li>
            <li>Integration Tests - Complete workflow testing</li>
            <li>Security Tests - Vulnerability and security testing</li>
            <li>Performance Tests - Gas optimization and throughput testing</li>
        </ul>
    </div>
    
    <div class="section">
        <h3>Coverage Report</h3>
        <p>Detailed coverage information available in the coverage directory.</p>
    </div>
</body>
</html>
EOF
    
    echo -e "${GREEN}📊 Test report generated: $RESULTS_DIR/test-report.html${NC}"
    echo -e "${BLUE}📈 Total Tests: $total_tests${NC}"
    echo -e "${GREEN}✅ Passed: $passed_tests${NC}"
    echo -e "${RED}❌ Failed: $failed_tests${NC}"
}

# Function to cleanup
cleanup() {
    print_section "Cleaning Up"
    
    # Kill validator if we started it
    if [ ! -z "$VALIDATOR_PID" ]; then
        echo -e "${YELLOW}🛑 Stopping test validator...${NC}"
        kill $VALIDATOR_PID 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Main execution
main() {
    local start_time=$(date +%s)
    local exit_code=0
    
    # Trap to ensure cleanup on exit
    trap cleanup EXIT
    
    # Check if running in CI mode
    if [ "$1" = "--ci" ]; then
        echo -e "${BLUE}🤖 Running in CI mode${NC}"
        CI_MODE=true
    fi
    
    # Run all test phases
    check_validator
    setup_environment
    
    # Run tests
    run_unit_tests || exit_code=$?
    run_integration_tests || exit_code=$?
    run_security_tests || exit_code=$?
    run_performance_tests || exit_code=$?
    
    # Run coverage if not in CI mode or if explicitly requested
    if [ "$CI_MODE" != "true" ] || [ "$2" = "--coverage" ]; then
        run_coverage || exit_code=$?
    fi
    
    # Generate report
    generate_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_section "Test Execution Complete"
    echo -e "${BLUE}⏱️  Total execution time: ${duration}s${NC}"
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests completed successfully!${NC}"
    else
        echo -e "${RED}💥 Some tests failed. Check logs for details.${NC}"
    fi
    
    exit $exit_code
}

# Run main function with all arguments
main "$@"
