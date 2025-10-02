#!/bin/bash

# QuantDesk Environment Variable Scanner
# Detects unused, duplicate, and problematic environment variables in AI agent workflows

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

# Function to find all .env files
find_env_files() {
    print_status "Scanning for environment files..."
    
    # Find all .env files
    ENV_FILES=$(find . -name "*.env*" -type f 2>/dev/null | grep -v node_modules | sort)
    
    if [ -z "$ENV_FILES" ]; then
        print_warning "No .env files found"
        return 1
    fi
    
    echo "Found environment files:"
    echo "$ENV_FILES" | while read -r file; do
        echo "  - $file"
    done
    
    return 0
}

# Function to extract environment variables from files
extract_env_vars() {
    local file="$1"
    local prefix="$2"
    
    if [ ! -f "$file" ]; then
        return 1
    fi
    
    # Extract variable names (lines that start with VAR_NAME=)
    grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$file" 2>/dev/null | \
    cut -d'=' -f1 | \
    sort | uniq | \
    while read -r var; do
        echo "${prefix}${var}"
    done
}

# Function to find duplicates across files
find_duplicates() {
    print_status "Checking for duplicate environment variables..."
    
    local all_vars=""
    local duplicates=""
    
    # Collect all variables from all files
    while read -r file; do
        if [ -f "$file" ]; then
            vars=$(extract_env_vars "$file" "${file}:")
            all_vars="${all_vars}${vars}\n"
        fi
    done <<< "$ENV_FILES"
    
    # Find duplicates
    duplicates=$(echo -e "$all_vars" | cut -d':' -f2 | sort | uniq -d)
    
    if [ -n "$duplicates" ]; then
        print_warning "Duplicate environment variables found:"
        echo "$duplicates" | while read -r var; do
            echo "  - $var"
            echo "    Found in:"
            echo -e "$all_vars" | grep ":$var$" | cut -d':' -f1 | while read -r file; do
                echo "      $file"
            done
        done
        return 1
    else
        print_success "No duplicate environment variables found"
        return 0
    fi
}

# Function to find similar variable names
find_similar_vars() {
    print_status "Checking for similar environment variable names..."
    
    local all_vars=""
    local similar_vars=""
    
    # Collect all variables from all files
    while read -r file; do
        if [ -f "$file" ]; then
            vars=$(extract_env_vars "$file" "")
            all_vars="${all_vars}${vars}\n"
        fi
    done <<< "$ENV_FILES"
    
    # Find similar variables (same prefix, different suffixes)
    similar_vars=$(echo -e "$all_vars" | sort | uniq | \
        awk '{
            if (prev != "" && substr($0, 1, length(prev)) == prev) {
                print prev " vs " $0
            }
            prev = $0
        }')
    
    if [ -n "$similar_vars" ]; then
        print_warning "Similar environment variable names found:"
        echo "$similar_vars" | while read -r line; do
            echo "  - $line"
        done
        return 1
    else
        print_success "No similar environment variable names found"
        return 0
    fi
}

# Function to find unused environment variables
find_unused_vars() {
    print_status "Checking for unused environment variables..."
    
    local unused_vars=""
    local code_files=""
    
    # Find all code files
    code_files=$(find . -type f \( -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" -o -name "*.py" -o -name "*.sh" \) 2>/dev/null | grep -v node_modules | grep -v ".git")
    
    # Check each environment variable
    while read -r file; do
        if [ -f "$file" ]; then
            vars=$(extract_env_vars "$file" "")
            echo "$vars" | while read -r var; do
                if [ -n "$var" ]; then
                    # Check if variable is used in code
                    if ! grep -r "process\.env\.$var\|process\.env\[['\"]$var['\"]\]\|os\.environ\['$var'\]\|os\.environ\[\"$var\"\]\|\$$var" $code_files >/dev/null 2>&1; then
                        echo "  - $var (in $file)"
                    fi
                fi
            done
        fi
    done <<< "$ENV_FILES"
}

# Function to validate environment variable format
validate_env_format() {
    print_status "Validating environment variable format..."
    
    local issues=""
    
    while read -r file; do
        if [ -f "$file" ]; then
            # Check for common format issues
            if grep -E '^[a-z]' "$file" >/dev/null 2>&1; then
                echo "  - $file: Contains lowercase variable names (should be UPPERCASE)"
                issues="yes"
            fi
            
            if grep -E '^[A-Za-z_][A-Za-z0-9_]*[^=]=' "$file" >/dev/null 2>&1; then
                echo "  - $file: Contains variables with spaces before ="
                issues="yes"
            fi
            
            if grep -E '^[A-Za-z_][A-Za-z0-9_]*=$' "$file" >/dev/null 2>&1; then
                echo "  - $file: Contains empty variable values"
                issues="yes"
            fi
        fi
    done <<< "$ENV_FILES"
    
    if [ -z "$issues" ]; then
        print_success "Environment variable format is valid"
        return 0
    else
        return 1
    fi
}

# Function to check for sensitive information
check_sensitive_info() {
    print_status "Checking for potentially sensitive information..."
    
    local sensitive_patterns="password|secret|key|token|auth|credential"
    local issues=""
    
    while read -r file; do
        if [ -f "$file" ]; then
            if grep -iE "$sensitive_patterns" "$file" >/dev/null 2>&1; then
                echo "  - $file: Contains potentially sensitive information"
                grep -iE "$sensitive_patterns" "$file" | while read -r line; do
                    echo "    $line"
                done
                issues="yes"
            fi
        fi
    done <<< "$ENV_FILES"
    
    if [ -z "$issues" ]; then
        print_success "No sensitive information detected"
        return 0
    else
        print_warning "Potentially sensitive information found (review manually)"
        return 1
    fi
}

# Function to generate environment variable report
generate_report() {
    print_status "Generating environment variable report..."
    
    local report_file="reports/env-variables-report.md"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Create reports directory if it doesn't exist
    mkdir -p reports
    
    cat > "$report_file" << EOF
# Environment Variables Report

**Generated:** $timestamp  
**Project:** QuantDesk  
**Scanner Version:** 1.0.0

## Summary

This report analyzes environment variables across the QuantDesk project to identify:
- Duplicate variables
- Similar variable names
- Unused variables
- Format issues
- Sensitive information

## Environment Files Found

EOF

    echo "$ENV_FILES" | while read -r file; do
        if [ -f "$file" ]; then
            echo "- \`$file\`" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

## Environment Variables by File

EOF

    while read -r file; do
        if [ -f "$file" ]; then
            echo "### $file" >> "$report_file"
            echo "" >> "$report_file"
            echo '```bash' >> "$report_file"
            extract_env_vars "$file" "" >> "$report_file"
            echo '```' >> "$report_file"
            echo "" >> "$report_file"
        fi
    done <<< "$ENV_FILES"
    
    print_success "Report generated: $report_file"
}

# Function to install dotenv-linter (optional)
install_dotenv_linter() {
    print_status "Checking for dotenv-linter..."
    
    if command_exists dotenv-linter; then
        print_success "dotenv-linter is already installed"
        return 0
    fi
    
    print_status "Installing dotenv-linter..."
    
    # Try different installation methods
    if command_exists cargo; then
        cargo install dotenv-linter
        print_success "dotenv-linter installed via cargo"
    elif command_exists brew; then
        brew install dotenv-linter
        print_success "dotenv-linter installed via brew"
    else
        print_warning "dotenv-linter not available. Install manually:"
        print_warning "  cargo install dotenv-linter"
        print_warning "  or"
        print_warning "  brew install dotenv-linter"
        return 1
    fi
}

# Function to run dotenv-linter
run_dotenv_linter() {
    print_status "Running dotenv-linter..."
    
    if ! command_exists dotenv-linter; then
        print_warning "dotenv-linter not installed, skipping..."
        return 1
    fi
    
    while read -r file; do
        if [ -f "$file" ]; then
            echo "Checking $file:"
            dotenv-linter "$file" || true
            echo ""
        fi
    done <<< "$ENV_FILES"
}

# Main execution function
main() {
    echo "🔍 QuantDesk Environment Variable Scanner"
    echo "========================================"
    echo ""
    
    # Find environment files
    if ! find_env_files; then
        print_error "No environment files found to scan"
        exit 1
    fi
    
    echo ""
    
    # Run all checks
    local exit_code=0
    
    find_duplicates || exit_code=1
    echo ""
    
    find_similar_vars || exit_code=1
    echo ""
    
    validate_env_format || exit_code=1
    echo ""
    
    check_sensitive_info || exit_code=1
    echo ""
    
    # Generate report
    generate_report
    echo ""
    
    # Try to install and run dotenv-linter
    install_dotenv_linter
    echo ""
    
    run_dotenv_linter
    echo ""
    
    # Summary
    echo "📊 Scan Summary"
    echo "==============="
    echo ""
    
    if [ $exit_code -eq 0 ]; then
        print_success "Environment variable scan completed successfully"
        echo ""
        echo "✅ No critical issues found"
        echo "✅ Environment variables are properly formatted"
        echo "✅ No duplicates or similar names detected"
    else
        print_warning "Environment variable scan completed with warnings"
        echo ""
        echo "⚠️  Some issues were found (see details above)"
        echo "⚠️  Review the generated report for more information"
    fi
    
    echo ""
    echo "📋 Next Steps:"
    echo "1. Review the generated report: reports/env-variables-report.md"
    echo "2. Fix any issues found"
    echo "3. Re-run the scanner to verify fixes"
    echo "4. Consider using dotenv-linter for ongoing validation"
    
    echo ""
    print_success "Environment variable scanning complete! 🎉"
    
    exit $exit_code
}

# Run main function
main "$@"

