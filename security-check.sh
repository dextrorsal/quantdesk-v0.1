#!/bin/bash

# QuantDesk Security Check Script
# Run this before committing to ensure no sensitive data is exposed

echo "🔒 QuantDesk Security Check"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env files exist
echo -e "\n📋 Checking for environment files..."
if [ -f ".env" ]; then
    echo -e "${RED}❌ .env file found - DO NOT COMMIT THIS FILE${NC}"
    exit 1
else
    echo -e "${GREEN}✅ No .env file found${NC}"
fi

# Check for sensitive patterns in tracked files
echo -e "\n🔍 Scanning for hardcoded secrets..."
SECRETS_FOUND=0

# Check for JWT tokens
if git grep -q "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" -- .; then
    echo -e "${RED}❌ JWT token found in code${NC}"
    SECRETS_FOUND=1
fi

# Check for common secret patterns
if git grep -q "your-super-secret" -- .; then
    echo -e "${RED}❌ Default JWT secret found${NC}"
    SECRETS_FOUND=1
fi

# Check for API keys
if git grep -q "api_key.*['\"][^'\"]{20,}['\"]" -- .; then
    echo -e "${RED}❌ Hardcoded API key found${NC}"
    SECRETS_FOUND=1
fi

# Check for database URLs with passwords
if git grep -q "postgresql://.*:.*@" -- .; then
    echo -e "${RED}❌ Database URL with password found${NC}"
    SECRETS_FOUND=1
fi

if [ $SECRETS_FOUND -eq 0 ]; then
    echo -e "${GREEN}✅ No hardcoded secrets found${NC}"
fi

# Check .gitignore is working
echo -e "\n📁 Verifying .gitignore effectiveness..."
IGNORED_FILES=(
    "test-ledger/"
    "backend/logs/"
    "services/ml-service/logs/"
    "contracts/smart-contracts/.anchor/"
    "contracts/smart-contracts/target/"
    "contracts/smart-contracts/node_modules/"
    "contracts/smart-contracts/yarn.lock"
    "contracts/smart-contracts/test-results.log"
)

for file in "${IGNORED_FILES[@]}"; do
    if git check-ignore "$file" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ $file is properly ignored${NC}"
    else
        echo -e "${YELLOW}⚠️  $file is not ignored (may not exist)${NC}"
    fi
done

# Check for node_modules
echo -e "\n📦 Checking for node_modules..."
if find . -name "node_modules" -type d | grep -q .; then
    echo -e "${YELLOW}⚠️  node_modules directories found${NC}"
    echo "These should be ignored by .gitignore"
else
    echo -e "${GREEN}✅ No node_modules found${NC}"
fi

# Check for Python cache
echo -e "\n🐍 Checking for Python cache..."
if find . -name "__pycache__" -type d | grep -q .; then
    echo -e "${YELLOW}⚠️  __pycache__ directories found${NC}"
    echo "These should be ignored by .gitignore"
else
    echo -e "${GREEN}✅ No __pycache__ found${NC}"
fi

# Check for log files
echo -e "\n📝 Checking for log files..."
if find . -name "*.log" -type f | grep -q .; then
    echo -e "${YELLOW}⚠️  Log files found${NC}"
    echo "These should be ignored by .gitignore"
else
    echo -e "${GREEN}✅ No log files found${NC}"
fi

# Check for keypair files
echo -e "\n🔑 Checking for keypair files..."
if find . -name "*.keypair.json" -type f | grep -q .; then
    echo -e "${RED}❌ Keypair files found - DO NOT COMMIT THESE${NC}"
    SECRETS_FOUND=1
else
    echo -e "${GREEN}✅ No keypair files found${NC}"
fi

# Final security assessment
echo -e "\n🎯 Security Assessment"
echo "===================="

if [ $SECRETS_FOUND -eq 0 ]; then
    echo -e "${GREEN}✅ SECURITY CHECK PASSED${NC}"
    echo -e "${GREEN}Your repository is safe to commit!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Copy env.example to .env"
    echo "2. Fill in your actual environment variables"
    echo "3. Test your application"
    echo "4. Commit your changes"
    exit 0
else
    echo -e "${RED}❌ SECURITY CHECK FAILED${NC}"
    echo -e "${RED}Please fix the issues above before committing${NC}"
    echo ""
    echo "Common fixes:"
    echo "1. Remove hardcoded secrets from code"
    echo "2. Use environment variables instead"
    echo "3. Ensure .env files are in .gitignore"
    echo "4. Remove sensitive files from git tracking"
    exit 1
fi
