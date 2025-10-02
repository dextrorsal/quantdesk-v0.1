#!/bin/bash

# 🔒 QuantDesk Pre-Push Security Check
# Run this before pushing to ensure no sensitive data is committed

echo "🔍 QuantDesk Security Pre-Push Check"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for sensitive files
echo -e "\n${YELLOW}1. Checking for sensitive files...${NC}"
SENSITIVE_FILES=$(git status --porcelain | grep -E "\.env|\.key|secret|password|token")
if [ -n "$SENSITIVE_FILES" ]; then
    echo -e "${RED}❌ Sensitive files detected:${NC}"
    echo "$SENSITIVE_FILES"
    echo -e "${RED}Please remove these files before pushing!${NC}"
    exit 1
else
    echo -e "${GREEN}✅ No sensitive files detected${NC}"
fi

# Check for API keys in tracked files
echo -e "\n${YELLOW}2. Checking for API keys in tracked files...${NC}"
API_KEYS=$(git ls-files | xargs grep -l -E "sk-proj-|pk_[a-zA-Z0-9]{20,}|AIzaSy[a-zA-Z0-9_-]{30,}|CG-[a-zA-Z0-9_-]{20,}|xai-[a-zA-Z0-9_-]{20,}|glsa_[a-zA-Z0-9_-]{20,}" 2>/dev/null)
if [ -n "$API_KEYS" ]; then
    echo -e "${RED}❌ API keys found in tracked files:${NC}"
    echo "$API_KEYS"
    echo -e "${RED}Please remove API keys from these files!${NC}"
    exit 1
else
    echo -e "${GREEN}✅ No API keys found in tracked files${NC}"
fi

# Check for hardcoded database URLs
echo -e "\n${YELLOW}3. Checking for hardcoded database URLs...${NC}"
DB_URLS=$(git ls-files | xargs grep -l -E "https://[a-zA-Z0-9-]+\.supabase\.co|postgresql://[^:]+:[^@]+@" 2>/dev/null | xargs grep -L -E "testpassword|localhost|example|placeholder|your-")
if [ -n "$DB_URLS" ]; then
    echo -e "${RED}❌ Hardcoded database URLs found:${NC}"
    echo "$DB_URLS"
    echo -e "${RED}Please replace with placeholder values!${NC}"
    exit 1
else
    echo -e "${GREEN}✅ No hardcoded database URLs found${NC}"
fi

# Check environment files are ignored
echo -e "\n${YELLOW}4. Checking environment files are ignored...${NC}"
ENV_FILES=".env backend/.env MIKEY-AI/.env data-ingestion/.env"
for file in $ENV_FILES; do
    if git check-ignore "$file" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ $file is ignored${NC}"
    else
        echo -e "${RED}❌ $file is NOT ignored${NC}"
        exit 1
    fi
done

# Check for private keys
echo -e "\n${YELLOW}5. Checking for private keys...${NC}"
PRIVATE_KEYS=$(git ls-files | xargs grep -l -E "wallet\.json|keypair\.json|\.key$|\.pem$" 2>/dev/null | xargs grep -L -E "example|placeholder|your-|test-|target/deploy|scripts/" | xargs grep -l -E "BEGIN.*PRIVATE|BEGIN.*RSA|BEGIN.*EC")
if [ -n "$PRIVATE_KEYS" ]; then
    echo -e "${RED}❌ Private key files found:${NC}"
    echo "$PRIVATE_KEYS"
    echo -e "${RED}Please remove private key files!${NC}"
    exit 1
else
    echo -e "${GREEN}✅ No private key files found${NC}"
fi

# Check .gitignore is comprehensive
echo -e "\n${YELLOW}6. Checking .gitignore coverage...${NC}"
if grep -q "\.env" .gitignore && grep -q "\.key" .gitignore && grep -q "secret" .gitignore; then
    echo -e "${GREEN}✅ .gitignore looks comprehensive${NC}"
else
    echo -e "${YELLOW}⚠️ .gitignore may need updates${NC}"
fi

# Final summary
echo -e "\n${GREEN}🎉 Security check passed!${NC}"
echo -e "${GREEN}✅ Repository is safe to push${NC}"
echo ""
echo -e "${YELLOW}Remember:${NC}"
echo "• Never commit .env files"
echo "• Never commit API keys"
echo "• Never commit private keys"
echo "• Use placeholder values in examples"
echo "• Keep proprietary components private"
echo ""
echo -e "${GREEN}Ready to push! 🚀${NC}"
