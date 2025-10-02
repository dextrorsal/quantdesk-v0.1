#!/bin/bash

# GitHub Security Audit Script for QuantDesk
# This script uses GitHub MCP tools to perform comprehensive security auditing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_OWNER="dex"
REPO_NAME="quantdesk"
REPORTS_DIR="reports/github-security"

echo -e "${BLUE}🔐 GitHub Security Audit for QuantDesk${NC}"
echo "=============================================="

# Create reports directory
mkdir -p "$REPORTS_DIR"

# Function to check GitHub authentication
check_auth() {
    echo -e "\n${BLUE}🔑 Checking GitHub authentication...${NC}"
    
    if ! mcp_GitHub_get_me >/dev/null 2>&1; then
        echo -e "${RED}❌ GitHub authentication failed${NC}"
        echo -e "${YELLOW}Please ensure your GitHub MCP is properly configured with a valid token${NC}"
        echo -e "${YELLOW}Required scopes: repo, security_events, read:org${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ GitHub authentication successful${NC}"
}

# Function to audit dependabot alerts
audit_dependabot() {
    echo -e "\n${BLUE}📦 Auditing Dependabot alerts...${NC}"
    
    # Get all dependabot alerts
    echo -e "${YELLOW}  Fetching all dependabot alerts...${NC}"
    mcp_GitHub_list_dependabot_alerts --owner "$REPO_OWNER" --repo "$REPO_NAME" > "$REPORTS_DIR/dependabot-alerts.json" 2>/dev/null || {
        echo -e "${RED}  ❌ Failed to fetch dependabot alerts${NC}"
        return 1
    }
    
    # Get high severity alerts
    echo -e "${YELLOW}  Fetching high severity alerts...${NC}"
    mcp_GitHub_list_dependabot_alerts --owner "$REPO_OWNER" --repo "$REPO_NAME" --severity high > "$REPORTS_DIR/dependabot-high.json" 2>/dev/null || {
        echo -e "${RED}  ❌ Failed to fetch high severity alerts${NC}"
    }
    
    # Get critical severity alerts
    echo -e "${YELLOW}  Fetching critical alerts...${NC}"
    mcp_GitHub_list_dependabot_alerts --owner "$REPO_OWNER" --repo "$REPO_NAME" --severity critical > "$REPORTS_DIR/dependabot-critical.json" 2>/dev/null || {
        echo -e "${RED}  ❌ Failed to fetch critical alerts${NC}"
    }
    
    echo -e "${GREEN}  ✅ Dependabot audit complete${NC}"
}

# Function to audit code scanning alerts
audit_code_scanning() {
    echo -e "\n${BLUE}🔍 Auditing code scanning alerts...${NC}"
    
    # Get all code scanning alerts
    echo -e "${YELLOW}  Fetching all code scanning alerts...${NC}"
    mcp_GitHub_list_code_scanning_alerts --owner "$REPO_OWNER" --repo "$REPO_NAME" > "$REPORTS_DIR/code-scanning-alerts.json" 2>/dev/null || {
        echo -e "${RED}  ❌ Failed to fetch code scanning alerts${NC}"
        return 1
    }
    
    # Get high severity alerts
    echo -e "${YELLOW}  Fetching high severity code alerts...${NC}"
    mcp_GitHub_list_code_scanning_alerts --owner "$REPO_OWNER" --repo "$REPO_NAME" --severity high > "$REPORTS_DIR/code-scanning-high.json" 2>/dev/null || {
        echo -e "${RED}  ❌ Failed to fetch high severity code alerts${NC}"
    }
    
    # Get critical severity alerts
    echo -e "${YELLOW}  Fetching critical code alerts...${NC}"
    mcp_GitHub_list_code_scanning_alerts --owner "$REPO_OWNER" --repo "$REPO_NAME" --severity critical > "$REPORTS_DIR/code-scanning-critical.json" 2>/dev/null || {
        echo -e "${RED}  ❌ Failed to fetch critical code alerts${NC}"
    }
    
    echo -e "${GREEN}  ✅ Code scanning audit complete${NC}"
}

# Function to audit secret scanning alerts
audit_secret_scanning() {
    echo -e "\n${BLUE}🔐 Auditing secret scanning alerts...${NC}"
    
    # Get all secret scanning alerts
    echo -e "${YELLOW}  Fetching all secret scanning alerts...${NC}"
    mcp_GitHub_list_secret_scanning_alerts --owner "$REPO_OWNER" --repo "$REPO_NAME" > "$REPORTS_DIR/secret-scanning-alerts.json" 2>/dev/null || {
        echo -e "${RED}  ❌ Failed to fetch secret scanning alerts${NC}"
        return 1
    }
    
    # Get open alerts
    echo -e "${YELLOW}  Fetching open secret alerts...${NC}"
    mcp_GitHub_list_secret_scanning_alerts --owner "$REPO_OWNER" --repo "$REPO_NAME" --state open > "$REPORTS_DIR/secret-scanning-open.json" 2>/dev/null || {
        echo -e "${RED}  ❌ Failed to fetch open secret alerts${NC}"
    }
    
    echo -e "${GREEN}  ✅ Secret scanning audit complete${NC}"
}

# Function to check security advisories for key packages
audit_security_advisories() {
    echo -e "\n${BLUE}📋 Checking security advisories for key packages...${NC}"
    
    # Key packages to check
    local packages=("react" "express" "@solana/web3.js" "@pythnetwork/client" "typescript" "node")
    
    for package in "${packages[@]}"; do
        echo -e "${YELLOW}  Checking advisories for $package...${NC}"
        mcp_GitHub_list_global_security_advisories --query "$package" --perPage 5 > "$REPORTS_DIR/advisories-$package.json" 2>/dev/null || {
            echo -e "${RED}  ❌ Failed to fetch advisories for $package${NC}"
        }
    done
    
    echo -e "${GREEN}  ✅ Security advisories audit complete${NC}"
}

# Function to generate summary report
generate_summary() {
    echo -e "\n${BLUE}📊 Generating summary report...${NC}"
    
    cat > "$REPORTS_DIR/security-summary.md" << 'EOF'
# GitHub Security Audit Summary

**Generated:** $(date)  
**Repository:** dex/quantdesk  
**Method:** GitHub MCP Tools

## Executive Summary

This report provides a comprehensive security audit of the QuantDesk repository using GitHub's security features.

## Security Features Audited

### 1. Dependabot Alerts
- **Purpose**: Dependency vulnerability detection
- **Coverage**: npm, pip, Maven, NuGet, Composer, Go modules
- **Status**: [Check dependabot-alerts.json for details]

### 2. Code Scanning (CodeQL)
- **Purpose**: Static analysis for security vulnerabilities
- **Coverage**: JavaScript, TypeScript, Python, Java, C#, Go, C/C++
- **Status**: [Check code-scanning-alerts.json for details]

### 3. Secret Scanning
- **Purpose**: Detect exposed secrets and credentials
- **Coverage**: API keys, tokens, passwords, certificates
- **Status**: [Check secret-scanning-alerts.json for details]

### 4. Security Advisories
- **Purpose**: Global vulnerability database
- **Coverage**: All major package ecosystems
- **Status**: [Check advisories-*.json for details]

## Key Findings

### Dependabot Alerts
- [Review dependabot-alerts.json for specific findings]
- [Review dependabot-high.json for high severity issues]
- [Review dependabot-critical.json for critical issues]

### Code Scanning
- [Review code-scanning-alerts.json for specific findings]
- [Review code-scanning-high.json for high severity issues]
- [Review code-scanning-critical.json for critical issues]

### Secret Scanning
- [Review secret-scanning-alerts.json for specific findings]
- [Review secret-scanning-open.json for open issues]

### Security Advisories
- [Review advisories-*.json for package-specific advisories]

## Recommendations

### Immediate Actions
1. Address any critical dependabot alerts
2. Fix high-severity code scanning issues
3. Resolve any exposed secrets
4. Update packages with known vulnerabilities

### Medium-term Actions
1. Enable automated security scanning
2. Set up security alerts and notifications
3. Implement security testing in CI/CD
4. Regular security reviews

### Long-term Actions
1. Security training for development team
2. Third-party security audits
3. Penetration testing
4. Security incident response plan

## Next Steps

1. Review all generated reports
2. Address critical and high-severity issues
3. Set up automated monitoring
4. Integrate with CI/CD pipeline
5. Document security procedures

## Resources

- [GitHub Security Documentation](https://docs.github.com/en/code-security)
- [Dependabot Documentation](https://docs.github.com/en/code-security/dependabot)
- [CodeQL Documentation](https://codeql.github.com/docs/)
- [Secret Scanning Documentation](https://docs.github.com/en/code-security/secret-scanning)

---

*Report generated using GitHub MCP tools*
*For questions or concerns, please contact the development team*
EOF
    
    echo -e "${GREEN}✅ Summary report generated${NC}"
}

# Function to display results
display_results() {
    echo -e "\n${BLUE}📋 Audit Results:${NC}"
    echo -e "=================="
    
    echo -e "\n${YELLOW}📁 Generated Reports:${NC}"
    ls -la "$REPORTS_DIR/"
    
    echo -e "\n${YELLOW}📊 Report Summary:${NC}"
    echo -e "  📦 Dependabot alerts: $REPORTS_DIR/dependabot-alerts.json"
    echo -e "  🔍 Code scanning: $REPORTS_DIR/code-scanning-alerts.json"
    echo -e "  🔐 Secret scanning: $REPORTS_DIR/secret-scanning-alerts.json"
    echo -e "  📋 Security advisories: $REPORTS_DIR/advisories-*.json"
    echo -e "  📄 Summary report: $REPORTS_DIR/security-summary.md"
    
    echo -e "\n${BLUE}🔍 To view results:${NC}"
    echo -e "  cat $REPORTS_DIR/security-summary.md"
    echo -e "  jq '.' $REPORTS_DIR/dependabot-alerts.json"
    echo -e "  jq '.' $REPORTS_DIR/code-scanning-alerts.json"
    echo -e "  jq '.' $REPORTS_DIR/secret-scanning-alerts.json"
}

# Main execution
main() {
    echo -e "${GREEN}🚀 Starting GitHub security audit...${NC}"
    
    check_auth
    audit_dependabot
    audit_code_scanning
    audit_secret_scanning
    audit_security_advisories
    generate_summary
    display_results
    
    echo -e "\n${GREEN}🎉 GitHub security audit complete!${NC}"
    echo -e "\n${BLUE}📋 Next steps:${NC}"
    echo -e "1. Review generated reports"
    echo -e "2. Address critical and high-severity issues"
    echo -e "3. Set up automated monitoring"
    echo -e "4. Integrate with CI/CD pipeline"
    
    echo -e "\n${GREEN}✨ Happy coding!${NC}"
}

# Run main function
main "$@"
