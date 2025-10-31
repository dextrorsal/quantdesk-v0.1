#!/bin/bash

# Socket.dev Dependency Audit Script for QuantDesk
# This script scans all package dependencies and generates comprehensive reports

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Starting Socket.dev dependency audit for QuantDesk...${NC}"
echo "=================================================="

# Check if Socket CLI is installed
if ! command -v socket &> /dev/null; then
    echo -e "${RED}❌ Socket CLI not found. Installing...${NC}"
    npm install -g @socketsecurity/cli
fi

# Check if authenticated
if ! socket whoami &> /dev/null; then
    echo -e "${YELLOW}⚠️  Not authenticated with Socket.dev${NC}"
    echo -e "${YELLOW}Please run: socket login${NC}"
    echo -e "${YELLOW}Or set SOCKET_API_TOKEN environment variable${NC}"
    exit 1
fi

# Create reports directory
mkdir -p reports/socket
echo -e "${GREEN}✅ Created reports directory${NC}"

# Function to scan a package
scan_package() {
    local package_name=$1
    local package_dir=$2
    
    echo -e "\n${BLUE}📦 Scanning $package_name...${NC}"
    
    if [ -d "$package_dir" ]; then
        cd "$package_dir"
        
        # Check if package.json exists
        if [ ! -f "package.json" ]; then
            echo -e "${YELLOW}⚠️  No package.json found in $package_dir, skipping...${NC}"
            cd ..
            return
        fi
        
        # Generate JSON report
        echo -e "${YELLOW}  Generating JSON report...${NC}"
        socket scan --json > "../reports/socket/${package_name}.json" 2>/dev/null || {
            echo -e "${RED}  ❌ Failed to generate JSON report for $package_name${NC}"
        }
        
        # Generate SBOM
        echo -e "${YELLOW}  Generating SBOM...${NC}"
        socket sbom --format cyclonedx > "../reports/socket/${package_name}.cdx.json" 2>/dev/null || {
            echo -e "${RED}  ❌ Failed to generate SBOM for $package_name${NC}"
        }
        
        # Generate human-readable report
        echo -e "${YELLOW}  Generating human-readable report...${NC}"
        socket scan --output "../reports/socket/${package_name}-report.txt" 2>/dev/null || {
            echo -e "${RED}  ❌ Failed to generate text report for $package_name${NC}"
        }
        
        # Get summary
        echo -e "${YELLOW}  Getting summary...${NC}"
        socket scan --summary > "../reports/socket/${package_name}-summary.txt" 2>/dev/null || {
            echo -e "${RED}  ❌ Failed to generate summary for $package_name${NC}"
        }
        
        cd ..
        echo -e "${GREEN}  ✅ $package_name scan complete${NC}"
    else
        echo -e "${YELLOW}⚠️  Directory $package_dir not found, skipping...${NC}"
    fi
}

# Function to analyze results
analyze_results() {
    echo -e "\n${BLUE}📊 Analyzing results...${NC}"
    
    local total_packages=0
    local total_vulnerabilities=0
    local high_risk_packages=0
    
    for report in reports/socket/*.json; do
        if [ -f "$report" ]; then
            local package_name=$(basename "$report" .json)
            echo -e "\n${YELLOW}📋 $package_name:${NC}"
            
            # Extract key metrics using jq if available
            if command -v jq &> /dev/null; then
                local vulns=$(jq '.vulnerabilities | length' "$report" 2>/dev/null || echo "0")
                local score=$(jq '.score' "$report" 2>/dev/null || echo "N/A")
                local deps=$(jq '.dependencies | length' "$report" 2>/dev/null || echo "0")
                
                echo -e "  Dependencies: $deps"
                echo -e "  Vulnerabilities: $vulns"
                echo -e "  Security Score: $score"
                
                total_packages=$((total_packages + 1))
                total_vulnerabilities=$((total_vulnerabilities + vulns))
                
                if [ "$score" != "N/A" ] && [ "$score" -lt 70 ]; then
                    high_risk_packages=$((high_risk_packages + 1))
                fi
            else
                echo -e "  ${YELLOW}Install jq for detailed analysis: sudo apt install jq${NC}"
            fi
        fi
    done
    
    echo -e "\n${BLUE}📈 Summary:${NC}"
    echo -e "  Total packages scanned: $total_packages"
    echo -e "  Total vulnerabilities: $total_vulnerabilities"
    echo -e "  High-risk packages: $high_risk_packages"
}

# Function to generate HTML report
generate_html_report() {
    echo -e "\n${BLUE}🌐 Generating HTML report...${NC}"
    
    cat > reports/socket/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantDesk Dependency Audit Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .package { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #3498db; }
        .package h3 { margin-top: 0; color: #2c3e50; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }
        .metric { background: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; font-size: 14px; }
        .file-list { background: #f8f9fa; padding: 15px; border-radius: 5px; }
        .file-list a { color: #3498db; text-decoration: none; }
        .file-list a:hover { text-decoration: underline; }
        .timestamp { color: #7f8c8d; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 QuantDesk Dependency Audit Report</h1>
        <p class="timestamp">Generated on: $(date)</p>
        
        <h2>📦 Scanned Packages</h2>
        <div class="package">
            <h3>Backend</h3>
            <p>Node.js backend API with Solana integration</p>
            <div class="file-list">
                <a href="backend.json">JSON Report</a> | 
                <a href="backend.cdx.json">SBOM (CycloneDX)</a> | 
                <a href="backend-report.txt">Text Report</a> | 
                <a href="backend-summary.txt">Summary</a>
            </div>
        </div>
        
        <div class="package">
            <h3>Admin Dashboard</h3>
            <p>React-based admin interface</p>
            <div class="file-list">
                <a href="admin-dashboard.json">JSON Report</a> | 
                <a href="admin-dashboard.cdx.json">SBOM (CycloneDX)</a> | 
                <a href="admin-dashboard-report.txt">Text Report</a> | 
                <a href="admin-dashboard-summary.txt">Summary</a>
            </div>
        </div>
        
        <div class="package">
            <h3>Data Ingestion</h3>
            <p>High-throughput data pipeline</p>
            <div class="file-list">
                <a href="data-ingestion.json">JSON Report</a> | 
                <a href="data-ingestion.cdx.json">SBOM (CycloneDX)</a> | 
                <a href="data-ingestion-report.txt">Text Report</a> | 
                <a href="data-ingestion-summary.txt">Summary</a>
            </div>
        </div>
        
        <div class="package">
            <h3>Frontend</h3>
            <p>User-facing trading interface</p>
            <div class="file-list">
                <a href="frontend.json">JSON Report</a> | 
                <a href="frontend.cdx.json">SBOM (CycloneDX)</a> | 
                <a href="frontend-report.txt">Text Report</a> | 
                <a href="frontend-summary.txt">Summary</a>
            </div>
        </div>
        
        <h2>📊 Quick Analysis</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value" id="total-packages">-</div>
                <div class="metric-label">Packages Scanned</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="total-vulnerabilities">-</div>
                <div class="metric-label">Vulnerabilities</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="high-risk">-</div>
                <div class="metric-label">High-Risk Packages</div>
            </div>
        </div>
        
        <h2>📋 Next Steps</h2>
        <ul>
            <li>Review high-risk packages and update if necessary</li>
            <li>Address any critical vulnerabilities</li>
            <li>Set up automated monitoring</li>
            <li>Integrate with CI/CD pipeline</li>
        </ul>
        
        <h2>🔗 Resources</h2>
        <ul>
            <li><a href="https://docs.socket.dev/">Socket.dev Documentation</a></li>
            <li><a href="https://cyclonedx.org/">CycloneDX SBOM Format</a></li>
            <li><a href="https://owasp.org/www-project-dependency-check/">OWASP Dependency Check</a></li>
        </ul>
    </div>
    
    <script>
        // Simple analysis script
        fetch('backend.json')
            .then(response => response.json())
            .then(data => {
                document.getElementById('total-packages').textContent = '4';
                document.getElementById('total-vulnerabilities').textContent = data.vulnerabilities ? data.vulnerabilities.length : '0';
                document.getElementById('high-risk').textContent = data.score < 70 ? '1' : '0';
            })
            .catch(() => {
                document.getElementById('total-packages').textContent = '4';
                document.getElementById('total-vulnerabilities').textContent = '?';
                document.getElementById('high-risk').textContent = '?';
            });
    </script>
</body>
</html>
EOF
    
    echo -e "${GREEN}✅ HTML report generated: reports/socket/index.html${NC}"
}

# Main execution
echo -e "${GREEN}🚀 Starting dependency audit...${NC}"

# Scan all packages
scan_package "backend" "backend"
scan_package "admin-dashboard" "admin-dashboard"
scan_package "data-ingestion" "data-ingestion"
scan_package "frontend" "frontend"

# Analyze results
analyze_results

# Generate HTML report
generate_html_report

echo -e "\n${GREEN}🎉 Dependency audit complete!${NC}"
echo -e "\n${BLUE}📊 Reports available in:${NC}"
echo -e "  📁 reports/socket/"
echo -e "  🌐 reports/socket/index.html (HTML report)"
echo -e "  📋 reports/socket/*.json (JSON reports)"
echo -e "  📄 reports/socket/*.txt (Text reports)"

echo -e "\n${YELLOW}📋 Summary of generated files:${NC}"
ls -la reports/socket/

echo -e "\n${BLUE}🔍 To view the HTML report:${NC}"
echo -e "  open reports/socket/index.html"

echo -e "\n${BLUE}📖 For detailed analysis:${NC}"
echo -e "  cat reports/socket/*-summary.txt"

echo -e "\n${GREEN}✨ Happy coding!${NC}"
