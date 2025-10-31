#!/bin/bash

# 📚 Documentation Organization Script for QuantDesk
# Organizes documentation files by category and moves outdated files to archive

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}📚 QuantDesk Documentation Organization${NC}"
echo "=========================================="
echo ""

# Configuration
DOCS_DIR="docs"
ARCHIVE_DIR="archive/docs"
BACKUP_DIR="docs-backup-$(date +%Y%m%d-%H%M%S)"

# Create directories
mkdir -p "$ARCHIVE_DIR"
mkdir -p "$BACKUP_DIR"

# Counters
TOTAL_DOCS=0
MOVED_TO_ARCHIVE=0
ORGANIZED_DOCS=0
KEPT_IN_PLACE=0

echo -e "${BLUE}🔍 Analyzing Documentation Structure...${NC}"
echo ""

# Function to categorize a document
categorize_doc() {
    local file=$1
    local filename=$(basename "$file")
    local dir=$(dirname "$file")
    
    TOTAL_DOCS=$((TOTAL_DOCS + 1))
    
    # Check if file is outdated
    local outdated_keywords=("OLD" "ARCHIVE" "DEPRECATED" "LEGACY" "HISTORICAL" "PHASE_1" "PHASE_2" "OLD_SCRIPTS")
    local is_outdated=false
    
    for keyword in "${outdated_keywords[@]}"; do
        if [[ "$filename" == *"$keyword"* ]]; then
            is_outdated=true
            break
        fi
    done
    
    # Check for project history indicators
    if [[ "$dir" == *"project_history"* ]] || [[ "$dir" == *"archive"* ]]; then
        is_outdated=true
    fi
    
    # Check file age (older than 3 months)
    local file_age=$(find "$file" -mtime +90 2>/dev/null)
    
    # Check content for outdated indicators
    if grep -q -i "deprecated\|legacy\|outdated\|old\|archive" "$file" 2>/dev/null; then
        is_outdated=true
    fi
    
    if [[ "$is_outdated" == true ]] || [[ -n "$file_age" ]]; then
        echo -e "${YELLOW}📄 $file${NC} - ${RED}OUTDATED${NC}"
        return 1
    else
        echo -e "${GREEN}📄 $file${NC} - ${GREEN}CURRENT${NC}"
        return 0
    fi
}

# Function to organize current documentation
organize_current_docs() {
    echo -e "\n${BLUE}📂 Organizing Current Documentation...${NC}"
    echo "=========================================="
    
    # Create organized directory structure
    mkdir -p "$DOCS_DIR"/{ci-cd,architecture,api,guides,deployment,getting-started,trading,admin,security,performance}
    
    # Move CI/CD docs
    for file in "$DOCS_DIR"/CI_CD_*.md; do
        if [ -f "$file" ]; then
            mv "$file" "$DOCS_DIR/ci-cd/"
            echo -e "${GREEN}✅ Moved $(basename "$file") to ci-cd/${NC}"
            ORGANIZED_DOCS=$((ORGANIZED_DOCS + 1))
        fi
    done
    
    # Move architecture docs (skip if already in architecture/)
    for file in "$DOCS_DIR"/architecture/*.md; do
        if [ -f "$file" ] && [[ "$file" != "$DOCS_DIR/architecture/"* ]]; then
            mv "$file" "$DOCS_DIR/architecture/"
            echo -e "${GREEN}✅ Moved $(basename "$file") to architecture/${NC}"
            ORGANIZED_DOCS=$((ORGANIZED_DOCS + 1))
        fi
    done
    
    # Move API docs
    for file in "$DOCS_DIR"/api/*.md; do
        if [ -f "$file" ]; then
            mv "$file" "$DOCS_DIR/api/"
            echo -e "${GREEN}✅ Moved $(basename "$file") to api/${NC}"
            ORGANIZED_DOCS=$((ORGANIZED_DOCS + 1))
        fi
    done
    
    # Move guides
    for file in "$DOCS_DIR"/guides/*.md; do
        if [ -f "$file" ]; then
            mv "$file" "$DOCS_DIR/guides/"
            echo -e "${GREEN}✅ Moved $(basename "$file") to guides/${NC}"
            ORGANIZED_DOCS=$((ORGANIZED_DOCS + 1))
        fi
    done
    
    # Move deployment docs
    for file in "$DOCS_DIR"/deployment/*.md; do
        if [ -f "$file" ]; then
            mv "$file" "$DOCS_DIR/deployment/"
            echo -e "${GREEN}✅ Moved $(basename "$file") to deployment/${NC}"
            ORGANIZED_DOCS=$((ORGANIZED_DOCS + 1))
        fi
    done
    
    # Move getting-started docs
    for file in "$DOCS_DIR"/getting-started/*.md; do
        if [ -f "$file" ]; then
            mv "$file" "$DOCS_DIR/getting-started/"
            echo -e "${GREEN}✅ Moved $(basename "$file") to getting-started/${NC}"
            ORGANIZED_DOCS=$((ORGANIZED_DOCS + 1))
        fi
    done
    
    # Move trading docs
    for file in "$DOCS_DIR"/trading/*.md; do
        if [ -f "$file" ]; then
            mv "$file" "$DOCS_DIR/trading/"
            echo -e "${GREEN}✅ Moved $(basename "$file") to trading/${NC}"
            ORGANIZED_DOCS=$((ORGANIZED_DOCS + 1))
        fi
    done
    
    # Categorize remaining root-level docs
    for file in "$DOCS_DIR"/*.md; do
        if [ -f "$file" ]; then
            local filename=$(basename "$file")
            
            case "$filename" in
                *ADMIN*|*admin*)
                    mv "$file" "$DOCS_DIR/admin/"
                    echo -e "${GREEN}✅ Moved $filename to admin/${NC}"
                    ORGANIZED_DOCS=$((ORGANIZED_DOCS + 1))
                    ;;
                *SECURITY*|*security*)
                    mv "$file" "$DOCS_DIR/security/"
                    echo -e "${GREEN}✅ Moved $filename to security/${NC}"
                    ORGANIZED_DOCS=$((ORGANIZED_DOCS + 1))
                    ;;
                *PERFORMANCE*|*performance*|*METRICS*|*metrics*)
                    mv "$file" "$DOCS_DIR/performance/"
                    echo -e "${GREEN}✅ Moved $filename to performance/${NC}"
                    ORGANIZED_DOCS=$((ORGANIZED_DOCS + 1))
                    ;;
                *TRADING*|*trading*|*TRADER*|*trader*)
                    mv "$file" "$DOCS_DIR/trading/"
                    echo -e "${GREEN}✅ Moved $filename to trading/${NC}"
                    ORGANIZED_DOCS=$((ORGANIZED_DOCS + 1))
                    ;;
                *)
                    echo -e "${CYAN}📄 $filename - Keeping in root${NC}"
                    KEPT_IN_PLACE=$((KEPT_IN_PLACE + 1))
                    ;;
            esac
        fi
    done
}

# Function to archive outdated documentation
archive_outdated_docs() {
    echo -e "\n${BLUE}🗂️ Archiving Outdated Documentation...${NC}"
    echo "======================================="
    
    # Create archive subdirectories
    mkdir -p "$ARCHIVE_DIR"/{project_history,deprecated,legacy,old_scripts}
    
    # Move project history
    if [ -d "$DOCS_DIR/project_history" ]; then
        mv "$DOCS_DIR/project_history" "$ARCHIVE_DIR/"
        echo -e "${YELLOW}📦 Moved project_history to archive/${NC}"
        MOVED_TO_ARCHIVE=$((MOVED_TO_ARCHIVE + 1))
    fi
    
    # Move outdated files
    for file in "$DOCS_DIR"/*.md; do
        if [ -f "$file" ]; then
            local filename=$(basename "$file")
            
            # Check if file should be archived
            if categorize_doc "$file"; then
                continue
            else
                # Move to appropriate archive subdirectory
                if [[ "$filename" == *"OLD"* ]] || [[ "$filename" == *"LEGACY"* ]]; then
                    mv "$file" "$ARCHIVE_DIR/legacy/"
                    echo -e "${YELLOW}📦 Moved $filename to archive/legacy/${NC}"
                elif [[ "$filename" == *"DEPRECATED"* ]]; then
                    mv "$file" "$ARCHIVE_DIR/deprecated/"
                    echo -e "${YELLOW}📦 Moved $filename to archive/deprecated/${NC}"
                else
                    mv "$file" "$ARCHIVE_DIR/"
                    echo -e "${YELLOW}📦 Moved $filename to archive/${NC}"
                fi
                MOVED_TO_ARCHIVE=$((MOVED_TO_ARCHIVE + 1))
            fi
        fi
    done
}

# Function to create documentation index
create_docs_index() {
    echo -e "\n${BLUE}📋 Creating Documentation Index...${NC}"
    echo "=================================="
    
    cat > "$DOCS_DIR/README.md" << 'EOF'
# 📚 QuantDesk Documentation

## 📂 Documentation Structure

### 🚀 CI/CD Pipeline
- [CI/CD Comprehensive Guide](./ci-cd/CI_CD_COMPREHENSIVE_GUIDE.md)
- [CI/CD Quick Reference](./ci-cd/CI_CD_QUICK_REFERENCE.md)
- [CI/CD Troubleshooting](./ci-cd/CI_CD_TROUBLESHOOTING.md)
- [CI/CD Architecture Diagrams](./ci-cd/CI_CD_ARCHITECTURE_DIAGRAMS.md)

### 🏗️ Architecture
- [Architecture Overview](./architecture/overview.md)
- [Complete Architecture](./architecture/complete-arch.md)
- [Professional Diagrams Guide](./architecture/PROFESSIONAL_DIAGRAMS_GUIDE.md)

### 🔌 API Documentation
- [API Reference](./api/API.md)
- [Postman Documentation](./api/postman-doc.md)

### 📖 Guides
- [Getting Started](./getting-started/README.md)
- [Installation Guide](./getting-started/installation.md)
- [Configuration Guide](./getting-started/configuration.md)
- [Quick Start](./getting-started/quick-start.md)

### 🚀 Deployment
- [Deployment Guide](./deployment/DEPLOYMENT_GUIDE.md)
- [Frontend Deployment](./deployment/FRONTEND_DEPLOYMENT.md)

### 💹 Trading
- [Trading Overview](./trading/overview.md)
- [Trading Strategies](./trading/)

### 👥 Admin
- [Admin Dashboard Access](./admin/ADMIN_DASHBOARD_ACCESS.md)
- [Admin User Management](./admin/ADMIN_USER_MANAGEMENT.md)

### 🔒 Security
- [Security Guide](./security/)
- [Security Checklist](./security/)

### 📊 Performance
- [Performance Metrics](./performance/PERFORMANCE_METRICS.md)
- [Performance Optimization](./performance/)

## 🗂️ Archived Documentation

Outdated and historical documentation has been moved to the `archive/docs/` directory:
- `project_history/` - Historical project documentation
- `deprecated/` - Deprecated features and guides
- `legacy/` - Legacy system documentation
- `old_scripts/` - Outdated scripts and tools

## 🔍 Finding Documentation

Use the search function in your editor or IDE to quickly find specific documentation:
- **CI/CD**: Search for "CI_CD" or "workflow"
- **API**: Search for "API" or "endpoint"
- **Trading**: Search for "trading" or "strategy"
- **Security**: Search for "security" or "audit"

## 📝 Contributing to Documentation

When adding new documentation:
1. Place files in the appropriate category directory
2. Update this README.md with links to new docs
3. Follow the naming convention: `CATEGORY_DESCRIPTION.md`
4. Include a brief description in the file header

---

**📚 This documentation is organized for easy navigation and maintenance.**
EOF

    echo -e "${GREEN}✅ Created documentation index${NC}"
}

# Main execution
echo -e "${BLUE}🚀 Starting Documentation Organization...${NC}"
echo ""

# Create backup
echo -e "${YELLOW}📦 Creating backup in $BACKUP_DIR...${NC}"
cp -r "$DOCS_DIR" "$BACKUP_DIR/"
echo -e "${GREEN}✅ Backup created${NC}"

# Organize current documentation
organize_current_docs

# Archive outdated documentation
archive_outdated_docs

# Create documentation index
create_docs_index

# Summary
echo -e "\n${BLUE}📊 Organization Summary${NC}"
echo "===================="
echo -e "Total Documents: $TOTAL_DOCS"
echo -e "${GREEN}Organized Documents: $ORGANIZED_DOCS${NC}"
echo -e "${YELLOW}Archived Documents: $MOVED_TO_ARCHIVE${NC}"
echo -e "${CYAN}Kept in Root: $KEPT_IN_PLACE${NC}"
echo -e "${PURPLE}Backup Location: $BACKUP_DIR${NC}"

echo ""
echo -e "${GREEN}🎉 Documentation organization completed!${NC}"
echo ""
echo -e "${BLUE}💡 Next Steps:${NC}"
echo "1. Review the organized structure in docs/"
echo "2. Check archived files in archive/docs/"
echo "3. Update any broken links in your documentation"
echo "4. Run: ./validate-docs.sh to check for issues"
echo "5. Run: ./consolidate-docs.sh to merge similar content"

echo ""
echo -e "${YELLOW}⚠️ Note: A backup has been created in $BACKUP_DIR${NC}"
