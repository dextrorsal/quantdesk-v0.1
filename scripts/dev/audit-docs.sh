#!/bin/bash

# 📚 Documentation Audit Script for QuantDesk
# Quick audit of documentation structure and health

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}📚 QuantDesk Documentation Audit${NC}"
echo "================================="
echo ""

# Configuration
DOCS_DIR="docs"
ARCHIVE_DIR="archive/docs"

# Counters
TOTAL_DOCS=0
CURRENT_DOCS=0
OUTDATED_DOCS=0
ARCHIVE_DOCS=0
EMPTY_DOCS=0
LARGE_DOCS=0

echo -e "${BLUE}🔍 Analyzing Documentation...${NC}"
echo ""

# Function to analyze a document
analyze_doc() {
    local file=$1
    local filename=$(basename "$file")
    local dir=$(dirname "$file")
    
    TOTAL_DOCS=$((TOTAL_DOCS + 1))
    
    # Check file size
    local file_size=$(wc -l < "$file" 2>/dev/null || echo "0")
    
    # Check if file is empty
    if [ "$file_size" -eq 0 ]; then
        EMPTY_DOCS=$((EMPTY_DOCS + 1))
        echo -e "${RED}📄 $file${NC} - ${RED}EMPTY${NC}"
        return
    fi
    
    # Check if file is very large (>1000 lines)
    if [ "$file_size" -gt 1000 ]; then
        LARGE_DOCS=$((LARGE_DOCS + 1))
        echo -e "${YELLOW}📄 $file${NC} - ${YELLOW}LARGE ($file_size lines)${NC}"
    fi
    
    # Check for outdated indicators
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
    
    if [[ "$is_outdated" == true ]] || [[ -n "$file_age" ]]; then
        OUTDATED_DOCS=$((OUTDATED_DOCS + 1))
        echo -e "${YELLOW}📄 $file${NC} - ${YELLOW}OUTDATED${NC}"
    elif [[ "$dir" == *"project_history"* ]] || [[ "$dir" == *"archive"* ]]; then
        ARCHIVE_DOCS=$((ARCHIVE_DOCS + 1))
        echo -e "${PURPLE}📄 $file${NC} - ${CYAN}ARCHIVE${NC}"
    else
        CURRENT_DOCS=$((CURRENT_DOCS + 1))
        echo -e "${GREEN}📄 $file${NC} - ${GREEN}CURRENT${NC}"
    fi
}

# Analyze all documentation files
echo -e "${BLUE}📋 Document Analysis:${NC}"
echo "──────────────────────────────────────────────────"

for doc in "$DOCS_DIR"/**/*.md; do
    if [ -f "$doc" ]; then
        analyze_doc "$doc"
    fi
done

# Analyze archive directory if it exists
if [ -d "$ARCHIVE_DIR" ]; then
    echo -e "\n${PURPLE}📦 Archive Analysis:${NC}"
    echo "──────────────────────────────────────────────────"
    
    for doc in "$ARCHIVE_DIR"/**/*.md; do
        if [ -f "$doc" ]; then
            analyze_doc "$doc"
        fi
    done
fi

echo ""
echo -e "${BLUE}📊 Documentation Statistics${NC}"
echo "=============================="
echo -e "Total Documents: $TOTAL_DOCS"
echo -e "${GREEN}Current Documents: $CURRENT_DOCS${NC}"
echo -e "${YELLOW}Outdated Documents: $OUTDATED_DOCS${NC}"
echo -e "${PURPLE}Archive Documents: $ARCHIVE_DOCS${NC}"
echo -e "${RED}Empty Documents: $EMPTY_DOCS${NC}"
echo -e "${YELLOW}Large Documents: $LARGE_DOCS${NC}"
echo ""

# Calculate health percentage
if [ $TOTAL_DOCS -gt 0 ]; then
    local health_percentage=$(( (CURRENT_DOCS * 100) / TOTAL_DOCS ))
    echo -e "${BLUE}📈 Documentation Health: $health_percentage%${NC}"
fi

echo ""
echo -e "${BLUE}💡 Recommendations${NC}"
echo "=================="

if [ $OUTDATED_DOCS -gt 0 ]; then
    echo -e "${YELLOW}1. Review $OUTDATED_DOCS outdated documents${NC}"
fi

if [ $EMPTY_DOCS -gt 0 ]; then
    echo -e "${RED}2. Fix $EMPTY_DOCS empty documents${NC}"
fi

if [ $LARGE_DOCS -gt 0 ]; then
    echo -e "${YELLOW}3. Consider splitting $LARGE_DOCS large documents${NC}"
fi

if [ $OUTDATED_DOCS -gt 5 ]; then
    echo -e "${BLUE}4. Run: ./organize-docs.sh to reorganize structure${NC}"
fi

if [ $CURRENT_DOCS -gt 50 ]; then
    echo -e "${BLUE}5. Run: ./validate-docs.sh to validate and consolidate${NC}"
fi

echo ""
echo -e "${GREEN}🎯 Quick Actions${NC}"
echo "==============="
echo "• Run: ./organize-docs.sh - Organize documentation structure"
echo "• Run: ./validate-docs.sh - Validate and consolidate content"
echo "• Check: reports/docs/ - For detailed analysis reports"

echo ""
echo -e "${BLUE}📚 Documentation audit completed!${NC}"