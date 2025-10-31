#!/bin/bash

# 📚 Documentation Validation and Consolidation Script
# Validates documentation structure and consolidates similar content

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}📚 QuantDesk Documentation Validation & Consolidation${NC}"
echo "======================================================"
echo ""

# Configuration
DOCS_DIR="docs"
REPORTS_DIR="reports/docs"

# Create reports directory
mkdir -p "$REPORTS_DIR"

# Counters
TOTAL_DOCS=0
VALID_DOCS=0
INVALID_DOCS=0
DUPLICATE_DOCS=0
CONSOLIDATED_DOCS=0

echo -e "${BLUE}🔍 Validating Documentation Structure...${NC}"
echo ""

# Function to validate a document
validate_doc() {
    local file=$1
    local filename=$(basename "$file")
    
    TOTAL_DOCS=$((TOTAL_DOCS + 1))
    local is_valid=true
    local issues=()
    
    # Check if file exists and is readable
    if [ ! -f "$file" ]; then
        issues+=("File not found")
        is_valid=false
    elif [ ! -r "$file" ]; then
        issues+=("File not readable")
        is_valid=false
    fi
    
    # Check file size (not empty)
    if [ -f "$file" ] && [ ! -s "$file" ]; then
        issues+=("Empty file")
        is_valid=false
    fi
    
    # Check for basic markdown structure
    if [ -f "$file" ] && ! grep -q "^#" "$file"; then
        issues+=("No headers found")
        is_valid=false
    fi
    
    # Check for broken internal links
    if [ -f "$file" ]; then
        local broken_links=$(grep -o '\[.*\]([^)]*)' "$file" | grep -v 'http' | grep -v 'mailto' | while read link; do
            local target=$(echo "$link" | sed 's/\[.*\](\(.*\))/\1/')
            if [ ! -f "$DOCS_DIR/$target" ] && [ ! -f "$target" ]; then
                echo "$link"
            fi
        done)
        
        if [ -n "$broken_links" ]; then
            issues+=("Broken internal links: $broken_links")
            is_valid=false
        fi
    fi
    
    if [ "$is_valid" == true ]; then
        VALID_DOCS=$((VALID_DOCS + 1))
        echo -e "${GREEN}✅ $filename${NC}"
    else
        INVALID_DOCS=$((INVALID_DOCS + 1))
        echo -e "${RED}❌ $filename${NC}"
        for issue in "${issues[@]}"; do
            echo -e "   ${YELLOW}⚠️ $issue${NC}"
        done
    fi
}

# Function to find duplicate content
find_duplicates() {
    echo -e "\n${BLUE}🔍 Finding Duplicate Content...${NC}"
    echo "==============================="
    
    # Create temporary file for content hashes
    local temp_file=$(mktemp)
    
    # Generate content hashes for all markdown files
    find "$DOCS_DIR" -name "*.md" -type f | while read file; do
        if [ -f "$file" ]; then
            local hash=$(md5sum "$file" | cut -d' ' -f1)
            local filename=$(basename "$file")
            echo "$hash $filename $file" >> "$temp_file"
        fi
    done
    
    # Find duplicates
    sort "$temp_file" | uniq -d -w 32 | while read line; do
        local hash=$(echo "$line" | cut -d' ' -f1)
        local files=$(grep "^$hash" "$temp_file" | cut -d' ' -f3-)
        
        echo -e "${YELLOW}🔄 Duplicate content found:${NC}"
        echo "$files" | while read file; do
            echo -e "   ${CYAN}• $file${NC}"
        done
        
        DUPLICATE_DOCS=$((DUPLICATE_DOCS + 1))
    done
    
    rm "$temp_file"
}

# Function to consolidate similar content
consolidate_content() {
    echo -e "\n${BLUE}🔄 Consolidating Similar Content...${NC}"
    echo "=================================="
    
    # Find files with similar names
    find "$DOCS_DIR" -name "*.md" -type f | while read file; do
        local filename=$(basename "$file")
        local base_name=$(echo "$filename" | sed 's/_[0-9]*\.md$//' | sed 's/\.md$//')
        
        # Find other files with similar base names
        local similar_files=$(find "$DOCS_DIR" -name "${base_name}*.md" -type f | grep -v "$file")
        
        if [ -n "$similar_files" ]; then
            echo -e "${YELLOW}🔄 Similar files found for $filename:${NC}"
            echo "$similar_files" | while read similar_file; do
                echo -e "   ${CYAN}• $(basename "$similar_file")${NC}"
            done
            
            # Check if files can be consolidated
            local file_size=$(wc -l < "$file")
            local similar_size=$(echo "$similar_files" | xargs wc -l | tail -1 | awk '{print $1}')
            
            if [ "$file_size" -lt 50 ] && [ "$similar_size" -lt 50 ]; then
                echo -e "   ${GREEN}💡 Consider consolidating these small files${NC}"
                CONSOLIDATED_DOCS=$((CONSOLIDATED_DOCS + 1))
            fi
        fi
    done
}

# Function to generate documentation report
generate_report() {
    echo -e "\n${BLUE}📊 Generating Documentation Report...${NC}"
    echo "===================================="
    
    local report_file="$REPORTS_DIR/documentation-report-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" << EOF
# 📚 QuantDesk Documentation Report

**Generated**: $(date)
**Total Documents**: $TOTAL_DOCS
**Valid Documents**: $VALID_DOCS
**Invalid Documents**: $INVALID_DOCS
**Duplicate Documents**: $DUPLICATE_DOCS
**Consolidation Opportunities**: $CONSOLIDATED_DOCS

## 📊 Summary

- **Documentation Health**: $(( (VALID_DOCS * 100) / TOTAL_DOCS ))%
- **Duplicate Content**: $DUPLICATE_DOCS files
- **Consolidation Needed**: $CONSOLIDATED_DOCS files

## 🔍 Issues Found

### Invalid Documents
EOF

    # Add invalid documents to report
    find "$DOCS_DIR" -name "*.md" -type f | while read file; do
        if [ ! -f "$file" ] || [ ! -r "$file" ] || [ ! -s "$file" ]; then
            echo "- $(basename "$file"): Invalid file" >> "$report_file"
        fi
    done

    cat >> "$report_file" << EOF

### Duplicate Content
EOF

    # Add duplicate content to report
    find "$DOCS_DIR" -name "*.md" -type f | while read file; do
        local filename=$(basename "$file")
        local base_name=$(echo "$filename" | sed 's/_[0-9]*\.md$//' | sed 's/\.md$//')
        local similar_files=$(find "$DOCS_DIR" -name "${base_name}*.md" -type f | grep -v "$file")
        
        if [ -n "$similar_files" ]; then
            echo "- $filename: Similar to $(echo "$similar_files" | xargs -I {} basename {})" >> "$report_file"
        fi
    done

    cat >> "$report_file" << EOF

## 💡 Recommendations

1. **Fix Invalid Documents**: Address $INVALID_DOCS invalid documents
2. **Consolidate Duplicates**: Merge $DUPLICATE_DOCS duplicate files
3. **Optimize Structure**: Consider consolidating $CONSOLIDATED_DOCS small files
4. **Update Links**: Fix any broken internal links
5. **Regular Maintenance**: Run this script monthly to maintain documentation health

## 🚀 Next Steps

1. Review invalid documents and fix issues
2. Consolidate duplicate content
3. Update documentation index
4. Set up regular documentation audits
5. Implement documentation standards

---

**📚 This report helps maintain documentation quality and organization.**
EOF

    echo -e "${GREEN}✅ Report generated: $report_file${NC}"
}

# Function to create documentation standards
create_standards() {
    echo -e "\n${BLUE}📋 Creating Documentation Standards...${NC}"
    echo "===================================="
    
    cat > "$DOCS_DIR/DOCUMENTATION_STANDARDS.md" << 'EOF'
# 📚 QuantDesk Documentation Standards

## 📝 File Naming Conventions

### Format
- **Category**: `CATEGORY_DESCRIPTION.md`
- **Examples**: 
  - `CI_CD_COMPREHENSIVE_GUIDE.md`
  - `API_ENDPOINT_REFERENCE.md`
  - `TRADING_STRATEGY_OVERVIEW.md`

### Categories
- `CI_CD_*` - CI/CD pipeline documentation
- `API_*` - API documentation
- `TRADING_*` - Trading strategies and systems
- `ADMIN_*` - Admin dashboard and management
- `SECURITY_*` - Security guides and checklists
- `PERFORMANCE_*` - Performance metrics and optimization
- `DEPLOYMENT_*` - Deployment guides
- `ARCHITECTURE_*` - System architecture

## 📋 Document Structure

### Required Sections
1. **Title**: Clear, descriptive title
2. **Overview**: Brief description of the document
3. **Table of Contents**: For documents > 500 words
4. **Main Content**: Organized with clear headings
5. **Examples**: Code examples and use cases
6. **References**: Links to related documentation

### Header Format
```markdown
# 📚 Document Title

## 🎯 Overview
Brief description of what this document covers.

## 📋 Table of Contents
- [Section 1](#section-1)
- [Section 2](#section-2)

## 📖 Main Content
### Section 1
Content here...

### Section 2
Content here...

## 🔗 References
- [Related Doc 1](./related-doc-1.md)
- [Related Doc 2](./related-doc-2.md)
```

## 🎨 Style Guidelines

### Headers
- Use descriptive headers with emojis
- Use proper markdown hierarchy (H1, H2, H3)
- Keep headers concise but descriptive

### Code Blocks
- Use appropriate language tags
- Include comments for complex code
- Provide context for code examples

### Links
- Use relative paths for internal links
- Use descriptive link text
- Verify all links work

### Lists
- Use consistent bullet styles
- Keep lists focused and organized
- Use numbered lists for procedures

## 🔍 Quality Checklist

### Before Publishing
- [ ] Document follows naming convention
- [ ] All headers are properly formatted
- [ ] Code examples are tested and working
- [ ] Internal links are verified
- [ ] Document is proofread for clarity
- [ ] Examples are relevant and helpful
- [ ] References are up-to-date

### Regular Maintenance
- [ ] Review documents monthly
- [ ] Update outdated information
- [ ] Fix broken links
- [ ] Consolidate duplicate content
- [ ] Archive outdated documents

## 📊 Documentation Metrics

### Health Indicators
- **File Size**: 100-2000 words optimal
- **Header Count**: 3-10 headers per document
- **Link Count**: 2-10 internal links per document
- **Code Blocks**: 1-5 code examples per document

### Quality Metrics
- **Readability**: Clear, concise language
- **Completeness**: All sections filled
- **Accuracy**: Information is current
- **Usability**: Easy to follow and understand

---

**📚 These standards ensure consistent, high-quality documentation across the project.**
EOF

    echo -e "${GREEN}✅ Documentation standards created${NC}"
}

# Main execution
echo -e "${BLUE}🚀 Starting Documentation Validation...${NC}"
echo ""

# Validate all documentation
find "$DOCS_DIR" -name "*.md" -type f | while read file; do
    validate_doc "$file"
done

# Find duplicates
find_duplicates

# Consolidate content
consolidate_content

# Generate report
generate_report

# Create standards
create_standards

# Summary
echo -e "\n${BLUE}📊 Validation Summary${NC}"
echo "===================="
echo -e "Total Documents: $TOTAL_DOCS"
echo -e "${GREEN}Valid Documents: $VALID_DOCS${NC}"
echo -e "${RED}Invalid Documents: $INVALID_DOCS${NC}"
echo -e "${YELLOW}Duplicate Documents: $DUPLICATE_DOCS${NC}"
echo -e "${PURPLE}Consolidation Opportunities: $CONSOLIDATED_DOCS${NC}"

echo ""
echo -e "${GREEN}🎉 Documentation validation completed!${NC}"
echo ""
echo -e "${BLUE}💡 Next Steps:${NC}"
echo "1. Review the generated report in reports/docs/"
echo "2. Fix any invalid documents"
echo "3. Consolidate duplicate content"
echo "4. Follow the documentation standards"
echo "5. Set up regular validation runs"

echo ""
echo -e "${YELLOW}⚠️ Note: Check reports/docs/ for detailed analysis${NC}"
