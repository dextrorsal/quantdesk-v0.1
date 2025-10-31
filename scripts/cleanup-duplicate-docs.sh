#!/bin/bash

# QuantDesk Documentation Cleanup Script
# Safely removes duplicate files from public_docs/ that exist in docs/

set -e  # Exit on any error

echo "🧹 QuantDesk Documentation Cleanup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
DOCS_DIR="docs"
PUBLIC_DOCS_DIR="public_docs"

# Check if directories exist
if [ ! -d "$DOCS_DIR" ]; then
    echo -e "${RED}❌ Error: $DOCS_DIR directory not found${NC}"
    exit 1
fi

if [ ! -d "$PUBLIC_DOCS_DIR" ]; then
    echo -e "${RED}❌ Error: $PUBLIC_DOCS_DIR directory not found${NC}"
    exit 1
fi

echo -e "${BLUE}📁 Scanning for duplicate files...${NC}"
echo ""

# Function to find all files recursively
find_files() {
    local dir="$1"
    find "$dir" -type f -name "*.md" -o -name "*.json" -o -name "*.html" -o -name "*.yml" -o -name "*.yaml" | sort
}

# Get all files from both directories
DOCS_FILES=$(find_files "$DOCS_DIR")
PUBLIC_DOCS_FILES=$(find_files "$PUBLIC_DOCS_DIR")

# Arrays to store results
declare -a DUPLICATES=()
declare -a UNIQUE_TO_PUBLIC=()
declare -a SIZE_DIFFERENCES=()

echo -e "${YELLOW}🔍 Analyzing files...${NC}"
echo ""

# Check for duplicates
while IFS= read -r public_file; do
    # Get relative path from public_docs
    relative_path="${public_file#$PUBLIC_DOCS_DIR/}"
    
    # Check if file exists in docs
    docs_file="$DOCS_DIR/$relative_path"
    
    if [ -f "$docs_file" ]; then
        # File exists in both directories
        DUPLICATES+=("$relative_path")
        
        # Check file sizes
        public_size=$(stat -f%z "$public_file" 2>/dev/null || stat -c%s "$public_file" 2>/dev/null)
        docs_size=$(stat -f%z "$docs_file" 2>/dev/null || stat -c%s "$docs_file" 2>/dev/null)
        
        if [ "$public_size" -ne "$docs_size" ]; then
            SIZE_DIFFERENCES+=("$relative_path (Public: ${public_size}b, Docs: ${docs_size}b)")
        fi
    else
        # File only exists in public_docs
        UNIQUE_TO_PUBLIC+=("$relative_path")
    fi
done <<< "$PUBLIC_DOCS_FILES"

# Display results
echo -e "${GREEN}📊 Analysis Results:${NC}"
echo "=================="
echo ""

echo -e "${BLUE}📋 Duplicate Files Found: ${#DUPLICATES[@]}${NC}"
if [ ${#DUPLICATES[@]} -gt 0 ]; then
    for file in "${DUPLICATES[@]}"; do
        echo "  ✅ $file"
    done
else
    echo "  No duplicates found"
fi
echo ""

echo -e "${YELLOW}⚠️  Files with Size Differences: ${#SIZE_DIFFERENCES[@]}${NC}"
if [ ${#SIZE_DIFFERENCES[@]} -gt 0 ]; then
    for file in "${SIZE_DIFFERENCES[@]}"; do
        echo "  ⚠️  $file"
    done
else
    echo "  All duplicate files have identical sizes"
fi
echo ""

echo -e "${RED}🔒 Files Unique to Public Docs: ${#UNIQUE_TO_PUBLIC[@]}${NC}"
if [ ${#UNIQUE_TO_PUBLIC[@]} -gt 0 ]; then
    for file in "${UNIQUE_TO_PUBLIC[@]}"; do
        echo "  🔒 $file"
    done
else
    echo "  No unique files found"
fi
echo ""

# Interactive cleanup
if [ ${#DUPLICATES[@]} -gt 0 ]; then
    echo -e "${YELLOW}🤔 Cleanup Options:${NC}"
    echo "=================="
    echo ""
    echo "1. Remove ALL duplicate files from public_docs/ (${#DUPLICATES[@]} files)"
    echo "2. Remove duplicates with identical sizes only"
    echo "3. Show detailed comparison for files with size differences"
    echo "4. Skip cleanup (exit)"
    echo ""
    
    read -p "Choose option (1-4): " choice
    
    case $choice in
        1)
            echo ""
            echo -e "${RED}⚠️  WARNING: This will delete ${#DUPLICATES[@]} files from public_docs/!${NC}"
            read -p "Are you sure? Type 'yes' to confirm: " confirm
            
            if [ "$confirm" = "yes" ]; then
                echo ""
                echo -e "${GREEN}🗑️  Removing duplicate files...${NC}"
                for file in "${DUPLICATES[@]}"; do
                    public_file="$PUBLIC_DOCS_DIR/$file"
                    echo "  Removing: $file"
                    rm "$public_file"
                done
                echo ""
                echo -e "${GREEN}✅ Cleanup complete! Removed ${#DUPLICATES[@]} duplicate files.${NC}"
            else
                echo -e "${YELLOW}❌ Cleanup cancelled.${NC}"
            fi
            ;;
        2)
            echo ""
            echo -e "${GREEN}🗑️  Removing duplicates with identical sizes...${NC}"
            removed_count=0
            for file in "${DUPLICATES[@]}"; do
                # Check if this file has size differences
                is_different=false
                for diff_file in "${SIZE_DIFFERENCES[@]}"; do
                    if [[ "$diff_file" == "$file"* ]]; then
                        is_different=true
                        break
                    fi
                done
                
                if [ "$is_different" = false ]; then
                    public_file="$PUBLIC_DOCS_DIR/$file"
                    echo "  Removing: $file"
                    rm "$public_file"
                    ((removed_count++))
                fi
            done
            echo ""
            echo -e "${GREEN}✅ Cleanup complete! Removed $removed_count files with identical sizes.${NC}"
            ;;
        3)
            echo ""
            echo -e "${BLUE}📊 Detailed Comparison:${NC}"
            echo "========================"
            for file in "${SIZE_DIFFERENCES[@]}"; do
                echo ""
                echo -e "${YELLOW}File: $file${NC}"
                echo "Public Docs:"
                head -5 "$PUBLIC_DOCS_DIR/$file" 2>/dev/null || echo "  (file not readable)"
                echo ""
                echo "Docs:"
                head -5 "$DOCS_DIR/$file" 2>/dev/null || echo "  (file not readable)"
                echo "---"
            done
            ;;
        4)
            echo -e "${YELLOW}❌ Cleanup skipped.${NC}"
            ;;
        *)
            echo -e "${RED}❌ Invalid option. Cleanup skipped.${NC}"
            ;;
    esac
else
    echo -e "${GREEN}✅ No duplicate files found. Nothing to clean up!${NC}"
fi

echo ""
echo -e "${BLUE}📈 Summary:${NC}"
echo "=========="
echo "• Duplicate files: ${#DUPLICATES[@]}"
echo "• Files with size differences: ${#SIZE_DIFFERENCES[@]}"
echo "• Files unique to public_docs: ${#UNIQUE_TO_PUBLIC[@]}"
echo ""

if [ ${#UNIQUE_TO_PUBLIC[@]} -gt 0 ]; then
    echo -e "${YELLOW}💡 Note: ${#UNIQUE_TO_PUBLIC[@]} files are unique to public_docs/ and were not removed.${NC}"
    echo "   These files may need manual review to determine if they should be:"
    echo "   • Moved to docs/"
    echo "   • Kept in public_docs/"
    echo "   • Deleted if obsolete"
fi

echo ""
echo -e "${GREEN}🎉 Script completed!${NC}"
