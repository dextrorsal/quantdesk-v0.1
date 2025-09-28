#!/bin/bash

# QuantDesk Documentation HTML Update Script
# This script converts all markdown files to HTML for the docs site

echo "🔄 Updating QuantDesk Documentation HTML..."
echo ""

# Run the conversion script
python3 convert_markdown.py

echo ""
echo "🔗 Fixing internal links in HTML files..."
python3 fix_html_links.py

echo ""
echo "✅ HTML documentation updated successfully!"
echo "🌐 Visit http://localhost:8080 to view the updated docs"
echo ""
echo "💡 Tip: Run this script whenever you update markdown files"
