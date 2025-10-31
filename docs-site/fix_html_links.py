#!/usr/bin/env python3
"""
Fix internal markdown links in HTML files to point to HTML versions
"""

import os
import re
from pathlib import Path

def fix_html_links():
    """Fix internal .md links in HTML files to point to .html versions"""
    
    html_dir = Path(__file__).parent / "html"
    
    if not html_dir.exists():
        print("❌ HTML directory not found")
        return
    
    fixed_count = 0
    
    # Find all HTML files
    for html_file in html_dir.rglob("*.html"):
        try:
            # Read the HTML file
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix internal links - replace .md with .html in href attributes
            # Pattern: href="something.md" -> href="something.html"
            content = re.sub(r'href="([^"]+)\.md"', r'href="\1.html"', content)
            
            # Only write if content changed
            if content != original_content:
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                fixed_count += 1
                print(f"✅ Fixed links in: {html_file}")
                
        except Exception as e:
            print(f"❌ Error processing {html_file}: {e}")
    
    print(f"\n🎉 Fixed internal links in {fixed_count} HTML files")

if __name__ == "__main__":
    fix_html_links()
