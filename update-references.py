#!/usr/bin/env python3
"""
Update all QuantDesk references to QuantDesk and fix domain references
"""

import os
import re
from pathlib import Path

def update_quantify_references():
    """Update all QuantDesk references to QuantDesk and fix domains"""
    
    project_root = Path(__file__).parent.parent
    
    # Files to update (exclude binary files and node_modules)
    files_to_update = []
    
    # Find all relevant files
    for pattern in ["**/*.md", "**/*.html", "**/*.js", "**/*.ts", "**/*.tsx", "**/*.py", "**/*.sh"]:
        files_to_update.extend(project_root.glob(pattern))
    
    # Exclude certain directories
    exclude_dirs = {"node_modules", ".git", "dist", "build", "coverage"}
    files_to_update = [f for f in files_to_update if not any(exclude in str(f) for exclude in exclude_dirs)]
    
    updated_files = 0
    
    for file_path in files_to_update:
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Update QuantDesk references (case-insensitive)
            content = re.sub(r'\bQuantify\b', 'QuantDesk', content, flags=re.IGNORECASE)
            content = re.sub(r'\bquantify\b', 'quantdesk', content, flags=re.IGNORECASE)
            
            # Update domain references
            content = re.sub(r'support@QuantDesk\.ai', 'contact@quantdesk.app', content)
            content = re.sub(r'support@quantdesk\.com', 'contact@quantdesk.app', content)
            content = re.sub(r'support@quantdesk\.app', 'contact@quantdesk.app', content)
            content = re.sub(r'https://api-dev\.quantdesk\.com', 'https://api-dev.quantdesk.app', content)
            content = re.sub(r'https://api-test\.quantdesk\.com', 'https://api-test.quantdesk.app', content)
            content = re.sub(r'https://status\.quantdesk\.com', 'https://status.quantdesk.app', content)
            
            # Update GitHub repository references
            content = re.sub(r'github\.com/yourusername/QuantDesk', 'github.com/quantdesk/quantdesk', content)
            content = re.sub(r'github\.com/quantdesk/quantdesk', 'github.com/quantdesk/quantdesk', content)
            
            # Update project references
            content = re.sub(r'QuantDesk-1\.0\.1', 'QuantDesk-1.0.1', content)
            content = re.sub(r'/home/dex/Desktop/QuantDesk-1\.0\.1', '/home/dex/Desktop/QuantDesk-1.0.1', content)
            
            # Update environment variable names
            content = re.sub(r'QUANTDESK_', 'QUANTDESK_', content)
            content = re.sub(r'quantdesk-env', 'quantdesk-env', content)
            
            # Update log file references
            content = re.sub(r'logs/QuantDesk\.log', 'logs/quantdesk.log', content)
            
            # Only write if content changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                updated_files += 1
                print(f"✅ Updated: {file_path.relative_to(project_root)}")
                
        except Exception as e:
            print(f"❌ Error updating {file_path}: {e}")
    
    print(f"\n🎉 Updated {updated_files} files")
    print("🔄 Run './update-html.sh' to regenerate HTML files with new references")

if __name__ == "__main__":
    update_quantify_references()
