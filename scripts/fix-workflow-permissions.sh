#!/bin/bash

# Script to fix GitHub Actions workflow permissions
# This adds proper permissions to all workflow files

WORKFLOWS_DIR="/home/dex/Desktop/quantdesk/.github/workflows"

echo "🔧 Fixing GitHub Actions workflow permissions..."

for workflow_file in "$WORKFLOWS_DIR"/*.yml "$WORKFLOWS_DIR"/*.yaml; do
    if [ -f "$workflow_file" ]; then
        echo "Processing: $(basename "$workflow_file")"
        
        # Create a temporary file
        temp_file=$(mktemp)
        
        # Add permissions at the top of the workflow (after name)
        awk '
        /^name:/ {
            print $0
            getline
            print $0
            print ""
            print "# Security: Explicit permissions for GITHUB_TOKEN"
            print "permissions:"
            print "  contents: read"
            print "  pull-requests: write"
            print "  issues: write"
            print "  checks: write"
            print "  statuses: write"
            print ""
            next
        }
        { print $0 }
        ' "$workflow_file" > "$temp_file"
        
        # Replace the original file
        mv "$temp_file" "$workflow_file"
        
        echo "✅ Fixed: $(basename "$workflow_file")"
    fi
done

echo "🎉 All workflow permissions fixed!"
echo "📊 This should resolve all 287 CodeQL alerts about missing workflow permissions."
