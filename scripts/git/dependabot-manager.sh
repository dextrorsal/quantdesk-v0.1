#!/bin/bash

# 🔧 QuantDesk Dependabot Management Script
# This script helps you handle Dependabot PRs safely

set -e

echo "🤖 QuantDesk Dependabot Management"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository!"
        exit 1
    fi
}

# Function to fetch latest changes
fetch_latest() {
    print_status "Fetching latest changes from remote..."
    git fetch origin
    print_success "Latest changes fetched"
}

# Function to list Dependabot branches
list_dependabot_branches() {
    print_status "Listing Dependabot branches..."
    echo ""
    echo "📋 Available Dependabot PRs:"
    echo "=============================="
    
    local branches=$(git branch -r | grep dependabot | sed 's/origin\///')
    
    if [ -z "$branches" ]; then
        print_warning "No Dependabot branches found"
        return 0
    fi
    
    local count=1
    for branch in $branches; do
        echo "$count. $branch"
        count=$((count + 1))
    done
    echo ""
}

# Function to handle a specific Dependabot PR
handle_dependabot_pr() {
    local branch_name=$1
    
    if [ -z "$branch_name" ]; then
        print_error "Branch name is required"
        return 1
    fi
    
    print_status "Handling Dependabot PR: $branch_name"
    echo ""
    
    # Checkout the Dependabot branch
    print_status "Checking out branch: $branch_name"
    git checkout "$branch_name"
    
    # Show what changed
    print_status "Changes in this PR:"
    echo "======================"
    git log --oneline main..HEAD
    echo ""
    
    # Show package.json changes
    print_status "Package.json changes:"
    echo "======================"
    git diff main..HEAD -- package.json package-lock.json || true
    echo ""
    
    # Ask user what to do
    echo "What would you like to do with this PR?"
    echo "1. Test the changes"
    echo "2. Merge the changes"
    echo "3. Close the PR"
    echo "4. Skip this PR"
    echo ""
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1)
            test_dependabot_changes "$branch_name"
            ;;
        2)
            merge_dependabot_pr "$branch_name"
            ;;
        3)
            close_dependabot_pr "$branch_name"
            ;;
        4)
            print_status "Skipping PR: $branch_name"
            ;;
        *)
            print_error "Invalid choice"
            ;;
    esac
}

# Function to test Dependabot changes
test_dependabot_changes() {
    local branch_name=$1
    
    print_status "Testing Dependabot changes..."
    echo ""
    
    # Install dependencies
    print_status "Installing dependencies..."
    if [ -f "package.json" ]; then
        npm install
        print_success "Dependencies installed"
    fi
    
    # Run tests if available
    print_status "Running tests..."
    if npm run test 2>/dev/null; then
        print_success "Tests passed"
    else
        print_warning "No tests found or tests failed"
    fi
    
    # Build project if possible
    print_status "Building project..."
    if npm run build 2>/dev/null; then
        print_success "Build successful"
    else
        print_warning "No build script found or build failed"
    fi
    
    echo ""
    print_success "Testing completed for $branch_name"
    echo ""
    echo "Would you like to merge this PR? (y/n)"
    read -p "Enter your choice: " merge_choice
    
    if [ "$merge_choice" = "y" ] || [ "$merge_choice" = "Y" ]; then
        merge_dependabot_pr "$branch_name"
    else
        print_status "PR not merged. You can merge it later from GitHub."
    fi
}

# Function to merge Dependabot PR
merge_dependabot_pr() {
    local branch_name=$1
    
    print_status "Merging Dependabot PR: $branch_name"
    echo ""
    
    # Switch to main branch
    git checkout main
    
    # Merge the Dependabot branch
    git merge "$branch_name" --no-ff -m "chore: merge Dependabot PR $branch_name"
    
    # Push to remote
    git push origin main
    
    # Delete the Dependabot branch
    git push origin --delete "$branch_name"
    
    print_success "Dependabot PR $branch_name merged successfully"
}

# Function to close Dependabot PR
close_dependabot_pr() {
    local branch_name=$1
    
    print_status "Closing Dependabot PR: $branch_name"
    echo ""
    
    # Switch to main branch
    git checkout main
    
    # Delete the Dependabot branch
    git push origin --delete "$branch_name"
    
    print_success "Dependabot PR $branch_name closed"
}

# Function to handle all Dependabot PRs
handle_all_dependabot_prs() {
    print_status "Handling all Dependabot PRs..."
    echo ""
    
    local branches=$(git branch -r | grep dependabot | sed 's/origin\///')
    
    if [ -z "$branches" ]; then
        print_warning "No Dependabot branches found"
        return 0
    fi
    
    for branch in $branches; do
        echo "Processing: $branch"
        handle_dependabot_pr "$branch"
        echo ""
    done
    
    print_success "All Dependabot PRs processed"
}

# Function to setup branch protection
setup_branch_protection() {
    print_status "Setting up branch protection..."
    echo ""
    echo "To set up branch protection on GitHub:"
    echo "1. Go to your repository on GitHub"
    echo "2. Click Settings > Branches"
    echo "3. Click 'Add rule'"
    echo "4. Set branch name pattern to 'main'"
    echo "5. Enable 'Require pull request reviews before merging'"
    echo "6. Enable 'Require status checks to pass before merging'"
    echo "7. Enable 'Require branches to be up to date before merging'"
    echo "8. Click 'Create'"
    echo ""
    print_success "Branch protection setup instructions provided"
}

# Function to create a feature branch
create_feature_branch() {
    local feature_name=$1
    
    if [ -z "$feature_name" ]; then
        read -p "Enter feature name: " feature_name
    fi
    
    local branch_name="feature/$feature_name"
    
    print_status "Creating feature branch: $branch_name"
    
    # Start from main branch
    git checkout main
    git pull origin main
    
    # Create feature branch
    git checkout -b "$branch_name"
    
    print_success "Feature branch $branch_name created"
    echo ""
    echo "You can now work on your feature and commit changes:"
    echo "git add ."
    echo "git commit -m 'feat: add $feature_name'"
    echo "git push origin $branch_name"
}

# Main menu
show_menu() {
    echo ""
    echo "🔧 QuantDesk Git Management Menu"
    echo "=================================="
    echo "1. List Dependabot PRs"
    echo "2. Handle specific Dependabot PR"
    echo "3. Handle all Dependabot PRs"
    echo "4. Create feature branch"
    echo "5. Setup branch protection"
    echo "6. Show Git status"
    echo "7. Exit"
    echo ""
}

# Main function
main() {
    check_git_repo
    fetch_latest
    
    while true; do
        show_menu
        read -p "Enter your choice (1-7): " choice
        
        case $choice in
            1)
                list_dependabot_branches
                ;;
            2)
                list_dependabot_branches
                echo ""
                read -p "Enter branch name: " branch_name
                handle_dependabot_pr "$branch_name"
                ;;
            3)
                handle_all_dependabot_prs
                ;;
            4)
                create_feature_branch
                ;;
            5)
                setup_branch_protection
                ;;
            6)
                git status
                ;;
            7)
                print_success "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice"
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@"
