#!/bin/bash

echo "🤖 QuantDesk Automated Test Runner"
echo "=================================="
echo ""
echo "Choose your test option:"
echo "1. Quick Test (Environment + Core)"
echo "2. Smart Contract Test (Build + Features)"
echo "3. Frontend Test (Build + Components)"
echo "4. Full Test Suite (Everything)"
echo "5. Custom Test (Specify components)"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Running Quick Test..."
        ./working-test.sh
        ;;
    2)
        echo ""
        echo "🔧 Running Smart Contract Test..."
        echo "Building smart contracts..."
        cd quantdesk-perp-dex && anchor build && cd ..
        echo ""
        echo "✅ Smart contract test complete!"
        ;;
    3)
        echo ""
        echo "🎨 Running Frontend Test..."
        echo "Installing dependencies..."
        cd frontend && npm install && cd ..
        echo ""
        echo "Building frontend..."
        cd frontend && npm run build && cd ..
        echo ""
        echo "✅ Frontend test complete!"
        ;;
    4)
        echo ""
        echo "🎯 Running Full Test Suite..."
        ./working-test.sh
        echo ""
        echo "Building smart contracts..."
        cd quantdesk-perp-dex && anchor build && cd ..
        echo ""
        echo "Building frontend..."
        cd frontend && npm run build && cd ..
        echo ""
        echo "✅ Full test suite complete!"
        ;;
    5)
        echo ""
        echo "🔧 Custom Test Options:"
        echo "a. Test environment only"
        echo "b. Test smart contracts only"
        echo "c. Test frontend only"
        echo "d. Test documentation only"
        echo ""
        read -p "Enter your choice (a-d): " custom_choice
        
        case $custom_choice in
            a)
                echo "Testing environment..."
                solana --version && anchor --version && rustc --version && node --version
                ;;
            b)
                echo "Testing smart contracts..."
                cd quantdesk-perp-dex && anchor build && cd ..
                ;;
            c)
                echo "Testing frontend..."
                cd frontend && npm run build && cd ..
                ;;
            d)
                echo "Testing documentation..."
                ls -la *.md
                ;;
            *)
                echo "Invalid choice. Exiting."
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "🎉 Test automation complete!"
echo "=================================="
