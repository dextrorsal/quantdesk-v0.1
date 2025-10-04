#!/bin/bash

# QuantDesk IDL Space Quick Setup Script
# This script helps you quickly set up IDL Space with your QuantDesk program

echo "🚀 QuantDesk IDL Space Quick Setup"
echo "=================================="
echo ""

# Program information
PROGRAM_ID="GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a"
IDL_FILE="/home/dex/Desktop/quantdesk/contracts/smart-contracts/target/idl/quantdesk_perp_dex.json"

echo "📋 Program Information:"
echo "   Program ID: $PROGRAM_ID"
echo "   IDL File: $IDL_FILE"
echo ""

# Check if IDL file exists
if [ -f "$IDL_FILE" ]; then
    echo "✅ IDL file found!"
    echo "   File size: $(du -h "$IDL_FILE" | cut -f1)"
    echo "   Last modified: $(date -r "$IDL_FILE")"
else
    echo "❌ IDL file not found!"
    echo "   Run 'anchor build' in the contracts/smart-contracts directory first"
    exit 1
fi

echo ""
echo "🌐 IDL Space Setup Steps:"
echo "========================="
echo ""
echo "1. Open IDL Space in your browser:"
echo "   https://idl.space"
echo ""
echo "2. Click the 'Enter App' button"
echo ""
echo "3. Look for 'Import IDL' or 'Upload IDL' option"
echo ""
echo "4. Upload this file:"
echo "   $IDL_FILE"
echo ""

# Try to open IDL Space in browser
echo "🔗 Opening IDL Space..."
if command -v xdg-open > /dev/null; then
    xdg-open "https://idl.space" &
    echo "   ✅ Opened in default browser"
elif command -v open > /dev/null; then
    open "https://idl.space" &
    echo "   ✅ Opened in default browser"
else
    echo "   ⚠️  Please manually open: https://idl.space"
fi

echo ""
echo "📋 Copy this IDL file path to clipboard:"
echo "   $IDL_FILE"
echo ""

# Try to copy to clipboard
if command -v xclip > /dev/null; then
    echo "$IDL_FILE" | xclip -selection clipboard
    echo "   ✅ Copied to clipboard!"
elif command -v pbcopy > /dev/null; then
    echo "$IDL_FILE" | pbcopy
    echo "   ✅ Copied to clipboard!"
else
    echo "   ⚠️  Please manually copy the file path above"
fi

echo ""
echo "🎯 What to do next:"
echo "==================="
echo ""
echo "1. In IDL Space, click 'Enter App'"
echo "2. Find the 'Import IDL' or 'Upload IDL' option"
echo "3. Upload your IDL file (path copied above)"
echo "4. Start exploring your program's 26 instructions!"
echo ""
echo "💡 Pro Tips:"
echo "   • Connect your Phantom wallet for testing"
echo "   • Start with 'create_user_account' instruction"
echo "   • Use the PDA finder to derive account addresses"
echo "   • Test on devnet before mainnet"
echo ""
echo "🔗 Useful Links:"
echo "   • IDL Space: https://idl.space"
echo "   • Solana Docs: https://solana.com/developers"
echo "   • Anchor Docs: https://anchor-lang.com"
echo ""
echo "Happy coding! 🚀"
