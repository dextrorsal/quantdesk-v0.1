#!/bin/bash

# QuantDesk Advanced Features Integration Script
# This script helps you integrate advanced features into your QuantDesk perpetual DEX

echo "🚀 QuantDesk Advanced Features Integration"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "contracts/smart-contracts/programs/quantdesk-perp-dex/Cargo.toml" ]; then
    echo "❌ Please run this script from the QuantDesk root directory"
    exit 1
fi

echo "📋 Available Advanced Features:"
echo "================================"
echo ""
echo "✅ Insurance Fund Management"
echo "   • initialize_insurance_fund"
echo "   • deposit_insurance_fund"
echo "   • withdraw_insurance_fund"
echo "   • update_risk_parameters"
echo ""
echo "✅ Emergency Controls"
echo "   • pause_program"
echo "   • resume_program"
echo "   • emergency_withdraw"
echo ""
echo "✅ Fee Management"
echo "   • update_trading_fees"
echo "   • update_funding_fees"
echo "   • collect_fees"
echo "   • distribute_fees"
echo ""
echo "✅ Oracle Management"
echo "   • add_oracle_feed"
echo "   • remove_oracle_feed"
echo "   • update_oracle_weights"
echo "   • emergency_oracle_override"
echo "   • update_pyth_price"
echo ""
echo "✅ Governance & Admin"
echo "   • update_program_authority"
echo "   • update_whitelist"
echo "   • update_market_parameters"
echo ""
echo "✅ Advanced Order Types"
echo "   • place_oco_order (One-Cancels-Other)"
echo "   • place_bracket_order"
echo ""
echo "✅ Cross-Program Integration"
echo "   • jupiter_swap"
echo "   • update_pyth_price"
echo ""

echo "🔧 Integration Steps:"
echo "====================="
echo ""
echo "1. Backup your current program:"
echo "   cp contracts/smart-contracts/programs/quantdesk-perp-dex/src/lib.rs contracts/smart-contracts/programs/quantdesk-perp-dex/src/lib_backup.rs"
echo ""
echo "2. Replace with enhanced version:"
echo "   cp contracts/smart-contracts/programs/quantdesk-perp-dex/src/lib_enhanced.rs contracts/smart-contracts/programs/quantdesk-perp-dex/src/lib.rs"
echo ""
echo "3. Build the enhanced program:"
echo "   cd contracts/smart-contracts && anchor build"
echo ""
echo "4. Test the new features:"
echo "   anchor test"
echo ""

# Ask user if they want to proceed
read -p "Do you want to proceed with the integration? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔄 Starting integration..."
    
    # Backup current lib.rs
    echo "📁 Backing up current lib.rs..."
    cp contracts/smart-contracts/programs/quantdesk-perp-dex/src/lib.rs contracts/smart-contracts/programs/quantdesk-perp-dex/src/lib_backup.rs
    echo "   ✅ Backup created: lib_backup.rs"
    
    # Replace with enhanced version
    echo "🔄 Replacing with enhanced version..."
    cp contracts/smart-contracts/programs/quantdesk-perp-dex/src/lib_enhanced.rs contracts/smart-contracts/programs/quantdesk-perp-dex/src/lib.rs
    echo "   ✅ Enhanced lib.rs installed"
    
    # Build the program
    echo "🔨 Building enhanced program..."
    cd contracts/smart-contracts
    if anchor build; then
        echo "   ✅ Build successful!"
        echo ""
        echo "🎉 Integration Complete!"
        echo "======================"
        echo ""
        echo "Your QuantDesk program now includes:"
        echo "• 26 original instructions"
        echo "• 15+ new advanced instructions"
        echo "• Insurance fund management"
        echo "• Emergency controls"
        echo "• Fee management"
        echo "• Oracle management"
        echo "• Governance controls"
        echo "• Advanced order types"
        echo "• Cross-program integration"
        echo ""
        echo "📋 Next Steps:"
        echo "1. Test the new features: anchor test"
        echo "2. Update your IDL: The new IDL will be generated"
        echo "3. Deploy to devnet: anchor deploy --provider.cluster devnet"
        echo "4. Test with IDL Space: ./setup-idl-space.sh"
        echo ""
        echo "🔗 New IDL will be available at:"
        echo "   contracts/smart-contracts/target/idl/quantdesk_perp_dex.json"
        echo ""
    else
        echo "   ❌ Build failed! Check the errors above."
        echo "   🔄 Restoring backup..."
        cp contracts/smart-contracts/programs/quantdesk-perp-dex/src/lib_backup.rs contracts/smart-contracts/programs/quantdesk-perp-dex/src/lib.rs
        echo "   ✅ Backup restored"
        exit 1
    fi
else
    echo "❌ Integration cancelled"
    exit 0
fi

echo ""
echo "💡 Pro Tips:"
echo "============="
echo ""
echo "• Test each feature individually before deploying"
echo "• Use IDL Space to test the new instructions"
echo "• Start with insurance fund initialization"
echo "• Test emergency controls in a safe environment"
echo "• Verify oracle integration with Pyth feeds"
echo ""
echo "🚀 Happy coding with your enhanced QuantDesk DEX!"
