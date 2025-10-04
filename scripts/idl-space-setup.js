#!/usr/bin/env node

/**
 * IDL Space Setup Script for QuantDesk Perpetual DEX
 * 
 * This script helps you interact with IDL Space using your existing Solana program.
 * IDL Space is a powerful tool for:
 * - Building and testing transactions
 * - Finding PDAs (Program Derived Addresses)
 * - Inspecting account states
 * - Debugging Solana programs
 */

const fs = require('fs');
const path = require('path');

// Configuration
const CONFIG = {
    programId: 'GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a',
    idlPath: './contracts/smart-contracts/target/idl/quantdesk_perp_dex.json',
    network: 'devnet', // Change to 'mainnet-beta' for production
    rpcUrl: 'https://api.devnet.solana.com'
};

console.log('🚀 QuantDesk IDL Space Setup');
console.log('============================');
console.log('');

// Read and validate IDL file
function loadIDL() {
    try {
        const idlContent = fs.readFileSync(CONFIG.idlPath, 'utf8');
        const idl = JSON.parse(idlContent);
        
        console.log('✅ IDL File Loaded Successfully');
        console.log(`   Program: ${idl.metadata.name}`);
        console.log(`   Version: ${idl.metadata.version}`);
        console.log(`   Program ID: ${CONFIG.programId}`);
        console.log(`   Instructions: ${idl.instructions.length}`);
        console.log(`   Accounts: ${idl.accounts.length}`);
        console.log(`   Types: ${idl.types.length}`);
        console.log('');
        
        return idl;
    } catch (error) {
        console.error('❌ Error loading IDL file:', error.message);
        process.exit(1);
    }
}

// Display available instructions
function displayInstructions(idl) {
    console.log('📋 Available Instructions:');
    console.log('===========================');
    
    const instructionGroups = {
        'Market Management': ['initialize_market', 'update_oracle_price', 'settle_funding'],
        'Position Management': ['open_position', 'close_position', 'liquidate_position', 'open_position_cross_collateral', 'liquidate_position_cross_collateral'],
        'Order Management': ['place_order', 'cancel_order', 'execute_conditional_order'],
        'Collateral Management': ['initialize_collateral_account', 'add_collateral', 'remove_collateral', 'update_collateral_value'],
        'Token Operations': ['initialize_token_vault', 'deposit_tokens', 'withdraw_tokens', 'create_user_token_account'],
        'SOL Operations': ['initialize_protocol_sol_vault', 'deposit_native_sol', 'withdraw_native_sol'],
        'User Account Management': ['create_user_account', 'update_user_account', 'close_user_account', 'check_user_permissions']
    };
    
    Object.entries(instructionGroups).forEach(([group, instructions]) => {
        console.log(`\n${group}:`);
        instructions.forEach(instructionName => {
            const instruction = idl.instructions.find(inst => inst.name === instructionName);
            if (instruction) {
                console.log(`   • ${instructionName} (${instruction.accounts.length} accounts)`);
            }
        });
    });
    
    console.log('');
}

// Display available account types
function displayAccountTypes(idl) {
    console.log('🏦 Available Account Types:');
    console.log('============================');
    
    idl.accounts.forEach(account => {
        console.log(`   • ${account.name}`);
    });
    
    console.log('');
}

// Generate IDL Space URLs
function generateIDLSpaceURLs() {
    console.log('🌐 IDL Space URLs:');
    console.log('==================');
    console.log('');
    
    const baseUrl = 'https://idl.space';
    const programId = CONFIG.programId;
    
    console.log('Main IDL Space Interface:');
    console.log(`   ${baseUrl}/program/${programId}`);
    console.log('');
    
    console.log('Direct Links:');
    console.log(`   • Build Transaction: ${baseUrl}/program/${programId}/build`);
    console.log(`   • Find PDAs: ${baseUrl}/program/${programId}/pda`);
    console.log(`   • View Accounts: ${baseUrl}/program/${programId}/accounts`);
    console.log(`   • Program Info: ${baseUrl}/program/${programId}/info`);
    console.log('');
}

// Generate example PDA derivations
function generatePDAExamples(idl) {
    console.log('🔍 Example PDA Derivations:');
    console.log('============================');
    console.log('');
    
    const examples = [
        {
            name: 'User Account PDA',
            seeds: ['user_account', 'authority', 'account_index'],
            description: 'Derives user account address'
        },
        {
            name: 'Market PDA',
            seeds: ['market', 'base_asset', 'quote_asset'],
            description: 'Derives market address (e.g., BTC/USDT)'
        },
        {
            name: 'Position PDA',
            seeds: ['position', 'user', 'market'],
            description: 'Derives position address for a user in a market'
        },
        {
            name: 'Order PDA',
            seeds: ['order', 'user', 'market'],
            description: 'Derives order address for a user in a market'
        },
        {
            name: 'Collateral Account PDA',
            seeds: ['collateral', 'user', 'asset_type'],
            description: 'Derives collateral account for a user and asset'
        },
        {
            name: 'Protocol SOL Vault PDA',
            seeds: ['protocol_sol_vault'],
            description: 'Derives protocol SOL vault address'
        }
    ];
    
    examples.forEach(example => {
        console.log(`${example.name}:`);
        console.log(`   Seeds: [${example.seeds.map(s => `"${s}"`).join(', ')}]`);
        console.log(`   Description: ${example.description}`);
        console.log('');
    });
}

// Generate transaction examples
function generateTransactionExamples(idl) {
    console.log('💸 Example Transaction Building:');
    console.log('=================================');
    console.log('');
    
    const examples = [
        {
            instruction: 'create_user_account',
            description: 'Create a new user account',
            required: ['authority (signer)', 'account_index (u16)']
        },
        {
            instruction: 'initialize_market',
            description: 'Initialize a new trading market',
            required: ['authority (signer)', 'base_asset (string)', 'quote_asset (string)', 'initial_price (u64)', 'max_leverage (u8)', 'initial_margin_ratio (u16)', 'maintenance_margin_ratio (u16)']
        },
        {
            instruction: 'deposit_native_sol',
            description: 'Deposit SOL to user account',
            required: ['user (signer)', 'amount (u64)']
        },
        {
            instruction: 'open_position',
            description: 'Open a new trading position',
            required: ['user (signer)', 'size (u64)', 'side (PositionSide)', 'leverage (u8)']
        },
        {
            instruction: 'place_order',
            description: 'Place a trading order',
            required: ['user (signer)', 'order_type (OrderType)', 'side (PositionSide)', 'size (u64)', 'price (u64)', 'leverage (u8)']
        }
    ];
    
    examples.forEach(example => {
        console.log(`${example.instruction}:`);
        console.log(`   Description: ${example.description}`);
        console.log(`   Required Parameters: ${example.required.join(', ')}`);
        console.log('');
    });
}

// Main execution
function main() {
    const idl = loadIDL();
    displayInstructions(idl);
    displayAccountTypes(idl);
    generateIDLSpaceURLs();
    generatePDAExamples(idl);
    generateTransactionExamples(idl);
    
    console.log('🎯 Next Steps:');
    console.log('==============');
    console.log('');
    console.log('1. Open IDL Space in your browser:');
    console.log(`   https://idl.space/program/${CONFIG.programId}`);
    console.log('');
    console.log('2. Upload your IDL file or use the program ID directly');
    console.log('');
    console.log('3. Start building and testing transactions!');
    console.log('');
    console.log('4. Use the PDA finder to derive account addresses');
    console.log('');
    console.log('5. Inspect account states and debug your program');
    console.log('');
    console.log('💡 Pro Tips:');
    console.log('============');
    console.log('');
    console.log('• Use devnet for testing (current config)');
    console.log('• Switch to mainnet-beta for production');
    console.log('• IDL Space works great with Phantom wallet');
    console.log('• You can build complex multi-instruction transactions');
    console.log('• PDA finder helps with account derivation');
    console.log('');
}

// Run the script
if (require.main === module) {
    main();
}

module.exports = {
    CONFIG,
    loadIDL,
    displayInstructions,
    generateIDLSpaceURLs
};
