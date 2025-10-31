#!/usr/bin/env node
// @ts-nocheck
// Run with: npx ts-node scripts/monitor-users.ts <wallet_address>

/**
 * QuantDesk User Account Monitor
 * 
 * View user accounts, collateral, positions, and activity
 * For admin monitoring and support
 */

import { Connection, PublicKey } from '@solana/web3.js';
import * as anchor from '@coral-xyz/anchor';

const RPC_URL = process.env.SOLANA_RPC_URL || 'https://api.devnet.solana.com';
const PROGRAM_ID = new PublicKey('HsSzXVuSwiqGQvocT2zMX8zc86KQ9TYfZFZcfmmejzso');

function deriveUserAccountPDA(userPubkey: PublicKey, accountIndex: number): [PublicKey, number] {
  const accountIndexBuffer = Buffer.alloc(2);
  accountIndexBuffer.writeUInt16LE(accountIndex, 0);
  
  return PublicKey.findProgramAddressSync(
    [Buffer.from('user_account'), userPubkey.toBuffer(), accountIndexBuffer],
    PROGRAM_ID
  );
}

async function monitorUser(walletAddress: string) {
  console.log('👤 QuantDesk User Account Monitor\n');
  console.log(`Network: ${RPC_URL}`);
  console.log(`User Wallet: ${walletAddress}\n`);

  const connection = new Connection(RPC_URL, 'confirmed');
  const userPubkey = new PublicKey(walletAddress);

  try {
    // 1. Check user's wallet balance
    console.log('💰 WALLET BALANCE');
    console.log('─'.repeat(70));
    const balance = await connection.getBalance(userPubkey);
    console.log(`SOL Balance: ${(balance / 1e9).toFixed(6)} SOL (~$${((balance / 1e9) * 100).toFixed(2)} USD @ $100/SOL)`);
    console.log();

    // 2. Find and fetch user account (index 0)
    console.log('📊 USER ACCOUNT (Index 0)');
    console.log('─'.repeat(70));
    const [userAccountPDA] = deriveUserAccountPDA(userPubkey, 0);
    console.log(`User Account PDA: ${userAccountPDA.toBase58()}`);

    const userAccountInfo = await connection.getAccountInfo(userAccountPDA);

    if (!userAccountInfo) {
      console.log('❌ User account not created yet');
      console.log('   User needs to create account before trading');
      console.log();
      console.log('📋 TO CREATE ACCOUNT:');
      console.log('   - Connect wallet in frontend');
      console.log('   - Click "Create Account"');
      console.log('   - Or call create_user_account instruction');
      return;
    }

    console.log('✅ User account exists\n');

    // Decode user account data (simplified - you'd use Anchor program here)
    // For now, we'll just show raw data
    const accountData = userAccountInfo.data;
    
    // Account structure (from UserAccount):
    // 0-31: authority (Pubkey)
    // 32-33: account_index (u16)
    // 34-41: total_collateral (u64)
    // 42-43: total_positions (u16)
    // 44-45: total_orders (u16)
    // ... etc

    const authority = new PublicKey(accountData.slice(8, 40)); // Skip 8-byte discriminator
    const accountIndex = accountData.readUInt16LE(40);
    const totalCollateral = Number(accountData.readBigUInt64LE(42));
    const totalPositions = accountData.readUInt16LE(50);
    const totalOrders = accountData.readUInt16LE(52);

    console.log('Account Details:');
    console.log(`  Authority: ${authority.toBase58()}`);
    console.log(`  Account Index: ${accountIndex}`);
    console.log(`  Status: ${userAccountInfo ? 'Active' : 'Inactive'}\n`);

    console.log('💵 COLLATERAL & DEPOSITS');
    console.log('  Total Collateral: ${(totalCollateral / 1e6).toFixed(6)} USDC');
    console.log(`  USD Value: $${(totalCollateral / 1e6).toFixed(2)}`);
    console.log(`  Account Rent: ${(userAccountInfo.lamports / 1e9).toFixed(6)} SOL\n`);

    console.log('📈 TRADING ACTIVITY');
    console.log(`  Open Positions: ${totalPositions}`);
    console.log(`  Active Orders: ${totalOrders}`);
    console.log(`  Max Positions Allowed: 50 (default)\n`);

    // 3. Find user positions
    console.log('🎯 POSITIONS');
    console.log('─'.repeat(70));
    
    const markets = [
      { symbol: 'SOL/USD', base: 'SOL', quote: 'USD' },
      { symbol: 'BTC/USD', base: 'BTC', quote: 'USD' },
      { symbol: 'ETH/USD', base: 'ETH', quote: 'USD' },
    ];

    for (const market of markets) {
      const [marketPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from('market'), Buffer.from(market.base), Buffer.from(market.quote)],
        PROGRAM_ID
      );

      const [positionPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from('position'), userPubkey.toBuffer(), marketPDA.toBuffer()],
        PROGRAM_ID
      );

      const positionInfo = await connection.getAccountInfo(positionPDA);

      if (positionInfo) {
        console.log(`✅ ${market.symbol}:`);
        console.log(`   PDA: ${positionPDA.toBase58()}`);
        console.log(`   Data Size: ${positionInfo.data.length} bytes`);
        console.log(`   Status: Active`);
        // You'd decode position data here to show:
        // - Side (long/short)
        // - Size
        // - Entry price
        // - Leverage
        // - PnL
      } else {
        console.log(`   ${market.symbol}: No position`);
      }
    }
    console.log();

    // 4. Risk metrics (would be decoded from account data)
    console.log('⚠️  RISK METRICS');
    console.log('─'.repeat(70));
    console.log('  Account Health: N/A (decode from account data)');
    console.log('  Liquidation Price: N/A (decode from account data)');
    console.log('  Available Margin: N/A (decode from account data)');
    console.log('  Margin Used: N/A (decode from account data)\n');

    // 5. Fees & Funding
    console.log('💸 FEES & FUNDING');
    console.log('─'.repeat(70));
    console.log('  Total Fees Paid: N/A (decode from account data)');
    console.log('  Total Rebates Earned: N/A (decode from account data)');
    console.log('  Funding Paid: N/A (decode from account data)');
    console.log('  Funding Received: N/A (decode from account data)\n');

    // 6. Timestamps
    console.log('⏰ ACTIVITY');
    console.log('─'.repeat(70));
    console.log('  Account Created: N/A (decode from account data)');
    console.log('  Last Activity: N/A (decode from account data)\n');

    // 7. Explorer links
    console.log('🔗 EXPLORER LINKS (Devnet)');
    console.log('─'.repeat(70));
    console.log(`Wallet: https://explorer.solana.com/address/${walletAddress}?cluster=devnet`);
    console.log(`User Account: https://explorer.solana.com/address/${userAccountPDA.toBase58()}?cluster=devnet`);
    console.log();

    // 8. Admin actions
    console.log('🛠️  ADMIN NOTES');
    console.log('─'.repeat(70));
    console.log('  To decode full data: Use Anchor program.account.userAccount.fetch()');
    console.log('  To view positions: Query position PDAs for each market');
    console.log('  To view collateral: Query collateral account PDAs');
    console.log('  Warning: User data is sensitive - admin access only');

  } catch (error) {
    console.error('Error monitoring user:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  const walletAddress = process.argv[2];
  
  if (!walletAddress) {
    console.log('Usage: npx ts-node scripts/monitor-users.ts <wallet_address>');
    console.log('\nExample:');
    console.log('  npx ts-node scripts/monitor-users.ts 6g2rsczYGk6oorSfbt2zGWyU8YUToU8yCmWFYdY1QM3N');
    process.exit(1);
  }

  monitorUser(walletAddress);
}

export { monitorUser };

