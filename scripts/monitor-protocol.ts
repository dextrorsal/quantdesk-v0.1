#!/usr/bin/env node
// @ts-nocheck
// Run with: npx ts-node scripts/monitor-protocol.ts

/**
 * QuantDesk Protocol Monitoring CLI
 * 
 * View protocol finances: fees, insurance fund, treasury
 * Safe for devnet testing
 */

import { Connection, PublicKey } from '@solana/web3.js';
import * as anchor from '@coral-xyz/anchor';
import { Program } from '@coral-xyz/anchor';

const RPC_URL = process.env.SOLANA_RPC_URL || 'https://api.devnet.solana.com';
const PROGRAM_ID = new PublicKey('HsSzXVuSwiqGQvocT2zMX8zc86KQ9TYfZFZcfmmejzso');

async function monitorProtocol() {
  console.log('🔍 QuantDesk Protocol Monitor\n');
  console.log(`Network: ${RPC_URL}`);
  console.log(`Program ID: ${PROGRAM_ID.toBase58()}\n`);

  const connection = new Connection(RPC_URL, 'confirmed');

  try {
    // 1. Check Program Account Balance (Infrastructure SOL)
    console.log('📊 PROGRAM ACCOUNT');
    console.log('─'.repeat(50));
    const programBalance = await connection.getBalance(PROGRAM_ID);
    console.log(`Balance: ${(programBalance / 1e9).toFixed(4)} SOL`);
    console.log(`Value: ~$${((programBalance / 1e9) * 100).toFixed(2)} USD (@ $100/SOL)`);
    console.log(`Note: This is locked rent, not revenue\n`);

    // 2. Find FeeCollector PDA
    console.log('💰 FEE COLLECTOR');
    console.log('─'.repeat(50));
    const [feeCollectorPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from('fee_collector')],
      PROGRAM_ID
    );
    console.log(`PDA: ${feeCollectorPDA.toBase58()}`);
    
    try {
      const feeAccountInfo = await connection.getAccountInfo(feeCollectorPDA);
      if (feeAccountInfo) {
        console.log(`✅ Fee collector initialized`);
        console.log(`Data size: ${feeAccountInfo.data.length} bytes`);
        
        // Decode fee data (simplified - you'd use Anchor program here)
        const feeBalance = await connection.getBalance(feeCollectorPDA);
        console.log(`SOL Balance: ${(feeBalance / 1e9).toFixed(6)} SOL`);
        
        // In production, you'd decode:
        // - trading_fees_collected
        // - funding_fees_collected
        // - maker/taker rates
      } else {
        console.log(`⚠️  Not initialized yet (normal for first deploy)`);
        console.log(`   Run: Initialize fee collector instruction`);
      }
    } catch (e) {
      console.log(`⚠️  Not found (needs initialization)`);
    }
    console.log();

    // 3. Find Insurance Fund PDA
    console.log('🛡️  INSURANCE FUND');
    console.log('─'.repeat(50));
    const [insuranceFundPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from('insurance_fund')],
      PROGRAM_ID
    );
    console.log(`PDA: ${insuranceFundPDA.toBase58()}`);
    
    try {
      const insuranceAccountInfo = await connection.getAccountInfo(insuranceFundPDA);
      if (insuranceAccountInfo) {
        console.log(`✅ Insurance fund initialized`);
        const insuranceBalance = await connection.getBalance(insuranceFundPDA);
        console.log(`SOL Balance: ${(insuranceBalance / 1e9).toFixed(6)} SOL`);
      } else {
        console.log(`⚠️  Not initialized yet`);
      }
    } catch (e) {
      console.log(`⚠️  Not found`);
    }
    console.log();

    // 4. Market Stats
    console.log('📈 MARKETS');
    console.log('─'.repeat(50));
    
    const markets = ['SOL/USD', 'BTC/USD', 'ETH/USD'];
    for (const symbol of markets) {
      const [base, quote] = symbol.split('/');
      const [marketPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from('market'), Buffer.from(base), Buffer.from(quote)],
        PROGRAM_ID
      );
      
      console.log(`${symbol}:`);
      console.log(`  PDA: ${marketPDA.toBase58()}`);
      
      const marketInfo = await connection.getAccountInfo(marketPDA);
      if (marketInfo) {
        console.log(`  ✅ Initialized`);
        const balance = await connection.getBalance(marketPDA);
        console.log(`  Balance: ${(balance / 1e9).toFixed(6)} SOL`);
      } else {
        console.log(`  ⚠️  Not initialized`);
      }
    }
    console.log();

    // 5. Explorer Links
    console.log('🔗 EXPLORER LINKS (Devnet)');
    console.log('─'.repeat(50));
    console.log(`Program: https://explorer.solana.com/address/${PROGRAM_ID.toBase58()}?cluster=devnet`);
    console.log(`Fee Collector: https://explorer.solana.com/address/${feeCollectorPDA.toBase58()}?cluster=devnet`);
    console.log(`Insurance Fund: https://explorer.solana.com/address/${insuranceFundPDA.toBase58()}?cluster=devnet`);
    console.log();

    // 6. Summary
    console.log('📋 SUMMARY');
    console.log('─'.repeat(50));
    console.log(`Total Protocol SOL: ${(programBalance / 1e9).toFixed(4)} SOL (infrastructure)`);
    console.log(`Status: ${programBalance > 0 ? '✅ Deployed' : '❌ Not deployed'}`);
    console.log(`\nℹ️  Note: Fee collection requires user trading activity`);

  } catch (error) {
    console.error('Error monitoring protocol:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  monitorProtocol();
}

export { monitorProtocol };

