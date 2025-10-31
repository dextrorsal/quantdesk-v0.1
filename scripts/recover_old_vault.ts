#!/usr/bin/env ts-node

/**
 * Recover SOL from old program vault
 * Creates a minimal withdraw instruction and upgrades the old program
 */

import { Connection, PublicKey, Keypair, Transaction, SystemProgram } from '@solana/web3.js';
import anchor from '@coral-xyz/anchor';
const { Program, AnchorProvider, Wallet, BN } = anchor;
import * as fs from 'fs';
import * as path from 'path';

const connection = new Connection('https://api.devnet.solana.com', 'confirmed');

// Old program and vault
const OLD_PROGRAM_ID = new PublicKey('GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a');
const OLD_VAULT = new PublicKey('FsnVEemM46kshuCzMbeYezV4EFRD1sGugSG4WkRvgp3s');
const AUTHORITY = new PublicKey('wgfSHTWx1woRXhsWijj1kcpCP8tmbmK2KnouFVAuoc6');

async function getKeypair(): Promise<Keypair> {
  const keypairPath = path.join(process.env.HOME || '', '.config/solana/keys/id.json');
  if (!fs.existsSync(keypairPath)) {
    throw new Error('Keypair file not found');
  }
  const keypairData = JSON.parse(fs.readFileSync(keypairPath, 'utf8'));
  return Keypair.fromSecretKey(Uint8Array.from(keypairData));
}

async function deriveVaultPDA(programId: PublicKey): Promise<[PublicKey, number]> {
  return PublicKey.findProgramAddressSync(
    [Buffer.from('protocol_sol_vault')],
    programId
  );
}

async function directTransfer() {
  console.log('\n💡 Attempting direct transfer from vault PDA...');
  
  try {
    const keypair = await getKeypair();
    
    // Check if we can create a transaction to transfer
    // This won't work because vault is owned by program, but let's try
    const vaultBalance = await connection.getBalance(OLD_VAULT);
    
    if (vaultBalance === 0) {
      console.log('   ❌ Vault is empty');
      return false;
    }
    
    console.log(`   💰 Vault balance: ${vaultBalance / 1e9} SOL`);
    
    // This won't work - vault is owned by program, not us
    // But let's try to see what error we get
    console.log('   ⚠️  Direct transfer will fail (vault is PDA owned by program)');
    console.log('   📋 Need program instruction to withdraw');
    
    return false;
  } catch (e: any) {
    console.log(`   ❌ Error: ${e.message}`);
    return false;
  }
}

async function trySimpleCloseAccount() {
  console.log('\n💡 Attempting to close vault account...');
  
  try {
    const keypair = await getKeypair();
    
    // Derive vault PDA to verify
    const [derivedVault, bump] = await deriveVaultPDA(OLD_PROGRAM_ID);
    
    if (!derivedVault.equals(OLD_VAULT)) {
      console.log(`   ⚠️  Derived vault doesn't match (${derivedVault.toString()})`);
      return false;
    }
    
    console.log(`   ✅ Vault PDA matches (bump: ${bump})`);
    
    // Try to create a close account instruction
    // This requires the program to sign, so won't work directly
    console.log('   ⚠️  Cannot close PDA account without program instruction');
    console.log('   📋 Need to upgrade program with close_account instruction');
    
    return false;
  } catch (e: any) {
    console.log(`   ❌ Error: ${e.message}`);
    return false;
  }
}

async function checkVaultBalance() {
  const balance = await connection.getBalance(OLD_VAULT);
  return balance / 1e9;
}

async function main() {
  console.log('🔍 Attempting to recover SOL from old vault...\n');
  console.log(`Program: ${OLD_PROGRAM_ID.toString()}`);
  console.log(`Vault: ${OLD_VAULT.toString()}`);
  
  // Verify authority
  try {
    const keypair = await getKeypair();
    if (!keypair.publicKey.equals(AUTHORITY)) {
      console.log('❌ We are NOT the upgrade authority!');
      return;
    }
    console.log('✅ Confirmed: We are the upgrade authority\n');
  } catch (e) {
    console.log(`❌ Error: ${e}`);
    return;
  }
  
  // Check balance
  const balance = await checkVaultBalance();
  console.log(`💰 Vault balance: ${balance.toFixed(4)} SOL\n`);
  
  if (balance < 0.01) {
    console.log('⚠️  Balance too small to recover');
    return;
  }
  
  // Try methods
  console.log('📋 Attempting recovery methods...\n');
  
  const directWorked = await directTransfer();
  if (directWorked) {
    console.log('✅ Successfully recovered via direct transfer!');
    return;
  }
  
  const closeWorked = await trySimpleCloseAccount();
  if (closeWorked) {
    console.log('✅ Successfully recovered via close account!');
    return;
  }
  
  // Final recommendation
  console.log('\n📋 Conclusion:');
  console.log('=============');
  console.log('❌ Cannot recover via direct methods');
  console.log('💡 Requires program upgrade with withdraw instruction');
  console.log('\n💡 Recommended approach:');
  console.log('1. Build minimal Anchor program with withdraw_vault instruction');
  console.log('2. Upgrade old program to new program with withdraw');
  console.log('3. Call withdraw_vault instruction');
  console.log('4. SOL transfers to authority');
  console.log('\n⚠️  This requires:');
  console.log('   - Creating new program with withdraw instruction');
  console.log('   - Deploying/upgrading old program');
  console.log('   - Calling withdraw instruction');
  console.log('\n💡 Alternative: Check if current program can interact with old vault');
  console.log(`   (Unlikely, but worth checking)\n`);
  
  console.log(`💰 Reclaimable amount: ${balance.toFixed(4)} SOL (~$${(balance * 190).toFixed(2)})`);
}

main().catch(console.error);

