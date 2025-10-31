#!/usr/bin/env ts-node

/**
 * Attempt to withdraw SOL from old vault
 * We have upgrade authority, so we can try to:
 * 1. Check if there's a withdraw instruction
 * 2. Or upgrade program with a withdraw instruction
 * 3. Or close the program entirely
 */

import { Connection, PublicKey, Keypair, Transaction, SystemProgram } from '@solana/web3.js';
import { Program, AnchorProvider, Wallet } from '@coral-xyz/anchor';
import * as fs from 'fs';
import * as path from 'path';

const connection = new Connection('https://api.devnet.solana.com', 'confirmed');

// Old program and vault
const OLD_PROGRAM_ID = 'GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a';
const OLD_VAULT = 'FsnVEemM46kshuCzMbeYezV4EFRD1sGugSG4WkRvgp3s';
const AUTHORITY = 'wgfSHTWx1woRXhsWijj1kcpCP8tmbmK2KnouFVAuoc6';

async function getKeypair(): Promise<Keypair> {
  const keypairPath = path.join(process.env.HOME || '', '.config/solana/keys/id.json');
  if (!fs.existsSync(keypairPath)) {
    throw new Error('Keypair file not found');
  }
  const keypairData = JSON.parse(fs.readFileSync(keypairPath, 'utf8'));
  return Keypair.fromSecretKey(Uint8Array.from(keypairData));
}

async function checkIDLExists() {
  try {
    const programPubkey = new PublicKey(OLD_PROGRAM_ID);
    const idlAccount = await Program.fetchIdl(programPubkey, { connection } as any);
    return idlAccount !== null;
  } catch (e) {
    return false;
  }
}

async function tryDirectWithdraw() {
  console.log('\n💡 Option 1: Check if program has withdraw instruction...');
  
  try {
    const idlExists = await checkIDLExists();
    if (!idlExists) {
      console.log('   ❌ No IDL found on-chain');
      console.log('   ⚠️  Cannot call program instructions without IDL');
      return false;
    }
    
    // Try to fetch IDL and check for withdraw/close instructions
    const programPubkey = new PublicKey(OLD_PROGRAM_ID);
    const idl = await Program.fetchIdl(programPubkey, { connection } as any);
    
    if (idl && idl.instructions) {
      console.log('   ✅ IDL found! Instructions:');
      idl.instructions.forEach((ix: any) => {
        console.log(`      - ${ix.name}`);
      });
      
      // Check for withdraw/close instructions
      const hasWithdraw = idl.instructions.some((ix: any) => 
        ix.name.toLowerCase().includes('withdraw') || 
        ix.name.toLowerCase().includes('close')
      );
      
      if (hasWithdraw) {
        console.log('   ✅ Found withdraw/close instruction!');
        console.log('   💡 Could potentially use this to reclaim SOL');
        return true;
      } else {
        console.log('   ❌ No withdraw/close instruction found');
        return false;
      }
    }
  } catch (e: any) {
    console.log(`   ❌ Error checking IDL: ${e.message}`);
    return false;
  }
  
  return false;
}

async function checkProgramBalance() {
  console.log('\n💰 Checking program balances...');
  
  const programInfo = await connection.getAccountInfo(new PublicKey(OLD_PROGRAM_ID));
  if (programInfo) {
    const programBalance = programInfo.lamports / 1e9;
    console.log(`   Program account: ${programBalance.toFixed(4)} SOL`);
    
    const vaultInfo = await connection.getAccountInfo(new PublicKey(OLD_VAULT));
    if (vaultInfo) {
      const vaultBalance = vaultInfo.lamports / 1e9;
      console.log(`   Vault PDA: ${vaultBalance.toFixed(4)} SOL`);
      console.log(`   Total: ${(programBalance + vaultBalance).toFixed(4)} SOL`);
      return vaultBalance;
    }
  }
  return 0;
}

async function main() {
  console.log('🔍 Checking options to reclaim SOL from old vault...\n');
  console.log(`Program: ${OLD_PROGRAM_ID}`);
  console.log(`Vault: ${OLD_VAULT}`);
  console.log(`Authority: ${AUTHORITY}\n`);
  
  // Check our wallet
  try {
    const keypair = await getKeypair();
    const ourWallet = keypair.publicKey.toString();
    console.log(`Our wallet: ${ourWallet}`);
    
    if (ourWallet === AUTHORITY) {
      console.log('✅ We ARE the upgrade authority!\n');
    } else {
      console.log('❌ We are NOT the upgrade authority');
      console.log('   Cannot reclaim SOL without authority\n');
      return;
    }
  } catch (e: any) {
    console.log(`❌ Error loading keypair: ${e.message}\n`);
    return;
  }
  
  // Check balances
  const vaultBalance = await checkProgramBalance();
  
  if (vaultBalance < 0.01) {
    console.log('\n⚠️  Vault balance too small to recover');
    return;
  }
  
  // Try to find withdraw instruction
  const hasWithdraw = await tryDirectWithdraw();
  
  if (!hasWithdraw) {
    console.log('\n💡 Option 2: Close entire program');
    console.log('   Since we have upgrade authority, we could:');
    console.log('   1. Upgrade program with empty buffer (closes it)');
    console.log('   2. This refunds rent from program account to authority');
    console.log('   ⚠️  BUT: Vault PDA SOL would be stuck (needs instruction)');
    console.log('   ⚠️  Need withdraw instruction to get vault SOL');
    console.log('\n💡 Option 3: Upgrade with withdraw instruction');
    console.log('   1. Build program with withdraw_vault instruction');
    console.log('   2. Upgrade old program');
    console.log('   3. Call withdraw_vault instruction');
    console.log('   4. Close program if no longer needed');
  }
  
  console.log(`\n💰 Total reclaimable: ~${vaultBalance.toFixed(4)} SOL`);
  console.log('\n⚠️  Note: Requires program modification or specific instructions');
}

main().catch(console.error);

