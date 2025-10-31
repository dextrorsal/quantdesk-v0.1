#!/usr/bin/env ts-node

/**
 * Check if we have authority to withdraw from old vault PDAs
 */

import { Connection, PublicKey, Keypair } from '@solana/web3.js';
import { Wallet } from '@coral-xyz/anchor';
import bs58 from 'bs58';
import * as fs from 'fs';
import * as path from 'path';

const connection = new Connection('https://api.devnet.solana.com', 'confirmed');

// Old vault with 3.1281 SOL
const OLD_VAULT = 'FsnVEemM46kshuCzMbeYezV4EFRD1sGugSG4WkRvgp3s';
const OLD_PROGRAM = 'GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a';

// Current program vault
const CURRENT_VAULT = '5pXGgCZiyhRWAbR29oebssF9Cb4tsSwZppvHBuTxUBZ4';
const CURRENT_PROGRAM = 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw';

async function getVaultInfo(vaultAddress: string, programId: string, vaultName: string) {
  console.log(`\n🔍 Checking ${vaultName}:`);
  console.log(`   Vault: ${vaultAddress}`);
  console.log(`   Program: ${programId}`);
  
  try {
    const vaultPubkey = new PublicKey(vaultAddress);
    const programPubkey = new PublicKey(programId);
    
    const accountInfo = await connection.getAccountInfo(vaultPubkey);
    if (!accountInfo) {
      console.log('   ❌ Account does not exist');
      return;
    }
    
    const balance = accountInfo.lamports / 1e9;
    console.log(`   💰 Balance: ${balance.toFixed(4)} SOL`);
    console.log(`   📊 Account Owner: ${accountInfo.owner.toString()}`);
    console.log(`   📏 Data Size: ${accountInfo.data.length} bytes`);
    
    // Check if owned by the program
    if (accountInfo.owner.equals(programPubkey)) {
      console.log('   ✅ Owned by program (PDA)');
      
      // Check if we can derive it
      try {
        const [derivedVault] = PublicKey.findProgramAddressSync(
          [Buffer.from('protocol_sol_vault')],
          programPubkey
        );
        
        if (derivedVault.equals(vaultPubkey)) {
          console.log('   ✅ Can derive vault PDA');
          console.log('   ⚠️  Would need program instruction to withdraw');
          console.log('   💡 Could close if program has close_account instruction');
        } else {
          console.log('   ⚠️  Derived PDA does not match');
        }
      } catch (e) {
        console.log('   ❌ Cannot derive vault PDA:', e);
      }
    } else {
      console.log(`   ⚠️  Owned by different program: ${accountInfo.owner.toString()}`);
    }
    
    // Check program info
    const programInfo = await connection.getAccountInfo(programPubkey);
    if (programInfo && programInfo.executable) {
      console.log('   📋 Program exists and is executable');
      
      // Get upgrade authority using solana CLI output format
      // We know from solana program show that authority is: wgfSHTWx1woRXhsWijj1kcpCP8tmbmK2KnouFVAuoc6
      console.log('   ⚠️  Upgrade authority check: Run `solana program show <program>` to get authority');
      
      // Check if we have the keypair
      const keypairPath = path.join(process.env.HOME || '', '.config/solana/keys/id.json');
      if (fs.existsSync(keypairPath)) {
        const keypairData = JSON.parse(fs.readFileSync(keypairPath, 'utf8'));
        const ourKeypair = Keypair.fromSecretKey(Uint8Array.from(keypairData));
        const ourPubkey = ourKeypair.publicKey;
        console.log(`   🔑 Our wallet: ${ourPubkey.toString()}`);
        console.log(`   💡 Check if this matches the program's upgrade authority`);
      }
    }
    
  } catch (error: any) {
    console.log(`   ❌ Error: ${error.message}`);
  }
}

async function main() {
  console.log('🔍 Checking vault authorities and reclaimability...\n');
  
  await getVaultInfo(OLD_VAULT, OLD_PROGRAM, 'OLD PROGRAM VAULT');
  await getVaultInfo(CURRENT_VAULT, CURRENT_PROGRAM, 'CURRENT PROGRAM VAULT');
  
  console.log('\n📋 Summary:');
  console.log('===========');
  console.log('If vault is owned by program (PDA), you need:');
  console.log('1. A program instruction to withdraw/close');
  console.log('2. OR upgrade authority to add such instruction');
  console.log('3. OR close the entire program if no longer needed');
  console.log('\n⚠️  Closing a program requires upgrade authority');
  console.log('   And typically needs to be done via upgrade instruction');
}

main().catch(console.error);

