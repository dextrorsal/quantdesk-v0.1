#!/usr/bin/env ts-node

/**
 * Check SOL balances in old programs and PDAs that might be reclaimable
 */

import { Connection, PublicKey } from '@solana/web3.js';
import * as anchor from '@coral-xyz/anchor';

const connection = new Connection('https://api.devnet.solana.com', 'confirmed');

// All known program IDs from Anchor.toml
const PROGRAM_IDS = {
  quantdesk_collateral: 'GPrakftrbBUUiir2MpQZv6G7UB5Jq8yNGHV5YTVYPQ5i',
  quantdesk_core: 'CNfhSBoMkRbDEQ2EC3RkfJ2S39Up6WJLr4U31ZL49LrU',
  quantdesk_oracle: '8gjwta4tMQshM7HbnEMsdFUMqjRe7XgVnxJVbcmf3cAC',
  quantdesk_perp_dex: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw', // Current active
  quantdesk_security: '84b7Khx4uj7mHDvn2V63kNSwkcgpagrBgZSdTJ7kTxWW',
  quantdesk_trading: 'AvxWXu25yWhDXJBy1V5GYcn2eVws4F2QWK5G3zV4t8sZ',
  
  // Old program IDs that might have existed before
  old_program_1: 'Gmz5q8cadQ5P8eZHTaQfVcCfm9dZVNSyjftYKbM9Dxpx', // We deployed this earlier
  old_program_2: 'GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a', // Frontend was using this
};

interface ProgramBalance {
  name: string;
  programId: string;
  balance: number;
  exists: boolean;
  isProgram: boolean;
  authority?: string;
  executable: boolean;
}

async function checkProgramBalance(name: string, programId: string): Promise<ProgramBalance> {
  try {
    const pubkey = new PublicKey(programId);
    const accountInfo = await connection.getAccountInfo(pubkey);
    
    if (!accountInfo) {
      return {
        name,
        programId,
        balance: 0,
        exists: false,
        isProgram: false,
        executable: false,
      };
    }
    
    const balance = accountInfo.lamports / 1e9; // Convert to SOL
    const isProgram = accountInfo.executable;
    
    // If it's a program, check authority (upgrade authority)
    let authority: string | undefined;
    if (isProgram) {
      try {
        const programInfo = await connection.getProgramAccounts(pubkey);
        // Authority would be in upgrade authority, but we'd need to parse it differently
      } catch (e) {
        // Ignore
      }
    }
    
    return {
      name,
      programId,
      balance,
      exists: true,
      isProgram,
      executable: accountInfo.executable,
      authority,
    };
  } catch (error) {
    return {
      name,
      programId,
      balance: 0,
      exists: false,
      isProgram: false,
      executable: false,
    };
  }
}

async function checkProtocolVault(programId: PublicKey): Promise<{ address: string; balance: number }> {
  try {
    const [vaultPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('protocol_sol_vault')],
      programId
    );
    
    const balance = await connection.getBalance(vaultPda);
    return {
      address: vaultPda.toString(),
      balance: balance / 1e9,
    };
  } catch (error) {
    return {
      address: 'ERROR',
      balance: 0,
    };
  }
}

async function main() {
  console.log('🔍 Checking old program balances on devnet...\n');
  
  const results: ProgramBalance[] = [];
  let totalReclaimable = 0;
  
  // Check all known programs
  for (const [name, programId] of Object.entries(PROGRAM_IDS)) {
    console.log(`Checking ${name}...`);
    const result = await checkProgramBalance(name, programId);
    results.push(result);
    
    if (result.exists && result.balance > 0) {
      const solAmount = result.balance.toFixed(4);
      console.log(`  ✅ ${name}: ${solAmount} SOL (${result.isProgram ? 'Program' : 'Account'})`);
      
      // Programs have minimum balance for rent, but if it's not executable, it might be reclaimable
      if (!result.isProgram || result.balance > 2.0) {
        // Only count if it's significant (> 2 SOL) or not a program
        totalReclaimable += result.balance;
      }
      
      // Check vault PDA for this program
      try {
        const vault = await checkProtocolVault(new PublicKey(programId));
        if (vault.balance > 0) {
          console.log(`  💰 Vault PDA: ${vault.address}`);
          console.log(`     Balance: ${vault.balance.toFixed(4)} SOL`);
          totalReclaimable += vault.balance;
        }
      } catch (e) {
        // Ignore vault check errors
      }
    } else {
      console.log(`  ❌ ${name}: Does not exist`);
    }
  }
  
  console.log('\n📊 Summary:');
  console.log('===========');
  
  const existing = results.filter(r => r.exists && r.balance > 0);
  const programs = results.filter(r => r.exists && r.isProgram);
  const nonPrograms = results.filter(r => r.exists && !r.isProgram);
  
  console.log(`Total programs/accounts checked: ${results.length}`);
  console.log(`Existing with balance: ${existing.length}`);
  console.log(`Programs: ${programs.length}`);
  console.log(`Non-program accounts: ${nonPrograms.length}`);
  console.log(`\n💰 Estimated reclaimable: ~${totalReclaimable.toFixed(4)} SOL`);
  console.log('\n⚠️  Note: Program accounts require minimum rent (usually ~2 SOL)');
  console.log('   Non-program accounts might be fully reclaimable if you have authority\n');
  
  // Show details
  console.log('📋 Detailed Results:');
  console.log('===================');
  for (const result of existing) {
    const type = result.isProgram ? 'PROGRAM' : 'ACCOUNT';
    const reclaimable = result.isProgram && result.balance <= 2.0 
      ? ' (Rent minimum)' 
      : result.isProgram 
        ? ` (~${(result.balance - 2.0).toFixed(4)} SOL potentially reclaimable)` 
        : ' (Check authority needed)';
    console.log(`${result.name.padEnd(25)} ${type.padEnd(8)} ${result.balance.toFixed(4).padStart(10)} SOL${reclaimable}`);
  }
}

main().catch(console.error);

