#!/usr/bin/env ts-node

/**
 * Deploy recovery program to old program ID and recover SOL
 */

import { Connection, PublicKey, Keypair, SystemProgram } from '@solana/web3.js';
import anchor from '@coral-xyz/anchor';
const { Program, AnchorProvider, Wallet, BN } = anchor;
import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';

const connection = new Connection('https://api.devnet.solana.com', 'confirmed');

// Old program ID (we'll deploy recovery program to this ID)
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

async function buildRecoveryProgram() {
  console.log('\n🔨 Building recovery program...');
  try {
    const contractsDir = path.resolve(process.cwd(), 'contracts');
    
    // Check if we need to add program to Anchor.toml
    const anchorToml = fs.readFileSync(path.join(contractsDir, 'Anchor.toml'), 'utf8');
    if (!anchorToml.includes('vault-recovery')) {
      console.log('⚠️  Need to add vault-recovery to Anchor.toml first');
      console.log('   Or manually configure Anchor.toml for program:');
      console.log(`   [programs.devnet]`);
      console.log(`   vault-recovery = "${OLD_PROGRAM_ID.toString()}"`);
      return false;
    }
    
    console.log('   Running: anchor build --program-name vault-recovery');
    execSync('anchor build --program-name vault-recovery', { 
      stdio: 'inherit',
      cwd: contractsDir
    });
    
    console.log('✅ Build successful!');
    return true;
  } catch (e: any) {
    console.log(`❌ Build failed: ${e.message}`);
    return false;
  }
}

async function deployRecoveryProgram() {
  console.log('\n🚀 Deploying recovery program...');
  
  try {
    const keypair = await getKeypair();
    
    console.log(`   Program will be deployed to: ${OLD_PROGRAM_ID.toString()}`);
    console.log('   ⚠️  This UPGRADES the old program!');
    
    // Deploy using solana program deploy with upgrade
    console.log('   Running: solana program deploy --program-id ...');
    const soPath = path.resolve(process.cwd(), 'contracts', 'target', 'deploy', 'vault_recovery.so');
    
    if (!fs.existsSync(soPath)) {
      console.log(`❌ SO file not found at: ${soPath}`);
      return false;
    }
    
    // Use the old program ID for upgrade
    execSync(
      `solana program deploy --program-id ${OLD_PROGRAM_ID.toString()} ${soPath} --url devnet`,
      { stdio: 'inherit' }
    );
    
    console.log('✅ Deployment successful!');
    return true;
  } catch (e: any) {
    console.log(`❌ Deployment failed: ${e.message}`);
    return false;
  }
}

async function recoverSOL() {
  console.log('\n💰 Recovering SOL from vault...');
  
  try {
    const keypair = await getKeypair();
    const wallet = new Wallet(keypair);
    const provider = new AnchorProvider(connection, wallet, { commitment: 'confirmed' });
    
    // Load IDL (should be generated after build)
    const idlPath = path.resolve(process.cwd(), 'contracts', 'target', 'idl', 'vault_recovery.json');
    
    if (!fs.existsSync(idlPath)) {
      console.log('❌ IDL not found. Build may have failed.');
      console.log(`   Expected at: ${idlPath}`);
      return false;
    }
    
    const idl = JSON.parse(fs.readFileSync(idlPath, 'utf8'));
    const program = new Program(idl, provider);
    
    // Check vault balance
    const vaultBalance = await connection.getBalance(OLD_VAULT);
    console.log(`   Vault balance: ${vaultBalance / 1e9} SOL`);
    
    if (vaultBalance === 0) {
      console.log('   ⚠️  Vault is empty');
      return false;
    }
    
    // Derive vault PDA with bump
    const [vaultPda, bump] = PublicKey.findProgramAddressSync(
      [Buffer.from('protocol_sol_vault')],
      OLD_PROGRAM_ID
    );
    
    console.log(`   Vault PDA: ${vaultPda.toString()} (bump: ${bump})`);
    
    // Call withdraw_vault instruction
    console.log('   Calling withdraw_vault instruction...');
    const sig = await program.methods
      .withdrawVault(new BN(vaultBalance))
      .accounts({
        protocolVault: vaultPda,
        authority: keypair.publicKey,
      })
      .rpc();
    
    console.log(`✅ Withdrawal successful!`);
    console.log(`   Signature: ${sig}`);
    console.log(`   Explorer: https://explorer.solana.com/tx/${sig}?cluster=devnet`);
    
    // Verify balance transferred
    const newBalance = await connection.getBalance(OLD_VAULT);
    const authorityBalance = await connection.getBalance(AUTHORITY);
    
    console.log(`\n📊 Final balances:`);
    console.log(`   Vault: ${newBalance / 1e9} SOL`);
    console.log(`   Authority: ${authorityBalance / 1e9} SOL`);
    
    return true;
  } catch (e: any) {
    console.log(`❌ Recovery failed: ${e.message}`);
    if (e.logs) {
      console.log('   Logs:', e.logs);
    }
    return false;
  }
}

async function main() {
  console.log('🔍 SOL Recovery Script\n');
  console.log(`Program ID: ${OLD_PROGRAM_ID.toString()}`);
  console.log(`Vault: ${OLD_VAULT.toString()}`);
  console.log(`Authority: ${AUTHORITY.toString()}\n`);
  
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
  
  // Step 1: Build
  const built = await buildRecoveryProgram();
  if (!built) {
    console.log('\n❌ Build failed. Cannot proceed.');
    return;
  }
  
  // Step 2: Deploy
  const deployed = await deployRecoveryProgram();
  if (!deployed) {
    console.log('\n❌ Deployment failed. Cannot proceed.');
    return;
  }
  
  // Step 3: Recover
  const recovered = await recoverSOL();
  
  if (recovered) {
    console.log('\n🎉 SOL RECOVERY SUCCESSFUL!');
  } else {
    console.log('\n❌ Recovery failed.');
  }
}

main().catch(console.error);

