/**
 * Standalone test suite for deposit functionality
 * Run this to debug AccountNotSigner issues
 * 
 * Usage:
 *   import { testDeposit } from './test-deposit';
 *   testDeposit();
 */

import { Connection, PublicKey, SystemProgram, Transaction } from '@solana/web3.js';
import { AnchorProvider, Program, Wallet } from '@coral-xyz/anchor';
import { BN } from '@coral-xyz/anchor';

// Import IDL - adjust path as needed
// import programIdl from '../contracts/target/idl/quantdesk_perp_dex.json';

const RPC_URL = 'https://api.devnet.solana.com';

/**
 * Test deposit with wallet adapter simulation
 */
export async function testDepositWithWallet(walletAdapter: any) {
  console.log('🧪 TEST: Starting deposit test with wallet adapter...');
  
  const connection = new Connection(RPC_URL, 'confirmed');
  const programId = new PublicKey(programIdl.address);
  
  // Create Anchor wallet from adapter
  const anchorWallet: Wallet = {
    publicKey: walletAdapter.publicKey,
    signTransaction: async (tx: Transaction) => {
      console.log('📝 TEST: Wallet adapter signing transaction...');
      const signed = await walletAdapter.signTransaction(tx);
      console.log('✅ TEST: Transaction signed:', {
        signaturesCount: signed.signatures.length,
        feePayer: signed.feePayer?.toBase58(),
        firstSignaturePresent: !!signed.signatures[0]?.signature,
      });
      return signed;
    },
    signAllTransactions: async (txs: Transaction[]) => {
      const signed = [];
      for (const tx of txs) {
        signed.push(await walletAdapter.signTransaction(tx));
      }
      return signed;
    },
  };
  
  const provider = new AnchorProvider(connection, anchorWallet, {
    commitment: 'confirmed',
    preflightCommitment: 'confirmed',
  });
  
  const program = new Program(programIdl as any, provider);
  
  // Derive PDAs
  const accountIndex = 0;
  const accountIndexBuffer = Buffer.alloc(2);
  accountIndexBuffer.writeUInt16LE(accountIndex, 0);
  
  const [userAccountPda] = PublicKey.findProgramAddressSync(
    [
      Buffer.from('user_account'),
      walletAdapter.publicKey.toBuffer(),
      accountIndexBuffer,
    ],
    programId
  );
  
  const [solCollateralAccount] = PublicKey.findProgramAddressSync(
    [Buffer.from('collateral'), walletAdapter.publicKey.toBuffer(), Buffer.from('SOL')],
    programId
  );
  
  const [protocolSOLVault] = PublicKey.findProgramAddressSync(
    [Buffer.from('protocol_sol_vault')],
    programId
  );
  
  const SOL_USD_PRICE_FEED = new PublicKey('H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG');
  
  console.log('📋 TEST: Account PDAs:', {
    userAccount: userAccountPda.toBase58(),
    collateralAccount: solCollateralAccount.toBase58(),
    protocolVault: protocolSOLVault.toBase58(),
    user: walletAdapter.publicKey.toBase58(),
  });
  
  // Test 1: Check account metas
  console.log('\n🔍 TEST 1: Checking instruction account structure...');
  const testIx = await program.methods
    .depositNativeSol(new BN(10000000)) // 0.01 SOL
    .accounts({
      userAccount: userAccountPda,
      collateralAccount: solCollateralAccount,
      protocolVault: protocolSOLVault,
      user: walletAdapter.publicKey,
      solUsdPriceFeed: SOL_USD_PRICE_FEED,
      systemProgram: SystemProgram.programId,
    })
    .instruction();
  
  console.log('✅ TEST 1: Instruction created');
  console.log('📋 Account metas:', testIx.keys.map((k, i) => ({
    index: i,
    pubkey: k.pubkey.toBase58(),
    isSigner: k.isSigner,
    isWritable: k.isWritable,
  })));
  
  const userIndex = testIx.keys.findIndex(k => k.pubkey.equals(walletAdapter.publicKey));
  console.log(`📍 User account index: ${userIndex} (isSigner: ${testIx.keys[userIndex]?.isSigner})`);
  
  if (!testIx.keys[userIndex]?.isSigner) {
    throw new Error('❌ TEST FAILED: User is NOT marked as signer in instruction!');
  }
  
  // Test 2: Try Anchor .rpc()
  console.log('\n🔍 TEST 2: Attempting deposit via Anchor .rpc()...');
  try {
    const signature = await program.methods
      .depositNativeSol(new BN(10000000))
      .accounts({
        userAccount: userAccountPda,
        collateralAccount: solCollateralAccount,
        protocolVault: protocolSOLVault,
        user: walletAdapter.publicKey,
        solUsdPriceFeed: SOL_USD_PRICE_FEED,
        systemProgram: SystemProgram.programId,
      })
      .rpc({
        commitment: 'confirmed',
        skipPreflight: false,
      });
    
    console.log('✅ TEST 2 PASSED: Deposit successful!');
    console.log('📤 Signature:', signature);
    return signature;
  } catch (error: any) {
    console.error('❌ TEST 2 FAILED:', error);
    console.error('📋 Error details:', {
      message: error?.message,
      code: error?.code,
      errorCode: error?.error?.errorCode,
      logs: error?.logs,
    });
    
    // Test 3: Manual transaction building to see what's wrong
    console.log('\n🔍 TEST 3: Building transaction manually to inspect...');
    const tx = new Transaction();
    tx.add(testIx);
    
    const { blockhash } = await connection.getLatestBlockhash('confirmed');
    tx.recentBlockhash = blockhash;
    tx.feePayer = walletAdapter.publicKey;
    
    console.log('📋 Transaction before signing:', {
      feePayer: tx.feePayer.toBase58(),
      instructionsCount: tx.instructions.length,
      signaturesCount: tx.signatures.length,
    });
    
    // Test signing
    console.log('\n🔍 TEST 4: Testing wallet adapter signing...');
    const signedTx = await walletAdapter.signTransaction(tx);
    console.log('✅ TEST 4: Transaction signed');
    console.log('📋 Signed transaction:', {
      signaturesCount: signedTx.signatures.length,
      feePayerSigned: signedTx.signatures[0]?.publicKey.equals(walletAdapter.publicKey),
      signaturePresent: !!signedTx.signatures[0]?.signature,
      signatureLength: signedTx.signatures[0]?.signature?.length,
    });
    
    // Test simulation
    console.log('\n🔍 TEST 5: Simulating signed transaction...');
    try {
      const simulation = await connection.simulateTransaction(signedTx, {
        commitment: 'confirmed',
        replaceRecentBlockhash: true,
      });
      
      if (simulation.value.err) {
        console.error('❌ TEST 5 FAILED: Simulation error:', simulation.value.err);
        console.error('📋 Simulation logs:', simulation.value.logs);
      } else {
        console.log('✅ TEST 5 PASSED: Simulation successful');
        console.log('📊 Units consumed:', simulation.value.unitsConsumed);
      }
    } catch (simError: any) {
      console.error('❌ TEST 5 ERROR:', simError.message);
    }
    
    throw error;
  }
}

/**
 * Full test suite entry point
 */
export async function runDepositTestSuite(walletAdapter: any) {
  console.log('═══════════════════════════════════════════════════════════');
  console.log('🧪 DEPOSIT TEST SUITE');
  console.log('═══════════════════════════════════════════════════════════');
  
  try {
    await testDepositWithWallet(walletAdapter);
    console.log('\n✅ ALL TESTS PASSED!');
  } catch (error: any) {
    console.log('\n❌ TEST SUITE FAILED');
    console.error('Final error:', error);
    throw error;
  }
}

