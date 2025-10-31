// Simple test to verify wallet can sign transactions
import { Connection, PublicKey, SystemProgram, Transaction } from '@solana/web3.js';

export async function testWalletSigning(wallet: any) {
  console.log('🧪 Testing wallet signing capability...');
  
  if (!wallet?.adapter?.publicKey) {
    throw new Error('Wallet not connected');
  }

  const connection = new Connection('https://api.devnet.solana.com', 'confirmed');
  
  try {
    // Create a simple SOL transfer to self (no actual transfer, just test signing)
    const transaction = new Transaction().add(
      SystemProgram.transfer({
        fromPubkey: wallet.adapter.publicKey,
        toPubkey: wallet.adapter.publicKey,
        lamports: 1, // 1 lamport to self
      })
    );

    transaction.feePayer = wallet.adapter.publicKey;
    transaction.recentBlockhash = (await connection.getLatestBlockhash()).blockhash;

    console.log('🔐 Transaction created, requesting signature...');
    const signed = await wallet.adapter.signTransaction(transaction);
    
    console.log('✅ Wallet CAN sign transactions!');
    console.log('Signature:', signed.signatures[0]?.signature?.toString('hex').substring(0, 20) + '...');
    
    return true;
  } catch (error: any) {
    console.error('❌ Wallet signing test FAILED:', error.message);
    return false;
  }
}
