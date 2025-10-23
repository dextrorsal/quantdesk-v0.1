import { Connection, Keypair, LAMPORTS_PER_SOL, PublicKey, SystemProgram, Transaction } from '@solana/web3.js';
import bs58 from 'bs58';

export interface PayoutResult {
  txSig: string;
  lamports: number;
}

export class ReferralPayoutService {
  private connection: Connection;
  private feePayer: Keypair;

  constructor(rpcUrl: string, payerPrivateKeyBase58: string) {
    this.connection = new Connection(rpcUrl, 'confirmed');
    this.feePayer = Keypair.fromSecretKey(bs58.decode(payerPrivateKeyBase58));
  }

  async sendSol(toPubkey: string, solAmount: number): Promise<PayoutResult> {
    const dest = new PublicKey(toPubkey);
    const lamports = Math.round(solAmount * LAMPORTS_PER_SOL);
    const ix = SystemProgram.transfer({ fromPubkey: this.feePayer.publicKey, toPubkey: dest, lamports });
    const { blockhash, lastValidBlockHeight } = await this.connection.getLatestBlockhash('confirmed');
    const tx = new Transaction({ feePayer: this.feePayer.publicKey, blockhash, lastValidBlockHeight }).add(ix);
    tx.sign(this.feePayer);
    const sig = await this.connection.sendRawTransaction(tx.serialize(), { skipPreflight: false });
    await this.connection.confirmTransaction({ signature: sig, blockhash, lastValidBlockHeight }, 'confirmed');
    return { txSig: sig, lamports };
  }

  getConnection(): Connection {
    return this.connection;
  }
}


