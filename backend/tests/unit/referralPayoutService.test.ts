import { ReferralPayoutService } from '../../src/services/referralPayout';
import { Connection, Keypair, LAMPORTS_PER_SOL, PublicKey, SystemProgram, Transaction } from '@solana/web3.js';
import bs58 from 'bs58';

// Mock Solana web3.js components
jest.mock('@solana/web3.js', () => ({
  Connection: jest.fn().mockImplementation(() => ({
    getLatestBlockhash: jest.fn().mockResolvedValue({ blockhash: 'mock_blockhash', lastValidBlockHeight: 123 }),
    sendRawTransaction: jest.fn().mockResolvedValue('mock_tx_signature'),
    confirmTransaction: jest.fn().mockResolvedValue({ value: { err: null } }),
  })),
  Keypair: {
    fromSecretKey: jest.fn().mockReturnValue({
      publicKey: new PublicKey('mock_payer_pubkey'),
      secretKey: new Uint8Array(64),
    }),
  },
  PublicKey: jest.fn().mockImplementation((key) => ({
    toBase58: () => key,
  })),
  SystemProgram: {
    transfer: jest.fn(() => ({ type: 'transfer' })),
  },
  Transaction: jest.fn().mockImplementation((options) => ({
    add: jest.fn(() => ({ type: 'transaction' })),
    sign: jest.fn(),
    serialize: jest.fn(() => Buffer.from('mock_serialized_tx')),
  })),
  LAMPORTS_PER_SOL: 1_000_000_000,
}));

describe('ReferralPayoutService', () => {
  let referralPayoutService: ReferralPayoutService;
  const mockRpcUrl = 'http://localhost:8899';
  const mockPayerPrivateKey = bs58.encode(new Uint8Array(Array(64).fill(1))); // A valid-looking base58 string
  const mockReferrerPubkey = 'referrer_wallet_pubkey';
  const mockSolAmount = 0.05;

  beforeEach(() => {
    referralPayoutService = new ReferralPayoutService(mockRpcUrl, mockPayerPrivateKey);
    jest.clearAllMocks();
    // Reset mock implementation for PublicKey to allow new instances
    (PublicKey as jest.Mock).mockImplementation((key) => ({ toBase58: () => key }));
  });

  it('should initialize with a connection and feePayer', () => {
    expect(Connection).toHaveBeenCalledWith(mockRpcUrl, 'confirmed');
    expect(Keypair.fromSecretKey).toHaveBeenCalledWith(bs58.decode(mockPayerPrivateKey));
    expect(referralPayoutService.getConnection()).toBeInstanceOf(Connection);
  });

  describe('sendSol', () => {
    it('should send SOL to the referrer and return transaction details', async () => {
      const result = await referralPayoutService.sendSol(mockReferrerPubkey, mockSolAmount);

      expect(PublicKey).toHaveBeenCalledWith(mockReferrerPubkey);
      expect(SystemProgram.transfer).toHaveBeenCalledWith({
        fromPubkey: new PublicKey('mock_payer_pubkey'),
        toPubkey: new PublicKey(mockReferrerPubkey),
        lamports: Math.round(mockSolAmount * LAMPORTS_PER_SOL),
      });
      expect(referralPayoutService['connection'].getLatestBlockhash).toHaveBeenCalledWith('confirmed');
      expect(Transaction).toHaveBeenCalledWith({
        feePayer: new PublicKey('mock_payer_pubkey'),
        blockhash: 'mock_blockhash',
        lastValidBlockHeight: 123,
      });
      expect(referralPayoutService['feePayer'].publicKey.toBase58()).toBe(
        new PublicKey('mock_payer_pubkey').toBase58()
      ); // Ensure publicKey is used from the mocked Keypair
      expect(referralPayoutService['connection'].sendRawTransaction).toHaveBeenCalled();
      expect(referralPayoutService['connection'].confirmTransaction).toHaveBeenCalled();

      expect(result).toEqual({
        txSig: 'mock_tx_signature',
        lamports: Math.round(mockSolAmount * LAMPORTS_PER_SOL),
      });
    });

    it('should throw an error if transaction confirmation fails', async () => {
      (referralPayoutService['connection'].confirmTransaction as jest.Mock).mockResolvedValueOnce({ value: { err: 'Transaction failed' } });

      await expect(referralPayoutService.sendSol(mockReferrerPubkey, mockSolAmount)).rejects.toThrow(
        'Transaction failed: Transaction failed'
      );
    });
  });
});
