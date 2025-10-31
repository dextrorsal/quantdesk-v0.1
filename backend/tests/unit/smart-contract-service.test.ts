import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SmartContractService } from '../../src/services/smartContractService';
import { Connection, Keypair, PublicKey } from '@solana/web3.js';
import { AnchorProvider, Program } from '@coral-xyz/anchor';

// Mock dependencies
vi.mock('@solana/web3.js');
vi.mock('@coral-xyz/anchor');
vi.mock('fs');

describe('SmartContractService Tests', () => {
  let smartContractService: SmartContractService;
  let mockConnection: any;
  let mockProvider: any;
  let mockProgram: any;
  let mockWallet: any;

  beforeEach(() => {
    vi.clearAllMocks();
    
    // Mock environment variables
    process.env.SOLANA_PRIVATE_KEY = Buffer.from('test-private-key').toString('base64');
    process.env.PROGRAM_ID = 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw';
    process.env.PROGRAM_IDL_PATH = './test-idl.json';

    // Mock Connection
    mockConnection = {
      getVersion: vi.fn().mockResolvedValue({ 'solana-core': '1.16.0' })
    };
    vi.mocked(Connection).mockImplementation(() => mockConnection);

    // Mock Keypair
    mockWallet = {
      publicKey: new PublicKey('11111111111111111111111111111111')
    };
    vi.mocked(Keypair.fromSecretKey).mockReturnValue(mockWallet);

    // Mock AnchorProvider
    mockProvider = {
      sendAndConfirm: vi.fn()
    };
    vi.mocked(AnchorProvider).mockImplementation(() => mockProvider);

    // Mock Program
    mockProgram = {
      programId: new PublicKey('C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw'),
      methods: {
        placeOrder: vi.fn().mockReturnValue({
          accounts: vi.fn().mockReturnValue({
            instruction: vi.fn().mockResolvedValue({})
          })
        }),
        executeConditionalOrder: vi.fn().mockReturnValue({
          accounts: vi.fn().mockReturnValue({
            instruction: vi.fn().mockResolvedValue({})
          })
        }),
        openPosition: vi.fn().mockReturnValue({
          accounts: vi.fn().mockReturnValue({
            instruction: vi.fn().mockResolvedValue({})
          })
        })
      }
    };

    // Mock fs
    const mockFs = {
      existsSync: vi.fn().mockReturnValue(true),
      readFileSync: vi.fn().mockReturnValue(JSON.stringify({
        name: 'quantdesk_perp_dex',
        instructions: []
      }))
    };
    vi.doMock('fs', () => mockFs);

    smartContractService = SmartContractService.getInstance();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('executeOrder', () => {
    it('should execute order successfully when program is initialized', async () => {
      // Mock program initialization
      (smartContractService as any).program = mockProgram;
      (smartContractService as any).wallet = mockWallet;

      // Mock PDA derivation
      const mockMarketPda = new PublicKey('22222222222222222222222222222222');
      const mockUserPda = new PublicKey('33333333333333333333333333333333');
      const mockOrderPda = new PublicKey('44444444444444444444444444444444');

      vi.spyOn(smartContractService as any, 'getMarketPda').mockResolvedValue(mockMarketPda);
      vi.spyOn(smartContractService as any, 'getUserPda').mockResolvedValue(mockUserPda);
      vi.spyOn(smartContractService as any, 'getOrderPda').mockResolvedValue(mockOrderPda);

      // Mock transaction confirmation
      mockProvider.sendAndConfirm.mockResolvedValue('tx-signature-123');

      const params = {
        orderId: 'order-123',
        userId: 'user-123',
        marketSymbol: 'BTC/USD',
        side: 'long' as const,
        size: 0.001,
        price: 50000,
        leverage: 1,
        orderType: 'market' as const
      };

      const result = await smartContractService.executeOrder(params);

      expect(result).toEqual({
        success: true,
        transactionSignature: 'tx-signature-123',
        positionId: mockOrderPda.toString()
      });

      expect(mockProvider.sendAndConfirm).toHaveBeenCalled();
    });

    it('should use mock execution when program is not initialized', async () => {
      // Don't set program (simulate mock mode)
      (smartContractService as any).program = null;

      const params = {
        orderId: 'order-123',
        userId: 'user-123',
        marketSymbol: 'BTC/USD',
        side: 'long' as const,
        size: 0.001,
        price: 50000,
        leverage: 1,
        orderType: 'market' as const
      };

      const result = await smartContractService.executeOrder(params);

      expect(result.success).toBe(true);
      expect(result.transactionSignature).toMatch(/^tx_\d+_[a-z0-9]+$/);
      expect(result.positionId).toMatch(/^pos_\d+_[a-z0-9]+$/);
    });

    it('should handle execution errors', async () => {
      (smartContractService as any).program = mockProgram;
      (smartContractService as any).wallet = mockWallet;

      // Mock PDA derivation to throw error
      vi.spyOn(smartContractService as any, 'getMarketPda').mockRejectedValue(new Error('PDA derivation failed'));

      const params = {
        orderId: 'order-123',
        userId: 'user-123',
        marketSymbol: 'BTC/USD',
        side: 'long' as const,
        size: 0.001,
        price: 50000,
        leverage: 1,
        orderType: 'market' as const
      };

      const result = await smartContractService.executeOrder(params);

      expect(result).toEqual({
        success: false,
        error: 'PDA derivation failed'
      });
    });
  });

  describe('createPosition', () => {
    it('should create position successfully when program is initialized', async () => {
      (smartContractService as any).program = mockProgram;
      (smartContractService as any).wallet = mockWallet;

      // Mock PDA derivation
      const mockMarketPda = new PublicKey('22222222222222222222222222222222');
      const mockUserPda = new PublicKey('33333333333333333333333333333333');
      const mockPositionPda = new PublicKey('55555555555555555555555555555555');

      vi.spyOn(smartContractService as any, 'getMarketPda').mockResolvedValue(mockMarketPda);
      vi.spyOn(smartContractService as any, 'getUserPda').mockResolvedValue(mockUserPda);
      vi.spyOn(smartContractService as any, 'getPositionPda').mockResolvedValue(mockPositionPda);

      // Mock transaction confirmation
      mockProvider.sendAndConfirm.mockResolvedValue('tx-signature-456');

      const result = await smartContractService.createPosition(
        'user-123',
        'BTC/USD',
        'long',
        0.001,
        50000,
        1
      );

      expect(result).toEqual({
        success: true,
        positionId: mockPositionPda.toString()
      });

      expect(mockProvider.sendAndConfirm).toHaveBeenCalled();
    });

    it('should use mock execution when program is not initialized', async () => {
      (smartContractService as any).program = null;

      const result = await smartContractService.createPosition(
        'user-123',
        'BTC/USD',
        'long',
        0.001,
        50000,
        1
      );

      expect(result.success).toBe(true);
      expect(result.positionId).toMatch(/^pos_\d+_[a-z0-9]+$/);
    });

    it('should handle position creation errors', async () => {
      (smartContractService as any).program = mockProgram;
      (smartContractService as any).wallet = mockWallet;

      // Mock PDA derivation to throw error
      vi.spyOn(smartContractService as any, 'getMarketPda').mockRejectedValue(new Error('Market not found'));

      const result = await smartContractService.createPosition(
        'user-123',
        'BTC/USD',
        'long',
        0.001,
        50000,
        1
      );

      expect(result).toEqual({
        success: false,
        error: 'Market not found'
      });
    });
  });

  describe('PDA derivation', () => {
    it('should derive market PDA correctly', async () => {
      (smartContractService as any).program = mockProgram;

      const mockPda = new PublicKey('22222222222222222222222222222222');
      const mockBump = 255;

      // Mock PublicKey.findProgramAddress
      const mockFindProgramAddress = vi.fn().mockResolvedValue([mockPda, mockBump]);
      vi.mocked(PublicKey.findProgramAddress).mockImplementation(mockFindProgramAddress);

      const result = await (smartContractService as any).getMarketPda('BTC/USD');

      expect(result).toBe(mockPda);
      expect(mockFindProgramAddress).toHaveBeenCalledWith(
        [Buffer.from('market'), Buffer.from('BTC'), Buffer.from('USD')],
        mockProgram.programId
      );
    });

    it('should derive user PDA correctly', async () => {
      (smartContractService as any).program = mockProgram;

      const mockPda = new PublicKey('33333333333333333333333333333333');
      const mockBump = 255;

      const mockFindProgramAddress = vi.fn().mockResolvedValue([mockPda, mockBump]);
      vi.mocked(PublicKey.findProgramAddress).mockImplementation(mockFindProgramAddress);

      const result = await (smartContractService as any).getUserPda('user-123');

      expect(result).toBe(mockPda);
      expect(mockFindProgramAddress).toHaveBeenCalledWith(
        [Buffer.from('user_account'), expect.any(Buffer), Buffer.from([0, 0])],
        mockProgram.programId
      );
    });

    it('should derive position PDA correctly', async () => {
      (smartContractService as any).program = mockProgram;

      const mockMarketPda = new PublicKey('22222222222222222222222222222222');
      const mockPositionPda = new PublicKey('55555555555555555555555555555555');
      const mockBump = 255;

      vi.spyOn(smartContractService as any, 'getMarketPda').mockResolvedValue(mockMarketPda);

      const mockFindProgramAddress = vi.fn().mockResolvedValue([mockPositionPda, mockBump]);
      vi.mocked(PublicKey.findProgramAddress).mockImplementation(mockFindProgramAddress);

      const result = await (smartContractService as any).getPositionPda('user-123', 'BTC/USD', 'long');

      expect(result).toBe(mockPositionPda);
      expect(mockFindProgramAddress).toHaveBeenCalledWith(
        [Buffer.from('position'), expect.any(Buffer), mockMarketPda.toBuffer()],
        mockProgram.programId
      );
    });
  });

  describe('health check', () => {
    it('should return true when connection is healthy', async () => {
      mockConnection.getVersion.mockResolvedValue({ 'solana-core': '1.16.0' });

      const result = await smartContractService.healthCheck();

      expect(result).toBe(true);
    });

    it('should return false when connection fails', async () => {
      mockConnection.getVersion.mockRejectedValue(new Error('Connection failed'));

      const result = await smartContractService.healthCheck();

      expect(result).toBe(false);
    });
  });
});
