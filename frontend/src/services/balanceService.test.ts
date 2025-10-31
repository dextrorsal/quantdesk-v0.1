import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { PublicKey } from '@solana/web3.js'
import { balanceService } from './balanceService'
import { smartContractService } from './smartContractService'

// Mock the dependencies
vi.mock('@solana/web3.js', () => ({
  Connection: vi.fn().mockImplementation(() => ({
    getBalance: vi.fn(),
    getAccountInfo: vi.fn(),
    getTokenAccountBalance: vi.fn(),
    getParsedTokenAccountsByOwner: vi.fn(),
  })),
  PublicKey: Object.assign(vi.fn().mockImplementation((key) => ({ 
    toString: () => key,
    toBuffer: () => Buffer.from(key),
  })), {
    findProgramAddress: vi.fn().mockImplementation(async (seeds, programId) => {
      // Create a deterministic PDA based on seeds and programId
      const seedString = seeds.map(seed => seed.toString()).join('');
      const programIdString = programId.toString();
      const combined = seedString + programIdString;
      
      // Create a mock PDA that's deterministic
      const mockPDA = {
        toString: () => `mock-pda-${combined.slice(0, 8)}`,
        toBuffer: () => Buffer.from(combined.slice(0, 32).padEnd(32, '0')),
      };
      
      // Return as a proper array for destructuring
      return [mockPDA, 255];
    }),
    findProgramAddressSync: vi.fn().mockImplementation((seeds, programId) => {
      // Create a deterministic PDA based on seeds and programId
      const seedString = seeds.map(seed => seed.toString()).join('');
      const programIdString = programId.toString();
      const combined = seedString + programIdString;
      
      // Create a mock PDA that's deterministic
      const mockPDA = {
        toString: () => `mock-pda-${combined.slice(0, 8)}`,
        toBuffer: () => Buffer.from(combined.slice(0, 32).padEnd(32, '0')),
      };
      
      // Return as a proper array for destructuring
      return [mockPDA, 255];
    }),
  }),
}))

vi.mock('@solana/spl-token', () => ({
  TOKEN_PROGRAM_ID: {
    toBuffer: () => Buffer.from('token-program-id'),
  },
  ASSOCIATED_TOKEN_PROGRAM_ID: {
    toBuffer: () => Buffer.from('associated-token-program-id'),
  },
}))

vi.mock('../config/tokens', () => ({
  getDepositTokens: vi.fn().mockReturnValue([
    { symbol: 'SOL', name: 'Solana', mintAddress: 'So11111111111111111111111111111111111111112' },
    { symbol: 'USDC', name: 'USD Coin', mintAddress: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v' },
  ]),
}))

vi.mock('../services/smartContractService', () => ({
  smartContractService: {
    getSOLCollateralBalance: vi.fn(),
  },
}))

// Mock fetch globally
global.fetch = vi.fn()

// Mock the token config
vi.mock('../config/tokens', () => ({
  getDepositTokens: vi.fn(() => [
    {
      mintAddress: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
      symbol: 'USDC',
      name: 'USD Coin',
      decimals: 6,
    },
    {
      mintAddress: 'So11111111111111111111111111111111111111112',
      symbol: 'SOL',
      name: 'Solana',
      decimals: 9,
    },
  ]),
}))

describe('BalanceService Price Integration', () => {
  let mockWalletAddress: PublicKey

  beforeEach(() => {
    mockWalletAddress = new PublicKey('test-wallet-address')
    // Override toString to return the actual string
    mockWalletAddress.toString = () => 'test-wallet-address'
    // Add toBuffer method
    mockWalletAddress.toBuffer = () => Buffer.from('test-wallet-address')
    
    // Clear all mocks
    vi.clearAllMocks()
    
    // Reset fetch mock
    global.fetch = vi.fn()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('getRealSOLPrice', () => {
    it('should return real SOL price from oracle API', async () => {
      const mockOracleResponse = {
        SOL: 150.25,
        BTC: 45000,
        ETH: 2800,
      }

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockOracleResponse),
      })

      // Access private method for testing
      const solPrice = await (balanceService as any).getRealSOLPrice()
      
      expect(solPrice).toBe(150.25)
      expect(global.fetch).toHaveBeenCalledWith('/api/oracle/prices')
    })

    it('should return fallback price when oracle API fails', async () => {
      global.fetch = vi.fn().mockRejectedValueOnce(new Error('Network error'))

      const solPrice = await (balanceService as any).getRealSOLPrice()
      
      expect(solPrice).toBe(100) // Fallback price
    })

    it('should return fallback price when oracle response is invalid', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ invalid: 'data' }),
      })

      const solPrice = await (balanceService as any).getRealSOLPrice()
      
      expect(solPrice).toBe(100) // Fallback price
    })

    it('should handle different SOL price field names', async () => {
      const testCases = [
        { sol: 120.50 },
        { 'SOL/USD': 130.75 },
        { SOL: 140.00 },
      ]

      for (const testCase of testCases) {
        global.fetch = vi.fn().mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(testCase),
        })

        const solPrice = await (balanceService as any).getRealSOLPrice()
        expect(solPrice).toBe(Object.values(testCase)[0])
      }
    })
  })

  describe('getRealTokenPrice', () => {
    it('should return real token price from oracle API', async () => {
      const mockOracleResponse = {
        BTC: 45000,
        ETH: 2800,
        USDC: 1.0,
      }

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockOracleResponse),
      })

      const btcPrice = await (balanceService as any).getRealTokenPrice('BTC')
      const ethPrice = await (balanceService as any).getRealTokenPrice('ETH')
      const usdcPrice = await (balanceService as any).getRealTokenPrice('USDC')
      
      expect(btcPrice).toBe(45000)
      expect(ethPrice).toBe(2800)
      expect(usdcPrice).toBe(1.0)
    })

    it('should return null when token price not found', async () => {
      const mockOracleResponse = {
        BTC: 45000,
        ETH: 2800,
      }

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockOracleResponse),
      })

      const unknownPrice = await (balanceService as any).getRealTokenPrice('UNKNOWN')
      
      expect(unknownPrice).toBeNull()
    })

    it('should return null when oracle API fails', async () => {
      global.fetch = vi.fn().mockRejectedValueOnce(new Error('Network error'))

      const tokenPrice = await (balanceService as any).getRealTokenPrice('BTC')
      
      expect(tokenPrice).toBeNull()
    })

    it('should handle case-insensitive token symbols', async () => {
      const mockOracleResponse = {
        btc: 45000,
        ETH: 2800,
        usdc: 1.0, // Changed to lowercase to match the test
      }

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockOracleResponse),
      })

      const btcPrice = await (balanceService as any).getRealTokenPrice('BTC')
      const ethPrice = await (balanceService as any).getRealTokenPrice('eth')
      const usdcPrice = await (balanceService as any).getRealTokenPrice('USDC')
      
      expect(btcPrice).toBe(45000)
      expect(ethPrice).toBe(2800)
      expect(usdcPrice).toBe(1.0)
    })
  })

  describe('getMockTokenPrice', () => {
    it('should return correct mock prices for known tokens', () => {
      const mockPrice = (balanceService as any).getMockTokenPrice('SOL')
      expect(mockPrice).toBe(100)

      const btcPrice = (balanceService as any).getMockTokenPrice('BTC')
      expect(btcPrice).toBe(50000)

      const ethPrice = (balanceService as any).getMockTokenPrice('ETH')
      expect(ethPrice).toBe(3000)

      const usdcPrice = (balanceService as any).getMockTokenPrice('USDC')
      expect(usdcPrice).toBe(1)
    })

    it('should return default price for unknown tokens', () => {
      const unknownPrice = (balanceService as any).getMockTokenPrice('UNKNOWN')
      expect(unknownPrice).toBe(0) // Default fallback
    })
  })

  describe('getUserBalances Integration', () => {
    beforeEach(() => {
      // Mock smart contract service
      vi.mocked(smartContractService.getSOLCollateralBalance).mockResolvedValue(5.5)
      
      // Mock oracle API response
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          SOL: 150.25,
          BTC: 45000,
          ETH: 2800,
        }),
      })
    })

    it('should use real SOL price and collateral balance for calculations', async () => {
      // Mock connection methods
      const mockConnection = {
        getBalance: vi.fn().mockResolvedValue(1000000000), // 1 SOL in lamports
        getAccountInfo: vi.fn().mockResolvedValue(null), // No token accounts
        getTokenAccountBalance: vi.fn(),
      }
      
      vi.mocked(balanceService as any).connection = mockConnection

      const balances = await balanceService.getUserBalances(mockWalletAddress)
      
      expect(balances.nativeSOL).toBe(5.5) // Collateral balance, not wallet balance
      expect(balances.totalValueUSD).toBe(5.5 * 150.25) // 5.5 SOL * $150.25
      expect(smartContractService.getSOLCollateralBalance).toHaveBeenCalledWith('test-wallet-address')
    })

    it('should fallback to mock prices when oracle fails', async () => {
      // Mock oracle failure
      global.fetch = vi.fn().mockRejectedValue(new Error('Oracle unavailable'))
      
      // Mock connection methods
      const mockConnection = {
        getBalance: vi.fn().mockResolvedValue(1000000000),
        getAccountInfo: vi.fn().mockResolvedValue(null),
        getTokenAccountBalance: vi.fn(),
      }
      
      vi.mocked(balanceService as any).connection = mockConnection

      const balances = await balanceService.getUserBalances(mockWalletAddress)
      
      expect(balances.nativeSOL).toBe(5.5)
      expect(balances.totalValueUSD).toBe(5.5 * 100) // 5.5 SOL * $100 (fallback price)
    })

    it('should handle errors gracefully and return zero balances', async () => {
      // Mock smart contract service to throw error
      vi.mocked(smartContractService.getSOLCollateralBalance).mockRejectedValue(new Error('Smart contract error'))

      const balances = await balanceService.getUserBalances(mockWalletAddress)
      
      expect(balances.nativeSOL).toBe(0)
      expect(balances.tokens).toEqual([])
      expect(balances.totalValueUSD).toBe(0)
    })

    it('should calculate total USD value with mixed real and mock prices', async () => {
      // Mock oracle with partial data
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          SOL: 150.25,
          BTC: 45000,
          // ETH not provided, should fallback to mock
        }),
      })

      // Mock token balances
      const mockTokens = [
        { symbol: 'BTC', uiAmount: 0.1 }, // Real price: $45000
        { symbol: 'ETH', uiAmount: 2.0 },  // Mock price: $3000
      ]

      // Mock getAllTokenBalances to return test tokens
      vi.spyOn(balanceService as any, 'getAllTokenBalances').mockResolvedValue(mockTokens)

      const balances = await balanceService.getUserBalances(mockWalletAddress)
      
      const expectedTotal = (5.5 * 150.25) + (0.1 * 45000) + (2.0 * 3000)
      expect(balances.totalValueUSD).toBe(expectedTotal)
    })
  })

  describe('getNativeSOLBalance', () => {
    it('should return SOL balance in SOL units (not lamports)', async () => {
      const mockConnection = {
        getBalance: vi.fn().mockResolvedValue(2500000000), // 2.5 SOL in lamports
      }
      
      vi.mocked(balanceService as any).connection = mockConnection

      const balance = await balanceService.getNativeSOLBalance(mockWalletAddress)
      
      expect(balance).toBe(2.5) // Should be in SOL, not lamports
      expect(mockConnection.getBalance).toHaveBeenCalledWith(mockWalletAddress)
    })

    it('should return 0 when connection fails', async () => {
      const mockConnection = {
        getBalance: vi.fn().mockRejectedValue(new Error('Connection error')),
      }
      
      vi.mocked(balanceService as any).connection = mockConnection

      const balance = await balanceService.getNativeSOLBalance(mockWalletAddress)
      
      expect(balance).toBe(0)
    })
  })

  describe('getTokenBalance', () => {
    it('should return null when token account does not exist', async () => {
      const mockConnection = {
        getAccountInfo: vi.fn().mockResolvedValue(null), // No account
        getTokenAccountBalance: vi.fn(),
      }
      
      vi.mocked(balanceService as any).connection = mockConnection

      const mintAddress = new PublicKey('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v')
      // Add toBuffer method to mint address
      mintAddress.toBuffer = () => Buffer.from('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v')
      
      const balance = await balanceService.getTokenBalance(mockWalletAddress, mintAddress)
      
      expect(balance).toBeNull()
    })

    it('should return token balance when account exists', async () => {
      // Mock the connection with proper methods
      const mockConnection = {
        getAccountInfo: vi.fn().mockResolvedValue({ data: Buffer.alloc(0) }), // Account exists
        getTokenAccountBalance: vi.fn().mockResolvedValue({
          value: {
            amount: '1000000',
            decimals: 6,
            uiAmount: 1.0,
          },
        }),
      }
      
      // Set the connection on the service instance
      vi.mocked(balanceService as any).connection = mockConnection

      // Create a simple mint address mock
      const mintAddress = {
        toString: () => 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
        toBuffer: () => Buffer.from('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'),
      }
      
      // Mock the getAssociatedTokenAddress method directly to avoid PDA issues
      const mockTokenAccount = {
        toString: () => 'mock-token-account-address',
        toBuffer: () => Buffer.from('mock-token-account-address'),
      }
      
      vi.spyOn(balanceService as any, 'getAssociatedTokenAddress').mockResolvedValue(mockTokenAccount)
      
      const balance = await balanceService.getTokenBalance(mockWalletAddress, mintAddress)
      
      expect(balance).toEqual({
        mint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
        symbol: 'USDC',
        name: 'USD Coin',
        balance: '1000000',
        decimals: 6,
        uiAmount: 1.0,
        tokenAccount: 'mock-token-account-address',
      })
    })
  })
})
