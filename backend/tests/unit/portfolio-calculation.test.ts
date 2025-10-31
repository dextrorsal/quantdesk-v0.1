/**
 * Test 1.1-UNIT-002: Portfolio calculation service with current prices
 * 
 * Tests the portfolio calculation logic including P&L calculations,
 * margin calculations, and health factor computations.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { PortfolioCalculationService } from '../../src/services/portfolioCalculationService'
import { PnLCalculationService } from '../../src/services/pnlCalculationService'
import { PythOracleService } from '../../src/services/pythOracleService'

// Mock dependencies
vi.mock('../../src/services/pnlCalculationService')
vi.mock('../../src/services/pythOracleService')
vi.mock('../../src/services/supabaseDatabase')

describe('1.1-UNIT-002: Portfolio Calculation Service', () => {
  let portfolioService: PortfolioCalculationService
  let mockPnLService: any
  let mockOracleService: any
  let mockDatabase: any

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks()

    // Mock PnL Calculation Service
    mockPnLService = {
      calculatePositionPnL: vi.fn(),
      calculatePortfolioPnL: vi.fn(),
    }

    // Mock Oracle Service
    mockOracleService = {
      getAllPrices: vi.fn(),
      getPrice: vi.fn(),
    }

    // Mock Database Service
    mockDatabase = {
      getUserByWallet: vi.fn(),
      getUserPositions: vi.fn(),
      getUserCollateral: vi.fn(),
    }

    // Create service instance with mocked dependencies
    portfolioService = new PortfolioCalculationService(
      mockDatabase,
      mockPnLService,
      mockOracleService
    )
  })

  describe('calculatePortfolio', () => {
    it('should calculate portfolio with current prices', async () => {
      const userId = 'test-user-123'
      const mockUser = { id: userId, wallet_address: 'test-wallet' }
      
      const mockPositions = [
        {
          id: 'pos-1',
          symbol: 'SOL',
          side: 'long',
          size: 10,
          entry_price: 100,
          current_price: 110,
          margin: 1000,
          leverage: 10,
          status: 'open'
        },
        {
          id: 'pos-2', 
          symbol: 'BTC',
          side: 'short',
          size: 0.5,
          entry_price: 50000,
          current_price: 48000,
          margin: 2500,
          leverage: 10,
          status: 'open'
        }
      ]

      const mockCollateral = [
        { amount: 5000, asset: 'USDC' }
      ]

      const mockPrices = {
        'SOL': 110,
        'BTC': 48000,
        'USDC': 1
      }

      // Setup mocks
      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue(mockPositions)
      mockDatabase.getUserCollateral.mockResolvedValue(mockCollateral)
      mockOracleService.getAllPrices.mockResolvedValue(mockPrices)
      
      mockPnLService.calculatePositionPnL
        .mockReturnValueOnce({ unrealizedPnl: 100, unrealizedPnlPercent: 10 }) // SOL position
        .mockReturnValueOnce({ unrealizedPnl: 1000, unrealizedPnlPercent: 40 }) // BTC position

      // Execute
      const result = await portfolioService.calculatePortfolio(userId)

      // Verify
      expect(result).toBeDefined()
      expect(result?.userId).toBe(userId)
      expect(result?.positions).toHaveLength(2)
      expect(result?.totalValue).toBeGreaterThan(0)
      expect(result?.totalUnrealizedPnl).toBe(1100) // 100 + 1000
    })

    it('should handle empty portfolio', async () => {
      const userId = 'test-user-empty'
      const mockUser = { id: userId, wallet_address: 'test-wallet' }

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockOracleService.getAllPrices.mockResolvedValue({})

      const result = await portfolioService.calculatePortfolio(userId)

      expect(result).toBeDefined()
      expect(result?.positions).toHaveLength(0)
      expect(result?.totalValue).toBe(0)
      expect(result?.totalUnrealizedPnl).toBe(0)
    })

    it('should handle missing user', async () => {
      const userId = 'non-existent-user'

      mockDatabase.getUserByWallet.mockResolvedValue(null)

      const result = await portfolioService.calculatePortfolio(userId)

      expect(result).toBeNull()
    })

    it('should calculate health factor correctly', async () => {
      const userId = 'test-user-health'
      const mockUser = { id: userId, wallet_address: 'test-wallet' }
      
      const mockPositions = [
        {
          id: 'pos-1',
          symbol: 'SOL',
          side: 'long',
          size: 10,
          entry_price: 100,
          current_price: 90, // Loss
          margin: 1000,
          leverage: 10,
          status: 'open'
        }
      ]

      const mockCollateral = [
        { amount: 2000, asset: 'USDC' }
      ]

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue(mockPositions)
      mockDatabase.getUserCollateral.mockResolvedValue(mockCollateral)
      mockOracleService.getAllPrices.mockResolvedValue({ 'SOL': 90, 'USDC': 1 })
      
      mockPnLService.calculatePositionPnL.mockReturnValue({
        unrealizedPnl: -100,
        unrealizedPnlPercent: -10
      })

      const result = await portfolioService.calculatePortfolio(userId)

      expect(result).toBeDefined()
      expect(result?.healthFactor).toBeDefined()
      expect(result?.marginRatio).toBeDefined()
      
      // Health factor should be calculated based on collateral vs losses
      expect(result?.healthFactor).toBeGreaterThan(0)
    })
  })

  describe('updatePositionsWithCurrentPrices', () => {
    it('should update positions with current market prices', async () => {
      const positions = [
        { symbol: 'SOL', entry_price: 100 },
        { symbol: 'BTC', entry_price: 50000 }
      ]

      const mockPrices = {
        'SOL': 110,
        'BTC': 48000
      }

      mockOracleService.getAllPrices.mockResolvedValue(mockPrices)

      const result = await portfolioService.updatePositionsWithCurrentPrices(positions)

      expect(result).toHaveLength(2)
      expect(result[0].currentPrice).toBe(110)
      expect(result[1].currentPrice).toBe(48000)
    })

    it('should handle missing prices gracefully', async () => {
      const positions = [
        { symbol: 'SOL', entry_price: 100 },
        { symbol: 'UNKNOWN', entry_price: 50 }
      ]

      const mockPrices = {
        'SOL': 110
        // UNKNOWN price missing
      }

      mockOracleService.getAllPrices.mockResolvedValue(mockPrices)

      const result = await portfolioService.updatePositionsWithCurrentPrices(positions)

      expect(result).toHaveLength(2)
      expect(result[0].currentPrice).toBe(110)
      expect(result[1].currentPrice).toBe(50) // Should fallback to entry price
    })
  })

  describe('caching', () => {
    it('should cache portfolio calculations', async () => {
      const userId = 'test-user-cache'
      const mockUser = { id: userId, wallet_address: 'test-wallet' }

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockOracleService.getAllPrices.mockResolvedValue({})

      // First call
      const result1 = await portfolioService.calculatePortfolio(userId)
      
      // Second call should use cache
      const result2 = await portfolioService.calculatePortfolio(userId)

      expect(result1).toEqual(result2)
      expect(mockDatabase.getUserByWallet).toHaveBeenCalledTimes(1) // Only called once due to caching
    })

    it('should invalidate cache when forced refresh', async () => {
      const userId = 'test-user-force-refresh'
      const mockUser = { id: userId, wallet_address: 'test-wallet' }

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockOracleService.getAllPrices.mockResolvedValue({})

      // First call
      await portfolioService.calculatePortfolio(userId)
      
      // Second call with force refresh
      await portfolioService.calculatePortfolio(userId, true)

      expect(mockDatabase.getUserByWallet).toHaveBeenCalledTimes(2) // Called twice due to force refresh
    })
  })

  describe('error handling', () => {
    it('should handle database errors gracefully', async () => {
      const userId = 'test-user-error'
      
      mockDatabase.getUserByWallet.mockRejectedValue(new Error('Database connection failed'))

      const result = await portfolioService.calculatePortfolio(userId)

      expect(result).toBeNull()
    })

    it('should handle oracle service errors gracefully', async () => {
      const userId = 'test-user-oracle-error'
      const mockUser = { id: userId, wallet_address: 'test-wallet' }

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockOracleService.getAllPrices.mockRejectedValue(new Error('Oracle service unavailable'))

      const result = await portfolioService.calculatePortfolio(userId)

      expect(result).toBeNull()
    })

    it('should handle PnL calculation errors gracefully', async () => {
      const userId = 'test-user-pnl-error'
      const mockUser = { id: userId, wallet_address: 'test-wallet' }
      
      const mockPositions = [
        {
          id: 'pos-1',
          symbol: 'SOL',
          side: 'long',
          size: 10,
          entry_price: 100,
          current_price: 110,
          margin: 1000,
          leverage: 10,
          status: 'open'
        }
      ]

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue(mockPositions)
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockOracleService.getAllPrices.mockResolvedValue({ 'SOL': 110 })
      
      mockPnLService.calculatePositionPnL.mockImplementation(() => {
        throw new Error('PnL calculation failed')
      })

      const result = await portfolioService.calculatePortfolio(userId)

      expect(result).toBeNull()
    })
  })
})
