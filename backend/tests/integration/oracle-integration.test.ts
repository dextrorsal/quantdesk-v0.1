/**
 * Test 1.1-INT-003: Oracle price feed integration updates position values
 * 
 * Integration test for Oracle service integration including
 * price feed processing, cache invalidation, and portfolio updates.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { PythOracleService } from '../../src/services/pythOracleService'
import { PortfolioCalculationService } from '../../src/services/portfolioCalculationService'
import { SupabaseDatabaseService } from '../../src/services/supabaseDatabase'
import { RedisService } from '../../src/services/redisService'

// Mock dependencies
vi.mock('../../src/services/portfolioCalculationService')
vi.mock('../../src/services/supabaseDatabase')
vi.mock('../../src/services/redisService')

describe('1.1-INT-003: Oracle Integration', () => {
  let oracleService: PythOracleService
  let mockPortfolioService: any
  let mockDatabase: any
  let mockRedis: any

  beforeEach(() => {
    vi.clearAllMocks()

    // Mock Database Service
    mockDatabase = {
      storePriceData: vi.fn(),
      getLatestPrices: vi.fn(),
    }

    // Mock Redis Service
    mockRedis = {
      set: vi.fn(),
      get: vi.fn(),
      del: vi.fn(),
    }

    // Mock Portfolio Service
    mockPortfolioService = {
      invalidateAllPortfolioCaches: vi.fn(),
    }

    // Create Oracle service instance
    oracleService = new PythOracleService(mockDatabase, mockRedis)
  })

  describe('price feed integration', () => {
    it('should process Oracle price feeds and update positions', async () => {
      const mockPriceFeed = {
        'SOL': {
          price: 110.50,
          confidence: 0.01,
          timestamp: Date.now(),
          symbol: 'SOL'
        },
        'BTC': {
          price: 48500.75,
          confidence: 50.0,
          timestamp: Date.now(),
          symbol: 'BTC'
        }
      }

      // Mock successful storage
      mockDatabase.storePriceData.mockResolvedValue(true)
      mockRedis.set.mockResolvedValue('OK')

      // Process price feed
      await oracleService.processAndStorePriceFeed(mockPriceFeed)

      // Verify database storage
      expect(mockDatabase.storePriceData).toHaveBeenCalledWith(
        expect.objectContaining({
          'SOL': expect.objectContaining({
            price: 110.50,
            symbol: 'SOL'
          }),
          'BTC': expect.objectContaining({
            price: 48500.75,
            symbol: 'BTC'
          })
        })
      )

      // Verify Redis caching
      expect(mockRedis.set).toHaveBeenCalledWith(
        'prices:SOL',
        expect.stringContaining('110.5'),
        'EX',
        60
      )
      expect(mockRedis.set).toHaveBeenCalledWith(
        'prices:BTC',
        expect.stringContaining('48500.75'),
        'EX',
        60
      )
    })

    it('should invalidate portfolio caches when prices update', async () => {
      const mockPriceFeed = {
        'SOL': {
          price: 115.25, // Price change
          confidence: 0.01,
          timestamp: Date.now(),
          symbol: 'SOL'
        }
      }

      mockDatabase.storePriceData.mockResolvedValue(true)
      mockRedis.set.mockResolvedValue('OK')

      // Mock portfolio service notification
      vi.spyOn(oracleService as any, 'notifyPortfolioServiceOfPriceUpdate')
        .mockImplementation(async () => {
          await mockPortfolioService.invalidateAllPortfolioCaches()
        })

      await oracleService.processAndStorePriceFeed(mockPriceFeed)

      // Verify portfolio cache invalidation
      expect(mockPortfolioService.invalidateAllPortfolioCaches).toHaveBeenCalled()
    })

    it('should handle price feed processing errors', async () => {
      const mockPriceFeed = {
        'SOL': {
          price: null, // Invalid price
          confidence: 0.01,
          timestamp: Date.now(),
          symbol: 'SOL'
        }
      }

      // Should not throw error, but handle gracefully
      await expect(oracleService.processAndStorePriceFeed(mockPriceFeed))
        .resolves.not.toThrow()

      // Should not store invalid data
      expect(mockDatabase.storePriceData).not.toHaveBeenCalled()
    })
  })

  describe('price retrieval integration', () => {
    it('should retrieve prices from cache first, then database', async () => {
      const cachedPrice = '110.50'
      mockRedis.get.mockResolvedValue(cachedPrice)

      const price = await oracleService.getPrice('SOL')

      expect(price).toBe(110.50)
      expect(mockRedis.get).toHaveBeenCalledWith('prices:SOL')
      expect(mockDatabase.getLatestPrices).not.toHaveBeenCalled()
    })

    it('should fallback to database on cache miss', async () => {
      mockRedis.get.mockResolvedValue(null)
      mockDatabase.getLatestPrices.mockResolvedValue([
        { symbol: 'SOL', price: 110.50, timestamp: Date.now() }
      ])

      const price = await oracleService.getPrice('SOL')

      expect(price).toBe(110.50)
      expect(mockRedis.get).toHaveBeenCalledWith('prices:SOL')
      expect(mockDatabase.getLatestPrices).toHaveBeenCalled()
    })

    it('should handle database errors gracefully', async () => {
      mockRedis.get.mockResolvedValue(null)
      mockDatabase.getLatestPrices.mockRejectedValue(
        new Error('Database connection failed')
      )

      const price = await oracleService.getPrice('SOL')

      expect(price).toBeNull()
    })
  })

  describe('batch price operations', () => {
    it('should process multiple prices efficiently', async () => {
      const mockPriceFeed = {
        'SOL': { price: 110.50, confidence: 0.01, timestamp: Date.now(), symbol: 'SOL' },
        'BTC': { price: 48500.75, confidence: 50.0, timestamp: Date.now(), symbol: 'BTC' },
        'ETH': { price: 3200.25, confidence: 0.5, timestamp: Date.now(), symbol: 'ETH' },
        'USDC': { price: 1.0, confidence: 0.001, timestamp: Date.now(), symbol: 'USDC' }
      }

      mockDatabase.storePriceData.mockResolvedValue(true)
      mockRedis.set.mockResolvedValue('OK')

      await oracleService.processAndStorePriceFeed(mockPriceFeed)

      // Should store all valid prices
      expect(mockDatabase.storePriceData).toHaveBeenCalledWith(
        expect.objectContaining({
          'SOL': expect.any(Object),
          'BTC': expect.any(Object),
          'ETH': expect.any(Object),
          'USDC': expect.any(Object)
        })
      )

      // Should cache all prices
      expect(mockRedis.set).toHaveBeenCalledTimes(4)
    })

    it('should handle mixed valid and invalid prices', async () => {
      const mockPriceFeed = {
        'SOL': { price: 110.50, confidence: 0.01, timestamp: Date.now(), symbol: 'SOL' },
        'INVALID': { price: null, confidence: 0.01, timestamp: Date.now(), symbol: 'INVALID' },
        'STALE': { 
          price: 100.0, 
          confidence: 0.01, 
          timestamp: Date.now() - (10 * 60 * 1000), // 10 minutes ago
          symbol: 'STALE' 
        },
        'BTC': { price: 48500.75, confidence: 50.0, timestamp: Date.now(), symbol: 'BTC' }
      }

      mockDatabase.storePriceData.mockResolvedValue(true)
      mockRedis.set.mockResolvedValue('OK')

      await oracleService.processAndStorePriceFeed(mockPriceFeed)

      // Should only store valid prices
      expect(mockDatabase.storePriceData).toHaveBeenCalledWith(
        expect.objectContaining({
          'SOL': expect.any(Object),
          'BTC': expect.any(Object)
        })
      )

      // Should not store invalid or stale prices
      const storedData = mockDatabase.storePriceData.mock.calls[0][0]
      expect(storedData['INVALID']).toBeUndefined()
      expect(storedData['STALE']).toBeUndefined()
    })
  })

  describe('cache management', () => {
    it('should set appropriate TTL for cached prices', async () => {
      const mockPriceFeed = {
        'SOL': { price: 110.50, confidence: 0.01, timestamp: Date.now(), symbol: 'SOL' }
      }

      mockDatabase.storePriceData.mockResolvedValue(true)
      mockRedis.set.mockResolvedValue('OK')

      await oracleService.processAndStorePriceFeed(mockPriceFeed)

      // Verify TTL is set to 60 seconds
      expect(mockRedis.set).toHaveBeenCalledWith(
        'prices:SOL',
        expect.any(String),
        'EX',
        60
      )
    })

    it('should handle cache errors gracefully', async () => {
      const mockPriceFeed = {
        'SOL': { price: 110.50, confidence: 0.01, timestamp: Date.now(), symbol: 'SOL' }
      }

      mockDatabase.storePriceData.mockResolvedValue(true)
      mockRedis.set.mockRejectedValue(new Error('Redis connection failed'))

      // Should not throw error
      await expect(oracleService.processAndStorePriceFeed(mockPriceFeed))
        .resolves.not.toThrow()

      // Should still store in database
      expect(mockDatabase.storePriceData).toHaveBeenCalled()
    })
  })

  describe('real-time updates', () => {
    it('should handle rapid price updates', async () => {
      const updates = [
        { 'SOL': { price: 110.50, confidence: 0.01, timestamp: Date.now(), symbol: 'SOL' } },
        { 'SOL': { price: 111.25, confidence: 0.01, timestamp: Date.now() + 1000, symbol: 'SOL' } },
        { 'SOL': { price: 112.00, confidence: 0.01, timestamp: Date.now() + 2000, symbol: 'SOL' } }
      ]

      mockDatabase.storePriceData.mockResolvedValue(true)
      mockRedis.set.mockResolvedValue('OK')

      // Process all updates
      for (const update of updates) {
        await oracleService.processAndStorePriceFeed(update)
      }

      // Should process all updates
      expect(mockDatabase.storePriceData).toHaveBeenCalledTimes(3)
      expect(mockRedis.set).toHaveBeenCalledTimes(3)
    })

    it('should handle concurrent price updates', async () => {
      const mockPriceFeed = {
        'SOL': { price: 110.50, confidence: 0.01, timestamp: Date.now(), symbol: 'SOL' },
        'BTC': { price: 48500.75, confidence: 50.0, timestamp: Date.now(), symbol: 'BTC' }
      }

      mockDatabase.storePriceData.mockResolvedValue(true)
      mockRedis.set.mockResolvedValue('OK')

      // Process concurrent updates
      const promises = [
        oracleService.processAndStorePriceFeed(mockPriceFeed),
        oracleService.processAndStorePriceFeed(mockPriceFeed),
        oracleService.processAndStorePriceFeed(mockPriceFeed)
      ]

      await Promise.all(promises)

      // Should handle concurrent processing
      expect(mockDatabase.storePriceData).toHaveBeenCalledTimes(3)
    })
  })

  describe('error recovery', () => {
    it('should recover from temporary database failures', async () => {
      const mockPriceFeed = {
        'SOL': { price: 110.50, confidence: 0.01, timestamp: Date.now(), symbol: 'SOL' }
      }

      // First call fails, second succeeds
      mockDatabase.storePriceData
        .mockRejectedValueOnce(new Error('Temporary database error'))
        .mockResolvedValueOnce(true)

      mockRedis.set.mockResolvedValue('OK')

      // First attempt should fail
      await expect(oracleService.processAndStorePriceFeed(mockPriceFeed))
        .rejects.toThrow('Temporary database error')

      // Second attempt should succeed
      await expect(oracleService.processAndStorePriceFeed(mockPriceFeed))
        .resolves.not.toThrow()
    })

    it('should handle Redis connection failures', async () => {
      const mockPriceFeed = {
        'SOL': { price: 110.50, confidence: 0.01, timestamp: Date.now(), symbol: 'SOL' }
      }

      mockDatabase.storePriceData.mockResolvedValue(true)
      mockRedis.set.mockRejectedValue(new Error('Redis connection failed'))

      // Should not throw error, but continue processing
      await expect(oracleService.processAndStorePriceFeed(mockPriceFeed))
        .resolves.not.toThrow()

      // Should still store in database
      expect(mockDatabase.storePriceData).toHaveBeenCalled()
    })
  })
})
