/**
 * Test 1.1-UNIT-007: Price feed processing and normalization
 * 
 * Tests the Oracle price feed processing, data transformation,
 * and normalization logic for real-time price updates.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { PythOracleService } from '../../src/services/pythOracleService'

// Mock dependencies
vi.mock('../../src/services/supabaseDatabase')
vi.mock('../../src/services/redisService')

describe('1.1-UNIT-007: Price Feed Processing', () => {
  let oracleService: PythOracleService
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

    // Create service instance
    oracleService = new PythOracleService(mockDatabase, mockRedis)
  })

  describe('price data processing', () => {
    it('should process and normalize price data correctly', async () => {
      const mockPriceData = {
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

      const processedData = await oracleService.processPriceData(mockPriceData)

      expect(processedData).toBeDefined()
      expect(processedData['SOL']).toBe(110.50)
      expect(processedData['BTC']).toBe(48500.75)
    })

    it('should handle scientific notation prices', async () => {
      const mockPriceData = {
        'SOL': {
          price: 1.105e2, // Scientific notation for 110.5
          confidence: 0.01,
          timestamp: Date.now(),
          symbol: 'SOL'
        }
      }

      const processedData = await oracleService.processPriceData(mockPriceData)

      expect(processedData['SOL']).toBe(110.5)
    })

    it('should filter out invalid prices', async () => {
      const mockPriceData = {
        'SOL': {
          price: 110.50,
          confidence: 0.01,
          timestamp: Date.now(),
          symbol: 'SOL'
        },
        'INVALID': {
          price: null,
          confidence: 0.01,
          timestamp: Date.now(),
          symbol: 'INVALID'
        },
        'NEGATIVE': {
          price: -50.0,
          confidence: 0.01,
          timestamp: Date.now(),
          symbol: 'NEGATIVE'
        }
      }

      const processedData = await oracleService.processPriceData(mockPriceData)

      expect(processedData['SOL']).toBe(110.50)
      expect(processedData['INVALID']).toBeUndefined()
      expect(processedData['NEGATIVE']).toBeUndefined()
    })

    it('should handle zero prices appropriately', async () => {
      const mockPriceData = {
        'ZERO': {
          price: 0,
          confidence: 0.01,
          timestamp: Date.now(),
          symbol: 'ZERO'
        }
      }

      const processedData = await oracleService.processPriceData(mockPriceData)

      // Zero prices might be valid for some assets, but should be handled carefully
      expect(processedData['ZERO']).toBe(0)
    })
  })

  describe('confidence filtering', () => {
    it('should filter prices with low confidence', async () => {
      const mockPriceData = {
        'HIGH_CONFIDENCE': {
          price: 100.0,
          confidence: 0.1, // Low confidence threshold
          timestamp: Date.now(),
          symbol: 'HIGH_CONFIDENCE'
        },
        'LOW_CONFIDENCE': {
          price: 100.0,
          confidence: 10.0, // High confidence threshold
          timestamp: Date.now(),
          symbol: 'LOW_CONFIDENCE'
        }
      }

      const processedData = await oracleService.processPriceData(mockPriceData)

      expect(processedData['HIGH_CONFIDENCE']).toBe(100.0)
      expect(processedData['LOW_CONFIDENCE']).toBeUndefined()
    })

    it('should handle missing confidence values', async () => {
      const mockPriceData = {
        'NO_CONFIDENCE': {
          price: 100.0,
          timestamp: Date.now(),
          symbol: 'NO_CONFIDENCE'
        }
      }

      const processedData = await oracleService.processPriceData(mockPriceData)

      // Should handle missing confidence gracefully
      expect(processedData['NO_CONFIDENCE']).toBeDefined()
    })
  })

  describe('timestamp validation', () => {
    it('should reject stale price data', async () => {
      const staleTimestamp = Date.now() - (5 * 60 * 1000) // 5 minutes ago
      const mockPriceData = {
        'STALE': {
          price: 100.0,
          confidence: 0.01,
          timestamp: staleTimestamp,
          symbol: 'STALE'
        }
      }

      const processedData = await oracleService.processPriceData(mockPriceData)

      // Should filter out stale data
      expect(processedData['STALE']).toBeUndefined()
    })

    it('should accept recent price data', async () => {
      const recentTimestamp = Date.now() - (30 * 1000) // 30 seconds ago
      const mockPriceData = {
        'RECENT': {
          price: 100.0,
          confidence: 0.01,
          timestamp: recentTimestamp,
          symbol: 'RECENT'
        }
      }

      const processedData = await oracleService.processPriceData(mockPriceData)

      expect(processedData['RECENT']).toBe(100.0)
    })
  })

  describe('data storage', () => {
    it('should store processed price data', async () => {
      const mockPriceData = {
        'SOL': {
          price: 110.50,
          confidence: 0.01,
          timestamp: Date.now(),
          symbol: 'SOL'
        }
      }

      mockDatabase.storePriceData.mockResolvedValue(true)

      await oracleService.storePriceData(mockPriceData)

      expect(mockDatabase.storePriceData).toHaveBeenCalledWith(
        expect.objectContaining({
          'SOL': expect.objectContaining({
            price: 110.50,
            symbol: 'SOL'
          })
        })
      )
    })

    it('should cache price data in Redis', async () => {
      const mockPriceData = {
        'SOL': {
          price: 110.50,
          confidence: 0.01,
          timestamp: Date.now(),
          symbol: 'SOL'
        }
      }

      mockRedis.set.mockResolvedValue('OK')

      await oracleService.cachePriceData(mockPriceData)

      expect(mockRedis.set).toHaveBeenCalledWith(
        'prices:SOL',
        expect.stringContaining('110.5'),
        'EX',
        60 // 60 seconds TTL
      )
    })
  })

  describe('error handling', () => {
    it('should handle malformed price data', async () => {
      const malformedData = {
        'MALFORMED': {
          // Missing required fields
          symbol: 'MALFORMED'
        }
      }

      const processedData = await oracleService.processPriceData(malformedData)

      expect(processedData['MALFORMED']).toBeUndefined()
    })

    it('should handle database storage errors', async () => {
      const mockPriceData = {
        'SOL': {
          price: 110.50,
          confidence: 0.01,
          timestamp: Date.now(),
          symbol: 'SOL'
        }
      }

      mockDatabase.storePriceData.mockRejectedValue(new Error('Database error'))

      await expect(oracleService.storePriceData(mockPriceData))
        .rejects.toThrow('Database error')
    })

    it('should handle Redis cache errors gracefully', async () => {
      const mockPriceData = {
        'SOL': {
          price: 110.50,
          confidence: 0.01,
          timestamp: Date.now(),
          symbol: 'SOL'
        }
      }

      mockRedis.set.mockRejectedValue(new Error('Redis connection failed'))

      // Should not throw, but log error
      await expect(oracleService.cachePriceData(mockPriceData))
        .resolves.not.toThrow()
    })
  })

  describe('price retrieval', () => {
    it('should retrieve cached prices from Redis', async () => {
      const cachedPrice = '110.50'
      mockRedis.get.mockResolvedValue(cachedPrice)

      const price = await oracleService.getCachedPrice('SOL')

      expect(price).toBe(110.50)
      expect(mockRedis.get).toHaveBeenCalledWith('prices:SOL')
    })

    it('should fallback to database when cache miss', async () => {
      mockRedis.get.mockResolvedValue(null)
      mockDatabase.getLatestPrices.mockResolvedValue([
        { symbol: 'SOL', price: 110.50, timestamp: Date.now() }
      ])

      const price = await oracleService.getPrice('SOL')

      expect(price).toBe(110.50)
      expect(mockDatabase.getLatestPrices).toHaveBeenCalled()
    })

    it('should return null for non-existent symbols', async () => {
      mockRedis.get.mockResolvedValue(null)
      mockDatabase.getLatestPrices.mockResolvedValue([])

      const price = await oracleService.getPrice('NONEXISTENT')

      expect(price).toBeNull()
    })
  })

  describe('batch operations', () => {
    it('should process multiple prices efficiently', async () => {
      const mockPriceData = {
        'SOL': { price: 110.50, confidence: 0.01, timestamp: Date.now(), symbol: 'SOL' },
        'BTC': { price: 48500.75, confidence: 50.0, timestamp: Date.now(), symbol: 'BTC' },
        'ETH': { price: 3200.25, confidence: 0.5, timestamp: Date.now(), symbol: 'ETH' }
      }

      const processedData = await oracleService.processPriceData(mockPriceData)

      expect(Object.keys(processedData)).toHaveLength(3)
      expect(processedData['SOL']).toBe(110.50)
      expect(processedData['BTC']).toBe(48500.75)
      expect(processedData['ETH']).toBe(3200.25)
    })

    it('should handle mixed valid and invalid prices', async () => {
      const mockPriceData = {
        'VALID': { price: 100.0, confidence: 0.01, timestamp: Date.now(), symbol: 'VALID' },
        'INVALID': { price: null, confidence: 0.01, timestamp: Date.now(), symbol: 'INVALID' },
        'STALE': { 
          price: 100.0, 
          confidence: 0.01, 
          timestamp: Date.now() - (10 * 60 * 1000), // 10 minutes ago
          symbol: 'STALE' 
        }
      }

      const processedData = await oracleService.processPriceData(mockPriceData)

      expect(Object.keys(processedData)).toHaveLength(1)
      expect(processedData['VALID']).toBe(100.0)
    })
  })
})
