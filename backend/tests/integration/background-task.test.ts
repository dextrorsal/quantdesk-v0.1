/**
 * Test 1.1-INT-002: Background task triggers portfolio recalculation every 5s
 * 
 * Integration test for background portfolio recalculation service
 * including timing, error handling, and WebSocket broadcasting.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { PortfolioBackgroundService } from '../../src/services/portfolioBackgroundService'
import { PortfolioCalculationService } from '../../src/services/portfolioCalculationService'
import { WebSocketService } from '../../src/services/websocket'

// Mock dependencies
vi.mock('../../src/services/portfolioCalculationService')
vi.mock('../../src/services/websocket')

describe('1.1-INT-002: Background Task Triggers', () => {
  let backgroundService: PortfolioBackgroundService
  let mockPortfolioService: any
  let mockWebSocketService: any

  beforeEach(() => {
    vi.clearAllMocks()

    // Mock Portfolio Calculation Service
    mockPortfolioService = {
      calculatePortfolio: vi.fn(),
    }

    // Mock WebSocket Service
    mockWebSocketService = {
      portfolioSubscriptions: new Map([
        ['user-1', new Set(['socket-1'])],
        ['user-2', new Set(['socket-2', 'socket-3'])]
      ]),
      broadcastPortfolioUpdate: vi.fn(),
    }

    // Create service instance
    backgroundService = new PortfolioBackgroundService(mockPortfolioService)
  })

  afterEach(() => {
    backgroundService.stop()
  })

  describe('background task execution', () => {
    it('should trigger portfolio recalculation every 5 seconds', async () => {
      const mockPortfolioData = {
        userId: 'user-1',
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      }

      mockPortfolioService.calculatePortfolio.mockResolvedValue(mockPortfolioData)

      // Mock WebSocket service instance
      vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocketService)

      // Start background service
      backgroundService.start()

      // Wait for first execution
      await new Promise(resolve => setTimeout(resolve, 100))

      // Verify portfolio calculation was called
      expect(mockPortfolioService.calculatePortfolio).toHaveBeenCalled()
    })

    it('should broadcast updates to all subscribed users', async () => {
      const mockPortfolioData1 = {
        userId: 'user-1',
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      }

      const mockPortfolioData2 = {
        userId: 'user-2',
        totalValue: 20000,
        totalUnrealizedPnl: 1000,
        totalRealizedPnl: 500,
        marginRatio: 75.0,
        healthFactor: 80.0,
        positions: [],
        timestamp: Date.now()
      }

      mockPortfolioService.calculatePortfolio
        .mockResolvedValueOnce(mockPortfolioData1)
        .mockResolvedValueOnce(mockPortfolioData2)

      // Mock WebSocket service instance
      vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocketService)

      // Start background service
      backgroundService.start()

      // Wait for execution
      await new Promise(resolve => setTimeout(resolve, 100))

      // Verify broadcasts were sent
      expect(mockWebSocketService.broadcastPortfolioUpdate).toHaveBeenCalledWith('user-1', mockPortfolioData1)
      expect(mockWebSocketService.broadcastPortfolioUpdate).toHaveBeenCalledWith('user-2', mockPortfolioData2)
    })

    it('should handle users with no portfolio data gracefully', async () => {
      mockPortfolioService.calculatePortfolio
        .mockResolvedValueOnce(null) // User 1 has no portfolio
        .mockResolvedValueOnce({     // User 2 has portfolio
          userId: 'user-2',
          totalValue: 20000,
          totalUnrealizedPnl: 1000,
          totalRealizedPnl: 500,
          marginRatio: 75.0,
          healthFactor: 80.0,
          positions: [],
          timestamp: Date.now()
        })

      // Mock WebSocket service instance
      vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocketService)

      // Start background service
      backgroundService.start()

      // Wait for execution
      await new Promise(resolve => setTimeout(resolve, 100))

      // Should only broadcast for user-2
      expect(mockWebSocketService.broadcastPortfolioUpdate).toHaveBeenCalledTimes(1)
      expect(mockWebSocketService.broadcastPortfolioUpdate).toHaveBeenCalledWith('user-2', expect.any(Object))
    })
  })

  describe('error handling', () => {
    it('should handle portfolio calculation errors gracefully', async () => {
      mockPortfolioService.calculatePortfolio.mockRejectedValue(
        new Error('Portfolio calculation failed')
      )

      // Mock WebSocket service instance
      vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocketService)

      // Start background service
      backgroundService.start()

      // Wait for execution
      await new Promise(resolve => setTimeout(resolve, 100))

      // Should not throw error, but continue running
      expect(mockWebSocketService.broadcastPortfolioUpdate).not.toHaveBeenCalled()
    })

    it('should handle WebSocket service errors gracefully', async () => {
      const mockPortfolioData = {
        userId: 'user-1',
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      }

      mockPortfolioService.calculatePortfolio.mockResolvedValue(mockPortfolioData)
      mockWebSocketService.broadcastPortfolioUpdate.mockRejectedValue(
        new Error('WebSocket broadcast failed')
      )

      // Mock WebSocket service instance
      vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocketService)

      // Start background service
      backgroundService.start()

      // Wait for execution
      await new Promise(resolve => setTimeout(resolve, 100))

      // Should not throw error, but continue running
      expect(mockPortfolioService.calculatePortfolio).toHaveBeenCalled()
    })

    it('should handle missing WebSocket service gracefully', async () => {
      const mockPortfolioData = {
        userId: 'user-1',
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      }

      mockPortfolioService.calculatePortfolio.mockResolvedValue(mockPortfolioData)

      // Mock WebSocket service instance to throw error
      vi.mocked(WebSocketService.getInstance).mockImplementation(() => {
        throw new Error('WebSocket service not available')
      })

      // Start background service
      backgroundService.start()

      // Wait for execution
      await new Promise(resolve => setTimeout(resolve, 100))

      // Should not throw error, but continue running
      expect(mockPortfolioService.calculatePortfolio).toHaveBeenCalled()
    })
  })

  describe('service lifecycle', () => {
    it('should start and stop background service correctly', () => {
      expect(backgroundService.isRunning()).toBe(false)

      backgroundService.start()
      expect(backgroundService.isRunning()).toBe(true)

      backgroundService.stop()
      expect(backgroundService.isRunning()).toBe(false)
    })

    it('should prevent multiple starts', () => {
      backgroundService.start()
      expect(backgroundService.isRunning()).toBe(true)

      // Second start should be ignored
      backgroundService.start()
      expect(backgroundService.isRunning()).toBe(true)
    })

    it('should handle stop when not running', () => {
      expect(backgroundService.isRunning()).toBe(false)

      // Stop when not running should not throw
      expect(() => backgroundService.stop()).not.toThrow()
    })
  })

  describe('metrics and monitoring', () => {
    it('should track execution metrics', async () => {
      const mockPortfolioData = {
        userId: 'user-1',
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      }

      mockPortfolioService.calculatePortfolio.mockResolvedValue(mockPortfolioData)

      // Mock WebSocket service instance
      vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocketService)

      // Start background service
      backgroundService.start()

      // Wait for execution
      await new Promise(resolve => setTimeout(resolve, 100))

      const metrics = backgroundService.getMetrics()

      expect(metrics.runCount).toBeGreaterThan(0)
      expect(metrics.lastRun).toBeGreaterThan(0)
      expect(metrics.errorCount).toBe(0)
    })

    it('should track error metrics', async () => {
      mockPortfolioService.calculatePortfolio.mockRejectedValue(
        new Error('Test error')
      )

      // Mock WebSocket service instance
      vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocketService)

      // Start background service
      backgroundService.start()

      // Wait for execution
      await new Promise(resolve => setTimeout(resolve, 100))

      const metrics = backgroundService.getMetrics()

      expect(metrics.runCount).toBeGreaterThan(0)
      expect(metrics.errorCount).toBeGreaterThan(0)
    })
  })

  describe('timing and intervals', () => {
    it('should respect the 5-second interval', async () => {
      const startTime = Date.now()

      mockPortfolioService.calculatePortfolio.mockResolvedValue({
        userId: 'user-1',
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      })

      // Mock WebSocket service instance
      vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocketService)

      // Start background service
      backgroundService.start()

      // Wait for multiple executions
      await new Promise(resolve => setTimeout(resolve, 6000))

      const endTime = Date.now()
      const totalTime = endTime - startTime

      // Should have run approximately every 5 seconds
      const expectedRuns = Math.floor(totalTime / 5000)
      expect(mockPortfolioService.calculatePortfolio).toHaveBeenCalledTimes(expectedRuns)
    })
  })

  describe('concurrent execution', () => {
    it('should handle concurrent portfolio calculations', async () => {
      const mockPortfolioData = {
        userId: 'user-1',
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      }

      // Mock slow portfolio calculation
      mockPortfolioService.calculatePortfolio.mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve(mockPortfolioData), 1000))
      )

      // Mock WebSocket service instance
      vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocketService)

      // Start background service
      backgroundService.start()

      // Wait for execution
      await new Promise(resolve => setTimeout(resolve, 2000))

      // Should handle concurrent executions gracefully
      expect(mockPortfolioService.calculatePortfolio).toHaveBeenCalled()
    })
  })
})
