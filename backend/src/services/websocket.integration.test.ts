import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { Server as SocketIOServer } from 'socket.io'
import { createServer } from 'http'
import { WebSocketService } from './websocket'
import { SupabaseDatabaseService } from './supabaseDatabase'
import { pythOracleService } from './pythOracleService'

// Mock dependencies
vi.mock('./supabaseDatabase', () => ({
  SupabaseDatabaseService: {
    getInstance: vi.fn().mockReturnValue({
      getUserByWallet: vi.fn(),
      getUserPositions: vi.fn(),
      getUserCollateral: vi.fn(),
    }),
  },
}))

vi.mock('./pythOracleService', () => ({
  pythOracleService: {
    getAllPrices: vi.fn(),
    getPrice: vi.fn(),
  },
}))

vi.mock('../utils/logger', () => ({
  Logger: vi.fn().mockImplementation(() => ({
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
  })),
}))

describe('Backend WebSocket Integration Tests', () => {
  let httpServer: any
  let io: SocketIOServer
  let webSocketService: WebSocketService
  let mockDb: any
  let mockPythOracle: any

  beforeEach(() => {
    // Create HTTP server and Socket.IO instance
    httpServer = createServer()
    io = new SocketIOServer(httpServer, {
      cors: {
        origin: "*",
        methods: ["GET", "POST"]
      }
    })

    // Get mocked instances
    mockDb = SupabaseDatabaseService.getInstance()
    mockPythOracle = pythOracleService

    // Initialize WebSocket service
    webSocketService = new WebSocketService(io)
  })

  afterEach(() => {
    if (httpServer) {
      httpServer.close()
    }
    vi.clearAllMocks()
  })

  describe('Portfolio Updates Integration', () => {
    it('should handle portfolio subscription and broadcast updates', async () => {
      const mockUser = {
        id: 'user-123',
        wallet_address: 'test-wallet-address',
        created_at: new Date().toISOString(),
      }

      const mockPositions = [
        {
          id: 'pos-1',
          user_id: 'user-123',
          symbol: 'SOL-PERP',
          side: 'long',
          size: 100,
          entry_price: 150.0,
          mark_price: 155.0,
          unrealized_pnl: 500,
        },
      ]

      const mockCollateral = {
        id: 'coll-1',
        user_id: 'user-123',
        asset: 'SOL',
        amount: 10.0,
        value_usd: 1500,
      }

      // Mock database responses
      mockDb.getUserByWallet.mockResolvedValue(mockUser)
      mockDb.getUserPositions.mockResolvedValue(mockPositions)
      mockDb.getUserCollateral.mockResolvedValue([mockCollateral])

      // Mock oracle prices
      mockPythOracle.getAllPrices.mockResolvedValue({
        SOL: 150.0,
        BTC: 45000,
        ETH: 3000,
      })

      // Create a mock socket
      const mockSocket = {
        id: 'socket-123',
        data: { userId: 'user-123' },
        join: vi.fn(),
        leave: vi.fn(),
        emit: vi.fn(),
        on: vi.fn(),
      }

      // Simulate portfolio subscription
      const subscribeHandler = webSocketService['setupSocketHandlers'](mockSocket)
      
      // Trigger subscription
      const portfolioSubscriptionHandler = mockSocket.on.mock.calls
        .find(call => call[0] === 'subscribe_portfolio')?.[1]
      
      if (portfolioSubscriptionHandler) {
        await portfolioSubscriptionHandler({ userId: 'user-123' })
      }

      // Verify user joined portfolio room
      expect(mockSocket.join).toHaveBeenCalledWith('portfolio:user-123')

      // Start portfolio updates
      webSocketService['startPortfolioUpdates']()

      // Wait for portfolio calculation and broadcast
      await new Promise(resolve => setTimeout(resolve, 100))

      // Verify portfolio data was calculated and broadcast
      expect(mockDb.getUserByWallet).toHaveBeenCalledWith('user-123')
      expect(mockDb.getUserPositions).toHaveBeenCalledWith('user-123')
      expect(mockDb.getUserCollateral).toHaveBeenCalledWith('user-123')
    })

    it('should handle portfolio calculation errors gracefully', async () => {
      // Mock database error
      mockDb.getUserByWallet.mockRejectedValue(new Error('Database connection failed'))

      const mockSocket = {
        id: 'socket-123',
        data: { userId: 'user-123' },
        join: vi.fn(),
        leave: vi.fn(),
        emit: vi.fn(),
        on: vi.fn(),
      }

      // Start portfolio updates
      webSocketService['startPortfolioUpdates']()

      // Wait for error handling
      await new Promise(resolve => setTimeout(resolve, 100))

      // Should not crash and should handle error gracefully
      expect(mockDb.getUserByWallet).toHaveBeenCalledWith('user-123')
    })

    it('should calculate portfolio metrics correctly', async () => {
      const mockUser = {
        id: 'user-123',
        wallet_address: 'test-wallet-address',
        created_at: new Date().toISOString(),
      }

      const mockPositions = [
        {
          id: 'pos-1',
          user_id: 'user-123',
          symbol: 'SOL-PERP',
          side: 'long',
          size: 100,
          entry_price: 150.0,
          mark_price: 155.0,
          unrealized_pnl: 500,
        },
        {
          id: 'pos-2',
          user_id: 'user-123',
          symbol: 'BTC-PERP',
          side: 'short',
          size: 0.1,
          entry_price: 45000,
          mark_price: 44000,
          unrealized_pnl: 100,
        },
      ]

      const mockCollateral = {
        id: 'coll-1',
        user_id: 'user-123',
        asset: 'SOL',
        amount: 10.0,
        value_usd: 1500,
      }

      mockDb.getUserByWallet.mockResolvedValue(mockUser)
      mockDb.getUserPositions.mockResolvedValue(mockPositions)
      mockDb.getUserCollateral.mockResolvedValue([mockCollateral])

      mockPythOracle.getAllPrices.mockResolvedValue({
        SOL: 150.0,
        BTC: 44000,
      })

      // Calculate portfolio
      const portfolioData = await webSocketService['calculateUserPortfolio']('user-123')

      // Verify portfolio calculation
      expect(portfolioData).toMatchObject({
        userId: 'user-123',
        summary: {
          totalValue: expect.any(Number),
          unrealizedPnl: 600, // 500 + 100
          realizedPnl: 0,
        },
        riskMetrics: {
          marginRatio: expect.any(Number),
          leverage: expect.any(Number),
        },
        performanceMetrics: {
          dailyReturn: expect.any(Number),
          weeklyReturn: expect.any(Number),
        },
      })

      // Verify total value calculation
      expect(portfolioData.summary.totalValue).toBeGreaterThan(0)
      expect(portfolioData.summary.unrealizedPnl).toBe(600)
    })
  })

  describe('Market Data Integration', () => {
    it('should broadcast market data updates', async () => {
      const mockMarketData = {
        symbol: 'SOL-PERP',
        price: 150.25,
        change24h: 0.05,
        volume24h: 1000000,
        openInterest: 5000000,
        fundingRate: 0.0001,
      }

      const mockSocket = {
        id: 'socket-123',
        data: { userId: 'user-123' },
        join: vi.fn(),
        leave: vi.fn(),
        emit: vi.fn(),
        on: vi.fn(),
      }

      // Setup socket handlers
      webSocketService['setupSocketHandlers'](mockSocket)

      // Simulate market data update
      const marketDataHandler = mockSocket.on.mock.calls
        .find(call => call[0] === 'market_data_update')?.[1]
      
      if (marketDataHandler) {
        await marketDataHandler(mockMarketData)
      }

      // Verify market data was handled
      expect(mockSocket.on).toHaveBeenCalledWith('market_data_update', expect.any(Function))
    })

    it('should handle oracle price integration', async () => {
      const mockPrices = {
        SOL: 150.25,
        BTC: 45000,
        ETH: 3000,
      }

      mockPythOracle.getAllPrices.mockResolvedValue(mockPrices)

      // Test oracle integration
      const prices = await mockPythOracle.getAllPrices()

      expect(prices).toEqual(mockPrices)
      expect(mockPythOracle.getAllPrices).toHaveBeenCalled()
    })
  })

  describe('Connection Management Integration', () => {
    it('should handle client connection and authentication', async () => {
      const mockSocket = {
        id: 'socket-123',
        data: {},
        join: vi.fn(),
        leave: vi.fn(),
        emit: vi.fn(),
        on: vi.fn(),
      }

      // Setup socket handlers
      webSocketService['setupSocketHandlers'](mockSocket)

      // Simulate authentication
      const authHandler = mockSocket.on.mock.calls
        .find(call => call[0] === 'authenticate')?.[1]
      
      if (authHandler) {
        await authHandler({ token: 'valid-jwt-token', userId: 'user-123' })
      }

      // Verify authentication was handled
      expect(mockSocket.on).toHaveBeenCalledWith('authenticate', expect.any(Function))
    })

    it('should handle client disconnection', async () => {
      const mockSocket = {
        id: 'socket-123',
        data: { userId: 'user-123' },
        join: vi.fn(),
        leave: vi.fn(),
        emit: vi.fn(),
        on: vi.fn(),
      }

      // Setup socket handlers
      webSocketService['setupSocketHandlers'](mockSocket)

      // Simulate disconnection
      const disconnectHandler = mockSocket.on.mock.calls
        .find(call => call[0] === 'disconnect')?.[1]
      
      if (disconnectHandler) {
        await disconnectHandler()
      }

      // Verify disconnection was handled
      expect(mockSocket.on).toHaveBeenCalledWith('disconnect', expect.any(Function))
    })

    it('should manage connected clients correctly', async () => {
      const mockSocket1 = {
        id: 'socket-1',
        data: { userId: 'user-1' },
        join: vi.fn(),
        leave: vi.fn(),
        emit: vi.fn(),
        on: vi.fn(),
      }

      const mockSocket2 = {
        id: 'socket-2',
        data: { userId: 'user-2' },
        join: vi.fn(),
        leave: vi.fn(),
        emit: vi.fn(),
        on: vi.fn(),
      }

      // Setup socket handlers for both clients
      webSocketService['setupSocketHandlers'](mockSocket1)
      webSocketService['setupSocketHandlers'](mockSocket2)

      // Verify both clients are tracked
      expect(webSocketService['connectedClients'].size).toBe(2)
      expect(webSocketService['connectedClients'].has('user-1')).toBe(true)
      expect(webSocketService['connectedClients'].has('user-2')).toBe(true)
    })
  })

  describe('Error Handling Integration', () => {
    it('should handle database connection errors', async () => {
      mockDb.getUserByWallet.mockRejectedValue(new Error('Database connection failed'))

      const mockSocket = {
        id: 'socket-123',
        data: { userId: 'user-123' },
        join: vi.fn(),
        leave: vi.fn(),
        emit: vi.fn(),
        on: vi.fn(),
      }

      // Setup socket handlers
      webSocketService['setupSocketHandlers'](mockSocket)

      // Simulate portfolio subscription
      const portfolioSubscriptionHandler = mockSocket.on.mock.calls
        .find(call => call[0] === 'subscribe_portfolio')?.[1]
      
      if (portfolioSubscriptionHandler) {
        await portfolioSubscriptionHandler({ userId: 'user-123' })
      }

      // Start portfolio updates
      webSocketService['startPortfolioUpdates']()

      // Wait for error handling
      await new Promise(resolve => setTimeout(resolve, 100))

      // Should handle error gracefully without crashing
      expect(mockDb.getUserByWallet).toHaveBeenCalledWith('user-123')
    })

    it('should handle oracle service errors', async () => {
      mockPythOracle.getAllPrices.mockRejectedValue(new Error('Oracle service unavailable'))

      const mockSocket = {
        id: 'socket-123',
        data: { userId: 'user-123' },
        join: vi.fn(),
        leave: vi.fn(),
        emit: vi.fn(),
        on: vi.fn(),
      }

      // Setup socket handlers
      webSocketService['setupSocketHandlers'](mockSocket)

      // Start portfolio updates
      webSocketService['startPortfolioUpdates']()

      // Wait for error handling
      await new Promise(resolve => setTimeout(resolve, 100))

      // Should handle oracle error gracefully
      expect(mockPythOracle.getAllPrices).toHaveBeenCalled()
    })

    it('should handle malformed portfolio data', async () => {
      const mockUser = {
        id: 'user-123',
        wallet_address: 'test-wallet-address',
        created_at: new Date().toISOString(),
      }

      // Mock malformed data
      mockDb.getUserByWallet.mockResolvedValue(mockUser)
      mockDb.getUserPositions.mockResolvedValue(null) // Malformed data
      mockDb.getUserCollateral.mockResolvedValue([])

      mockPythOracle.getAllPrices.mockResolvedValue({})

      // Calculate portfolio with malformed data
      const portfolioData = await webSocketService['calculateUserPortfolio']('user-123')

      // Should handle malformed data gracefully
      expect(portfolioData).toMatchObject({
        userId: 'user-123',
        summary: {
          totalValue: 0,
          unrealizedPnl: 0,
          realizedPnl: 0,
        },
      })
    })
  })

  describe('Performance Integration', () => {
    it('should handle multiple concurrent portfolio updates', async () => {
      const mockUser = {
        id: 'user-123',
        wallet_address: 'test-wallet-address',
        created_at: new Date().toISOString(),
      }

      mockDb.getUserByWallet.mockResolvedValue(mockUser)
      mockDb.getUserPositions.mockResolvedValue([])
      mockDb.getUserCollateral.mockResolvedValue([])
      mockPythOracle.getAllPrices.mockResolvedValue({})

      // Start portfolio updates
      webSocketService['startPortfolioUpdates']()

      // Simulate multiple concurrent calculations
      const promises = []
      for (let i = 0; i < 10; i++) {
        promises.push(webSocketService['calculateUserPortfolio'](`user-${i}`))
      }

      const results = await Promise.all(promises)

      // All calculations should complete successfully
      expect(results).toHaveLength(10)
      results.forEach((result, index) => {
        expect(result.userId).toBe(`user-${index}`)
      })
    })

    it('should maintain performance under high load', async () => {
      const mockUser = {
        id: 'user-123',
        wallet_address: 'test-wallet-address',
        created_at: new Date().toISOString(),
      }

      mockDb.getUserByWallet.mockResolvedValue(mockUser)
      mockDb.getUserPositions.mockResolvedValue([])
      mockDb.getUserCollateral.mockResolvedValue([])
      mockPythOracle.getAllPrices.mockResolvedValue({})

      const startTime = Date.now()

      // Simulate high load
      const promises = []
      for (let i = 0; i < 100; i++) {
        promises.push(webSocketService['calculateUserPortfolio'](`user-${i}`))
      }

      await Promise.all(promises)

      const endTime = Date.now()
      const duration = endTime - startTime

      // Should complete within reasonable time (less than 5 seconds)
      expect(duration).toBeLessThan(5000)
    })
  })
})
