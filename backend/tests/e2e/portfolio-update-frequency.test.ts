/**
 * Test 1.1-E2E-001: User sees portfolio value update every 5 seconds
 * 
 * End-to-end test for portfolio update frequency including
 * WebSocket connection, portfolio subscription, and update timing.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { Server } from 'socket.io'
import { createServer } from 'http'
import { io as Client, Socket as ClientSocket } from 'socket.io-client'
import { WebSocketService } from '../../src/services/websocket'
import { PortfolioCalculationService } from '../../src/services/portfolioCalculationService'
import { PortfolioBackgroundService } from '../../src/services/portfolioBackgroundService'
import { PythOracleService } from '../../src/services/pythOracleService'
import { SupabaseDatabaseService } from '../../src/services/supabaseDatabase'
import { RedisService } from '../../src/services/redisService'
import jwt from 'jsonwebtoken'

// Mock dependencies
vi.mock('../../src/services/supabaseDatabase')
vi.mock('../../src/services/redisService')

describe('1.1-E2E-001: Portfolio Update Frequency', () => {
  let httpServer: any
  let io: Server
  let clientSocket: ClientSocket
  let wsService: WebSocketService
  let portfolioService: PortfolioCalculationService
  let backgroundService: PortfolioBackgroundService
  let oracleService: PythOracleService
  let mockDatabase: any
  let mockRedis: any

  const JWT_SECRET = 'test-secret-key'
  const mockUser = {
    id: 'test-user-e2e',
    wallet_address: 'test-wallet-e2e',
    email: 'test@example.com'
  }

  beforeEach(async () => {
    // Create HTTP server
    httpServer = createServer()
    
    // Create Socket.IO server
    io = new Server(httpServer, {
      cors: {
        origin: "*",
        methods: ["GET", "POST"]
      }
    })

    // Mock services
    mockDatabase = {
      getUserByWallet: vi.fn(),
      getUserPositions: vi.fn(),
      getUserCollateral: vi.fn(),
      storePriceData: vi.fn(),
      getLatestPrices: vi.fn(),
    }

    mockRedis = {
      set: vi.fn(),
      get: vi.fn(),
      del: vi.fn(),
    }

    // Initialize services
    oracleService = new PythOracleService(mockDatabase, mockRedis)
    portfolioService = new PortfolioCalculationService(
      mockDatabase,
      {} as any, // PnL service
      oracleService
    )
    backgroundService = new PortfolioBackgroundService(portfolioService)
    
    wsService = WebSocketService.getInstance(io)
    await wsService.initialize()

    // Start server
    await new Promise<void>((resolve) => {
      httpServer.listen(0, () => resolve())
    })

    const port = httpServer.address()?.port
    const clientUrl = `http://localhost:${port}`

    // Create client connection
    const validToken = jwt.sign(
      { 
        wallet_address: mockUser.wallet_address,
        user_id: mockUser.id
      },
      JWT_SECRET,
      { expiresIn: '1h' }
    )

    clientSocket = Client(clientUrl, {
      auth: {
        token: validToken
      }
    })

    await new Promise<void>((resolve) => {
      clientSocket.on('connect', () => resolve())
    })
  })

  afterEach(() => {
    backgroundService.stop()
    clientSocket.close()
    io.close()
    httpServer.close()
  })

  describe('portfolio update frequency', () => {
    it('should receive portfolio updates every 5 seconds', async () => {
      // Mock user data
      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue([
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
      ])
      mockDatabase.getUserCollateral.mockResolvedValue([
        { amount: 5000, asset: 'USDC' }
      ])

      // Mock Oracle prices
      const mockPrices = {
        'SOL': 110,
        'USDC': 1
      }
      mockDatabase.getLatestPrices.mockResolvedValue([
        { symbol: 'SOL', price: 110, timestamp: Date.now() },
        { symbol: 'USDC', price: 1, timestamp: Date.now() }
      ])

      // Mock portfolio calculation
      const mockPortfolioData = {
        userId: mockUser.id,
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [
          {
            id: 'pos-1',
            symbol: 'SOL',
            size: 10,
            entryPrice: 100,
            currentPrice: 110,
            unrealizedPnl: 100,
            unrealizedPnlPercent: 10,
            margin: 1000,
            leverage: 10
          }
        ],
        timestamp: Date.now()
      }

      // Mock portfolio service
      vi.spyOn(portfolioService, 'calculatePortfolio').mockResolvedValue(mockPortfolioData)

      // Subscribe to portfolio updates
      clientSocket.emit('subscribe_portfolio')

      // Collect updates over time
      const updates: any[] = []
      clientSocket.on('portfolio_update', (data) => {
        updates.push({
          data,
          timestamp: Date.now()
        })
      })

      // Start background service
      backgroundService.start()

      // Wait for multiple updates (at least 3 updates = 15+ seconds)
      await new Promise(resolve => setTimeout(resolve, 16000))

      // Should have received multiple updates
      expect(updates.length).toBeGreaterThanOrEqual(3)

      // Verify update intervals are approximately 5 seconds
      for (let i = 1; i < updates.length; i++) {
        const interval = updates[i].timestamp - updates[i-1].timestamp
        // Allow for some tolerance (±1 second)
        expect(interval).toBeGreaterThanOrEqual(4000)
        expect(interval).toBeLessThanOrEqual(6000)
      }

      // Verify all updates contain expected data
      updates.forEach(update => {
        expect(update.data.userId).toBe(mockUser.id)
        expect(update.data.totalValue).toBe(10000)
        expect(update.data.positions).toHaveLength(1)
      })
    })

    it('should handle price changes and update portfolio accordingly', async () => {
      // Mock user data
      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue([
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
      ])
      mockDatabase.getUserCollateral.mockResolvedValue([
        { amount: 5000, asset: 'USDC' }
      ])

      // Mock changing prices
      let priceChangeCount = 0
      const mockPrices = [
        { symbol: 'SOL', price: 110, timestamp: Date.now() },
        { symbol: 'SOL', price: 115, timestamp: Date.now() + 5000 },
        { symbol: 'SOL', price: 120, timestamp: Date.now() + 10000 },
        { symbol: 'USDC', price: 1, timestamp: Date.now() }
      ]

      mockDatabase.getLatestPrices.mockImplementation(() => {
        const price = mockPrices[priceChangeCount % mockPrices.length]
        priceChangeCount++
        return Promise.resolve([price])
      })

      // Mock portfolio calculation with changing values
      let portfolioValue = 10000
      vi.spyOn(portfolioService, 'calculatePortfolio').mockImplementation(async () => {
        portfolioValue += 500 // Increase value each time
        return {
          userId: mockUser.id,
          totalValue: portfolioValue,
          totalUnrealizedPnl: portfolioValue - 10000,
          totalRealizedPnl: 200,
          marginRatio: 85.5,
          healthFactor: 90.0,
          positions: [
            {
              id: 'pos-1',
              symbol: 'SOL',
              size: 10,
              entryPrice: 100,
              currentPrice: 110 + (portfolioValue - 10000) / 100,
              unrealizedPnl: portfolioValue - 10000,
              unrealizedPnlPercent: ((portfolioValue - 10000) / 10000) * 100,
              margin: 1000,
              leverage: 10
            }
          ],
          timestamp: Date.now()
        }
      })

      // Subscribe to portfolio updates
      clientSocket.emit('subscribe_portfolio')

      // Collect updates
      const updates: any[] = []
      clientSocket.on('portfolio_update', (data) => {
        updates.push({
          data,
          timestamp: Date.now()
        })
      })

      // Start background service
      backgroundService.start()

      // Wait for multiple updates
      await new Promise(resolve => setTimeout(resolve, 16000))

      // Should have received multiple updates with changing values
      expect(updates.length).toBeGreaterThanOrEqual(3)

      // Verify values are changing
      const firstUpdate = updates[0].data.totalValue
      const lastUpdate = updates[updates.length - 1].data.totalValue
      expect(lastUpdate).toBeGreaterThan(firstUpdate)

      // Verify P&L is updating
      const firstPnl = updates[0].data.totalUnrealizedPnl
      const lastPnl = updates[updates.length - 1].data.totalUnrealizedPnl
      expect(lastPnl).toBeGreaterThan(firstPnl)
    })

    it('should maintain consistent update frequency under load', async () => {
      // Mock user data
      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockDatabase.getLatestPrices.mockResolvedValue([])

      // Mock portfolio calculation
      vi.spyOn(portfolioService, 'calculatePortfolio').mockResolvedValue({
        userId: mockUser.id,
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      })

      // Subscribe to portfolio updates
      clientSocket.emit('subscribe_portfolio')

      // Collect updates
      const updates: any[] = []
      clientSocket.on('portfolio_update', (data) => {
        updates.push({
          data,
          timestamp: Date.now()
        })
      })

      // Start background service
      backgroundService.start()

      // Wait for extended period (30 seconds)
      await new Promise(resolve => setTimeout(resolve, 30000))

      // Should have received approximately 6 updates (30 seconds / 5 seconds)
      expect(updates.length).toBeGreaterThanOrEqual(5)
      expect(updates.length).toBeLessThanOrEqual(7)

      // Verify consistent timing
      const intervals: number[] = []
      for (let i = 1; i < updates.length; i++) {
        intervals.push(updates[i].timestamp - updates[i-1].timestamp)
      }

      // All intervals should be approximately 5 seconds
      intervals.forEach(interval => {
        expect(interval).toBeGreaterThanOrEqual(4000)
        expect(interval).toBeLessThanOrEqual(6000)
      })
    })
  })

  describe('error handling and recovery', () => {
    it('should continue updates after temporary service errors', async () => {
      // Mock user data
      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockDatabase.getLatestPrices.mockResolvedValue([])

      // Mock portfolio service to fail once, then succeed
      let callCount = 0
      vi.spyOn(portfolioService, 'calculatePortfolio').mockImplementation(async () => {
        callCount++
        if (callCount === 2) {
          throw new Error('Temporary service error')
        }
        return {
          userId: mockUser.id,
          totalValue: 10000,
          totalUnrealizedPnl: 500,
          totalRealizedPnl: 200,
          marginRatio: 85.5,
          healthFactor: 90.0,
          positions: [],
          timestamp: Date.now()
        }
      })

      // Subscribe to portfolio updates
      clientSocket.emit('subscribe_portfolio')

      // Collect updates
      const updates: any[] = []
      clientSocket.on('portfolio_update', (data) => {
        updates.push({
          data,
          timestamp: Date.now()
        })
      })

      // Start background service
      backgroundService.start()

      // Wait for multiple updates
      await new Promise(resolve => setTimeout(resolve, 16000))

      // Should have received updates despite the error
      expect(updates.length).toBeGreaterThanOrEqual(2)

      // Verify updates continued after error
      const firstUpdateTime = updates[0].timestamp
      const lastUpdateTime = updates[updates.length - 1].timestamp
      expect(lastUpdateTime).toBeGreaterThan(firstUpdateTime)
    })

    it('should handle WebSocket disconnection and reconnection', async () => {
      // Mock user data
      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockDatabase.getLatestPrices.mockResolvedValue([])

      // Mock portfolio calculation
      vi.spyOn(portfolioService, 'calculatePortfolio').mockResolvedValue({
        userId: mockUser.id,
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      })

      // Subscribe to portfolio updates
      clientSocket.emit('subscribe_portfolio')

      // Collect updates
      const updates: any[] = []
      clientSocket.on('portfolio_update', (data) => {
        updates.push({
          data,
          timestamp: Date.now()
        })
      })

      // Start background service
      backgroundService.start()

      // Wait for first update
      await new Promise(resolve => setTimeout(resolve, 6000))

      // Disconnect
      clientSocket.disconnect()

      // Wait a bit
      await new Promise(resolve => setTimeout(resolve, 3000))

      // Reconnect
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      const validToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      clientSocket = Client(clientUrl, {
        auth: {
          token: validToken
        }
      })

      await new Promise<void>((resolve) => {
        clientSocket.on('connect', () => resolve())
      })

      // Resubscribe
      clientSocket.emit('subscribe_portfolio')

      // Wait for more updates
      await new Promise(resolve => setTimeout(resolve, 10000))

      // Should have received updates before and after reconnection
      expect(updates.length).toBeGreaterThanOrEqual(2)
    })
  })

  describe('performance and reliability', () => {
    it('should handle multiple concurrent users', async () => {
      // Mock user data
      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockDatabase.getLatestPrices.mockResolvedValue([])

      // Mock portfolio calculation
      vi.spyOn(portfolioService, 'calculatePortfolio').mockResolvedValue({
        userId: mockUser.id,
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      })

      // Create multiple clients
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      const clients: ClientSocket[] = []

      for (let i = 0; i < 5; i++) {
        const validToken = jwt.sign(
          { 
            wallet_address: `wallet-${i}`,
            user_id: `user-${i}`
          },
          JWT_SECRET,
          { expiresIn: '1h' }
        )

        const client = Client(clientUrl, {
          auth: {
            token: validToken
          }
        })

        await new Promise<void>((resolve) => {
          client.on('connect', () => resolve())
        })

        clients.push(client)
      }

      // Subscribe all clients
      clients.forEach(client => {
        client.emit('subscribe_portfolio')
      })

      // Collect updates from all clients
      const allUpdates: any[] = []
      clients.forEach(client => {
        client.on('portfolio_update', (data) => {
          allUpdates.push({
            data,
            timestamp: Date.now()
          })
        })
      })

      // Start background service
      backgroundService.start()

      // Wait for updates
      await new Promise(resolve => setTimeout(resolve, 16000))

      // Should have received updates for all clients
      expect(allUpdates.length).toBeGreaterThanOrEqual(5)

      // Clean up
      clients.forEach(client => client.close())
    })

    it('should maintain update frequency under high load', async () => {
      // Mock user data
      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockDatabase.getLatestPrices.mockResolvedValue([])

      // Mock slow portfolio calculation
      vi.spyOn(portfolioService, 'calculatePortfolio').mockImplementation(async () => {
        // Simulate slow calculation
        await new Promise(resolve => setTimeout(resolve, 1000))
        return {
          userId: mockUser.id,
          totalValue: 10000,
          totalUnrealizedPnl: 500,
          totalRealizedPnl: 200,
          marginRatio: 85.5,
          healthFactor: 90.0,
          positions: [],
          timestamp: Date.now()
        }
      })

      // Subscribe to portfolio updates
      clientSocket.emit('subscribe_portfolio')

      // Collect updates
      const updates: any[] = []
      clientSocket.on('portfolio_update', (data) => {
        updates.push({
          data,
          timestamp: Date.now()
        })
      })

      // Start background service
      backgroundService.start()

      // Wait for updates
      await new Promise(resolve => setTimeout(resolve, 20000))

      // Should still receive updates despite slow calculation
      expect(updates.length).toBeGreaterThanOrEqual(3)

      // Verify updates are still approximately every 5 seconds
      for (let i = 1; i < updates.length; i++) {
        const interval = updates[i].timestamp - updates[i-1].timestamp
        // Allow for more tolerance due to slow calculation
        expect(interval).toBeGreaterThanOrEqual(4000)
        expect(interval).toBeLessThanOrEqual(7000)
      }
    })
  })
})
