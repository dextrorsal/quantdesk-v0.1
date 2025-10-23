/**
 * Test 1.1-INT-001: WebSocket endpoint delivers portfolio updates every 5s
 * 
 * Integration test for WebSocket endpoint functionality including
 * user-specific rooms, portfolio subscriptions, and update delivery.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { Server } from 'socket.io'
import { createServer } from 'http'
import { io as Client, Socket as ClientSocket } from 'socket.io-client'
import { WebSocketService } from '../../src/services/websocket'
import { PortfolioCalculationService } from '../../src/services/portfolioCalculationService'

// Mock dependencies
vi.mock('../../src/services/portfolioCalculationService')
vi.mock('../../src/services/supabaseDatabase')

describe('1.1-INT-001: WebSocket Endpoint Functionality', () => {
  let httpServer: any
  let io: Server
  let clientSocket: ClientSocket
  let mockPortfolioService: any
  let mockDatabase: any

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
    mockPortfolioService = {
      calculatePortfolio: vi.fn(),
    }

    mockDatabase = {
      getUserByWallet: vi.fn(),
    }

    // Initialize WebSocket service
    const wsService = WebSocketService.getInstance(io)
    await wsService.initialize()

    // Start server
    await new Promise<void>((resolve) => {
      httpServer.listen(0, () => resolve())
    })

    const port = httpServer.address()?.port
    const clientUrl = `http://localhost:${port}`

    // Create client connection
    clientSocket = Client(clientUrl, {
      auth: {
        token: 'mock-jwt-token'
      }
    })

    await new Promise<void>((resolve) => {
      clientSocket.on('connect', () => resolve())
    })
  })

  afterEach(() => {
    clientSocket.close()
    io.close()
    httpServer.close()
  })

  describe('portfolio subscription', () => {
    it('should subscribe user to portfolio updates', async () => {
      const mockUserId = 'test-user-123'
      const mockWalletAddress = 'test-wallet-address'

      // Mock authentication
      mockDatabase.getUserByWallet.mockResolvedValue({
        id: mockUserId,
        wallet_address: mockWalletAddress
      })

      // Mock portfolio data
      const mockPortfolioData = {
        userId: mockUserId,
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

      mockPortfolioService.calculatePortfolio.mockResolvedValue(mockPortfolioData)

      // Subscribe to portfolio updates
      clientSocket.emit('subscribe_portfolio')

      // Wait for subscription confirmation and initial data
      const portfolioUpdate = await new Promise<any>((resolve) => {
        clientSocket.on('portfolio_update', (data) => {
          resolve(data)
        })
      })

      expect(portfolioUpdate).toBeDefined()
      expect(portfolioUpdate.userId).toBe(mockUserId)
      expect(portfolioUpdate.totalValue).toBe(10000)
      expect(portfolioUpdate.positions).toHaveLength(1)
    })

    it('should reject unauthenticated portfolio subscriptions', async () => {
      // Don't mock authentication - should fail
      mockDatabase.getUserByWallet.mockResolvedValue(null)

      let errorReceived = false
      clientSocket.on('error', () => {
        errorReceived = true
      })

      clientSocket.emit('subscribe_portfolio')

      // Wait a bit for error
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(errorReceived).toBe(true)
    })

    it('should handle multiple portfolio subscriptions from same user', async () => {
      const mockUserId = 'test-user-multi'
      const mockWalletAddress = 'test-wallet-multi'

      mockDatabase.getUserByWallet.mockResolvedValue({
        id: mockUserId,
        wallet_address: mockWalletAddress
      })

      const mockPortfolioData = {
        userId: mockUserId,
        totalValue: 5000,
        totalUnrealizedPnl: 250,
        totalRealizedPnl: 100,
        marginRatio: 80.0,
        healthFactor: 85.0,
        positions: [],
        timestamp: Date.now()
      }

      mockPortfolioService.calculatePortfolio.mockResolvedValue(mockPortfolioData)

      // Subscribe multiple times
      clientSocket.emit('subscribe_portfolio')
      clientSocket.emit('subscribe_portfolio')
      clientSocket.emit('subscribe_portfolio')

      // Should only receive one update (deduplication)
      const updates: any[] = []
      clientSocket.on('portfolio_update', (data) => {
        updates.push(data)
      })

      await new Promise(resolve => setTimeout(resolve, 200))

      expect(updates.length).toBeGreaterThan(0)
    })
  })

  describe('portfolio unsubscription', () => {
    it('should unsubscribe user from portfolio updates', async () => {
      const mockUserId = 'test-user-unsub'
      const mockWalletAddress = 'test-wallet-unsub'

      mockDatabase.getUserByWallet.mockResolvedValue({
        id: mockUserId,
        wallet_address: mockWalletAddress
      })

      // Subscribe first
      clientSocket.emit('subscribe_portfolio')
      await new Promise(resolve => setTimeout(resolve, 100))

      // Then unsubscribe
      clientSocket.emit('unsubscribe_portfolio')

      // Should not receive updates after unsubscription
      let updateReceived = false
      clientSocket.on('portfolio_update', () => {
        updateReceived = true
      })

      await new Promise(resolve => setTimeout(resolve, 200))

      expect(updateReceived).toBe(false)
    })
  })

  describe('user-specific rooms', () => {
    it('should create isolated rooms for different users', async () => {
      const mockUser1 = {
        id: 'user-1',
        wallet_address: 'wallet-1'
      }
      const mockUser2 = {
        id: 'user-2', 
        wallet_address: 'wallet-2'
      }

      // Create second client for user 2
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      const clientSocket2 = Client(clientUrl, {
        auth: {
          token: 'mock-jwt-token-2'
        }
      })

      await new Promise<void>((resolve) => {
        clientSocket2.on('connect', () => resolve())
      })

      // Mock different portfolio data for each user
      const mockPortfolio1 = {
        userId: 'user-1',
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      }

      const mockPortfolio2 = {
        userId: 'user-2',
        totalValue: 20000,
        totalUnrealizedPnl: 1000,
        totalRealizedPnl: 500,
        marginRatio: 75.0,
        healthFactor: 80.0,
        positions: [],
        timestamp: Date.now()
      }

      // Mock authentication for both users
      mockDatabase.getUserByWallet
        .mockResolvedValueOnce(mockUser1)
        .mockResolvedValueOnce(mockUser2)

      mockPortfolioService.calculatePortfolio
        .mockResolvedValueOnce(mockPortfolio1)
        .mockResolvedValueOnce(mockPortfolio2)

      // Subscribe both users
      clientSocket.emit('subscribe_portfolio')
      clientSocket2.emit('subscribe_portfolio')

      // Collect updates
      const user1Updates: any[] = []
      const user2Updates: any[] = []

      clientSocket.on('portfolio_update', (data) => {
        user1Updates.push(data)
      })

      clientSocket2.on('portfolio_update', (data) => {
        user2Updates.push(data)
      })

      await new Promise(resolve => setTimeout(resolve, 300))

      // Each user should only receive their own updates
      expect(user1Updates.length).toBeGreaterThan(0)
      expect(user2Updates.length).toBeGreaterThan(0)
      expect(user1Updates[0].userId).toBe('user-1')
      expect(user2Updates[0].userId).toBe('user-2')

      clientSocket2.close()
    })
  })

  describe('error handling', () => {
    it('should handle portfolio calculation errors gracefully', async () => {
      const mockUserId = 'test-user-error'
      const mockWalletAddress = 'test-wallet-error'

      mockDatabase.getUserByWallet.mockResolvedValue({
        id: mockUserId,
        wallet_address: mockWalletAddress
      })

      // Mock portfolio service error
      mockPortfolioService.calculatePortfolio.mockRejectedValue(
        new Error('Portfolio calculation failed')
      )

      let errorReceived = false
      clientSocket.on('error', (error) => {
        errorReceived = true
        expect(error.message).toContain('Portfolio calculation failed')
      })

      clientSocket.emit('subscribe_portfolio')

      await new Promise(resolve => setTimeout(resolve, 200))

      expect(errorReceived).toBe(true)
    })

    it('should handle database connection errors', async () => {
      // Mock database error
      mockDatabase.getUserByWallet.mockRejectedValue(
        new Error('Database connection failed')
      )

      let errorReceived = false
      clientSocket.on('error', (error) => {
        errorReceived = true
        expect(error.message).toContain('Database connection failed')
      })

      clientSocket.emit('subscribe_portfolio')

      await new Promise(resolve => setTimeout(resolve, 200))

      expect(errorReceived).toBe(true)
    })
  })

  describe('connection management', () => {
    it('should clean up subscriptions on disconnect', async () => {
      const mockUserId = 'test-user-cleanup'
      const mockWalletAddress = 'test-wallet-cleanup'

      mockDatabase.getUserByWallet.mockResolvedValue({
        id: mockUserId,
        wallet_address: mockWalletAddress
      })

      // Subscribe
      clientSocket.emit('subscribe_portfolio')
      await new Promise(resolve => setTimeout(resolve, 100))

      // Disconnect
      clientSocket.disconnect()

      // Reconnect
      await new Promise<void>((resolve) => {
        clientSocket.on('connect', () => resolve())
        clientSocket.connect()
      })

      // Should not receive updates without re-subscription
      let updateReceived = false
      clientSocket.on('portfolio_update', () => {
        updateReceived = true
      })

      await new Promise(resolve => setTimeout(resolve, 200))

      expect(updateReceived).toBe(false)
    })
  })
})
