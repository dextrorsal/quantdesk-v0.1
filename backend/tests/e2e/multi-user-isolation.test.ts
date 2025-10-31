/**
 * Test 1.1-E2E-005: Multiple users receive isolated portfolio updates
 * 
 * End-to-end test for multi-user isolation including
 * user-specific rooms, data privacy, and security validation.
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

describe('1.1-E2E-005: Multi-user Isolation', () => {
  let httpServer: any
  let io: Server
  let wsService: WebSocketService
  let portfolioService: PortfolioCalculationService
  let backgroundService: PortfolioBackgroundService
  let oracleService: PythOracleService
  let mockDatabase: any
  let mockRedis: any

  const JWT_SECRET = 'test-secret-key'
  const mockUsers = [
    {
      id: 'user-1',
      wallet_address: 'wallet-1',
      email: 'user1@example.com'
    },
    {
      id: 'user-2',
      wallet_address: 'wallet-2',
      email: 'user2@example.com'
    },
    {
      id: 'user-3',
      wallet_address: 'wallet-3',
      email: 'user3@example.com'
    }
  ]

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
  })

  afterEach(() => {
    backgroundService.stop()
    io.close()
    httpServer.close()
  })

  describe('user isolation', () => {
    it('should isolate portfolio updates between different users', async () => {
      // Mock user data for each user
      mockUsers.forEach((user, index) => {
        mockDatabase.getUserByWallet.mockResolvedValueOnce(user)
        mockDatabase.getUserPositions.mockResolvedValueOnce([
          {
            id: `pos-${index}`,
            symbol: 'SOL',
            side: 'long',
            size: 10 + index,
            entry_price: 100,
            current_price: 110,
            margin: 1000 + index * 100,
            leverage: 10,
            status: 'open'
          }
        ])
        mockDatabase.getUserCollateral.mockResolvedValueOnce([
          { amount: 5000 + index * 1000, asset: 'USDC' }
        ])
      })

      mockDatabase.getLatestPrices.mockResolvedValue([
        { symbol: 'SOL', price: 110, timestamp: Date.now() },
        { symbol: 'USDC', price: 1, timestamp: Date.now() }
      ])

      // Mock different portfolio data for each user
      mockUsers.forEach((user, index) => {
        vi.spyOn(portfolioService, 'calculatePortfolio').mockResolvedValueOnce({
          userId: user.id,
          totalValue: 10000 + index * 5000,
          totalUnrealizedPnl: 500 + index * 250,
          totalRealizedPnl: 200 + index * 100,
          marginRatio: 85.5 - index * 5,
          healthFactor: 90.0 - index * 5,
          positions: [
            {
              id: `pos-${index}`,
              symbol: 'SOL',
              size: 10 + index,
              entryPrice: 100,
              currentPrice: 110,
              unrealizedPnl: 100 + index * 50,
              unrealizedPnlPercent: 10 + index * 5,
              margin: 1000 + index * 100,
              leverage: 10
            }
          ],
          timestamp: Date.now()
        })
      })

      // Create clients for each user
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      const clients: ClientSocket[] = []

      for (let i = 0; i < mockUsers.length; i++) {
        const validToken = jwt.sign(
          { 
            wallet_address: mockUsers[i].wallet_address,
            user_id: mockUsers[i].id
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

      // Subscribe all clients to portfolio updates
      clients.forEach(client => {
        client.emit('subscribe_portfolio')
      })

      // Collect updates from each client
      const userUpdates: { [key: string]: any[] } = {}
      clients.forEach((client, index) => {
        userUpdates[mockUsers[index].id] = []
        client.on('portfolio_update', (data) => {
          userUpdates[mockUsers[index].id].push({
            data,
            timestamp: Date.now()
          })
        })
      })

      // Start background service
      backgroundService.start()

      // Wait for updates
      await new Promise(resolve => setTimeout(resolve, 16000))

      // Verify each user received their own updates
      mockUsers.forEach(user => {
        expect(userUpdates[user.id].length).toBeGreaterThan(0)
        userUpdates[user.id].forEach(update => {
          expect(update.data.userId).toBe(user.id)
          expect(update.data.totalValue).toBe(10000 + (mockUsers.indexOf(user) * 5000))
        })
      })

      // Verify no cross-contamination
      mockUsers.forEach(user => {
        const userIndex = mockUsers.indexOf(user)
        userUpdates[user.id].forEach(update => {
          // Should not receive other users' data
          expect(update.data.userId).not.toBe(mockUsers[(userIndex + 1) % mockUsers.length].id)
          expect(update.data.userId).not.toBe(mockUsers[(userIndex + 2) % mockUsers.length].id)
        })
      })

      // Clean up
      clients.forEach(client => client.close())
    })

    it('should prevent users from accessing other users portfolio data', async () => {
      // Mock user data
      mockDatabase.getUserByWallet.mockResolvedValue(mockUsers[0])
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockDatabase.getLatestPrices.mockResolvedValue([])

      // Mock portfolio calculation
      vi.spyOn(portfolioService, 'calculatePortfolio').mockResolvedValue({
        userId: mockUsers[0].id,
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      })

      // Create client for user 1
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      const validToken = jwt.sign(
        { 
          wallet_address: mockUsers[0].wallet_address,
          user_id: mockUsers[0].id
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

      // Try to access another user's portfolio (should fail)
      let errorReceived = false
      client.on('error', (error) => {
        errorReceived = true
        expect(error.message).toContain('Unauthorized')
      })

      // Attempt to subscribe to another user's portfolio
      client.emit('subscribe_portfolio', { userId: mockUsers[1].id })

      await new Promise(resolve => setTimeout(resolve, 100))

      expect(errorReceived).toBe(true)

      client.close()
    })

    it('should handle users with different portfolio sizes', async () => {
      // Mock different portfolio sizes for each user
      const portfolioSizes = [
        { positions: 1, collateral: 5000, totalValue: 10000 },
        { positions: 3, collateral: 15000, totalValue: 25000 },
        { positions: 5, collateral: 25000, totalValue: 50000 }
      ]

      mockUsers.forEach((user, index) => {
        mockDatabase.getUserByWallet.mockResolvedValueOnce(user)
        
        // Mock different number of positions
        const positions = Array(portfolioSizes[index].positions).fill(null).map((_, posIndex) => ({
          id: `pos-${index}-${posIndex}`,
          symbol: 'SOL',
          side: 'long',
          size: 10,
          entry_price: 100,
          current_price: 110,
          margin: 1000,
          leverage: 10,
          status: 'open'
        }))
        
        mockDatabase.getUserPositions.mockResolvedValueOnce(positions)
        mockDatabase.getUserCollateral.mockResolvedValueOnce([
          { amount: portfolioSizes[index].collateral, asset: 'USDC' }
        ])
      })

      mockDatabase.getLatestPrices.mockResolvedValue([
        { symbol: 'SOL', price: 110, timestamp: Date.now() },
        { symbol: 'USDC', price: 1, timestamp: Date.now() }
      ])

      // Mock different portfolio data for each user
      mockUsers.forEach((user, index) => {
        const positions = Array(portfolioSizes[index].positions).fill(null).map((_, posIndex) => ({
          id: `pos-${index}-${posIndex}`,
          symbol: 'SOL',
          size: 10,
          entryPrice: 100,
          currentPrice: 110,
          unrealizedPnl: 100,
          unrealizedPnlPercent: 10,
          margin: 1000,
          leverage: 10
        }))

        vi.spyOn(portfolioService, 'calculatePortfolio').mockResolvedValueOnce({
          userId: user.id,
          totalValue: portfolioSizes[index].totalValue,
          totalUnrealizedPnl: 500 + index * 250,
          totalRealizedPnl: 200 + index * 100,
          marginRatio: 85.5 - index * 5,
          healthFactor: 90.0 - index * 5,
          positions,
          timestamp: Date.now()
        })
      })

      // Create clients for each user
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      const clients: ClientSocket[] = []

      for (let i = 0; i < mockUsers.length; i++) {
        const validToken = jwt.sign(
          { 
            wallet_address: mockUsers[i].wallet_address,
            user_id: mockUsers[i].id
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

      // Subscribe all clients to portfolio updates
      clients.forEach(client => {
        client.emit('subscribe_portfolio')
      })

      // Collect updates from each client
      const userUpdates: { [key: string]: any[] } = {}
      clients.forEach((client, index) => {
        userUpdates[mockUsers[index].id] = []
        client.on('portfolio_update', (data) => {
          userUpdates[mockUsers[index].id].push({
            data,
            timestamp: Date.now()
          })
        })
      })

      // Start background service
      backgroundService.start()

      // Wait for updates
      await new Promise(resolve => setTimeout(resolve, 16000))

      // Verify each user received their own portfolio data
      mockUsers.forEach((user, index) => {
        expect(userUpdates[user.id].length).toBeGreaterThan(0)
        userUpdates[user.id].forEach(update => {
          expect(update.data.userId).toBe(user.id)
          expect(update.data.totalValue).toBe(portfolioSizes[index].totalValue)
          expect(update.data.positions).toHaveLength(portfolioSizes[index].positions)
        })
      })

      // Clean up
      clients.forEach(client => client.close())
    })
  })

  describe('security validation', () => {
    it('should prevent unauthorized access to portfolio data', async () => {
      // Mock user data
      mockDatabase.getUserByWallet.mockResolvedValue(mockUsers[0])
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockDatabase.getLatestPrices.mockResolvedValue([])

      // Mock portfolio calculation
      vi.spyOn(portfolioService, 'calculatePortfolio').mockResolvedValue({
        userId: mockUsers[0].id,
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      })

      // Create client with invalid token
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      const invalidToken = 'invalid-jwt-token'

      const client = Client(clientUrl, {
        auth: {
          token: invalidToken
        }
      })

      let connectionError = false
      client.on('connect_error', () => {
        connectionError = true
      })

      // Wait for connection attempt
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(connectionError).toBe(true)

      client.close()
    })

    it('should handle users with expired tokens', async () => {
      // Mock user data
      mockDatabase.getUserByWallet.mockResolvedValue(mockUsers[0])
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockDatabase.getLatestPrices.mockResolvedValue([])

      // Mock portfolio calculation
      vi.spyOn(portfolioService, 'calculatePortfolio').mockResolvedValue({
        userId: mockUsers[0].id,
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      })

      // Create client with expired token
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      const expiredToken = jwt.sign(
        { 
          wallet_address: mockUsers[0].wallet_address,
          user_id: mockUsers[0].id
        },
        JWT_SECRET,
        { expiresIn: '-1h' } // Expired
      )

      const client = Client(clientUrl, {
        auth: {
          token: expiredToken
        }
      })

      let connectionError = false
      client.on('connect_error', () => {
        connectionError = true
      })

      // Wait for connection attempt
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(connectionError).toBe(true)

      client.close()
    })

    it('should prevent token hijacking attempts', async () => {
      // Mock user data
      mockDatabase.getUserByWallet.mockResolvedValue(mockUsers[0])
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockDatabase.getLatestPrices.mockResolvedValue([])

      // Mock portfolio calculation
      vi.spyOn(portfolioService, 'calculatePortfolio').mockResolvedValue({
        userId: mockUsers[0].id,
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      })

      // Create client with token for user 1
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      const user1Token = jwt.sign(
        { 
          wallet_address: mockUsers[0].wallet_address,
          user_id: mockUsers[0].id
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      const client = Client(clientUrl, {
        auth: {
          token: user1Token
        }
      })

      await new Promise<void>((resolve) => {
        client.on('connect', () => resolve())
      })

      // Try to modify the token to access user 2's data
      let errorReceived = false
      client.on('error', (error) => {
        errorReceived = true
        expect(error.message).toContain('Unauthorized')
      })

      // Attempt to subscribe to user 2's portfolio
      client.emit('subscribe_portfolio', { userId: mockUsers[1].id })

      await new Promise(resolve => setTimeout(resolve, 100))

      expect(errorReceived).toBe(true)

      client.close()
    })
  })

  describe('concurrent user handling', () => {
    it('should handle multiple users connecting simultaneously', async () => {
      // Mock user data for all users
      mockUsers.forEach((user, index) => {
        mockDatabase.getUserByWallet.mockResolvedValueOnce(user)
        mockDatabase.getUserPositions.mockResolvedValueOnce([])
        mockDatabase.getUserCollateral.mockResolvedValueOnce([])
      })

      mockDatabase.getLatestPrices.mockResolvedValue([])

      // Mock portfolio calculation for all users
      mockUsers.forEach((user, index) => {
        vi.spyOn(portfolioService, 'calculatePortfolio').mockResolvedValueOnce({
          userId: user.id,
          totalValue: 10000 + index * 5000,
          totalUnrealizedPnl: 500 + index * 250,
          totalRealizedPnl: 200 + index * 100,
          marginRatio: 85.5 - index * 5,
          healthFactor: 90.0 - index * 5,
          positions: [],
          timestamp: Date.now()
        })
      })

      // Create clients for all users simultaneously
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      const clients: ClientSocket[] = []

      const connectionPromises = mockUsers.map(async (user, index) => {
        const validToken = jwt.sign(
          { 
            wallet_address: user.wallet_address,
            user_id: user.id
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
        return client
      })

      await Promise.all(connectionPromises)

      // Subscribe all clients to portfolio updates
      clients.forEach(client => {
        client.emit('subscribe_portfolio')
      })

      // Collect updates from each client
      const userUpdates: { [key: string]: any[] } = {}
      clients.forEach((client, index) => {
        userUpdates[mockUsers[index].id] = []
        client.on('portfolio_update', (data) => {
          userUpdates[mockUsers[index].id].push({
            data,
            timestamp: Date.now()
          })
        })
      })

      // Start background service
      backgroundService.start()

      // Wait for updates
      await new Promise(resolve => setTimeout(resolve, 16000))

      // Verify all users received their own updates
      mockUsers.forEach(user => {
        expect(userUpdates[user.id].length).toBeGreaterThan(0)
        userUpdates[user.id].forEach(update => {
          expect(update.data.userId).toBe(user.id)
        })
      })

      // Clean up
      clients.forEach(client => client.close())
    })

    it('should handle users disconnecting and reconnecting', async () => {
      // Mock user data
      mockDatabase.getUserByWallet.mockResolvedValue(mockUsers[0])
      mockDatabase.getUserPositions.mockResolvedValue([])
      mockDatabase.getUserCollateral.mockResolvedValue([])
      mockDatabase.getLatestPrices.mockResolvedValue([])

      // Mock portfolio calculation
      vi.spyOn(portfolioService, 'calculatePortfolio').mockResolvedValue({
        userId: mockUsers[0].id,
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      })

      // Create client
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      const validToken = jwt.sign(
        { 
          wallet_address: mockUsers[0].wallet_address,
          user_id: mockUsers[0].id
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

      // Subscribe to portfolio updates
      client.emit('subscribe_portfolio')

      // Collect updates
      const updates: any[] = []
      client.on('portfolio_update', (data) => {
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
      client.disconnect()

      // Wait a bit
      await new Promise(resolve => setTimeout(resolve, 3000))

      // Reconnect
      const newClient = Client(clientUrl, {
        auth: {
          token: validToken
        }
      })

      await new Promise<void>((resolve) => {
        newClient.on('connect', () => resolve())
      })

      // Resubscribe
      newClient.emit('subscribe_portfolio')

      // Collect updates from new client
      const newUpdates: any[] = []
      newClient.on('portfolio_update', (data) => {
        newUpdates.push({
          data,
          timestamp: Date.now()
        })
      })

      // Wait for more updates
      await new Promise(resolve => setTimeout(resolve, 10000))

      // Should have received updates before and after reconnection
      expect(updates.length).toBeGreaterThan(0)
      expect(newUpdates.length).toBeGreaterThan(0)

      // Verify all updates are for the correct user
      [...updates, ...newUpdates].forEach(update => {
        expect(update.data.userId).toBe(mockUsers[0].id)
      })

      newClient.close()
    })
  })
})
