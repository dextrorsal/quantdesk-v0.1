/**
 * Test 1.1-INT-007: WebSocket endpoint /ws/portfolio with user-specific rooms
 * 
 * Integration test for WebSocket authentication including
 * JWT validation, user identification, and room management.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { Server } from 'socket.io'
import { createServer } from 'http'
import { io as Client, Socket as ClientSocket } from 'socket.io-client'
import { WebSocketService } from '../../src/services/websocket'
import jwt from 'jsonwebtoken'

// Mock dependencies
vi.mock('../../src/services/supabaseDatabase')
vi.mock('../../src/services/portfolioCalculationService')

describe('1.1-INT-007: WebSocket Authentication', () => {
  let httpServer: any
  let io: Server
  let clientSocket: ClientSocket
  let mockDatabase: any
  let mockPortfolioService: any

  const JWT_SECRET = 'test-secret-key'
  const mockUser = {
    id: 'test-user-123',
    wallet_address: 'test-wallet-address',
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
    }

    mockPortfolioService = {
      calculatePortfolio: vi.fn(),
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
  })

  afterEach(() => {
    if (clientSocket) {
      clientSocket.close()
    }
    io.close()
    httpServer.close()
  })

  describe('JWT authentication', () => {
    it('should authenticate user with valid JWT token', async () => {
      const validToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id 
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)

      // Create client with valid token
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      clientSocket = Client(clientUrl, {
        auth: {
          token: validToken
        }
      })

      await new Promise<void>((resolve) => {
        clientSocket.on('connect', () => resolve())
      })

      expect(clientSocket.connected).toBe(true)
    })

    it('should reject connection with invalid JWT token', async () => {
      const invalidToken = 'invalid-jwt-token'

      // Create client with invalid token
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      clientSocket = Client(clientUrl, {
        auth: {
          token: invalidToken
        }
      })

      let connectionError = false
      clientSocket.on('connect_error', () => {
        connectionError = true
      })

      // Wait for connection attempt
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(connectionError).toBe(true)
    })

    it('should reject connection with expired JWT token', async () => {
      const expiredToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id 
        },
        JWT_SECRET,
        { expiresIn: '-1h' } // Expired
      )

      // Create client with expired token
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      clientSocket = Client(clientUrl, {
        auth: {
          token: expiredToken
        }
      })

      let connectionError = false
      clientSocket.on('connect_error', () => {
        connectionError = true
      })

      // Wait for connection attempt
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(connectionError).toBe(true)
    })

    it('should reject connection without JWT token', async () => {
      // Create client without token
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      clientSocket = Client(clientUrl)

      let connectionError = false
      clientSocket.on('connect_error', () => {
        connectionError = true
      })

      // Wait for connection attempt
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(connectionError).toBe(true)
    })
  })

  describe('user identification', () => {
    it('should identify user from JWT token', async () => {
      const validToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id 
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)

      // Create client with valid token
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      clientSocket = Client(clientUrl, {
        auth: {
          token: validToken
        }
      })

      await new Promise<void>((resolve) => {
        clientSocket.on('connect', () => resolve())
      })

      // Verify user is identified
      expect(clientSocket.connected).toBe(true)
      
      // Test portfolio subscription (requires authentication)
      const mockPortfolioData = {
        userId: mockUser.id,
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      }

      mockPortfolioService.calculatePortfolio.mockResolvedValue(mockPortfolioData)

      clientSocket.emit('subscribe_portfolio')

      const portfolioUpdate = await new Promise<any>((resolve) => {
        clientSocket.on('portfolio_update', (data) => {
          resolve(data)
        })
      })

      expect(portfolioUpdate.userId).toBe(mockUser.id)
    })

    it('should handle user not found in database', async () => {
      const validToken = jwt.sign(
        { 
          wallet_address: 'non-existent-wallet',
          user_id: 'non-existent-user' 
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockResolvedValue(null)

      // Create client with valid token but non-existent user
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      clientSocket = Client(clientUrl, {
        auth: {
          token: validToken
        }
      })

      let connectionError = false
      clientSocket.on('connect_error', () => {
        connectionError = true
      })

      // Wait for connection attempt
      await new Promise(resolve => setTimeout(resolve, 100))

      expect(connectionError).toBe(true)
    })
  })

  describe('user-specific rooms', () => {
    it('should create isolated rooms for different users', async () => {
      const user1Token = jwt.sign(
        { 
          wallet_address: 'wallet-1',
          user_id: 'user-1' 
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      const user2Token = jwt.sign(
        { 
          wallet_address: 'wallet-2',
          user_id: 'user-2' 
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      const mockUser1 = { id: 'user-1', wallet_address: 'wallet-1' }
      const mockUser2 = { id: 'user-2', wallet_address: 'wallet-2' }

      mockDatabase.getUserByWallet
        .mockResolvedValueOnce(mockUser1)
        .mockResolvedValueOnce(mockUser2)

      // Create two clients
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      
      const clientSocket1 = Client(clientUrl, {
        auth: { token: user1Token }
      })
      
      const clientSocket2 = Client(clientUrl, {
        auth: { token: user2Token }
      })

      await Promise.all([
        new Promise<void>((resolve) => {
          clientSocket1.on('connect', () => resolve())
        }),
        new Promise<void>((resolve) => {
          clientSocket2.on('connect', () => resolve())
        })
      ])

      // Mock different portfolio data
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

      mockPortfolioService.calculatePortfolio
        .mockResolvedValueOnce(mockPortfolio1)
        .mockResolvedValueOnce(mockPortfolio2)

      // Subscribe both users
      clientSocket1.emit('subscribe_portfolio')
      clientSocket2.emit('subscribe_portfolio')

      // Collect updates
      const user1Updates: any[] = []
      const user2Updates: any[] = []

      clientSocket1.on('portfolio_update', (data) => {
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

      clientSocket1.close()
      clientSocket2.close()
    })

    it('should prevent cross-user data access', async () => {
      const user1Token = jwt.sign(
        { 
          wallet_address: 'wallet-1',
          user_id: 'user-1' 
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      const mockUser1 = { id: 'user-1', wallet_address: 'wallet-1' }
      mockDatabase.getUserByWallet.mockResolvedValue(mockUser1)

      // Create client
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      clientSocket = Client(clientUrl, {
        auth: { token: user1Token }
      })

      await new Promise<void>((resolve) => {
        clientSocket.on('connect', () => resolve())
      })

      // Try to access another user's data (should fail)
      let errorReceived = false
      clientSocket.on('error', (error) => {
        errorReceived = true
        expect(error.message).toContain('Unauthorized')
      })

      // Attempt to subscribe to another user's portfolio
      clientSocket.emit('subscribe_portfolio', { userId: 'user-2' })

      await new Promise(resolve => setTimeout(resolve, 100))

      expect(errorReceived).toBe(true)
    })
  })

  describe('authentication middleware', () => {
    it('should validate JWT token on every request', async () => {
      const validToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id 
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)

      // Create client
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      clientSocket = Client(clientUrl, {
        auth: { token: validToken }
      })

      await new Promise<void>((resolve) => {
        clientSocket.on('connect', () => resolve())
      })

      // Multiple portfolio subscriptions should all be authenticated
      clientSocket.emit('subscribe_portfolio')
      clientSocket.emit('subscribe_portfolio')
      clientSocket.emit('subscribe_portfolio')

      // Should not receive authentication errors
      let authError = false
      clientSocket.on('error', (error) => {
        if (error.message.includes('Not authenticated')) {
          authError = true
        }
      })

      await new Promise(resolve => setTimeout(resolve, 200))

      expect(authError).toBe(false)
    })

    it('should handle malformed JWT tokens', async () => {
      const malformedToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.malformed'

      // Create client with malformed token
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      clientSocket = Client(clientUrl, {
        auth: { token: malformedToken }
      })

      let connectionError = false
      clientSocket.on('connect_error', () => {
        connectionError = true
      })

      await new Promise(resolve => setTimeout(resolve, 100))

      expect(connectionError).toBe(true)
    })
  })

  describe('session management', () => {
    it('should handle token refresh during connection', async () => {
      const validToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id 
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)

      // Create client
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      clientSocket = Client(clientUrl, {
        auth: { token: validToken }
      })

      await new Promise<void>((resolve) => {
        clientSocket.on('connect', () => resolve())
      })

      // Connection should remain stable
      expect(clientSocket.connected).toBe(true)

      // Test portfolio subscription
      const mockPortfolioData = {
        userId: mockUser.id,
        totalValue: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85.5,
        healthFactor: 90.0,
        positions: [],
        timestamp: Date.now()
      }

      mockPortfolioService.calculatePortfolio.mockResolvedValue(mockPortfolioData)

      clientSocket.emit('subscribe_portfolio')

      const portfolioUpdate = await new Promise<any>((resolve) => {
        clientSocket.on('portfolio_update', (data) => {
          resolve(data)
        })
      })

      expect(portfolioUpdate.userId).toBe(mockUser.id)
    })

    it('should clean up user sessions on disconnect', async () => {
      const validToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id 
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)

      // Create client
      const port = httpServer.address()?.port
      const clientUrl = `http://localhost:${port}`
      clientSocket = Client(clientUrl, {
        auth: { token: validToken }
      })

      await new Promise<void>((resolve) => {
        clientSocket.on('connect', () => resolve())
      })

      // Subscribe to portfolio
      clientSocket.emit('subscribe_portfolio')
      await new Promise(resolve => setTimeout(resolve, 100))

      // Disconnect
      clientSocket.disconnect()

      // Reconnect with same token
      await new Promise<void>((resolve) => {
        clientSocket.on('connect', () => resolve())
        clientSocket.connect()
      })

      // Should need to resubscribe
      let updateReceived = false
      clientSocket.on('portfolio_update', () => {
        updateReceived = true
      })

      await new Promise(resolve => setTimeout(resolve, 200))

      expect(updateReceived).toBe(false)
    })
  })
})
