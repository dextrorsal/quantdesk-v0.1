/**
 * Test Setup Configuration
 * 
 * Global test setup for Story 1.1 - Real-time Portfolio Updates
 * Includes mocks, utilities, and test environment configuration.
 */

import { beforeAll, afterAll, beforeEach, afterEach, vi } from 'vitest'

// Global test timeout
vi.setConfig({
  testTimeout: 30000,
  hookTimeout: 30000,
  teardownTimeout: 30000
})

// Mock console methods to reduce noise in tests
const originalConsole = console
beforeAll(() => {
  global.console = {
    ...originalConsole,
    log: vi.fn(),
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn()
  }
})

afterAll(() => {
  global.console = originalConsole
})

// Global mocks
beforeEach(() => {
  // Mock Date.now() for consistent timestamps in tests
  vi.useFakeTimers()
  vi.setSystemTime(new Date('2025-10-01T00:00:00Z'))
})

afterEach(() => {
  vi.useRealTimers()
  vi.clearAllMocks()
})

// Mock environment variables
process.env.NODE_ENV = 'test'
process.env.JWT_SECRET = 'test-secret-key'
process.env.REDIS_URL = 'redis://localhost:6379'
process.env.DATABASE_URL = 'postgresql://test:test@localhost:5432/test'
process.env.PYTH_NETWORK_URL = 'https://hermes.pyth.network'

// Mock external dependencies
vi.mock('socket.io', () => ({
  Server: vi.fn().mockImplementation(() => ({
    on: vi.fn(),
    emit: vi.fn(),
    to: vi.fn().mockReturnThis(),
    close: vi.fn(),
    listen: vi.fn()
  })),
  Socket: vi.fn()
}))

vi.mock('socket.io-client', () => ({
  io: vi.fn().mockImplementation(() => ({
    on: vi.fn(),
    emit: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
    close: vi.fn(),
    connected: true
  }))
}))

// Test utilities
export const createMockUser = (overrides = {}) => ({
  id: 'test-user-123',
  wallet_address: 'test-wallet-address',
  email: 'test@example.com',
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  ...overrides
})

export const createMockPosition = (overrides = {}) => ({
  id: 'pos-123',
  user_id: 'test-user-123',
  symbol: 'SOL',
  side: 'long',
  size: 10,
  entry_price: 100,
  current_price: 110,
  margin: 1000,
  leverage: 10,
  status: 'open',
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  ...overrides
})

export const createMockPortfolioData = (overrides = {}) => ({
  userId: 'test-user-123',
  totalValue: 10000,
  totalUnrealizedPnl: 500,
  totalRealizedPnl: 200,
  marginRatio: 85.5,
  healthFactor: 90.0,
  positions: [
    {
      id: 'pos-123',
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
  timestamp: Date.now(),
  ...overrides
})

export const createMockPriceData = (overrides = {}) => ({
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
  },
  ...overrides
})

// Test database utilities
export const mockDatabaseResponses = {
  getUserByWallet: vi.fn(),
  getUserPositions: vi.fn(),
  getUserCollateral: vi.fn(),
  storePriceData: vi.fn(),
  getLatestPrices: vi.fn(),
  createUser: vi.fn(),
  updateUser: vi.fn(),
  deleteUser: vi.fn()
}

// Test Redis utilities
export const mockRedisResponses = {
  set: vi.fn(),
  get: vi.fn(),
  del: vi.fn(),
  exists: vi.fn(),
  expire: vi.fn(),
  ttl: vi.fn()
}

// Test WebSocket utilities
export const createMockSocket = (overrides = {}) => ({
  id: 'socket-123',
  data: {
    userId: 'test-user-123',
    walletAddress: 'test-wallet-address'
  },
  on: vi.fn(),
  emit: vi.fn(),
  join: vi.fn(),
  leave: vi.fn(),
  disconnect: vi.fn(),
  ...overrides
})

// Test JWT utilities
export const createMockJWT = (payload = {}, secret = 'test-secret-key') => {
  const jwt = require('jsonwebtoken')
  return jwt.sign(payload, secret, { expiresIn: '1h' })
}

// Test timing utilities
export const waitFor = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

export const waitForCondition = async (
  condition: () => boolean,
  timeout = 5000,
  interval = 100
) => {
  const startTime = Date.now()
  
  while (Date.now() - startTime < timeout) {
    if (condition()) {
      return true
    }
    await waitFor(interval)
  }
  
  throw new Error(`Condition not met within ${timeout}ms`)
}

// Test data generators
export const generateTestUsers = (count: number) => {
  return Array.from({ length: count }, (_, index) => 
    createMockUser({
      id: `user-${index}`,
      wallet_address: `wallet-${index}`,
      email: `user${index}@example.com`
    })
  )
}

export const generateTestPositions = (userId: string, count: number) => {
  return Array.from({ length: count }, (_, index) => 
    createMockPosition({
      id: `pos-${index}`,
      user_id: userId,
      symbol: ['SOL', 'BTC', 'ETH'][index % 3],
      size: 10 + index,
      entry_price: 100 + index * 10,
      current_price: 110 + index * 10
    })
  )
}

// Test assertions
export const expectPortfolioUpdate = (update: any, expectedUserId: string) => {
  expect(update).toBeDefined()
  expect(update.userId).toBe(expectedUserId)
  expect(update.totalValue).toBeGreaterThan(0)
  expect(update.positions).toBeDefined()
  expect(Array.isArray(update.positions)).toBe(true)
  expect(update.timestamp).toBeDefined()
  expect(typeof update.timestamp).toBe('number')
}

export const expectWebSocketMessage = (message: any, expectedType: string) => {
  expect(message).toBeDefined()
  expect(message.type).toBe(expectedType)
  expect(message.data).toBeDefined()
  expect(message.timestamp).toBeDefined()
}

// Test cleanup utilities
export const cleanupTestData = async () => {
  // Clear all mocks
  vi.clearAllMocks()
  
  // Reset timers
  vi.useRealTimers()
  
  // Clear any test data
  // This would be implemented based on your test database setup
}

// Export test configuration
export const testConfig = {
  timeout: 30000,
  retries: 3,
  parallel: true,
  coverage: {
    enabled: true,
    threshold: 80
  }
}
