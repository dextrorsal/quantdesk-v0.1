import { vi } from 'vitest'

// Mock environment variables
process.env.NODE_ENV = 'test'
process.env.PORT = '3002'
process.env.SUPABASE_URL = 'http://localhost:54321'
process.env.SUPABASE_ANON_KEY = 'test-anon-key'
process.env.JWT_SECRET = 'test-jwt-secret'
process.env.REDIS_URL = 'redis://localhost:6379'
process.env.PYTH_RPC_ENDPOINT = 'https://api.devnet.solana.com'

// Mock console methods to reduce noise in tests
global.console = {
  ...console,
  log: vi.fn(),
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
}
