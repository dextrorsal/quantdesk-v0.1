/**
 * Test 1.1-INT-008: JWT validation for WebSocket connections
 * 
 * Integration test for JWT token validation including
 * token parsing, signature verification, and expiration handling.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import jwt from 'jsonwebtoken'
import { JWTService } from '../../src/services/jwtService'

// Mock dependencies
vi.mock('../../src/services/supabaseDatabase')

describe('1.1-INT-008: JWT Validation', () => {
  let jwtService: JWTService
  let mockDatabase: any

  const JWT_SECRET = 'test-secret-key'
  const mockUser = {
    id: 'test-user-123',
    wallet_address: 'test-wallet-address',
    email: 'test@example.com'
  }

  beforeEach(() => {
    vi.clearAllMocks()

    // Mock Database Service
    mockDatabase = {
      getUserByWallet: vi.fn(),
    }

    // Create JWT service instance
    jwtService = new JWTService(mockDatabase, JWT_SECRET)
  })

  describe('token validation', () => {
    it('should validate valid JWT token', async () => {
      const validToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id,
          email: mockUser.email
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)

      const result = await jwtService.validateToken(validToken)

      expect(result).toBeDefined()
      expect(result?.user_id).toBe(mockUser.id)
      expect(result?.wallet_address).toBe(mockUser.wallet_address)
      expect(result?.email).toBe(mockUser.email)
    })

    it('should reject invalid JWT token', async () => {
      const invalidToken = 'invalid-jwt-token'

      const result = await jwtService.validateToken(invalidToken)

      expect(result).toBeNull()
    })

    it('should reject JWT token with wrong signature', async () => {
      const wrongSecretToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id
        },
        'wrong-secret-key',
        { expiresIn: '1h' }
      )

      const result = await jwtService.validateToken(wrongSecretToken)

      expect(result).toBeNull()
    })

    it('should reject expired JWT token', async () => {
      const expiredToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id
        },
        JWT_SECRET,
        { expiresIn: '-1h' } // Expired
      )

      const result = await jwtService.validateToken(expiredToken)

      expect(result).toBeNull()
    })

    it('should reject JWT token without required claims', async () => {
      const incompleteToken = jwt.sign(
        { 
          // Missing wallet_address and user_id
          email: mockUser.email
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      const result = await jwtService.validateToken(incompleteToken)

      expect(result).toBeNull()
    })
  })

  describe('token parsing', () => {
    it('should parse JWT token correctly', () => {
      const tokenPayload = {
        wallet_address: mockUser.wallet_address,
        user_id: mockUser.id,
        email: mockUser.email,
        iat: Math.floor(Date.now() / 1000),
        exp: Math.floor(Date.now() / 1000) + 3600
      }

      const token = jwt.sign(tokenPayload, JWT_SECRET)

      const parsed = jwtService.parseToken(token)

      expect(parsed).toBeDefined()
      expect(parsed?.wallet_address).toBe(mockUser.wallet_address)
      expect(parsed?.user_id).toBe(mockUser.id)
      expect(parsed?.email).toBe(mockUser.email)
    })

    it('should handle malformed JWT tokens', () => {
      const malformedTokens = [
        'not.a.jwt',
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.malformed',
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.malformed',
        '',
        null,
        undefined
      ]

      malformedTokens.forEach(token => {
        const parsed = jwtService.parseToken(token as any)
        expect(parsed).toBeNull()
      })
    })

    it('should handle JWT tokens with wrong algorithm', async () => {
      const wrongAlgToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id
        },
        JWT_SECRET,
        { 
          expiresIn: '1h',
          algorithm: 'HS512' // Different algorithm
        }
      )

      const result = await jwtService.validateToken(wrongAlgToken)

      expect(result).toBeNull()
    })
  })

  describe('user verification', () => {
    it('should verify user exists in database', async () => {
      const validToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)

      const result = await jwtService.validateToken(validToken)

      expect(result).toBeDefined()
      expect(mockDatabase.getUserByWallet).toHaveBeenCalledWith(mockUser.wallet_address)
    })

    it('should reject token for non-existent user', async () => {
      const validToken = jwt.sign(
        { 
          wallet_address: 'non-existent-wallet',
          user_id: 'non-existent-user'
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockResolvedValue(null)

      const result = await jwtService.validateToken(validToken)

      expect(result).toBeNull()
    })

    it('should handle database errors during user verification', async () => {
      const validToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockRejectedValue(
        new Error('Database connection failed')
      )

      const result = await jwtService.validateToken(validToken)

      expect(result).toBeNull()
    })
  })

  describe('token generation', () => {
    it('should generate valid JWT token for user', () => {
      const token = jwtService.generateToken(mockUser)

      expect(token).toBeDefined()
      expect(typeof token).toBe('string')

      // Verify token can be parsed
      const parsed = jwtService.parseToken(token)
      expect(parsed).toBeDefined()
      expect(parsed?.wallet_address).toBe(mockUser.wallet_address)
      expect(parsed?.user_id).toBe(mockUser.id)
    })

    it('should generate token with correct expiration', () => {
      const token = jwtService.generateToken(mockUser)
      const parsed = jwtService.parseToken(token)

      expect(parsed?.exp).toBeDefined()
      expect(parsed?.exp).toBeGreaterThan(Math.floor(Date.now() / 1000))
    })

    it('should generate token with correct issued at time', () => {
      const token = jwtService.generateToken(mockUser)
      const parsed = jwtService.parseToken(token)

      expect(parsed?.iat).toBeDefined()
      expect(parsed?.iat).toBeLessThanOrEqual(Math.floor(Date.now() / 1000))
    })
  })

  describe('token refresh', () => {
    it('should refresh valid token', async () => {
      const validToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)

      const newToken = await jwtService.refreshToken(validToken)

      expect(newToken).toBeDefined()
      expect(newToken).not.toBe(validToken)

      // Verify new token is valid
      const result = await jwtService.validateToken(newToken)
      expect(result).toBeDefined()
    })

    it('should reject refresh for invalid token', async () => {
      const invalidToken = 'invalid-token'

      const newToken = await jwtService.refreshToken(invalidToken)

      expect(newToken).toBeNull()
    })

    it('should reject refresh for expired token', async () => {
      const expiredToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id
        },
        JWT_SECRET,
        { expiresIn: '-1h' } // Expired
      )

      const newToken = await jwtService.refreshToken(expiredToken)

      expect(newToken).toBeNull()
    })
  })

  describe('security considerations', () => {
    it('should handle token with suspicious claims', async () => {
      const suspiciousToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id,
          admin: true, // Suspicious claim
          role: 'admin' // Suspicious claim
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)

      const result = await jwtService.validateToken(suspiciousToken)

      // Should still validate but ignore suspicious claims
      expect(result).toBeDefined()
      expect(result?.admin).toBeUndefined()
      expect(result?.role).toBeUndefined()
    })

    it('should handle token with very long expiration', async () => {
      const longExpToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id
        },
        JWT_SECRET,
        { expiresIn: '365d' } // 1 year
      )

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)

      const result = await jwtService.validateToken(longExpToken)

      // Should validate but consider security implications
      expect(result).toBeDefined()
    })

    it('should handle token with very short expiration', async () => {
      const shortExpToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id
        },
        JWT_SECRET,
        { expiresIn: '1s' } // 1 second
      )

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)

      const result = await jwtService.validateToken(shortExpToken)

      expect(result).toBeDefined()

      // Wait for expiration
      await new Promise(resolve => setTimeout(resolve, 1100))

      const expiredResult = await jwtService.validateToken(shortExpToken)
      expect(expiredResult).toBeNull()
    })
  })

  describe('error handling', () => {
    it('should handle JWT library errors gracefully', async () => {
      // Mock jwt.verify to throw error
      const originalVerify = jwt.verify
      vi.spyOn(jwt, 'verify').mockImplementation(() => {
        throw new Error('JWT library error')
      })

      const result = await jwtService.validateToken('any-token')

      expect(result).toBeNull()

      // Restore original function
      vi.spyOn(jwt, 'verify').mockImplementation(originalVerify)
    })

    it('should handle malformed token structure', async () => {
      const malformedTokens = [
        'header.payload', // Missing signature
        'header.payload.signature.extra', // Extra parts
        'header', // Only header
        'not-a-token-at-all'
      ]

      for (const token of malformedTokens) {
        const result = await jwtService.validateToken(token)
        expect(result).toBeNull()
      }
    })

    it('should handle token with invalid JSON payload', async () => {
      const invalidJsonToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpbnZhbGlkLWpzb24iOiJ7dW5jbG9zZWQifQ.invalid-signature'

      const result = await jwtService.validateToken(invalidJsonToken)

      expect(result).toBeNull()
    })
  })

  describe('performance', () => {
    it('should validate tokens efficiently', async () => {
      const validToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)

      const startTime = Date.now()
      
      // Validate multiple times
      for (let i = 0; i < 100; i++) {
        await jwtService.validateToken(validToken)
      }

      const endTime = Date.now()
      const totalTime = endTime - startTime

      // Should complete 100 validations in reasonable time (< 1 second)
      expect(totalTime).toBeLessThan(1000)
    })

    it('should handle concurrent token validations', async () => {
      const validToken = jwt.sign(
        { 
          wallet_address: mockUser.wallet_address,
          user_id: mockUser.id
        },
        JWT_SECRET,
        { expiresIn: '1h' }
      )

      mockDatabase.getUserByWallet.mockResolvedValue(mockUser)

      // Validate multiple tokens concurrently
      const promises = Array(10).fill(null).map(() => 
        jwtService.validateToken(validToken)
      )

      const results = await Promise.all(promises)

      // All should succeed
      results.forEach(result => {
        expect(result).toBeDefined()
        expect(result?.user_id).toBe(mockUser.id)
      })
    })
  })
})
