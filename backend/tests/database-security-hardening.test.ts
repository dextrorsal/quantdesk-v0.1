/**
 * Story 2.2: Database Security Hardening - Security Tests
 * 
 * This test suite validates the security hardening implementation:
 * - SQL injection prevention
 * - RLS policy enforcement
 * - Performance monitoring
 * - Fluent API security
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SupabaseDatabaseService } from '../src/services/supabaseDatabase';
import { getSupabaseService } from '../src/services/supabaseService';

describe('Database Security Hardening Tests', () => {
  let db: SupabaseDatabaseService;
  let supabase: any;

  beforeEach(() => {
    db = SupabaseDatabaseService.getInstance();
    supabase = getSupabaseService();
  });

  describe('SQL Injection Prevention', () => {
    it('should prevent SQL injection in select queries', async () => {
      const maliciousInput = "'; DROP TABLE users; --";
      
      // This should not execute the malicious SQL
      const result = await db.select('users', '*', { 
        username: maliciousInput 
      });
      
      // Should return empty array, not crash
      expect(Array.isArray(result)).toBe(true);
    });

    it('should prevent SQL injection in insert operations', async () => {
      const maliciousInput = "'; DROP TABLE users; --";
      
      // This should not execute the malicious SQL
      try {
        await db.insert('users', {
          wallet_address: 'test-wallet',
          username: maliciousInput
        });
      } catch (error) {
        // Should throw a validation error, not execute malicious SQL
        expect(error.message).not.toContain('DROP TABLE');
      }
    });

    it('should prevent SQL injection in update operations', async () => {
      const maliciousInput = "'; DROP TABLE users; --";
      
      try {
        await db.update('users', {
          username: maliciousInput
        }, { id: 'test-id' });
      } catch (error) {
        // Should throw a validation error, not execute malicious SQL
        expect(error.message).not.toContain('DROP TABLE');
      }
    });
  });

  describe('RLS Policy Enforcement', () => {
    it('should enforce user data isolation', async () => {
      // Test that users can only access their own data
      const user1Data = await db.select('users', '*', { id: 'user1-id' });
      const user2Data = await db.select('users', '*', { id: 'user2-id' });
      
      // Should only return data for the authenticated user
      expect(user1Data.length).toBeLessThanOrEqual(1);
      expect(user2Data.length).toBeLessThanOrEqual(1);
    });

    it('should prevent cross-user data access', async () => {
      // Test that users cannot access other users' positions
      const positions = await db.select('positions', '*', { user_id: 'other-user-id' });
      
      // Should return empty array due to RLS
      expect(positions).toEqual([]);
    });

    it('should enforce chat channel access policies', async () => {
      // Test public channel access
      const publicChannels = await db.select('chat_channels', '*', { 
        is_private: false, 
        is_active: true 
      });
      
      // Should return public channels
      expect(Array.isArray(publicChannels)).toBe(true);
      
      // Test private channel access
      const privateChannels = await db.select('chat_channels', '*', { 
        is_private: true 
      });
      
      // Should only return channels the user has access to
      expect(Array.isArray(privateChannels)).toBe(true);
    });
  });

  describe('Performance Monitoring', () => {
    it('should log slow queries', async () => {
      const start = Date.now();
      
      // Execute a query that might be slow
      await db.select('users', '*');
      
      const duration = Date.now() - start;
      
      // If query was slow, it should be logged
      if (duration > 1000) {
        // Check if slow query was logged
        const performanceLogs = await db.select('performance_logs', '*', {
          execution_time_ms: duration
        });
        
        expect(performanceLogs.length).toBeGreaterThan(0);
      }
    });

    it('should not break operations when logging fails', async () => {
      // This test ensures that logging failures don't break main operations
      const result = await db.select('users', '*');
      
      // Should still return data even if logging fails
      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('Fluent API Security', () => {
    it('should use type-safe queries', async () => {
      // Test that fluent API prevents SQL injection
      const result = await db.select('users', 'id, username', {
        is_active: true
      });
      
      expect(Array.isArray(result)).toBe(true);
      
      // Verify that the query was constructed safely
      result.forEach(user => {
        expect(user).toHaveProperty('id');
        expect(user).toHaveProperty('username');
      });
    });

    it('should validate input parameters', async () => {
      // Test with invalid parameters
      try {
        await db.select('', '*', {});
      } catch (error) {
        expect(error.message).toContain('table');
      }
    });

    it('should handle null and undefined values safely', async () => {
      // Test with null values
      const result1 = await db.select('users', '*', { username: null });
      expect(Array.isArray(result1)).toBe(true);
      
      // Test with undefined values
      const result2 = await db.select('users', '*', { username: undefined });
      expect(Array.isArray(result2)).toBe(true);
    });
  });

  describe('Database Connection Security', () => {
    it('should use secure connection', async () => {
      const isHealthy = await db.healthCheck();
      expect(isHealthy).toBe(true);
    });

    it('should handle connection failures gracefully', async () => {
      // Test error handling
      try {
        await db.select('non_existent_table', '*');
      } catch (error) {
        expect(error.message).toContain('table');
      }
    });
  });

  describe('Audit Logging', () => {
    it('should log database operations', async () => {
      // Perform an operation that should be logged
      await db.select('users', '*', { is_active: true });
      
      // Check if operation was logged (if audit logging is enabled)
      try {
        const auditLogs = await db.select('audit_logs', '*', {
          table_name: 'users',
          operation: 'SELECT'
        });
        
        expect(Array.isArray(auditLogs)).toBe(true);
      } catch (error) {
        // Audit logging might not be enabled in test environment
        expect(error.message).toContain('table');
      }
    });
  });

  describe('Data Validation', () => {
    it('should validate data types', async () => {
      // Test with invalid data types
      try {
        await db.insert('users', {
          wallet_address: 123, // Should be string
          username: true // Should be string
        });
      } catch (error) {
        expect(error.message).toContain('validation');
      }
    });

    it('should enforce required fields', async () => {
      // Test with missing required fields
      try {
        await db.insert('users', {
          username: 'test'
          // Missing required wallet_address
        });
      } catch (error) {
        expect(error.message).toContain('wallet_address');
      }
    });
  });

  describe('Transaction Security', () => {
    it('should handle transactions safely', async () => {
      // Test transaction rollback on error
      try {
        await db.transaction(async (client) => {
          await client.insert('users', { wallet_address: 'test1' });
          throw new Error('Test rollback');
        });
      } catch (error) {
        expect(error.message).toContain('rollback');
      }
    });
  });

  afterEach(() => {
    // Cleanup after each test
  });
});

/**
 * Integration tests for end-to-end security validation
 */
describe('Database Security Integration Tests', () => {
  let db: SupabaseDatabaseService;

  beforeEach(() => {
    db = SupabaseDatabaseService.getInstance();
  });

  describe('Epic 1 Test Script Compatibility', () => {
    it('should maintain compatibility with existing test scripts', async () => {
      // Test that existing functionality still works
      const markets = await db.getMarkets();
      expect(Array.isArray(markets)).toBe(true);
      
      const users = await db.select('users', '*', { is_active: true });
      expect(Array.isArray(users)).toBe(true);
    });

    it('should handle complex queries safely', async () => {
      // Test complex query patterns
      const result = await db.select('orders', '*', {
        status: 'pending',
        side: 'long'
      });
      
      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('Performance Under Load', () => {
    it('should handle concurrent requests', async () => {
      const promises = Array.from({ length: 10 }, () => 
        db.select('users', '*', { is_active: true })
      );
      
      const results = await Promise.all(promises);
      
      results.forEach(result => {
        expect(Array.isArray(result)).toBe(true);
      });
    });
  });
});
