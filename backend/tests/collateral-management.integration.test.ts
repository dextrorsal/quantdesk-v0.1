import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import request from 'supertest';
import express from 'express';
import { collateralSyncService } from '../src/services/collateralSyncService';
import { databaseService } from '../src/services/supabaseDatabase';
import { smartContractService } from '../../frontend/src/services/smartContractService';

/**
 * Comprehensive Integration Tests for Collateral Management
 * 
 * These tests validate the complete deposit/withdraw flow including:
 * - Backend validation
 * - Smart contract integration
 * - Database synchronization
 * - Error handling
 * - Security measures
 */

describe('Collateral Management Integration Tests', () => {
  let app: express.Application;
  let mockUserId: string;
  let mockWalletAddress: string;

  beforeEach(() => {
    // Setup test environment
    mockUserId = 'test-user-' + Date.now();
    mockWalletAddress = 'test-wallet-address-123456789012345678901234567890';
    
    // Mock external dependencies
    vi.mock('../src/services/supabaseDatabase');
    vi.mock('../../frontend/src/services/smartContractService');
    vi.mock('../src/services/collateralSyncService');
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Deposit Flow Tests', () => {
    it('should successfully process a valid SOL deposit', async () => {
      // Arrange
      const depositAmount = 1.5; // 1.5 SOL
      const mockAccountState = {
        exists: true,
        canDeposit: true,
        canTrade: true,
        totalCollateral: 150, // $150 USD at $100/SOL
        accountHealth: 10000,
        totalPositions: 0,
        totalOrders: 0,
        isActive: true
      };

      // Mock smart contract service
      vi.mocked(smartContractService.getUserAccountState).mockResolvedValue(mockAccountState);
      vi.mocked(smartContractService.getSOLCollateralBalance).mockResolvedValue(depositAmount);

      // Mock database service
      vi.mocked(databaseService.insert).mockResolvedValue([{
        id: 'deposit-123',
        user_id: mockUserId,
        asset: 'SOL',
        amount: depositAmount,
        status: 'pending'
      }]);

      vi.mocked(databaseService.update).mockResolvedValue([]);
      vi.mocked(databaseService.upsert).mockResolvedValue([]);

      // Mock collateral sync service
      vi.mocked(collateralSyncService.syncUserCollateral).mockResolvedValue();

      // Act
      const response = await request(app)
        .post('/api/deposits/deposit')
        .send({
          asset: 'SOL',
          amount: depositAmount
        })
        .set('Authorization', `Bearer ${mockUserId}`);

      // Assert
      expect(response.status).toBe(201);
      expect(response.body.success).toBe(true);
      expect(response.body.deposit.amount).toBe(depositAmount);
      expect(response.body.deposit.asset).toBe('SOL');
    });

    it('should reject deposit with invalid amount precision', async () => {
      // Act
      const response = await request(app)
        .post('/api/deposits/deposit')
        .send({
          asset: 'SOL',
          amount: 1.1234567890 // Too many decimal places
        })
        .set('Authorization', `Bearer ${mockUserId}`);

      // Assert
      expect(response.status).toBe(400);
      expect(response.body.code).toBe('INVALID_PRECISION');
      expect(response.body.details.maxDecimals).toBe(9);
    });

    it('should reject deposit exceeding maximum limit', async () => {
      // Act
      const response = await request(app)
        .post('/api/deposits/deposit')
        .send({
          asset: 'SOL',
          amount: 2000000 // Exceeds $1M limit
        })
        .set('Authorization', `Bearer ${mockUserId}`);

      // Assert
      expect(response.status).toBe(400);
      expect(response.body.code).toBe('AMOUNT_TOO_LARGE');
      expect(response.body.details.maxAmount).toBe(1000000);
    });

    it('should reject deposit with unsupported asset', async () => {
      // Act
      const response = await request(app)
        .post('/api/deposits/deposit')
        .send({
          asset: 'INVALID_TOKEN',
          amount: 1.0
        })
        .set('Authorization', `Bearer ${mockUserId}`);

      // Assert
      expect(response.status).toBe(400);
      expect(response.body.code).toBe('UNSUPPORTED_TOKEN');
    });
  });

  describe('Withdrawal Flow Tests', () => {
    it('should successfully process a valid SOL withdrawal', async () => {
      // Arrange
      const withdrawalAmount = 0.5; // 0.5 SOL
      const mockBalance = {
        available_balance: 2.0 // User has 2 SOL available
      };

      // Mock database service
      vi.mocked(databaseService.complexQuery).mockResolvedValue([mockBalance]);
      vi.mocked(databaseService.insert).mockResolvedValue([{
        id: 'withdrawal-123',
        user_id: mockUserId,
        asset: 'SOL',
        amount: withdrawalAmount,
        status: 'pending'
      }]);

      // Act
      const response = await request(app)
        .post('/api/deposits/withdraw')
        .send({
          asset: 'SOL',
          amount: withdrawalAmount
        })
        .set('Authorization', `Bearer ${mockUserId}`);

      // Assert
      expect(response.status).toBe(201);
      expect(response.body.success).toBe(true);
      expect(response.body.withdrawal.amount).toBe(withdrawalAmount);
    });

    it('should reject withdrawal with insufficient balance', async () => {
      // Arrange
      const withdrawalAmount = 5.0; // 5 SOL
      const mockBalance = {
        available_balance: 1.0 // User only has 1 SOL
      };

      vi.mocked(databaseService.complexQuery).mockResolvedValue([mockBalance]);

      // Act
      const response = await request(app)
        .post('/api/deposits/withdraw')
        .send({
          asset: 'SOL',
          amount: withdrawalAmount
        })
        .set('Authorization', `Bearer ${mockUserId}`);

      // Assert
      expect(response.status).toBe(400);
      expect(response.body.code).toBe('INSUFFICIENT_BALANCE');
    });

    it('should reject withdrawal with invalid destination address', async () => {
      // Act
      const response = await request(app)
        .post('/api/deposits/withdraw')
        .send({
          asset: 'SOL',
          amount: 1.0,
          destinationAddress: 'invalid-address' // Too short
        })
        .set('Authorization', `Bearer ${mockUserId}`);

      // Assert
      expect(response.status).toBe(400);
      expect(response.body.code).toBe('INVALID_DESTINATION');
    });
  });

  describe('Database Synchronization Tests', () => {
    it('should sync collateral data when on-chain and database differ', async () => {
      // Arrange
      const onChainData = {
        totalCollateral: 200,
        solAmount: 2.0,
        usdValue: 200,
        lastUpdated: Date.now()
      };

      const dbData = {
        totalCollateral: 150,
        solAmount: 1.5,
        usdValue: 150,
        lastUpdated: '2024-01-01T00:00:00Z'
      };

      // Mock services
      vi.mocked(smartContractService.getUserAccountState).mockResolvedValue({
        exists: true,
        totalCollateral: onChainData.totalCollateral
      });
      vi.mocked(smartContractService.getSOLCollateralBalance).mockResolvedValue(onChainData.solAmount);
      vi.mocked(databaseService.select).mockResolvedValue([{
        balance: dbData.solAmount,
        updated_at: dbData.lastUpdated
      }]);
      vi.mocked(databaseService.upsert).mockResolvedValue([]);
      vi.mocked(databaseService.insert).mockResolvedValue([]);

      // Act
      await collateralSyncService.syncUserCollateral(mockUserId, mockWalletAddress);

      // Assert
      expect(databaseService.upsert).toHaveBeenCalledWith('user_balances', {
        user_id: mockUserId,
        asset: 'SOL',
        balance: onChainData.solAmount,
        locked_balance: 0,
        updated_at: expect.any(String)
      });
    });

    it('should not sync when data is within tolerance', async () => {
      // Arrange
      const onChainData = {
        totalCollateral: 100.01, // Within 1 cent tolerance
        solAmount: 1.0001,
        usdValue: 100.01,
        lastUpdated: Date.now()
      };

      const dbData = {
        totalCollateral: 100.00,
        solAmount: 1.0000,
        usdValue: 100.00,
        lastUpdated: '2024-01-01T00:00:00Z'
      };

      // Mock services
      vi.mocked(smartContractService.getUserAccountState).mockResolvedValue({
        exists: true,
        totalCollateral: onChainData.totalCollateral
      });
      vi.mocked(smartContractService.getSOLCollateralBalance).mockResolvedValue(onChainData.solAmount);
      vi.mocked(databaseService.select).mockResolvedValue([{
        balance: dbData.solAmount,
        updated_at: dbData.lastUpdated
      }]);

      // Act
      await collateralSyncService.syncUserCollateral(mockUserId, mockWalletAddress);

      // Assert
      expect(databaseService.upsert).not.toHaveBeenCalled();
    });
  });

  describe('Security Tests', () => {
    it('should validate all input parameters', async () => {
      // Test missing asset
      const response1 = await request(app)
        .post('/api/deposits/deposit')
        .send({ amount: 1.0 })
        .set('Authorization', `Bearer ${mockUserId}`);

      expect(response1.status).toBe(400);
      expect(response1.body.code).toBe('INVALID_PARAMS');

      // Test missing amount
      const response2 = await request(app)
        .post('/api/deposits/deposit')
        .send({ asset: 'SOL' })
        .set('Authorization', `Bearer ${mockUserId}`);

      expect(response2.status).toBe(400);
      expect(response2.body.code).toBe('INVALID_PARAMS');

      // Test negative amount
      const response3 = await request(app)
        .post('/api/deposits/deposit')
        .send({ asset: 'SOL', amount: -1.0 })
        .set('Authorization', `Bearer ${mockUserId}`);

      expect(response3.status).toBe(400);
      expect(response3.body.code).toBe('INVALID_PARAMS');
    });

    it('should handle smart contract errors gracefully', async () => {
      // Arrange
      vi.mocked(smartContractService.getUserAccountState).mockRejectedValue(
        new Error('Smart contract error')
      );

      // Act
      const response = await request(app)
        .post('/api/deposits/deposit')
        .send({
          asset: 'SOL',
          amount: 1.0
        })
        .set('Authorization', `Bearer ${mockUserId}`);

      // Assert
      expect(response.status).toBe(500);
      expect(response.body.code).toBe('DEPOSIT_ERROR');
    });

    it('should handle database errors gracefully', async () => {
      // Arrange
      vi.mocked(databaseService.insert).mockRejectedValue(
        new Error('Database connection failed')
      );

      // Act
      const response = await request(app)
        .post('/api/deposits/deposit')
        .send({
          asset: 'SOL',
          amount: 1.0
        })
        .set('Authorization', `Bearer ${mockUserId}`);

      // Assert
      expect(response.status).toBe(500);
      expect(response.body.code).toBe('DEPOSIT_ERROR');
    });
  });

  describe('Edge Cases', () => {
    it('should handle zero amount deposits', async () => {
      const response = await request(app)
        .post('/api/deposits/deposit')
        .send({
          asset: 'SOL',
          amount: 0
        })
        .set('Authorization', `Bearer ${mockUserId}`);

      expect(response.status).toBe(400);
      expect(response.body.code).toBe('INVALID_PARAMS');
    });

    it('should handle very small amounts', async () => {
      const response = await request(app)
        .post('/api/deposits/deposit')
        .send({
          asset: 'SOL',
          amount: 0.000000001 // 1 lamport
        })
        .set('Authorization', `Bearer ${mockUserId}`);

      expect(response.status).toBe(201);
      expect(response.body.success).toBe(true);
    });

    it('should handle maximum precision amounts', async () => {
      const response = await request(app)
        .post('/api/deposits/deposit')
        .send({
          asset: 'SOL',
          amount: 1.123456789 // Exactly 9 decimal places
        })
        .set('Authorization', `Bearer ${mockUserId}`);

      expect(response.status).toBe(201);
      expect(response.body.success).toBe(true);
    });
  });

  describe('Performance Tests', () => {
    it('should complete deposit within acceptable time', async () => {
      const startTime = Date.now();

      const response = await request(app)
        .post('/api/deposits/deposit')
        .send({
          asset: 'SOL',
          amount: 1.0
        })
        .set('Authorization', `Bearer ${mockUserId}`);

      const endTime = Date.now();
      const duration = endTime - startTime;

      expect(response.status).toBe(201);
      expect(duration).toBeLessThan(5000); // Should complete within 5 seconds
    });

    it('should handle concurrent deposits', async () => {
      const promises = Array.from({ length: 10 }, (_, i) =>
        request(app)
          .post('/api/deposits/deposit')
          .send({
            asset: 'SOL',
            amount: 0.1
          })
          .set('Authorization', `Bearer ${mockUserId}-${i}`)
      );

      const responses = await Promise.all(promises);

      // All requests should succeed
      responses.forEach(response => {
        expect(response.status).toBe(201);
        expect(response.body.success).toBe(true);
      });
    });
  });
});
