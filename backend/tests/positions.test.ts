import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import request from 'supertest';
import express from 'express';
import positionsRouter from '../src/routes/positions';
import { SupabaseDatabaseService } from '../src/services/supabaseDatabase';
import { pnlCalculationService } from '../src/services/pnlCalculationService';
import { pythOracleService } from '../src/services/pythOracleService';

// Mock dependencies
vi.mock('../src/services/supabaseDatabase');
vi.mock('../src/services/pnlCalculationService');
vi.mock('../src/services/pythOracleService');

describe('Positions Routes', () => {
  let app: express.Application;
  let mockDb: any;
  let mockPnlService: any;
  let mockOracleService: any;

  beforeEach(() => {
    app = express();
    app.use(express.json());
    
    // Mock user authentication middleware
    app.use((req, res, next) => {
      req.user = { id: 'test-user-id' };
      next();
    });
    
    app.use('/api/positions', positionsRouter);

    // Setup mocks
    mockDb = {
      select: vi.fn(),
      update: vi.fn()
    };
    (SupabaseDatabaseService.getInstance as any).mockReturnValue(mockDb);

    mockPnlService = {
      updatePositionPnlWithCurrentPrices: vi.fn(),
      validatePnlConsistency: vi.fn()
    };
    (pnlCalculationService as any).mockReturnValue(mockPnlService);

    mockOracleService = {
      getPrice: vi.fn()
    };
    (pythOracleService as any).mockReturnValue(mockOracleService);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('GET /api/positions', () => {
    it('should return positions successfully', async () => {
      const mockPositions = [
        {
          id: '1',
          symbol: 'BTC-PERP',
          side: 'long',
          size: '1.0',
          entry_price: '50000',
          current_price: '55000',
          margin: '5000',
          leverage: '10',
          status: 'open',
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z'
        }
      ];

      const mockPnlResult = [
        {
          id: '1',
          symbol: 'BTC-PERP',
          side: 'long',
          size: 1.0,
          entryPrice: 50000,
          currentPrice: 55000,
          margin: 5000,
          leverage: 10,
          unrealizedPnl: 5000,
          unrealizedPnlPercent: 100,
          liquidationPrice: 45000,
          healthFactor: 1.1,
          marginRatio: 100
        }
      ];

      mockDb.select.mockResolvedValue(mockPositions);
      mockPnlService.updatePositionPnlWithCurrentPrices.mockResolvedValue(mockPnlResult);
      mockPnlService.validatePnlConsistency.mockReturnValue({
        isValid: true,
        errors: []
      });

      const response = await request(app)
        .get('/api/positions')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.positions).toHaveLength(1);
      expect(response.body.positions[0].id).toBe('1');
      expect(response.body.positions[0].unrealizedPnl).toBe(5000);
    });

    it('should return empty array when no positions', async () => {
      mockDb.select.mockResolvedValue([]);
      mockPnlService.updatePositionPnlWithCurrentPrices.mockResolvedValue([]);
      mockPnlService.validatePnlConsistency.mockReturnValue({
        isValid: true,
        errors: []
      });

      const response = await request(app)
        .get('/api/positions')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.positions).toHaveLength(0);
      expect(response.body.message).toBe('No open positions found');
    });

    it('should handle database errors', async () => {
      mockDb.select.mockRejectedValue(new Error('Database connection failed'));

      const response = await request(app)
        .get('/api/positions')
        .expect(500);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toBe('Failed to retrieve positions');
    });

    it('should handle P&L validation warnings', async () => {
      const mockPositions = [
        {
          id: '1',
          symbol: 'BTC-PERP',
          side: 'long',
          size: '1.0',
          entry_price: '50000',
          current_price: '55000',
          margin: '5000',
          leverage: '10',
          status: 'open',
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z'
        }
      ];

      const mockPnlResult = [
        {
          id: '1',
          symbol: 'BTC-PERP',
          side: 'long',
          size: 1.0,
          entryPrice: 50000,
          currentPrice: 55000,
          margin: 5000,
          leverage: 10,
          unrealizedPnl: 5000,
          unrealizedPnlPercent: 100,
          liquidationPrice: 45000,
          healthFactor: 1.1,
          marginRatio: 100
        }
      ];

      mockDb.select.mockResolvedValue(mockPositions);
      mockPnlService.updatePositionPnlWithCurrentPrices.mockResolvedValue(mockPnlResult);
      mockPnlService.validatePnlConsistency.mockReturnValue({
        isValid: false,
        errors: ['Position 1: Invalid unrealized P&L']
      });

      const response = await request(app)
        .get('/api/positions')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.validation.isValid).toBe(false);
      expect(response.body.validation.errors).toContain('Position 1: Invalid unrealized P&L');
    });
  });

  describe('GET /api/positions/:id', () => {
    it('should return specific position', async () => {
      const mockPosition = [
        {
          id: '1',
          symbol: 'BTC-PERP',
          side: 'long',
          size: '1.0',
          entry_price: '50000',
          current_price: '55000',
          margin: '5000',
          leverage: '10',
          status: 'open',
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z'
        }
      ];

      const mockPrice = { price: 55000 };

      mockDb.select.mockResolvedValue(mockPosition);
      mockOracleService.getPrice.mockResolvedValue(mockPrice);
      mockPnlService.calculatePositionPnl.mockReturnValue({
        unrealizedPnl: 5000,
        unrealizedPnlPercent: 100,
        liquidationPrice: 45000,
        healthFactor: 1.1,
        marginRatio: 100
      });

      const response = await request(app)
        .get('/api/positions/1')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.position.id).toBe('1');
      expect(response.body.position.currentPrice).toBe(55000);
    });

    it('should return 404 for non-existent position', async () => {
      mockDb.select.mockResolvedValue([]);

      const response = await request(app)
        .get('/api/positions/999')
        .expect(404);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toBe('Position not found');
    });
  });

  describe('POST /api/positions/:id/close', () => {
    it('should close position successfully', async () => {
      mockDb.update.mockResolvedValue([{ id: '1' }]);

      const response = await request(app)
        .post('/api/positions/1/close')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.message).toBe('Position closed successfully');
      expect(response.body.positionId).toBe('1');
    });

    it('should return 404 for non-existent position', async () => {
      mockDb.update.mockResolvedValue([]);

      const response = await request(app)
        .post('/api/positions/999/close')
        .expect(404);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toBe('Position not found or already closed');
    });

    it('should handle database errors during close', async () => {
      mockDb.update.mockRejectedValue(new Error('Database error'));

      const response = await request(app)
        .post('/api/positions/1/close')
        .expect(500);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toBe('Failed to close position');
    });
  });

  describe('Authentication', () => {
    it('should return 401 when user is not authenticated', async () => {
      const appWithoutAuth = express();
      appWithoutAuth.use(express.json());
      appWithoutAuth.use('/api/positions', positionsRouter);

      const response = await request(appWithoutAuth)
        .get('/api/positions')
        .expect(401);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toBe('User not authenticated');
    });
  });
});
