import { describe, it, expect, beforeEach } from 'vitest';
import { PnlCalculationService, PositionPnlData } from '../src/services/pnlCalculationService';

describe('PnlCalculationService', () => {
  let pnlService: PnlCalculationService;

  beforeEach(() => {
    pnlService = PnlCalculationService.getInstance();
  });

  describe('calculatePositionPnl', () => {
    it('should calculate correct P&L for long position with profit', () => {
      const position: PositionPnlData = {
        id: '1',
        symbol: 'BTC-PERP',
        side: 'long',
        size: 1000000, // 1 BTC in smallest units
        entryPrice: 50000000, // $50,000
        currentPrice: 55000000, // $55,000
        leverage: 10,
        margin: 5000000 // $5,000 margin
      };

      const result = pnlService.calculatePositionPnl(position);

      // Expected P&L: ((55,000 - 50,000) * 1,000,000) / 1,000,000 = 5,000
      expect(result.unrealizedPnl).toBe(5000000);
      expect(result.unrealizedPnlPercent).toBe(100); // 5,000 / 5,000 * 100
      expect(result.healthFactor).toBeGreaterThan(0);
      expect(result.liquidationPrice).toBeGreaterThan(0);
    });

    it('should calculate correct P&L for long position with loss', () => {
      const position: PositionPnlData = {
        id: '2',
        symbol: 'BTC-PERP',
        side: 'long',
        size: 1000000,
        entryPrice: 50000000,
        currentPrice: 45000000, // $45,000
        leverage: 10,
        margin: 5000000
      };

      const result = pnlService.calculatePositionPnl(position);

      // Expected P&L: ((45,000 - 50,000) * 1,000,000) / 1,000,000 = -5,000
      expect(result.unrealizedPnl).toBe(-5000000);
      expect(result.unrealizedPnlPercent).toBe(-100); // -5,000 / 5,000 * 100
      expect(result.healthFactor).toBeLessThan(1);
    });

    it('should calculate correct P&L for short position with profit', () => {
      const position: PositionPnlData = {
        id: '3',
        symbol: 'BTC-PERP',
        side: 'short',
        size: 1000000,
        entryPrice: 50000000,
        currentPrice: 45000000, // Price dropped
        leverage: 10,
        margin: 5000000
      };

      const result = pnlService.calculatePositionPnl(position);

      // Expected P&L: ((50,000 - 45,000) * 1,000,000) / 1,000,000 = 5,000
      expect(result.unrealizedPnl).toBe(5000000);
      expect(result.unrealizedPnlPercent).toBe(100);
    });

    it('should calculate correct P&L for short position with loss', () => {
      const position: PositionPnlData = {
        id: '4',
        symbol: 'BTC-PERP',
        side: 'short',
        size: 1000000,
        entryPrice: 50000000,
        currentPrice: 55000000, // Price rose
        leverage: 10,
        margin: 5000000
      };

      const result = pnlService.calculatePositionPnl(position);

      // Expected P&L: ((50,000 - 55,000) * 1,000,000) / 1,000,000 = -5,000
      expect(result.unrealizedPnl).toBe(-5000000);
      expect(result.unrealizedPnlPercent).toBe(-100);
    });

    it('should handle zero size position', () => {
      const position: PositionPnlData = {
        id: '5',
        symbol: 'BTC-PERP',
        side: 'long',
        size: 0,
        entryPrice: 50000000,
        currentPrice: 55000000,
        leverage: 10,
        margin: 5000000
      };

      const result = pnlService.calculatePositionPnl(position);

      expect(result.unrealizedPnl).toBe(0);
      expect(result.unrealizedPnlPercent).toBe(0);
      expect(result.healthFactor).toBe(0);
    });

    it('should handle invalid values gracefully', () => {
      const position: PositionPnlData = {
        id: '6',
        symbol: 'BTC-PERP',
        side: 'long',
        size: NaN,
        entryPrice: Infinity,
        currentPrice: -Infinity,
        leverage: 10,
        margin: 5000000
      };

      const result = pnlService.calculatePositionPnl(position);

      expect(result.unrealizedPnl).toBe(0);
      expect(result.unrealizedPnlPercent).toBe(0);
      expect(result.liquidationPrice).toBe(0);
      expect(result.healthFactor).toBe(0);
    });
  });

  describe('calculatePortfolioPnl', () => {
    it('should calculate portfolio P&L correctly', () => {
      const positions: PositionPnlData[] = [
        {
          id: '1',
          symbol: 'BTC-PERP',
          side: 'long',
          size: 1000000,
          entryPrice: 50000000,
          currentPrice: 55000000,
          leverage: 10,
          margin: 5000000
        },
        {
          id: '2',
          symbol: 'ETH-PERP',
          side: 'short',
          size: 10000000,
          entryPrice: 3000000,
          currentPrice: 2800000,
          leverage: 5,
          margin: 6000000
        }
      ];

      const result = pnlService.calculatePortfolioPnl(positions);

      expect(result.totalUnrealizedPnl).toBe(5000000 + 2000000); // 5,000 + 2,000 = 7,000
      expect(result.totalMargin).toBe(11000000); // 5,000 + 6,000 = 11,000
      expect(result.portfolioHealth).toBeGreaterThanOrEqual(100); // Portfolio health can be exactly 100%
      expect(result.positions).toHaveLength(2);
    });

    it('should handle empty portfolio', () => {
      const result = pnlService.calculatePortfolioPnl([]);

      expect(result.totalUnrealizedPnl).toBe(0);
      expect(result.totalMargin).toBe(0);
      expect(result.portfolioHealth).toBe(100);
      expect(result.positions).toHaveLength(0);
    });
  });

  describe('validatePnlConsistency', () => {
    it('should validate correct P&L calculations', () => {
      const positions = [
        {
          id: '1',
          symbol: 'BTC-PERP',
          side: 'long' as const,
          size: 1000000,
          entryPrice: 50000000,
          currentPrice: 55000000,
          leverage: 10,
          margin: 5000000,
          unrealizedPnl: 5000000,
          unrealizedPnlPercent: 100,
          liquidationPrice: 45000000,
          healthFactor: 1.1,
          marginRatio: 100
        }
      ];

      const result = pnlService.validatePnlConsistency(positions);

      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should detect invalid P&L calculations', () => {
      const positions = [
        {
          id: '1',
          symbol: 'BTC-PERP',
          side: 'long' as const,
          size: 1000000,
          entryPrice: 50000000,
          currentPrice: 55000000,
          leverage: 10,
          margin: 5000000,
          unrealizedPnl: NaN,
          unrealizedPnlPercent: Infinity,
          liquidationPrice: -1000,
          healthFactor: -1,
          marginRatio: 2000
        }
      ];

      const result = pnlService.validatePnlConsistency(positions);

      expect(result.isValid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });
  });
});
