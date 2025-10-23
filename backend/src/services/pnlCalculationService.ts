import { Logger } from '../utils/logger';
import { pythOracleService } from './pythOracleService';

const logger = new Logger();

export interface PositionPnlData {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice?: number;
  leverage: number;
  margin: number;
}

export interface PnlCalculationResult {
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  liquidationPrice: number;
  healthFactor: number;
  marginRatio: number;
}

export class PnlCalculationService {
  private static instance: PnlCalculationService;

  private constructor() {}

  public static getInstance(): PnlCalculationService {
    if (!PnlCalculationService.instance) {
      PnlCalculationService.instance = new PnlCalculationService();
    }
    return PnlCalculationService.instance;
  }

  /**
   * Calculate P&L for a single position
   */
  public calculatePositionPnl(position: PositionPnlData, currentPrice?: number): PnlCalculationResult {
    try {
      const price = currentPrice || position.currentPrice || position.entryPrice;
      
      // Calculate unrealized P&L
      let unrealizedPnl: number;
      if (position.side === 'long') {
        unrealizedPnl = (price - position.entryPrice) * position.size;
      } else {
        unrealizedPnl = (position.entryPrice - price) * position.size;
      }

      // Calculate unrealized P&L percentage
      const positionValue = position.size * position.entryPrice;
      const unrealizedPnlPercent = positionValue > 0 ? (unrealizedPnl / positionValue) * 100 : 0;

      // Calculate liquidation price
      const liquidationPrice = this.calculateLiquidationPrice(position);

      // Calculate health factor
      const equity = position.margin + unrealizedPnl;
      const healthFactor = equity > 0 ? equity / (position.size * price) : 0;

      // Calculate margin ratio
      const marginRatio = position.margin > 0 ? (unrealizedPnl / position.margin) * 100 : 0;

      return {
        unrealizedPnl,
        unrealizedPnlPercent,
        liquidationPrice,
        healthFactor,
        marginRatio
      };

    } catch (error) {
      logger.error('Error calculating position P&L:', error);
      return {
        unrealizedPnl: 0,
        unrealizedPnlPercent: 0,
        liquidationPrice: 0,
        healthFactor: 1,
        marginRatio: 0
      };
    }
  }

  /**
   * Calculate liquidation price for a position
   */
  private calculateLiquidationPrice(position: PositionPnlData): number {
    try {
      // Liquidation occurs when equity = 0
      // For long: liquidation_price = entry_price - (margin / size)
      // For short: liquidation_price = entry_price + (margin / size)
      
      if (position.side === 'long') {
        return Math.max(0, position.entryPrice - (position.margin / position.size));
      } else {
        return position.entryPrice + (position.margin / position.size);
      }
    } catch (error) {
      logger.error('Error calculating liquidation price:', error);
      return 0;
    }
  }

  /**
   * Calculate portfolio P&L for multiple positions
   */
  public calculatePortfolioPnl(positions: PositionPnlData[]): {
    totalUnrealizedPnl: number;
    totalUnrealizedPnlPercent: number;
    totalMargin: number;
    portfolioHealth: number;
    positions: Array<PositionPnlData & PnlCalculationResult>;
  } {
    try {
      let totalUnrealizedPnl = 0;
      let totalMargin = 0;
      let totalPositionValue = 0;

      const positionsWithPnl = positions.map(position => {
        const pnlResult = this.calculatePositionPnl(position);
        totalUnrealizedPnl += pnlResult.unrealizedPnl;
        totalMargin += position.margin;
        totalPositionValue += position.size * (position.currentPrice || position.entryPrice);

        return {
          ...position,
          ...pnlResult
        };
      });

      const totalUnrealizedPnlPercent = totalPositionValue > 0 
        ? (totalUnrealizedPnl / totalPositionValue) * 100 
        : 0;

      const portfolioHealth = totalMargin > 0 
        ? Math.max(0, Math.min(100, ((totalMargin + totalUnrealizedPnl) / totalMargin) * 100))
        : 100;

      return {
        totalUnrealizedPnl,
        totalUnrealizedPnlPercent,
        totalMargin,
        portfolioHealth,
        positions: positionsWithPnl
      };

    } catch (error) {
      logger.error('Error calculating portfolio P&L:', error);
      return {
        totalUnrealizedPnl: 0,
        totalUnrealizedPnlPercent: 0,
        totalMargin: 0,
        portfolioHealth: 100,
        positions: []
      };
    }
  }

  /**
   * Update position P&L with current market prices
   */
  public async updatePositionPnlWithCurrentPrices(positions: PositionPnlData[]): Promise<Array<PositionPnlData & PnlCalculationResult>> {
    try {
      const updatedPositions = [];

      for (const position of positions) {
        try {
          // Get current price from oracle
          const symbol = position.symbol.replace('-PERP', ''); // Remove -PERP suffix
          const currentPrice = await pythOracleService.getPrice(symbol);
          
          if (currentPrice) {
            const pnlResult = this.calculatePositionPnl(position, currentPrice.price);
            updatedPositions.push({
              ...position,
              currentPrice: currentPrice.price,
              ...pnlResult
            });
          } else {
            // Fallback to entry price if no current price available
            const pnlResult = this.calculatePositionPnl(position);
            updatedPositions.push({
              ...position,
              ...pnlResult
            });
          }
        } catch (error) {
          logger.error(`Error updating P&L for position ${position.id}:`, error);
          // Fallback to entry price calculation
          const pnlResult = this.calculatePositionPnl(position);
          updatedPositions.push({
            ...position,
            ...pnlResult
          });
        }
      }

      return updatedPositions;

    } catch (error) {
      logger.error('Error updating position P&L with current prices:', error);
      return positions.map(position => ({
        ...position,
        ...this.calculatePositionPnl(position)
      }));
    }
  }

  /**
   * Validate P&L calculation consistency
   */
  public validatePnlConsistency(positions: Array<PositionPnlData & PnlCalculationResult>): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    try {
      for (const position of positions) {
        // Check for NaN or infinite values
        if (!isFinite(position.unrealizedPnl)) {
          errors.push(`Position ${position.id}: Invalid unrealized P&L`);
        }

        if (!isFinite(position.unrealizedPnlPercent)) {
          errors.push(`Position ${position.id}: Invalid unrealized P&L percentage`);
        }

        if (!isFinite(position.liquidationPrice) || position.liquidationPrice < 0) {
          errors.push(`Position ${position.id}: Invalid liquidation price`);
        }

        if (!isFinite(position.healthFactor) || position.healthFactor < 0) {
          errors.push(`Position ${position.id}: Invalid health factor`);
        }

        // Check for reasonable values
        if (Math.abs(position.unrealizedPnlPercent) > 1000) {
          errors.push(`Position ${position.id}: Unrealistic P&L percentage`);
        }

        if (position.healthFactor > 10) {
          errors.push(`Position ${position.id}: Unrealistic health factor`);
        }
      }

      return {
        isValid: errors.length === 0,
        errors
      };

    } catch (error) {
      logger.error('Error validating P&L consistency:', error);
      return {
        isValid: false,
        errors: ['Validation error occurred']
      };
    }
  }
}

// Export singleton instance
export const pnlCalculationService = PnlCalculationService.getInstance();
