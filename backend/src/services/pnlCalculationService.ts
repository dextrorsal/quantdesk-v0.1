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
   * FIXED: Consistent with smart contract calculation formula
   */
  public calculatePositionPnl(position: PositionPnlData, currentPrice?: number): PnlCalculationResult {
    try {
      const price = currentPrice || position.currentPrice || position.entryPrice;
      
      // FIXED: Use consistent P&L calculation formula matching smart contract
      // Smart contract formula: pnl = (current_price - entry_price) * size / 1_000_000 for long
      // For short: pnl = (entry_price - current_price) * size / 1_000_000
      // The division by 1,000,000 is the price scaling factor used in smart contracts
      let unrealizedPnl: number;
      if (position.side === 'long') {
        // Long position: profit when price goes up
        unrealizedPnl = ((price - position.entryPrice) * position.size) / 1_000_000;
      } else {
        // Short position: profit when price goes down
        unrealizedPnl = ((position.entryPrice - price) * position.size) / 1_000_000;
      }

      // FIXED: Calculate unrealized P&L percentage based on margin, not position value
      // This matches the smart contract's approach to percentage calculations
      const unrealizedPnlPercent = position.margin > 0 ? (unrealizedPnl / position.margin) * 100 : 0;

      // FIXED: Calculate liquidation price using consistent formula
      const liquidationPrice = this.calculateLiquidationPrice(position, price);

      // FIXED: Calculate health factor using proper equity calculation
      const equity = position.margin + unrealizedPnl;
      const positionValue = position.size * price;
      const healthFactor = positionValue > 0 ? equity / positionValue : 0;

      // FIXED: Calculate margin ratio as percentage of margin used
      const marginRatio = position.margin > 0 ? (unrealizedPnl / position.margin) * 100 : 0;

      // FIXED: Add validation to prevent invalid values
      const validatedResult = {
        unrealizedPnl: isFinite(unrealizedPnl) ? unrealizedPnl : 0,
        unrealizedPnlPercent: isFinite(unrealizedPnlPercent) ? unrealizedPnlPercent : 0,
        liquidationPrice: isFinite(liquidationPrice) && liquidationPrice >= 0 ? liquidationPrice : 0,
        healthFactor: isFinite(healthFactor) && healthFactor >= 0 ? healthFactor : 0,
        marginRatio: isFinite(marginRatio) ? marginRatio : 0
      };

      logger.debug(`P&L calculated for position ${position.id}:`, {
        side: position.side,
        entryPrice: position.entryPrice,
        currentPrice: price,
        size: position.size,
        margin: position.margin,
        unrealizedPnl: validatedResult.unrealizedPnl,
        unrealizedPnlPercent: validatedResult.unrealizedPnlPercent,
        liquidationPrice: validatedResult.liquidationPrice,
        healthFactor: validatedResult.healthFactor
      });

      return validatedResult;

    } catch (error) {
      logger.error('Error calculating position P&L:', error);
      return {
        unrealizedPnl: 0,
        unrealizedPnlPercent: 0,
        liquidationPrice: 0,
        healthFactor: 0,
        marginRatio: 0
      };
    }
  }

  /**
   * Calculate liquidation price for a position
   * FIXED: Consistent with smart contract liquidation logic
   */
  private calculateLiquidationPrice(position: PositionPnlData, currentPrice?: number): number {
    try {
      const price = currentPrice || position.currentPrice || position.entryPrice;
      
      // FIXED: Use consistent liquidation price calculation matching smart contract
      // Liquidation occurs when equity = maintenance_margin
      // For long: liquidation_price = entry_price - (margin / size) * leverage_factor
      // For short: liquidation_price = entry_price + (margin / size) * leverage_factor
      
      // Calculate maintenance margin (typically 80% of initial margin)
      const maintenanceMarginRatio = 0.8; // 80% maintenance margin
      const maintenanceMargin = position.margin * maintenanceMarginRatio;
      
      if (position.side === 'long') {
        // Long position liquidation: price drops below maintenance margin threshold
        const liquidationPrice = position.entryPrice - (maintenanceMargin / position.size);
        return Math.max(0, liquidationPrice);
      } else {
        // Short position liquidation: price rises above maintenance margin threshold
        const liquidationPrice = position.entryPrice + (maintenanceMargin / position.size);
        return liquidationPrice;
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
