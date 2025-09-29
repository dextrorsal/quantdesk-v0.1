import { Connection, PublicKey, Transaction, TransactionInstruction } from '@solana/web3.js';
import { DatabaseService } from './database';
import { OracleService } from './oracle';
import { Logger } from '../utils/logger';
import { config } from '../config/environment';

const logger = new Logger();

export interface LiquidationCandidate {
  positionId: string;
  userId: string;
  marketId: string;
  healthFactor: number;
  liquidationPrice: number;
  estimatedReward: number;
}

export class LiquidationBot {
  private static instance: LiquidationBot;
  private connection: Connection;
  private db: DatabaseService;
  private oracle: OracleService;
  private isRunning: boolean = false;
  private liquidationThreshold: number = 0.95; // 95% health factor threshold
  private minLiquidationReward: number = 0.01; // Minimum 0.01 SOL reward

  private constructor() {
    this.connection = new Connection(config.RPC_URL, 'confirmed');
    this.db = DatabaseService.getInstance();
    this.oracle = OracleService.getInstance();
  }

  public static getInstance(): LiquidationBot {
    if (!LiquidationBot.instance) {
      LiquidationBot.instance = new LiquidationBot();
    }
    return LiquidationBot.instance;
  }

  /**
   * Start the liquidation monitoring service
   */
  public start(): void {
    if (this.isRunning) {
      logger.warn('Liquidation bot is already running');
      return;
    }

    this.isRunning = true;
    logger.info('Starting liquidation bot...');

    // Monitor positions every 5 seconds
    setInterval(async () => {
      try {
        await this.monitorPositions();
      } catch (error) {
        logger.error('Error in liquidation monitoring:', error);
      }
    }, 5000);

    logger.info('Liquidation bot started successfully');
  }

  /**
   * Stop the liquidation monitoring service
   */
  public stop(): void {
    this.isRunning = false;
    logger.info('Liquidation bot stopped');
  }

  /**
   * Monitor all positions for liquidation opportunities
   */
  private async monitorPositions(): Promise<void> {
    try {
      // Get all active positions
      const positions = await this.db.query(
        `SELECT p.*, m.symbol, m.maintenance_margin_ratio, m.oracle_account
         FROM positions p
         JOIN markets m ON p.market_id = m.id
         WHERE p.size > 0 AND NOT p.is_liquidated`
      );

      const liquidationCandidates: LiquidationCandidate[] = [];

      for (const position of positions.rows) {
        const healthFactor = await this.calculateHealthFactor(position);
        
        if (healthFactor < this.liquidationThreshold) {
          liquidationCandidates.push({
            positionId: position.id,
            userId: position.user_id,
            marketId: position.market_id,
            healthFactor,
            liquidationPrice: position.liquidation_price || 0,
            estimatedReward: this.calculateLiquidationReward(position)
          });
        }
      }

      if (liquidationCandidates.length > 0) {
        logger.info(`Found ${liquidationCandidates.length} liquidation candidates`);
        await this.processLiquidationCandidates(liquidationCandidates);
      }

    } catch (error) {
      logger.error('Error monitoring positions:', error);
    }
  }

  /**
   * Calculate health factor for a position
   */
  private async calculateHealthFactor(position: any): Promise<number> {
    try {
      // Get current oracle price
      const oraclePrice = await this.oracle.getPrice(position.symbol.replace('-PERP', ''));
      if (!oraclePrice) {
        logger.warn(`No oracle price available for ${position.symbol}`);
        return 1.0; // Safe default
      }

      // Calculate unrealized P&L
      const unrealizedPnl = this.calculateUnrealizedPnl(position, oraclePrice.price);
      
      // Calculate equity
      const equity = position.margin + unrealizedPnl;
      
      // Calculate position value
      const positionValue = position.size * oraclePrice.price;
      
      // Calculate health factor
      const healthFactor = equity / positionValue;
      
      return healthFactor;
    } catch (error) {
      logger.error(`Error calculating health factor for position ${position.id}:`, error);
      return 1.0; // Safe default
    }
  }

  /**
   * Calculate unrealized P&L for a position
   */
  private calculateUnrealizedPnl(position: any, currentPrice: number): number {
    const entryPrice = position.entry_price;
    const size = position.size;
    
    if (position.side === 'long') {
      return (currentPrice - entryPrice) * size;
    } else {
      return (entryPrice - currentPrice) * size;
    }
  }

  /**
   * Calculate estimated liquidation reward
   */
  private calculateLiquidationReward(position: any): number {
    // Liquidation reward is typically 5% of the position value
    const positionValue = position.size * position.entry_price;
    return positionValue * 0.05;
  }

  /**
   * Process liquidation candidates
   */
  private async processLiquidationCandidates(candidates: LiquidationCandidate[]): Promise<void> {
    for (const candidate of candidates) {
      try {
        // Check if liquidation is still valid
        const position = await this.db.getPositionById(candidate.positionId);
        if (!position || position.is_liquidated) {
          continue;
        }

        // Check if reward is sufficient
        if (candidate.estimatedReward < this.minLiquidationReward) {
          logger.info(`Skipping liquidation for position ${candidate.positionId} - reward too low`);
          continue;
        }

        // Execute liquidation
        await this.executeLiquidation(candidate);

      } catch (error) {
        logger.error(`Error processing liquidation candidate ${candidate.positionId}:`, error);
      }
    }
  }

  /**
   * Execute liquidation transaction
   */
  private async executeLiquidation(candidate: LiquidationCandidate): Promise<void> {
    try {
      logger.info(`Executing liquidation for position ${candidate.positionId}`);

      // Create liquidation transaction
      const transaction = new Transaction();
      
      // Add liquidation instruction
      const liquidationInstruction = await this.createLiquidationInstruction(candidate);
      transaction.add(liquidationInstruction);

      // Set recent blockhash
      const { blockhash } = await this.connection.getLatestBlockhash();
      transaction.recentBlockhash = blockhash;

      // Sign and send transaction
      // Note: In production, you would need a liquidator keypair
      // For now, we'll just log the transaction
      logger.info(`Liquidation transaction prepared for position ${candidate.positionId}`);

      // Update position in database
      await this.updatePositionAfterLiquidation(candidate);

      // Record liquidation event
      await this.recordLiquidation(candidate);

      logger.info(`Liquidation completed for position ${candidate.positionId}`);

    } catch (error) {
      logger.error(`Error executing liquidation for position ${candidate.positionId}:`, error);
    }
  }

  /**
   * Create liquidation instruction
   */
  private async createLiquidationInstruction(candidate: LiquidationCandidate): Promise<TransactionInstruction> {
    // This would create the actual liquidation instruction
    // For now, return a placeholder
    return new TransactionInstruction({
      keys: [],
      programId: new PublicKey(config.PROGRAM_ID),
      data: Buffer.from([])
    });
  }

  /**
   * Update position after liquidation
   */
  private async updatePositionAfterLiquidation(candidate: LiquidationCandidate): Promise<void> {
    try {
      await this.db.updatePosition(candidate.positionId, {
        is_liquidated: true,
        size: 0,
        closed_at: new Date()
      });

      logger.info(`Position ${candidate.positionId} marked as liquidated`);
    } catch (error) {
      logger.error(`Error updating position ${candidate.positionId} after liquidation:`, error);
    }
  }

  /**
   * Record liquidation event
   */
  private async recordLiquidation(candidate: LiquidationCandidate): Promise<void> {
    try {
      await this.db.query(
        `INSERT INTO liquidations (user_id, market_id, position_id, liquidator_address, 
         liquidation_type, liquidated_size, liquidation_price, liquidation_fee, remaining_margin)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
        [
          candidate.userId,
          candidate.marketId,
          candidate.positionId,
          'LIQUIDATION_BOT', // In production, use actual liquidator address
          'market',
          0, // liquidated_size - would be calculated
          candidate.liquidationPrice,
          candidate.estimatedReward,
          0 // remaining_margin - would be calculated
        ]
      );

      logger.info(`Liquidation recorded for position ${candidate.positionId}`);
    } catch (error) {
      logger.error(`Error recording liquidation for position ${candidate.positionId}:`, error);
    }
  }

  /**
   * Get liquidation statistics
   */
  public async getLiquidationStats(): Promise<any> {
    try {
      const stats = await this.db.query(
        `SELECT 
           COUNT(*) as total_liquidations,
           SUM(liquidation_fee) as total_fees,
           AVG(liquidation_fee) as avg_fee,
           COUNT(CASE WHEN created_at >= NOW() - INTERVAL '24 hours' THEN 1 END) as liquidations_24h
         FROM liquidations`
      );

      return stats.rows[0];
    } catch (error) {
      logger.error('Error getting liquidation stats:', error);
      return null;
    }
  }

  /**
   * Health check for liquidation bot
   */
  public async healthCheck(): Promise<boolean> {
    try {
      // Check if we can query positions
      const positions = await this.db.query('SELECT COUNT(*) FROM positions WHERE size > 0');
      return positions.rows.length > 0;
    } catch (error) {
      logger.error('Liquidation bot health check failed:', error);
      return false;
    }
  }
}
