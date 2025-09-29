// System Mode Service
// Manages demo/live mode switching and related functionality

import { Logger } from '../utils/logger';
import { DatabaseService } from './database';
import { WebSocketService } from './websocket';

export type SystemMode = 'demo' | 'live';

export interface ModeChangeEvent {
  oldMode: SystemMode;
  newMode: SystemMode;
  timestamp: string;
  userId?: string;
  reason?: string;
}

export interface SystemConfig {
  mode: SystemMode;
  tradingEnabled: boolean;
  apiEndpoints: {
    demo: string[];
    live: string[];
  };
  riskLimits: {
    demo: {
      maxPositionSize: number;
      maxLeverage: number;
      maxDailyLoss: number;
    };
    live: {
      maxPositionSize: number;
      maxLeverage: number;
      maxDailyLoss: number;
    };
  };
}

class SystemModeService {
  private static instance: SystemModeService;
  private currentMode: SystemMode = 'demo';
  private logger: Logger;
  private db: DatabaseService;
  private wsService: WebSocketService;
  private config: SystemConfig;
  private modeChangeListeners: ((event: ModeChangeEvent) => void)[] = [];

  private constructor() {
    this.logger = new Logger();
    this.db = DatabaseService.getInstance();
    this.wsService = WebSocketService.getInstance();
    this.config = this.getDefaultConfig();
    this.loadModeFromDatabase();
  }

  public static getInstance(): SystemModeService {
    if (!SystemModeService.instance) {
      SystemModeService.instance = new SystemModeService();
    }
    return SystemModeService.instance;
  }

  private getDefaultConfig(): SystemConfig {
    return {
      mode: 'demo',
      tradingEnabled: true,
      apiEndpoints: {
        demo: [
          'https://api-devnet.solana.com',
          'https://devnet.helius-rpc.com'
        ],
        live: [
          'https://api.mainnet-beta.solana.com',
          'https://mainnet.helius-rpc.com'
        ]
      },
      riskLimits: {
        demo: {
          maxPositionSize: 10000, // $10,000
          maxLeverage: 100, // 100x
          maxDailyLoss: 1000 // $1,000
        },
        live: {
          maxPositionSize: 100000, // $100,000
          maxLeverage: 50, // 50x
          maxDailyLoss: 10000 // $10,000
        }
      }
    };
  }

  private async loadModeFromDatabase(): Promise<void> {
    try {
      const result = await this.db.query(
        'SELECT mode FROM system_config WHERE key = $1',
        ['current_mode']
      );

      if (result.rows.length > 0) {
        this.currentMode = result.rows[0].mode as SystemMode;
        this.config.mode = this.currentMode;
        this.logger.info(`Loaded system mode from database: ${this.currentMode}`);
      } else {
        // Initialize with demo mode
        await this.saveModeToDatabase('demo');
        this.logger.info('Initialized system mode to demo');
      }
    } catch (error) {
      this.logger.error('Failed to load system mode from database:', error);
      // Fallback to demo mode
      this.currentMode = 'demo';
    }
  }

  private async saveModeToDatabase(mode: SystemMode): Promise<void> {
    try {
      await this.db.query(
        `INSERT INTO system_config (key, value, mode, updated_at) 
         VALUES ($1, $2, $3, NOW()) 
         ON CONFLICT (key) 
         DO UPDATE SET value = $2, mode = $3, updated_at = NOW()`,
        ['current_mode', mode, mode]
      );
    } catch (error) {
      this.logger.error('Failed to save system mode to database:', error);
      throw error;
    }
  }

  public getCurrentMode(): SystemMode {
    return this.currentMode;
  }

  public getConfig(): SystemConfig {
    return { ...this.config };
  }

  public getRiskLimits() {
    return this.config.riskLimits[this.currentMode];
  }

  public getApiEndpoints(): string[] {
    return this.config.apiEndpoints[this.currentMode];
  }

  public async setMode(newMode: SystemMode, userId?: string, reason?: string): Promise<void> {
    if (newMode === this.currentMode) {
      this.logger.info(`System mode is already ${newMode}`);
      return;
    }

    const oldMode = this.currentMode;
    const event: ModeChangeEvent = {
      oldMode,
      newMode,
      timestamp: new Date().toISOString(),
      userId,
      reason
    };

    try {
      this.logger.info(`Changing system mode from ${oldMode} to ${newMode}`, {
        userId,
        reason
      });

      // Validate mode change
      await this.validateModeChange(oldMode, newMode);

      // Update configuration
      this.currentMode = newMode;
      this.config.mode = newMode;

      // Save to database
      await this.saveModeToDatabase(newMode);

      // Apply mode-specific settings
      await this.applyModeSettings(newMode);

      // Notify listeners
      this.notifyModeChangeListeners(event);

      // Broadcast to WebSocket clients
      this.wsService.broadcast('system_mode_changed', {
        mode: newMode,
        timestamp: event.timestamp,
        userId,
        reason
      });

      this.logger.info(`System mode successfully changed to ${newMode}`);

    } catch (error) {
      this.logger.error(`Failed to change system mode to ${newMode}:`, error);
      throw error;
    }
  }

  private async validateModeChange(oldMode: SystemMode, newMode: SystemMode): Promise<void> {
    // Check if there are any active positions
    if (newMode === 'demo' && oldMode === 'live') {
      const activePositions = await this.db.query(
        'SELECT COUNT(*) as count FROM positions WHERE size > 0'
      );

      if (parseInt(activePositions.rows[0].count) > 0) {
        throw new Error('Cannot switch to demo mode with active live positions');
      }
    }

    // Check system health
    const healthCheck = await this.db.healthCheck();
    if (!healthCheck.healthy) {
      throw new Error('Cannot change mode: system health check failed');
    }
  }

  private async applyModeSettings(mode: SystemMode): Promise<void> {
    try {
      // Update trading engine settings
      await this.updateTradingEngineSettings(mode);

      // Update risk management settings
      await this.updateRiskManagementSettings(mode);

      // Update API endpoints
      await this.updateApiEndpoints(mode);

      // Update logging configuration
      await this.updateLoggingConfiguration(mode);

      this.logger.info(`Applied ${mode} mode settings`);

    } catch (error) {
      this.logger.error(`Failed to apply ${mode} mode settings:`, error);
      throw error;
    }
  }

  private async updateTradingEngineSettings(mode: SystemMode): Promise<void> {
    // TODO: Implement trading engine settings update
    this.logger.info(`Updated trading engine settings for ${mode} mode`);
  }

  private async updateRiskManagementSettings(mode: SystemMode): Promise<void> {
    const limits = this.config.riskLimits[mode];
    
    // Update risk limits in database
    await this.db.query(
      `INSERT INTO risk_limits (mode, max_position_size, max_leverage, max_daily_loss, updated_at) 
       VALUES ($1, $2, $3, $4, NOW()) 
       ON CONFLICT (mode) 
       DO UPDATE SET max_position_size = $2, max_leverage = $3, max_daily_loss = $4, updated_at = NOW()`,
      [mode, limits.maxPositionSize, limits.maxLeverage, limits.maxDailyLoss]
    );

    this.logger.info(`Updated risk management settings for ${mode} mode`);
  }

  private async updateApiEndpoints(mode: SystemMode): Promise<void> {
    const endpoints = this.config.apiEndpoints[mode];
    
    // Update API endpoints in database
    await this.db.query(
      `INSERT INTO api_endpoints (mode, endpoints, updated_at) 
       VALUES ($1, $2, NOW()) 
       ON CONFLICT (mode) 
       DO UPDATE SET endpoints = $2, updated_at = NOW()`,
      [mode, JSON.stringify(endpoints)]
    );

    this.logger.info(`Updated API endpoints for ${mode} mode`);
  }

  private async updateLoggingConfiguration(mode: SystemMode): Promise<void> {
    // TODO: Implement logging configuration update
    this.logger.info(`Updated logging configuration for ${mode} mode`);
  }

  public addModeChangeListener(listener: (event: ModeChangeEvent) => void): void {
    this.modeChangeListeners.push(listener);
  }

  public removeModeChangeListener(listener: (event: ModeChangeEvent) => void): void {
    const index = this.modeChangeListeners.indexOf(listener);
    if (index > -1) {
      this.modeChangeListeners.splice(index, 1);
    }
  }

  private notifyModeChangeListeners(event: ModeChangeEvent): void {
    this.modeChangeListeners.forEach(listener => {
      try {
        listener(event);
      } catch (error) {
        this.logger.error('Error in mode change listener:', error);
      }
    });
  }

  public isDemoMode(): boolean {
    return this.currentMode === 'demo';
  }

  public isLiveMode(): boolean {
    return this.currentMode === 'live';
  }

  public getModeDisplayName(): string {
    return this.currentMode.toUpperCase();
  }

  public async getModeHistory(limit: number = 50): Promise<ModeChangeEvent[]> {
    try {
      const result = await this.db.query(
        `SELECT old_mode, new_mode, timestamp, user_id, reason 
         FROM mode_change_history 
         ORDER BY timestamp DESC 
         LIMIT $1`,
        [limit]
      );

      return result.rows.map(row => ({
        oldMode: row.old_mode as SystemMode,
        newMode: row.new_mode as SystemMode,
        timestamp: row.timestamp,
        userId: row.user_id,
        reason: row.reason
      }));

    } catch (error) {
      this.logger.error('Failed to get mode history:', error);
      return [];
    }
  }

  public async logModeChange(event: ModeChangeEvent): Promise<void> {
    try {
      await this.db.query(
        `INSERT INTO mode_change_history (old_mode, new_mode, timestamp, user_id, reason) 
         VALUES ($1, $2, $3, $4, $5)`,
        [event.oldMode, event.newMode, event.timestamp, event.userId, event.reason]
      );
    } catch (error) {
      this.logger.error('Failed to log mode change:', error);
    }
  }
}

export const systemModeService = SystemModeService.getInstance();
