// System Mode Service
// Manages demo/live mode switching and related functionality

import { Logger } from '../utils/logger';
import { databaseService } from './supabaseDatabase'; // Updated import
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
  private supabase: typeof databaseService; // Changed from private db: DatabaseService;
  private wsService?: WebSocketService;
  private config: SystemConfig;
  private modeChangeListeners: ((event: ModeChangeEvent) => void)[] = [];

  private constructor() {
    this.logger = new Logger();
    this.supabase = databaseService; // Directly assign the imported singleton
    // this.wsService = WebSocketService.getInstance(); // Will be initialized later with SocketIOServer
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
      const result = await this.supabase.select('system_config', 'mode', { key: 'current_mode' });

      if (result.length > 0) {
        this.currentMode = result[0].mode as SystemMode;
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
      await this.supabase.upsert('system_config', {
        key: 'current_mode',
        value: mode,
        mode: mode,
        updated_at: new Date().toISOString()
      });
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
      if (this.wsService) {
        this.wsService.broadcast('system_mode_changed', {
          mode: newMode,
          timestamp: event.timestamp,
          userId,
          reason
        });
      }

      this.logger.info(`System mode successfully changed to ${newMode}`);

    } catch (error) {
      this.logger.error(`Failed to change system mode to ${newMode}:`, error);
      throw error;
    }
  }

  private async validateModeChange(oldMode: SystemMode, newMode: SystemMode): Promise<void> {
    // Check if there are any active positions
    if (newMode === 'demo' && oldMode === 'live') {
      const activePositions = await this.supabase.count('positions', { size: { gt: 0 } });

      if (activePositions > 0) {
        throw new Error('Cannot switch to demo mode with active live positions');
      }
    }

    // Check system health
    const healthCheck = await this.supabase.healthCheck();
    if (!healthCheck) {
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
    await this.supabase.upsert('risk_limits', {
      mode: mode,
      max_position_size: limits.maxPositionSize,
      max_leverage: limits.maxLeverage,
      max_daily_loss: limits.maxDailyLoss,
      updated_at: new Date().toISOString()
    });

    this.logger.info(`Updated risk management settings for ${mode} mode`);
  }

  private async updateApiEndpoints(mode: SystemMode): Promise<void> {
    const endpoints = this.config.apiEndpoints[mode];
    
    // Update API endpoints in database
    await this.supabase.upsert('api_endpoints', {
      mode: mode,
      endpoints: JSON.stringify(endpoints),
      updated_at: new Date().toISOString()
    });

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
      const result = await this.supabase.select('mode_change_history', 'old_mode, new_mode, timestamp, user_id, reason', {}, { 
        orderBy: 'timestamp', 
        orderDirection: 'desc', 
        limit: limit 
      });

      return result.map(row => ({
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
      await this.supabase.insert('mode_change_history', {
        old_mode: event.oldMode,
        new_mode: event.newMode,
        timestamp: event.timestamp,
        user_id: event.userId,
        reason: event.reason
      });
    } catch (error) {
      this.logger.error('Failed to log mode change:', error);
    }
  }
}

export const systemModeService = SystemModeService.getInstance();
