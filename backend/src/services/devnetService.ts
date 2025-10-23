import { Connection, Keypair, PublicKey } from '@solana/web3.js';
import { AnchorProvider, Program } from '@coral-xyz/anchor';
import { DEVNET_CONFIG } from '../config/devnet';

export interface OpenPositionParams {
  userId: string;
  market: string;
  side: 'long' | 'short';
  size: number;
  leverage: number;
}

export interface ClosePositionParams {
  positionId: string;
  userId: string;
}

export class DevnetService {
  private connection: Connection;
  private provider: AnchorProvider;
  private program: Program | null = null;
  
  constructor() {
    this.connection = new Connection(
      DEVNET_CONFIG.rpcEndpoint,
      DEVNET_CONFIG.commitment
    );
    
    // Initialize provider (will need wallet keypair)
    try {
      this.provider = new AnchorProvider(
        this.connection,
        {} as any, // Wallet placeholder
        { commitment: DEVNET_CONFIG.commitment }
      );
    } catch (error) {
      console.log('DevnetService: Provider initialization failed:', error);
    }
  }
  
  async getMarkets() {
    try {
      if (!this.program) {
        throw new Error('Program not initialized');
      }
      
      // Fetch markets from program
      const markets = await (this.program.account as any).market?.all() || [];
      return markets.map(market => ({
        id: market.publicKey.toString(),
        symbol: market.account.symbol || 'Unknown',
        baseAsset: market.account.baseAsset || 'Unknown',
        quoteAsset: market.account.quoteAsset || 'USDC',
        isActive: market.account.isActive || false,
      }));
    } catch (error) {
      console.log('DevnetService: Error fetching markets:', error);
      return [];
    }
  }
  
  async openPosition(params: OpenPositionParams) {
    try {
      if (!this.program) {
        throw new Error('Program not initialized');
      }
      
      // Create position transaction
      console.log('DevnetService: Opening position:', params);
      
      // For now, return mock success
      return {
        success: true,
        positionId: `pos_${Date.now()}`,
        transactionId: `tx_${Date.now()}`,
      };
    } catch (error) {
      console.log('DevnetService: Error opening position:', error);
      throw error;
    }
  }
  
  async closePosition(params: ClosePositionParams) {
    try {
      if (!this.program) {
        throw new Error('Program not initialized');
      }
      
      // Close position transaction
      console.log('DevnetService: Closing position:', params);
      
      // For now, return mock success
      return {
        success: true,
        transactionId: `tx_${Date.now()}`,
      };
    } catch (error) {
      console.log('DevnetService: Error closing position:', error);
      throw error;
    }
  }
  
  async getConnection(): Promise<Connection> {
    return this.connection;
  }
  
  async isConnected(): Promise<boolean> {
    try {
      const version = await this.connection.getVersion();
      return !!version;
    } catch (error) {
      return false;
    }
  }
  
  async getProgramId(): Promise<string> {
    return DEVNET_CONFIG.programs.quantdeskPerp || '';
  }
}

// Export singleton instance
export const devnetService = new DevnetService();
