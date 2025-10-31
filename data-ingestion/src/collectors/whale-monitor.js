const { Connection, PublicKey } = require('@solana/web3.js');
const { redis, logger, STREAMS, STREAM_CONFIG } = require('../config');

class WhaleMonitor {
  constructor() {
    console.log('SOLANA_RPC_URL:', process.env.SOLANA_RPC_URL);
    console.log('SOLANA_WS_URL:', process.env.SOLANA_WS_URL);
    this.connection = new Connection(process.env.SOLANA_RPC_URL);
    // Use the same RPC URL for WebSocket connection (Solana Web3.js handles this)
    this.wsConnection = new Connection(process.env.SOLANA_RPC_URL);
    this.whaleThreshold = parseFloat(process.env.WHALE_THRESHOLD) || 100000;
    this.whaleWallets = new Set();
    this.isRunning = false;
    this.subscriptionId = null;
  }

  async start() {
    if (this.isRunning) {
      logger.warn('Whale monitor already running');
      return;
    }

    logger.info('Starting whale monitor...');
    this.isRunning = true;

    // Load whale wallets
    await this.loadWhaleWallets();

    // Subscribe to account changes
    await this.subscribeToAccountChanges();

    logger.info(`Whale monitor started, monitoring ${this.whaleWallets.size} wallets`);
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    logger.info('Stopping whale monitor...');
    this.isRunning = false;

    if (this.subscriptionId) {
      await this.wsConnection.removeAccountChangeListener(this.subscriptionId);
    }

    logger.info('Whale monitor stopped');
  }

  async loadWhaleWallets() {
    try {
      // Load from Redis cache first
      const cachedWallets = await redis.smembers('whale_wallets');
      if (cachedWallets.length > 0) {
        cachedWallets.forEach(wallet => this.whaleWallets.add(wallet));
        logger.info(`Loaded ${cachedWallets.length} whale wallets from cache`);
        return;
      }

      // Skip database loading for now due to SSL issues
      // Add some sample whale wallets for testing
      const sampleWhales = [
        '9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM', // Sample whale wallet
        'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'  // USDC mint (for testing)
      ];
      
      sampleWhales.forEach(wallet => {
        this.whaleWallets.add(wallet);
      });

      // Cache in Redis
      if (this.whaleWallets.size > 0) {
        await redis.sadd('whale_wallets', ...this.whaleWallets);
        await redis.expire('whale_wallets', 3600); // Cache for 1 hour
      }

      logger.info(`Loaded ${this.whaleWallets.size} sample whale wallets (database disabled)`);
    } catch (error) {
      logger.error('Error loading whale wallets:', error);
      // Don't throw error, just start with empty set
      logger.warn('Starting whale monitor with empty wallet set');
    }
  }

  async subscribeToAccountChanges() {
    try {
      // Subscribe to program account changes for Drift program
      const programId = new PublicKey(process.env.PROGRAM_ID || 'G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J');
      
      this.subscriptionId = await this.wsConnection.onProgramAccountChange(
        programId,
        async (accountInfo, context) => {
          try {
            await this.processAccountChange(accountInfo, context);
          } catch (error) {
            logger.error('Error processing account change:', error);
          }
        },
        'confirmed'
      );

      logger.info('Subscribed to program account changes');
    } catch (error) {
      logger.error('Error subscribing to account changes:', error);
      throw error;
    }
  }

  async processAccountChange(accountInfo, context) {
    try {
      // Parse account data to detect whale activity
      const accountData = accountInfo.accountInfo.data;
      
      // This is simplified - you'd need to implement proper Drift account parsing
      const whaleEvent = await this.detectWhaleActivity(accountInfo, context);
      
      if (whaleEvent) {
        await this.publishWhaleEvent(whaleEvent);
        logger.info(`Whale activity detected: ${whaleEvent.type} - ${whaleEvent.wallet_address}`);
      }
    } catch (error) {
      logger.error('Error processing account change:', error);
    }
  }

  async detectWhaleActivity(accountInfo, context) {
    try {
      // Simplified whale detection logic
      // In reality, you'd parse the account data to extract:
      // - Position sizes
      // - Trade amounts
      // - Wallet addresses
      
      const accountPubkey = accountInfo.accountId.toString();
      
      // Check if this account belongs to a known whale wallet
      if (this.whaleWallets.has(accountPubkey)) {
        return {
          type: 'whale_position_change',
          wallet_address: accountPubkey,
          account_type: 'user_account',
          slot: context.slot,
          timestamp: new Date().toISOString(),
          // Additional fields would be extracted from account data
          position_size: 0, // Placeholder
          notional_value_usd: 0, // Placeholder
          action: 'unknown' // Placeholder
        };
      }

      return null;
    } catch (error) {
      logger.error('Error detecting whale activity:', error);
      return null;
    }
  }

  async publishWhaleEvent(whaleEvent) {
    try {
      await redis.xadd(
        STREAMS.WHALES_RAW,
        'MAXLEN',
        '~',
        STREAM_CONFIG.maxLen,
        '*',
        'data', JSON.stringify(whaleEvent),
        'wallet_address', whaleEvent.wallet_address,
        'event_type', whaleEvent.type,
        'timestamp', whaleEvent.timestamp
      );
    } catch (error) {
      logger.error('Error publishing whale event:', error);
      throw error;
    }
  }

  async addWhaleWallet(walletAddress) {
    try {
      this.whaleWallets.add(walletAddress);
      await redis.sadd('whale_wallets', walletAddress);
      logger.info(`Added whale wallet: ${walletAddress}`);
    } catch (error) {
      logger.error('Error adding whale wallet:', error);
      throw error;
    }
  }

  async removeWhaleWallet(walletAddress) {
    try {
      this.whaleWallets.delete(walletAddress);
      await redis.srem('whale_wallets', walletAddress);
      logger.info(`Removed whale wallet: ${walletAddress}`);
    } catch (error) {
      logger.error('Error removing whale wallet:', error);
      throw error;
    }
  }

  async refreshWhaleWallets() {
    try {
      logger.info('Refreshing whale wallets...');
      this.whaleWallets.clear();
      await redis.del('whale_wallets');
      await this.loadWhaleWallets();
      logger.info(`Refreshed whale wallets: ${this.whaleWallets.size} wallets`);
    } catch (error) {
      logger.error('Error refreshing whale wallets:', error);
      throw error;
    }
  }
}

// Start monitor if run directly
if (require.main === module) {
  const monitor = new WhaleMonitor();
  
  monitor.start().catch(error => {
    logger.error('Failed to start whale monitor:', error);
    process.exit(1);
  });
}

module.exports = WhaleMonitor;
