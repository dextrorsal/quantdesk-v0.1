const { Connection, PublicKey } = require('@solana/web3.js');
const axios = require('axios');
const { redis, logger, STREAMS, STREAM_CONFIG } = require('../config');

class RealWhaleMonitor {
  constructor() {
    this.connection = new Connection(process.env.SOLANA_RPC_URL);
    this.whaleThreshold = parseFloat(process.env.WHALE_THRESHOLD) || 100000;
    this.isRunning = false;
    this.subscriptionId = null;
    this.knownWhales = new Set();
    this.recentTransactions = new Map(); // Cache to avoid duplicates
  }

  async start() {
    if (this.isRunning) {
      logger.warn('Real whale monitor already running');
      return;
    }

    logger.info('Starting real whale monitor...');
    this.isRunning = true;

    // Load known whale wallets
    await this.loadKnownWhales();

    // Start monitoring large transactions
    await this.startTransactionMonitoring();

    logger.info(`Real whale monitor started, threshold: $${this.whaleThreshold}`);
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    logger.info('Stopping real whale monitor...');
    this.isRunning = false;

    if (this.subscriptionId) {
      await this.connection.removeAccountChangeListener(this.subscriptionId);
    }

    logger.info('Real whale monitor stopped');
  }

  async loadKnownWhales() {
    try {
      // Load from Redis cache first
      const cachedWhales = await redis.smembers('known_whales');
      if (cachedWhales.length > 0) {
        cachedWhales.forEach(wallet => this.knownWhales.add(wallet));
        logger.info(`Loaded ${cachedWhales.length} known whales from cache`);
        return;
      }

      // Load from popular whale wallets (you can expand this list)
      const popularWhales = [
        '9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM', // Example whale
        'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  // USDC mint
        'So11111111111111111111111111111111111111112',     // SOL mint
        // Add more known whale addresses here
      ];

      popularWhales.forEach(wallet => {
        this.knownWhales.add(wallet);
      });

      // Cache in Redis
      if (this.knownWhales.size > 0) {
        await redis.sadd('known_whales', ...this.knownWhales);
        await redis.expire('known_whales', 3600); // Cache for 1 hour
      }

      logger.info(`Loaded ${this.knownWhales.size} known whale wallets`);
    } catch (error) {
      logger.error('Error loading known whales:', error);
    }
  }

  async startTransactionMonitoring() {
    try {
      // Monitor program account changes for Drift Protocol
      const driftProgramId = new PublicKey(process.env.DRIFT_PROGRAM_ID || 'dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH');
      
      this.subscriptionId = await this.connection.onProgramAccountChange(
        driftProgramId,
        async (accountInfo, context) => {
          try {
            await this.processAccountChange(accountInfo, context);
          } catch (error) {
            logger.error('Error processing account change:', error);
          }
        },
        'confirmed'
      );

      // Also monitor for large SOL transfers
      await this.monitorLargeTransfers();

      logger.info('Started monitoring program account changes and large transfers');
    } catch (error) {
      logger.error('Error starting transaction monitoring:', error);
      throw error;
    }
  }

  async monitorLargeTransfers() {
    try {
      // Monitor recent blocks for large transactions
      setInterval(async () => {
        try {
          await this.scanRecentBlocks();
        } catch (error) {
          logger.error('Error scanning recent blocks:', error);
        }
      }, 30000); // Check every 30 seconds

      logger.info('Started monitoring recent blocks for large transfers');
    } catch (error) {
      logger.error('Error starting block monitoring:', error);
    }
  }

  async scanRecentBlocks() {
    try {
      const currentSlot = await this.connection.getSlot();
      const startSlot = Math.max(0, currentSlot - 10); // Check last 10 slots

      for (let slot = startSlot; slot <= currentSlot; slot++) {
        try {
          const block = await this.connection.getBlock(slot, {
            commitment: 'confirmed',
            maxSupportedTransactionVersion: 0
          });

          if (block && block.transactions) {
            for (const tx of block.transactions) {
              await this.analyzeTransaction(tx, slot);
            }
          }
        } catch (slotError) {
          // Skip failed slot reads
          continue;
        }
      }
    } catch (error) {
      logger.error('Error scanning recent blocks:', error);
    }
  }

  async analyzeTransaction(tx, slot) {
    try {
      if (!tx.transaction || !tx.meta) return;

      const signature = tx.transaction.signatures[0];
      if (!signature) return;

      // Skip if we've already processed this transaction
      if (this.recentTransactions.has(signature)) return;
      this.recentTransactions.set(signature, Date.now());

      // Clean old transactions from cache
      if (this.recentTransactions.size > 1000) {
        const now = Date.now();
        for (const [sig, timestamp] of this.recentTransactions.entries()) {
          if (now - timestamp > 300000) { // 5 minutes
            this.recentTransactions.delete(sig);
          }
        }
      }

      // Analyze transaction for whale activity
      const whaleActivity = await this.detectWhaleActivity(tx, slot);
      
      if (whaleActivity) {
        await this.publishWhaleEvent(whaleActivity);
        logger.info(`🐋 Whale activity detected: ${whaleActivity.type} - ${whaleActivity.wallet_address}`);
      }
    } catch (error) {
      logger.error('Error analyzing transaction:', error);
    }
  }

  async detectWhaleActivity(tx, slot) {
    try {
      const signature = tx.transaction.signatures[0];
      const meta = tx.meta;
      const message = tx.transaction.message;

      // Check for large SOL transfers
      const preBalances = meta.preBalances || [];
      const postBalances = meta.postBalances || [];
      const accountKeys = message.accountKeys || [];

      for (let i = 0; i < preBalances.length; i++) {
        const preBalance = preBalances[i];
        const postBalance = postBalances[i];
        const balanceChange = postBalance - preBalance;

        // Convert lamports to SOL
        const solChange = balanceChange / 1e9;

        // Check if this is a significant transfer (> 100 SOL or > $10k)
        if (Math.abs(solChange) > 100) {
          const walletAddress = accountKeys[i]?.toString();
          if (walletAddress) {
            // Check if this is a known whale
            const isKnownWhale = this.knownWhales.has(walletAddress);
            
            return {
              type: 'large_sol_transfer',
              wallet_address: walletAddress,
              signature: signature,
              slot: slot,
              amount_sol: solChange,
              amount_usd: solChange * 200, // Approximate SOL price
              is_known_whale: isKnownWhale,
              timestamp: new Date().toISOString(),
              transaction_type: solChange > 0 ? 'incoming' : 'outgoing'
            };
          }
        }
      }

      // Check for token transfers (simplified)
      const tokenTransfers = this.extractTokenTransfers(tx);
      for (const transfer of tokenTransfers) {
        if (transfer.amount_usd > this.whaleThreshold) {
          return {
            type: 'large_token_transfer',
            wallet_address: transfer.from_address,
            signature: signature,
            slot: slot,
            token_mint: transfer.mint,
            amount: transfer.amount,
            amount_usd: transfer.amount_usd,
            is_known_whale: this.knownWhales.has(transfer.from_address),
            timestamp: new Date().toISOString(),
            transaction_type: 'token_transfer'
          };
        }
      }

      return null;
    } catch (error) {
      logger.error('Error detecting whale activity:', error);
      return null;
    }
  }

  extractTokenTransfers(tx) {
    // Simplified token transfer extraction
    // In production, you'd use a proper token program parser
    const transfers = [];
    
    try {
      const meta = tx.meta;
      if (meta && meta.postTokenBalances) {
        for (const balance of meta.postTokenBalances) {
          if (balance.uiTokenAmount && balance.uiTokenAmount.uiAmount) {
            const amount = balance.uiTokenAmount.uiAmount;
            const mint = balance.mint;
            
            // Estimate USD value (simplified)
            let amountUsd = 0;
            if (mint === 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v') {
              amountUsd = amount; // USDC
            } else if (mint === 'So11111111111111111111111111111111111111112') {
              amountUsd = amount * 200; // SOL
            }

            if (amountUsd > 1000) { // Only track transfers > $1k
              transfers.push({
                mint: mint,
                amount: amount,
                amount_usd: amountUsd,
                from_address: balance.owner,
                to_address: balance.owner // Simplified
              });
            }
          }
        }
      }
    } catch (error) {
      logger.error('Error extracting token transfers:', error);
    }

    return transfers;
  }

  async processAccountChange(accountInfo, context) {
    try {
      // This would be called for Drift program account changes
      // You'd parse the account data to detect position changes, liquidations, etc.
      
      const accountPubkey = accountInfo.accountId.toString();
      
      // Check if this account belongs to a known whale
      if (this.knownWhales.has(accountPubkey)) {
        const whaleEvent = {
          type: 'whale_account_change',
          wallet_address: accountPubkey,
          slot: context.slot,
          timestamp: new Date().toISOString(),
          account_type: 'drift_position',
          // Additional fields would be extracted from account data
          position_size: 0, // Placeholder - would parse actual account data
          notional_value_usd: 0, // Placeholder
          action: 'position_update' // Placeholder
        };

        await this.publishWhaleEvent(whaleEvent);
        logger.info(`🐋 Whale account change detected: ${accountPubkey}`);
      }
    } catch (error) {
      logger.error('Error processing account change:', error);
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
        'amount_usd', whaleEvent.amount_usd?.toString() || '0',
        'timestamp', whaleEvent.timestamp
      );

      logger.info(`Published whale event: ${whaleEvent.type}`);
    } catch (error) {
      logger.error('Error publishing whale event:', error);
      throw error;
    }
  }

  async addWhaleWallet(walletAddress) {
    try {
      this.knownWhales.add(walletAddress);
      await redis.sadd('known_whales', walletAddress);
      logger.info(`Added whale wallet: ${walletAddress}`);
    } catch (error) {
      logger.error('Error adding whale wallet:', error);
      throw error;
    }
  }

  async removeWhaleWallet(walletAddress) {
    try {
      this.knownWhales.delete(walletAddress);
      await redis.srem('known_whales', walletAddress);
      logger.info(`Removed whale wallet: ${walletAddress}`);
    } catch (error) {
      logger.error('Error removing whale wallet:', error);
      throw error;
    }
  }

  async getWhaleStats() {
    try {
      const stats = {
        known_whales: this.knownWhales.size,
        recent_transactions: this.recentTransactions.size,
        threshold_usd: this.whaleThreshold,
        is_running: this.isRunning
      };

      return stats;
    } catch (error) {
      logger.error('Error getting whale stats:', error);
      throw error;
    }
  }
}

// Start monitor if run directly
if (require.main === module) {
  const monitor = new RealWhaleMonitor();
  
  monitor.start().catch(error => {
    logger.error('Failed to start real whale monitor:', error);
    process.exit(1);
  });

  // Graceful shutdown
  process.on('SIGINT', async () => {
    logger.info('Shutting down real whale monitor...');
    await monitor.stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    logger.info('SIGTERM received, shutting down real whale monitor...');
    await monitor.stop();
    process.exit(0);
  });
}

module.exports = RealWhaleMonitor;
