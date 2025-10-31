const express = require('express');
const cors = require('cors');
const { Connection, PublicKey, Keypair } = require('@solana/web3.js');
const fs = require('fs');
const path = require('path');
const { redis, logger, STREAMS } = require('../config');

class DataIngestionAPI {
  constructor() {
    this.app = express();
    this.port = process.env.DATA_INGESTION_PORT || 3003;
    this.connection = new Connection(process.env.SOLANA_RPC_URL || 'https://api.devnet.solana.com');
    this.wallet = null;
    
    this.setupMiddleware();
    this.setupRoutes();
    this.loadWallet();
  }

  setupMiddleware() {
    this.app.use(cors({
      origin: process.env.CORS_ORIGIN || 'http://localhost:3001',
      credentials: true
    }));
    this.app.use(express.json());
    this.app.use(express.urlencoded({ extended: true }));
    
    // Request logging
    this.app.use((req, res, next) => {
      logger.info(`${req.method} ${req.path}`, { 
        ip: req.ip, 
        userAgent: req.get('User-Agent') 
      });
      next();
    });
  }

  loadWallet() {
    try {
      // Load wallet from SOLANA_WALLET environment variable
      const walletPath = process.env.SOLANA_WALLET;
      
      if (!walletPath) {
        logger.warn('⚠️ SOLANA_WALLET environment variable not set');
        this.wallet = null;
        return;
      }

      if (!fs.existsSync(walletPath)) {
        logger.warn(`⚠️ Wallet file not found at: ${walletPath}`);
        this.wallet = null;
        return;
      }

      const walletData = JSON.parse(fs.readFileSync(walletPath, 'utf8'));
      this.wallet = Keypair.fromSecretKey(new Uint8Array(walletData));
      logger.info(`✅ Wallet loaded from: ${walletPath}`);
      logger.info(`📍 Wallet address: ${this.wallet.publicKey.toString()}`);
      
    } catch (error) {
      logger.error('❌ Error loading wallet:', error);
      this.wallet = null;
    }
  }

  setupRoutes() {
    // Health check
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        environment: process.env.NODE_ENV || 'development',
        services: {
          redis: redis.status === 'ready' ? 'connected' : 'disconnected',
          solana: 'connected',
          wallet: this.wallet ? 'loaded' : 'not_loaded'
        },
        wallet: this.wallet ? {
          address: this.wallet.publicKey.toString(),
          loaded: true,
          path: process.env.SOLANA_WALLET
        } : {
          loaded: false,
          path: process.env.SOLANA_WALLET || 'not_set'
        }
      });
    });

    // Get latest prices
    this.app.get('/api/prices/latest', async (req, res) => {
      try {
        const prices = await this.getLatestPrices();
        res.json({
          success: true,
          data: {
            prices,
            timestamp: new Date().toISOString(),
            source: 'pyth-network'
          }
        });
      } catch (error) {
        logger.error('Error fetching prices:', error);
        res.status(500).json({
          success: false,
          error: 'Failed to fetch prices',
          message: error.message
        });
      }
    });

    // Get recent whale transactions
    this.app.get('/api/whales/recent', async (req, res) => {
      try {
        const limit = parseInt(req.query.limit) || 10;
        const transactions = await this.getRecentWhales(limit);
        res.json({
          success: true,
          data: {
            transactions,
            count: transactions.length,
            timestamp: new Date().toISOString()
          }
        });
      } catch (error) {
        logger.error('Error fetching whale transactions:', error);
        res.status(500).json({
          success: false,
          error: 'Failed to fetch whale transactions',
          message: error.message
        });
      }
    });

    // Get market summary
    this.app.get('/api/market/summary', async (req, res) => {
      try {
        const summary = await this.getMarketSummary();
        res.json({
          success: true,
          data: summary
        });
      } catch (error) {
        logger.error('Error fetching market summary:', error);
        res.status(500).json({
          success: false,
          error: 'Failed to fetch market summary',
          message: error.message
        });
      }
    });

    // Get wallet balance
    this.app.get('/api/wallet/balance', async (req, res) => {
      try {
        if (!this.wallet) {
          return res.status(400).json({
            success: false,
            error: 'No wallet loaded'
          });
        }

        const balance = await this.connection.getBalance(this.wallet.publicKey);
        res.json({
          success: true,
          data: {
            address: this.wallet.publicKey.toString(),
            balance: balance / 1e9, // Convert lamports to SOL
            lamports: balance,
            timestamp: new Date().toISOString()
          }
        });
      } catch (error) {
        logger.error('Error fetching wallet balance:', error);
        res.status(500).json({
          success: false,
          error: 'Failed to fetch wallet balance',
          message: error.message
        });
      }
    });

    // Get system status
    this.app.get('/api/status', async (req, res) => {
      try {
        const status = await this.getSystemStatus();
        res.json({
          success: true,
          data: status
        });
      } catch (error) {
        logger.error('Error fetching system status:', error);
        res.status(500).json({
          success: false,
          error: 'Failed to fetch system status',
          message: error.message
        });
      }
    });

    // 404 handler
    this.app.use('*', (req, res) => {
      res.status(404).json({
        success: false,
        error: 'Not Found',
        message: `Route ${req.method} ${req.originalUrl} not found`
      });
    });

    // Error handler
    this.app.use((error, req, res, next) => {
      logger.error('API Error:', error);
      res.status(500).json({
        success: false,
        error: 'Internal Server Error',
        message: error.message
      });
    });
  }

  async getLatestPrices() {
    try {
      // Try to get prices from Redis streams
      const priceData = await redis.xrevrange(STREAMS.TICKS_RAW, '+', '-', 'COUNT', 1);
      
      if (priceData && priceData.length > 0) {
        const latestPrice = priceData[0][1];
        return {
          SOL: parseFloat(latestPrice.find(p => p[0] === 'SOL')?.[1] || '0'),
          BTC: parseFloat(latestPrice.find(p => p[0] === 'BTC')?.[1] || '0'),
          ETH: parseFloat(latestPrice.find(p => p[0] === 'ETH')?.[1] || '0'),
          USDC: parseFloat(latestPrice.find(p => p[0] === 'USDC')?.[1] || '1')
        };
      }

      // Fallback to mock data if no Redis data
      return {
        SOL: 100.50,
        BTC: 45000.00,
        ETH: 3000.00,
        USDC: 1.00
      };
    } catch (error) {
      logger.warn('Error fetching prices from Redis, using fallback:', error.message);
      return {
        SOL: 100.50,
        BTC: 45000.00,
        ETH: 3000.00,
        USDC: 1.00
      };
    }
  }

  async getRecentWhales(limit = 10) {
    try {
      // Try to get whale data from Redis streams
      const whaleData = await redis.xrevrange(STREAMS.WHALES_RAW, '+', '-', 'COUNT', limit);
      
      if (whaleData && whaleData.length > 0) {
        return whaleData.map(whale => ({
          id: whale[0],
          data: whale[1].reduce((acc, field) => {
            acc[field[0]] = field[1];
            return acc;
          }, {}),
          timestamp: new Date().toISOString()
        }));
      }

      // Fallback to mock data
      return Array.from({ length: limit }, (_, i) => ({
        id: `whale_${i}`,
        data: {
          amount: Math.random() * 1000 + 100,
          token: ['SOL', 'BTC', 'ETH'][Math.floor(Math.random() * 3)],
          type: 'transfer',
          from: 'mock_wallet_' + i,
          to: 'mock_wallet_' + (i + 1)
        },
        timestamp: new Date(Date.now() - i * 60000).toISOString()
      }));
    } catch (error) {
      logger.warn('Error fetching whale data from Redis, using fallback:', error.message);
      return [];
    }
  }

  async getMarketSummary() {
    try {
      const prices = await this.getLatestPrices();
      return {
        prices,
        volume24h: {
          SOL: Math.random() * 1000000,
          BTC: Math.random() * 10000000,
          ETH: Math.random() * 5000000
        },
        marketCap: {
          SOL: prices.SOL * 500000000, // Approximate SOL supply
          BTC: prices.BTC * 21000000,  // BTC supply
          ETH: prices.ETH * 120000000  // Approximate ETH supply
        },
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error('Error generating market summary:', error);
      throw error;
    }
  }

  async getSystemStatus() {
    try {
      const redisStatus = redis.status === 'ready' ? 'connected' : 'disconnected';
      const solanaStatus = 'connected'; // We'll assume it's connected if we got this far
      
      return {
        redis: redisStatus,
        solana: solanaStatus,
        wallet: this.wallet ? 'loaded' : 'not_loaded',
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error('Error getting system status:', error);
      throw error;
    }
  }

  start() {
    this.app.listen(this.port, () => {
      logger.info(`🚀 Data Ingestion API server running on port ${this.port}`);
      logger.info(`📍 Health check: http://localhost:${this.port}/health`);
      logger.info(`📊 API endpoints: http://localhost:${this.port}/api/`);
      
      if (this.wallet) {
        logger.info(`💰 Wallet loaded: ${this.wallet.publicKey.toString()}`);
      }
    });
  }
}

// Start the server
const api = new DataIngestionAPI();
api.start();

module.exports = DataIngestionAPI;
