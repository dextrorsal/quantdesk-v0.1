import express, { Request, Response } from 'express';
import { asyncHandler } from '../middleware/errorHandling';
import { databaseService } from '../services/supabaseDatabase';
import { pythOracleService } from '../services/pythOracleService';
import { Logger } from '../utils/logger';

const router = express.Router();
const logger = new Logger();

// ===== DEVELOPMENT AI ASSISTANCE ENDPOINTS =====
// These endpoints help Cursor AI, GitHub Copilot, and other development AI agents
// understand the QuantDesk protocol structure and data flow

/**
 * GET /api/dev/market-summary
 * Get aggregated market data for development AI assistance
 * Helps Cursor AI understand market structure and data flow
 */
router.get('/market-summary', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  try {
    // Get all markets with live data
    const markets = await databaseService.select('markets', '*', { is_active: true });
    
    // Get current oracle prices
    const oraclePrices = await pythOracleService.getAllPrices();
    
    // Aggregate market data
    const marketSummary = markets.map(market => {
      const currentPrice = oraclePrices[market.base_asset] || 0;
      return {
        id: market.id,
        symbol: market.symbol,
        baseAsset: market.base_asset,
        quoteAsset: market.quote_asset,
        currentPrice: currentPrice,
        priceChange24h: market.change24h || 0,
        volume24h: market.volume24h || 0,
        openInterest: market.open_interest || 0,
        maxLeverage: market.max_leverage,
        isActive: market.is_active,
        lastUpdated: new Date().toISOString()
      };
    });

    res.json({
      success: true,
      data: {
        markets: marketSummary,
        totalMarkets: marketSummary.length,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    logger.error('Error fetching market summary:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch market summary',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}));

/**
 * GET /api/dev/user-portfolio/:wallet
 * Get complete portfolio view for development AI assistance
 * Helps Cursor AI understand user data structure and relationships
 */
router.get('/user-portfolio/:wallet', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const { wallet } = req.params;
  
  if (!wallet) {
    res.status(400).json({
      success: false,
      error: 'Wallet address required'
    });
    return;
  }

  try {
    // Get user data
    const user = await databaseService.getUserByWallet(wallet);
    if (!user) {
      res.status(404).json({
        success: false,
        error: 'User not found'
      });
      return;
    }

    // Get user positions
    const positions = await databaseService.select('positions', '*', { user_id: user.id });
    
    // Get user orders
    const orders = await databaseService.select('orders', '*', { user_id: user.id });
    
    // Get user trades
    const trades = await databaseService.select('trades', '*', { user_id: user.id });

    // Calculate portfolio metrics
    const totalPositions = positions.length;
    const totalOrders = orders.length;
    const totalTrades = trades.length;
    
    const openPositions = positions.filter(p => p.status === 'open');
    const totalPnL = positions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0);

    res.json({
      success: true,
      data: {
        user: {
          id: user.id,
          wallet_address: user.wallet_address,
          created_at: user.created_at,
          total_volume: user.total_volume,
          total_trades: user.total_trades
        },
        portfolio: {
          totalPositions,
          openPositions: openPositions.length,
          totalOrders,
          totalTrades,
          totalPnL,
          positions: openPositions.map(p => ({
            id: p.id,
            market: p.market,
            side: p.side,
            size: p.size,
            entryPrice: p.entry_price,
            currentPrice: p.current_price,
            unrealizedPnl: p.unrealized_pnl,
            leverage: p.leverage,
            status: p.status
          }))
        },
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    logger.error('Error fetching user portfolio:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch user portfolio',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}));

/**
 * GET /api/ai/trading-signals
 * Get market analysis and trading signals for AI agents
 */
router.get('/trading-signals', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  try {
    // Get market data
    const markets = await databaseService.select('markets', '*', { is_active: true });
    const oraclePrices = await pythOracleService.getAllPrices();
    
    // Generate trading signals based on price movements and volume
    const tradingSignals = markets.map(market => {
      const currentPrice = oraclePrices[market.base_asset] || 0;
      if (!currentPrice) return null;

      // Simple signal generation based on price change and volume
      const priceChangePercent = market.change24h / (currentPrice || 1) * 100;
      const volumeScore = Math.min(market.volume24h / 1000000, 10); // Normalize volume
      
      let signal = 'HOLD';
      let confidence = 0.5;
      
      if (priceChangePercent > 5 && volumeScore > 5) {
        signal = 'STRONG_BUY';
        confidence = 0.8;
      } else if (priceChangePercent > 2 && volumeScore > 3) {
        signal = 'BUY';
        confidence = 0.6;
      } else if (priceChangePercent < -5 && volumeScore > 5) {
        signal = 'STRONG_SELL';
        confidence = 0.8;
      } else if (priceChangePercent < -2 && volumeScore > 3) {
        signal = 'SELL';
        confidence = 0.6;
      }

      return {
        market: market.symbol,
        baseAsset: market.base_asset,
        currentPrice: currentPrice,
        priceChange24h: market.change24h,
        priceChangePercent,
        volume24h: market.volume24h,
        signal,
        confidence,
        maxLeverage: market.max_leverage,
        timestamp: new Date().toISOString()
      };
    }).filter(Boolean);

    res.json({
      success: true,
      data: {
        signals: tradingSignals,
        totalSignals: tradingSignals.length,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    logger.error('Error generating trading signals:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to generate trading signals',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}));

/**
 * GET /api/ai/liquidation-risk
 * Get liquidation risk analysis for AI agents
 */
router.get('/liquidation-risk', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  try {
    // Get all open positions
    const positions = await databaseService.select('positions', '*', { status: 'open' });
    
    // Get current prices
    const oraclePrices = await pythOracleService.getAllPrices();
    
    // Calculate liquidation risk for each position
    const liquidationRisks = positions.map(position => {
      const currentPrice = oraclePrices[position.market] || 0;
      if (!currentPrice) return null;
      const liquidationPrice = position.liquidation_price;
      const entryPrice = position.entry_price;
      
      // Calculate distance to liquidation
      const distanceToLiquidation = Math.abs(currentPrice - liquidationPrice) / currentPrice * 100;
      
      // Risk level based on distance to liquidation
      let riskLevel = 'LOW';
      if (distanceToLiquidation < 5) riskLevel = 'HIGH';
      else if (distanceToLiquidation < 10) riskLevel = 'MEDIUM';

      return {
        positionId: position.id,
        userId: position.user_id,
        market: position.market,
        side: position.side,
        size: position.size,
        leverage: position.leverage,
        entryPrice,
        currentPrice,
        liquidationPrice,
        distanceToLiquidation,
        riskLevel,
        unrealizedPnl: position.unrealized_pnl,
        timestamp: new Date().toISOString()
      };
    }).filter(Boolean);

    // Sort by risk level
    liquidationRisks.sort((a, b) => {
      const riskOrder = { 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1 };
      return riskOrder[b.riskLevel] - riskOrder[a.riskLevel];
    });

    res.json({
      success: true,
      data: {
        risks: liquidationRisks,
        totalPositions: liquidationRisks.length,
        highRiskPositions: liquidationRisks.filter(r => r.riskLevel === 'HIGH').length,
        mediumRiskPositions: liquidationRisks.filter(r => r.riskLevel === 'MEDIUM').length,
        lowRiskPositions: liquidationRisks.filter(r => r.riskLevel === 'LOW').length,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    logger.error('Error analyzing liquidation risk:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to analyze liquidation risk',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}));

/**
 * GET /api/ai/funding-rates
 * Get funding rate analysis for AI agents
 */
router.get('/funding-rates', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  try {
    // Get funding rates for all markets
    const fundingRates = await databaseService.select('funding_rates', '*', {});
    
    // Group by market and get latest rates
    const latestRates = fundingRates.reduce((acc: any, rate: any) => {
      if (!acc[rate.market_id]) {
        acc[rate.market_id] = rate;
      }
      return acc;
    }, {});

    // Get market info for each funding rate
    const markets = await databaseService.select('markets', '*', { is_active: true });
    
    const fundingAnalysis = Object.values(latestRates).map((rate: any) => {
      const market = markets.find(m => m.id === rate.market_id);
      return {
        marketId: rate.market_id,
        marketSymbol: market?.symbol || 'Unknown',
        baseAsset: market?.base_asset || 'Unknown',
        fundingRate: rate.funding_rate,
        fundingRatePercent: (rate.funding_rate / 1000000) * 100, // Convert to percentage
        timestamp: rate.timestamp,
        nextFundingTime: new Date(new Date(rate.timestamp).getTime() + 8 * 60 * 60 * 1000).toISOString() // 8 hours later
      };
    });

    res.json({
      success: true,
      data: {
        fundingRates: fundingAnalysis,
        totalMarkets: fundingAnalysis.length,
        averageFundingRate: fundingAnalysis.reduce((sum, r) => sum + r.fundingRatePercent, 0) / fundingAnalysis.length,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    logger.error('Error fetching funding rates:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch funding rates',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}));

/**
 * GET /api/dev/codebase-structure
 * Get codebase structure information for development AI assistance
 * Helps Cursor AI understand the QuantDesk architecture and relationships
 */
router.get('/codebase-structure', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  try {
    // Get database schema information
    const tables = await databaseService.getClient().from('information_schema.tables')
      .select('table_name, table_type')
      .eq('table_schema', 'public');

    // Get market structure
    const markets = await databaseService.select('markets', 'id, symbol, base_asset, quote_asset, is_active', {});
    
    // Get user structure
    const userCount = await databaseService.getClient().from('users').select('id', { count: 'exact' });
    
    // Get position structure
    const positionCount = await databaseService.getClient().from('positions').select('id', { count: 'exact' });

    const codebaseStructure = {
      database: {
        tables: tables.data || [],
        totalTables: tables.data?.length || 0
      },
      markets: {
        total: markets.length,
        active: markets.filter(m => m.is_active).length,
        structure: markets.map(m => ({
          id: m.id,
          symbol: m.symbol,
          baseAsset: m.base_asset,
          quoteAsset: m.quote_asset
        }))
      },
      users: {
        total: userCount.count || 0
      },
      positions: {
        total: positionCount.count || 0
      },
      architecture: {
        backend: {
          services: ['databaseService', 'pythOracleService', 'devnetService'],
          routes: ['markets', 'oracle', 'users', 'positions', 'orders', 'trades'],
          middleware: ['authMiddleware', 'rateLimiting', 'errorHandling']
        },
        smartContract: {
          program: 'quantdesk-perp-dex',
          accounts: ['Market', 'Position', 'UserAccount', 'CollateralAccount'],
          instructions: ['initialize_market', 'open_position', 'close_position', 'update_collateral']
        }
      },
      timestamp: new Date().toISOString()
    };

    res.json({
      success: true,
      data: codebaseStructure
    });

  } catch (error) {
    logger.error('Error fetching codebase structure:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch codebase structure',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}));

export default router;
