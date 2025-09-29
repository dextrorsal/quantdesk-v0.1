import { Router, Request, Response } from 'express';
import { 
  jitLiquidityService,
  LiquidityAuction,
  AuctionStatus,
  MarketMakerTier,
  StrategyType,
  StrategyParameters
} from '../services/jitLiquidityService';
import { Logger } from '../utils/logger';

const router = Router();
const logger = new Logger();

/**
 * GET /api/jit-liquidity/auctions
 * Get all active liquidity auctions
 */
router.get('/auctions', async (req: Request, res: Response) => {
  try {
    const auctions = jitLiquidityService.getActiveAuctions();
    
    res.json({
      success: true,
      data: {
        auctions,
        totalAuctions: auctions.length,
        timestamp: Date.now()
      }
    });
    
  } catch (error) {
    logger.error('Error fetching liquidity auctions:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch liquidity auctions',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/jit-liquidity/auctions/:auctionId
 * Get specific auction details
 */
router.get('/auctions/:auctionId', async (req: Request, res: Response) => {
  try {
    const { auctionId } = req.params;
    const auction = jitLiquidityService.getAuction(auctionId);
    
    if (!auction) {
      return res.status(404).json({
        success: false,
        error: 'Auction not found'
      });
    }
    
    res.json({
      success: true,
      data: auction
    });
    
  } catch (error) {
    logger.error('Error fetching auction:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch auction',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/jit-liquidity/auctions
 * Create a new liquidity auction
 */
router.post('/auctions', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const { marketId, side, size, price, deadline, minPrice, maxPrice } = req.body;
    
    // Validate required fields
    if (!marketId || !side || !size || !price || !deadline) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: marketId, side, size, price, deadline'
      });
    }
    
    // Validate side
    if (side !== 'buy' && side !== 'sell') {
      return res.status(400).json({
        success: false,
        error: 'Invalid side. Must be "buy" or "sell"'
      });
    }
    
    // Validate values
    if (size <= 0 || price <= 0) {
      return res.status(400).json({
        success: false,
        error: 'Size and price must be positive'
      });
    }
    
    // Validate deadline
    if (deadline <= Date.now()) {
      return res.status(400).json({
        success: false,
        error: 'Deadline must be in the future'
      });
    }
    
    const auction = jitLiquidityService.createAuction(
      marketId,
      side,
      size,
      price,
      deadline,
      minPrice,
      maxPrice
    );
    
    res.json({
      success: true,
      data: auction,
      message: 'Liquidity auction created successfully'
    });
    
  } catch (error) {
    logger.error('Error creating liquidity auction:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create liquidity auction',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/jit-liquidity/auctions/:auctionId/bid
 * Submit a bid to an auction
 */
router.post('/auctions/:auctionId/bid', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    const { auctionId } = req.params;
    const { bidPrice, bidSize } = req.body;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    // Validate bid parameters
    if (!bidPrice || !bidSize) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: bidPrice, bidSize'
      });
    }
    
    if (bidPrice <= 0 || bidSize <= 0) {
      return res.status(400).json({
        success: false,
        error: 'Bid price and size must be positive'
      });
    }
    
    const success = jitLiquidityService.submitBid(
      auctionId,
      userId,
      bidPrice,
      bidSize
    );
    
    if (success) {
      res.json({
        success: true,
        message: 'Bid submitted successfully'
      });
    } else {
      res.status(400).json({
        success: false,
        error: 'Failed to submit bid'
      });
    }
    
  } catch (error) {
    logger.error('Error submitting bid:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to submit bid',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/jit-liquidity/auctions/:auctionId/close
 * Close an auction and determine winner
 */
router.post('/auctions/:auctionId/close', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    const { auctionId } = req.params;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const winningBid = jitLiquidityService.closeAuction(auctionId);
    
    if (winningBid) {
      res.json({
        success: true,
        data: winningBid,
        message: 'Auction closed successfully'
      });
    } else {
      res.status(400).json({
        success: false,
        error: 'Failed to close auction'
      });
    }
    
  } catch (error) {
    logger.error('Error closing auction:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to close auction',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/jit-liquidity/market-makers
 * Get all market makers
 */
router.get('/market-makers', async (req: Request, res: Response) => {
  try {
    const marketMakers = jitLiquidityService.getAllMarketMakers();
    
    res.json({
      success: true,
      data: {
        marketMakers,
        totalMarketMakers: marketMakers.length,
        timestamp: Date.now()
      }
    });
    
  } catch (error) {
    logger.error('Error fetching market makers:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch market makers',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/jit-liquidity/market-makers/:makerId
 * Get specific market maker details
 */
router.get('/market-makers/:makerId', async (req: Request, res: Response) => {
  try {
    const { makerId } = req.params;
    const marketMaker = jitLiquidityService.getMarketMaker(makerId);
    
    if (!marketMaker) {
      return res.status(404).json({
        success: false,
        error: 'Market maker not found'
      });
    }
    
    res.json({
      success: true,
      data: marketMaker
    });
    
  } catch (error) {
    logger.error('Error fetching market maker:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch market maker',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/jit-liquidity/liquidity-mining
 * Get all liquidity mining programs
 */
router.get('/liquidity-mining', async (req: Request, res: Response) => {
  try {
    const programs = jitLiquidityService.getAllLiquidityMiningPrograms();
    
    res.json({
      success: true,
      data: {
        programs,
        totalPrograms: programs.length,
        timestamp: Date.now()
      }
    });
    
  } catch (error) {
    logger.error('Error fetching liquidity mining programs:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch liquidity mining programs',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/jit-liquidity/liquidity-mining/:programId/join
 * Join a liquidity mining program
 */
router.post('/liquidity-mining/:programId/join', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    const { programId } = req.params;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const success = jitLiquidityService.joinLiquidityMiningProgram(programId, userId);
    
    if (success) {
      res.json({
        success: true,
        message: 'Successfully joined liquidity mining program'
      });
    } else {
      res.status(400).json({
        success: false,
        error: 'Failed to join liquidity mining program'
      });
    }
    
  } catch (error) {
    logger.error('Error joining liquidity mining program:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to join liquidity mining program',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/jit-liquidity/liquidity-mining/:programId/rewards/:makerId
 * Calculate liquidity mining rewards
 */
router.get('/liquidity-mining/:programId/rewards/:makerId', async (req: Request, res: Response) => {
  try {
    const { programId, makerId } = req.params;
    
    const rewards = jitLiquidityService.calculateLiquidityMiningRewards(programId, makerId);
    
    res.json({
      success: true,
      data: {
        programId,
        makerId,
        rewards,
        timestamp: Date.now()
      }
    });
    
  } catch (error) {
    logger.error('Error calculating liquidity mining rewards:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to calculate liquidity mining rewards',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/jit-liquidity/price-improvements
 * Get price improvements
 */
router.get('/price-improvements', async (req: Request, res: Response) => {
  try {
    const limit = parseInt(req.query.limit as string) || 100;
    const priceImprovements = jitLiquidityService.getPriceImprovements(limit);
    
    res.json({
      success: true,
      data: {
        priceImprovements,
        totalImprovements: priceImprovements.length,
        timestamp: Date.now()
      }
    });
    
  } catch (error) {
    logger.error('Error fetching price improvements:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch price improvements',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/jit-liquidity/strategies
 * Get market making strategies
 */
router.get('/strategies', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    const strategies = jitLiquidityService.getMarketMakingStrategies(userId);
    
    res.json({
      success: true,
      data: {
        strategies,
        totalStrategies: strategies.length,
        timestamp: Date.now()
      }
    });
    
  } catch (error) {
    logger.error('Error fetching market making strategies:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch market making strategies',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/jit-liquidity/strategies
 * Create a new market making strategy
 */
router.post('/strategies', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    const { marketId, strategyType, parameters } = req.body;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    // Validate required fields
    if (!marketId || !strategyType || !parameters) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: marketId, strategyType, parameters'
      });
    }
    
    // Validate strategy type
    if (!Object.values(StrategyType).includes(strategyType)) {
      return res.status(400).json({
        success: false,
        error: 'Invalid strategy type'
      });
    }
    
    const strategy = jitLiquidityService.createMarketMakingStrategy(
      userId,
      marketId,
      strategyType,
      parameters
    );
    
    res.json({
      success: true,
      data: strategy,
      message: 'Market making strategy created successfully'
    });
    
  } catch (error) {
    logger.error('Error creating market making strategy:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create market making strategy',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/jit-liquidity/stats
 * Get JIT liquidity statistics
 */
router.get('/stats', async (req: Request, res: Response) => {
  try {
    const stats = jitLiquidityService.getJITLiquidityStats();
    
    res.json({
      success: true,
      data: stats
    });
    
  } catch (error) {
    logger.error('Error fetching JIT liquidity stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch JIT liquidity stats',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
