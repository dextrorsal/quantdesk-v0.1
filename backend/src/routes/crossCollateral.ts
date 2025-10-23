import { Router, Request, Response } from 'express';
import { 
  crossCollateralService, 
  CollateralType, 
  CollateralSwapRequest 
} from '../services/crossCollateralService';
import { Logger } from '../utils/logger';

const router = Router();
const logger = new Logger();

/**
 * POST /api/cross-collateral/initialize
 * Initialize a collateral account for a user
 */
router.post('/initialize', async (req: Request, res: Response) => {
  try {
    const { user_id, asset_type, initial_amount } = req.body;
    
    // Validate required fields
    if (!user_id || !asset_type || !initial_amount) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: user_id, asset_type, initial_amount'
      });
    }

    // Validate asset type
    if (!Object.values(CollateralType).includes(asset_type)) {
      return res.status(400).json({
        success: false,
        error: 'Invalid asset type. Supported types: SOL, USDC, BTC, ETH, USDT'
      });
    }

    const collateralAccount = await crossCollateralService.initializeCollateralAccount(
      user_id, 
      asset_type, 
      initial_amount
    );
    
    res.status(201).json({
      success: true,
      data: collateralAccount,
      message: 'Collateral account initialized successfully'
    });

  } catch (error) {
    logger.error('Error initializing collateral account:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to initialize collateral account',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/cross-collateral/add
 * Add collateral to an existing account
 */
router.post('/add', async (req: Request, res: Response) => {
  try {
    const { account_id, amount, user_id } = req.body;
    
    // Validate required fields
    if (!account_id || !amount || !user_id) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: account_id, amount, user_id'
      });
    }

    const collateralAccount = await crossCollateralService.addCollateral(
      account_id, 
      amount, 
      user_id
    );
    
    res.json({
      success: true,
      data: collateralAccount,
      message: 'Collateral added successfully'
    });

  } catch (error) {
    logger.error('Error adding collateral:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to add collateral',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/cross-collateral/remove
 * Remove collateral from an account
 */
router.post('/remove', async (req: Request, res: Response) => {
  try {
    const { account_id, amount, user_id } = req.body;
    
    // Validate required fields
    if (!account_id || !amount || !user_id) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: account_id, amount, user_id'
      });
    }

    const collateralAccount = await crossCollateralService.removeCollateral(
      account_id, 
      amount, 
      user_id
    );
    
    res.json({
      success: true,
      data: collateralAccount,
      message: 'Collateral removed successfully'
    });

  } catch (error) {
    logger.error('Error removing collateral:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to remove collateral',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/cross-collateral/portfolio/:userId
 * Get user's collateral portfolio
 */
router.get('/portfolio/:userId', async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;

    const portfolio = await crossCollateralService.getUserCollateralPortfolio(userId);
    
    res.json({
      success: true,
      data: portfolio
    });

  } catch (error) {
    logger.error('Error fetching collateral portfolio:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch collateral portfolio',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/cross-collateral/swap
 * Swap collateral between different assets
 */
router.post('/swap', async (req: Request, res: Response) => {
  try {
    const swapRequest: CollateralSwapRequest = req.body;
    
    // Validate required fields
    if (!swapRequest.from_asset || !swapRequest.to_asset || !swapRequest.amount || !swapRequest.user_id) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: from_asset, to_asset, amount, user_id'
      });
    }

    // Validate asset types
    if (!Object.values(CollateralType).includes(swapRequest.from_asset) || 
        !Object.values(CollateralType).includes(swapRequest.to_asset)) {
      return res.status(400).json({
        success: false,
        error: 'Invalid asset type. Supported types: SOL, USDC, BTC, ETH, USDT'
      });
    }

    const result = await crossCollateralService.swapCollateral(swapRequest);
    
    if (result.success) {
      res.json({
        success: true,
        data: result,
        message: 'Collateral swap completed successfully'
      });
    } else {
      res.status(400).json({
        success: false,
        error: result.error || 'Collateral swap failed'
      });
    }

  } catch (error) {
    logger.error('Error swapping collateral:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to swap collateral',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/cross-collateral/max-borrowable/:userId
 * Calculate maximum borrowable amount for a user
 */
router.get('/max-borrowable/:userId', async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;

    const maxBorrowable = await crossCollateralService.calculateMaxBorrowableAmount(userId);
    
    res.json({
      success: true,
      data: {
        max_borrowable_usd: maxBorrowable,
        max_borrowable_formatted: `$${maxBorrowable.toLocaleString()}`
      }
    });

  } catch (error) {
    logger.error('Error calculating max borrowable amount:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to calculate max borrowable amount',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/cross-collateral/update-values
 * Update all collateral values using current market prices
 */
router.post('/update-values', async (req: Request, res: Response) => {
  try {
    await crossCollateralService.updateCollateralValues();
    
    res.json({
      success: true,
      message: 'Collateral values updated successfully'
    });

  } catch (error) {
    logger.error('Error updating collateral values:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to update collateral values',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/cross-collateral/types
 * Get supported collateral types and their configurations
 */
router.get('/types', (req: Request, res: Response) => {
  res.json({
    success: true,
    data: {
      supported_types: Object.values(CollateralType),
      configurations: {
        SOL: { max_ltv: 0.8, liquidation_threshold: 0.85 },
        USDC: { max_ltv: 0.95, liquidation_threshold: 0.97 },
        BTC: { max_ltv: 0.85, liquidation_threshold: 0.9 },
        ETH: { max_ltv: 0.85, liquidation_threshold: 0.9 },
        USDT: { max_ltv: 0.95, liquidation_threshold: 0.97 }
      }
    }
  });
});

/**
 * GET /api/cross-collateral/account/:accountId
 * Get a specific collateral account by ID
 */
router.get('/account/:accountId', async (req: Request, res: Response) => {
  try {
    const { accountId } = req.params;

    const account = await crossCollateralService.getCollateralAccountById(accountId);
    
    if (!account) {
      return res.status(404).json({
        success: false,
        error: 'Collateral account not found'
      });
    }

    res.json({
      success: true,
      data: account
    });

  } catch (error) {
    logger.error('Error fetching collateral account:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch collateral account',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/cross-collateral/stats
 * Get cross-collateralization statistics
 */
router.get('/stats', async (req: Request, res: Response) => {
  try {
    // This would typically query the database for statistics
    const stats = {
      total_collateral_accounts: 0,
      total_collateral_value_usd: 0,
      collateral_by_type: {},
      average_utilization_rate: 0,
      total_swaps_completed: 0
    };

    res.json({
      success: true,
      data: stats
    });

  } catch (error) {
    logger.error('Error fetching cross-collateral stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch cross-collateral statistics',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
