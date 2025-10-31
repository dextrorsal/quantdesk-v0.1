import express from 'express';
import { accountStateService } from '../services/accountStateService';
import { asyncHandler } from '../middleware/errorHandling';
import { authMiddleware } from '../middleware/auth';
import { Logger } from '../utils/logger';

const router = express.Router();
const logger = new Logger();

// Get complete account state
router.get('/state', authMiddleware, asyncHandler(async (req: any, res) => {
  const userId = req.user.id;

  try {
    const stateResponse = await accountStateService.getUserAccountState(userId);
    
    res.json(stateResponse);
  } catch (error) {
    logger.error('Error getting account state:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get account state',
      code: 'STATE_FETCH_ERROR'
    });
  }
}));

// Get account summary for dashboard
router.get('/summary', authMiddleware, asyncHandler(async (req: any, res) => {
  const userId = req.user.id;

  try {
    const summary = await accountStateService.getAccountSummary(userId);
    
    res.json({
      success: true,
      summary
    });
  } catch (error) {
    logger.error('Error getting account summary:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get account summary',
      code: 'SUMMARY_FETCH_ERROR'
    });
  }
}));

// Check if user can perform specific action
router.post('/can-perform', authMiddleware, asyncHandler(async (req: any, res) => {
  const userId = req.user.id;
  const { action } = req.body;

  if (!action || !['deposit', 'withdraw', 'trade', 'create_account'].includes(action)) {
    return res.status(400).json({
      success: false,
      error: 'Invalid action specified',
      code: 'INVALID_ACTION'
    });
  }

  try {
    const canPerform = await accountStateService.canUserPerformAction(userId, action);
    
    res.json({
      success: true,
      canPerform,
      action,
      message: canPerform ? `User can ${action}` : `User cannot ${action}`
    });
  } catch (error) {
    logger.error('Error checking user action permission:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to check action permission',
      code: 'PERMISSION_CHECK_ERROR'
    });
  }
}));

// Get user's trading accounts
router.get('/trading-accounts', authMiddleware, asyncHandler(async (req: any, res) => {
  const userId = req.user.id;

  try {
    const stateResponse = await accountStateService.getUserAccountState(userId);
    
    res.json({
      success: true,
      tradingAccounts: stateResponse.state.tradingAccounts
    });
  } catch (error) {
    logger.error('Error getting trading accounts:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get trading accounts',
      code: 'ACCOUNTS_FETCH_ERROR'
    });
  }
}));

// Get user's balances
router.get('/balances', authMiddleware, asyncHandler(async (req: any, res) => {
  const userId = req.user.id;

  try {
    const stateResponse = await accountStateService.getUserAccountState(userId);
    
    res.json({
      success: true,
      balances: stateResponse.state.balances,
      totalBalance: stateResponse.state.totalBalance,
      availableBalance: stateResponse.state.availableBalance
    });
  } catch (error) {
    logger.error('Error getting user balances:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get user balances',
      code: 'BALANCES_FETCH_ERROR'
    });
  }
}));

// Get account health and risk metrics
router.get('/health', authMiddleware, asyncHandler(async (req: any, res) => {
  const userId = req.user.id;

  try {
    const stateResponse = await accountStateService.getUserAccountState(userId);
    const state = stateResponse.state;
    
    res.json({
      success: true,
      health: {
        accountHealth: state.accountHealth,
        riskLevel: state.riskLevel,
        liquidationPrice: state.liquidationPrice,
        totalBalance: state.totalBalance,
        availableBalance: state.availableBalance,
        canTrade: state.canTrade
      }
    });
  } catch (error) {
    logger.error('Error getting account health:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get account health',
      code: 'HEALTH_FETCH_ERROR'
    });
  }
}));

// Get account state for specific trading account
router.get('/trading-accounts/:accountId/state', authMiddleware, asyncHandler(async (req: any, res) => {
  const userId = req.user.id;
  const { accountId } = req.params;

  try {
    // Get general account state
    const stateResponse = await accountStateService.getUserAccountState(userId);
    
    // Filter balances for specific trading account
    const accountBalances = stateResponse.state.balances.filter(
      balance => balance.tradingAccountId === accountId
    );
    
    // Find the trading account
    const tradingAccount = stateResponse.state.tradingAccounts.find(
      account => account.id === accountId
    );
    
    if (!tradingAccount) {
      return res.status(404).json({
        success: false,
        error: 'Trading account not found',
        code: 'ACCOUNT_NOT_FOUND'
      });
    }
    
    res.json({
      success: true,
      tradingAccount,
      balances: accountBalances,
      totalBalance: accountBalances.reduce((sum, balance) => sum + balance.balance, 0),
      availableBalance: accountBalances.reduce((sum, balance) => sum + balance.availableBalance, 0)
    });
  } catch (error) {
    logger.error('Error getting trading account state:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get trading account state',
      code: 'ACCOUNT_STATE_FETCH_ERROR'
    });
  }
}));

export default router;
