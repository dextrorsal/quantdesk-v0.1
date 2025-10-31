import express from 'express';
import { SupabaseDatabaseService } from '../services/supabaseDatabase';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandling';
import { AuthenticatedRequest } from '../middleware/auth';
import { transactionVerificationService } from '../services/transactionVerificationService';

const router = express.Router();
const logger = new Logger();
const db = SupabaseDatabaseService.getInstance();

// Verify account creation transaction
router.post('/verify-creation', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { transactionSignature, accountIndex } = req.body;
  const userId = req.userId;

  if (!transactionSignature) {
    return res.status(400).json({
      error: 'Transaction signature is required',
      code: 'MISSING_SIGNATURE'
    });
  }

  try {
    // Get user wallet address using fluent API
    const user = await db.select('users', 'wallet_address', { id: userId });
    
    if (user.length === 0) {
      return res.status(404).json({
        error: 'User not found',
        code: 'USER_NOT_FOUND'
      });
    }

    const userWallet = user[0].wallet_address;

    // Verify account creation transaction
    logger.info(`🔍 Verifying account creation transaction: ${transactionSignature}`);
    
    const verificationResult = await transactionVerificationService.verifyAccountCreationTransaction(
      transactionSignature,
      {
        userWallet,
        accountIndex: accountIndex || 0,
        expectedProgramId: process.env.QUANTDESK_PROGRAM_ID
      }
    );

    if (!verificationResult.isValid) {
      logger.error(`❌ Account creation verification failed: ${verificationResult.error}`);
      return res.status(400).json({
        error: 'Account creation verification failed',
        details: verificationResult.error,
        code: 'ACCOUNT_CREATION_VERIFICATION_FAILED'
      });
    }

    logger.info(`✅ Account creation verified successfully: ${transactionSignature}`);

    // Update user's account state in database using fluent API
    await db.update('users', {
      account_created: true,
      account_created_at: new Date().toISOString(),
      last_activity: new Date().toISOString()
    }, { id: userId });

    res.json({
      success: true,
      message: 'Account creation verified successfully',
      transactionSignature,
      verificationDetails: {
        accounts: verificationResult.accounts,
        logs: verificationResult.logs,
        programIds: verificationResult.programIds
      }
    });

  } catch (error) {
    logger.error('Error verifying account creation:', error);
    res.status(500).json({
      error: 'Failed to verify account creation',
      code: 'VERIFICATION_ERROR'
    });
  }
}));

// Get user's trading accounts
router.get('/trading-accounts', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const userId = req.userId;
    
    // Use fluent API instead of query() for security
    const tradingAccounts = await db.select('trading_accounts', '*', { 
      master_account_id: userId, 
      is_active: true 
    });

    res.json({
      success: true,
      tradingAccounts: tradingAccounts.map(account => ({
        id: account.id,
        name: account.name,
        accountIndex: account.account_index,
        isActive: account.is_active,
        createdAt: account.created_at
      }))
    });

  } catch (error) {
    logger.error('Error fetching trading accounts:', error);
    res.status(500).json({
      error: 'Failed to fetch trading accounts',
      code: 'FETCH_ERROR'
    });
  }
}));

// Create new trading account
router.post('/trading-accounts', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { name } = req.body;
  const userId = req.userId;

  if (!name || name.trim().length === 0) {
    return res.status(400).json({
      error: 'Trading account name is required',
      code: 'MISSING_NAME'
    });
  }

  try {
    // Get next trading account index using fluent API
    const existingAccounts = await db.select('trading_accounts', 'account_index', { 
      master_account_id: userId 
    });
    
    const nextIndex = existingAccounts.length > 0 
      ? Math.max(...existingAccounts.map(acc => acc.account_index)) + 1 
      : 1;

    // Create trading account using fluent API
    const newAccount = await db.insert('trading_accounts', {
      master_account_id: userId,
      name: name.trim(),
      account_index: nextIndex,
      is_active: true,
      created_at: new Date().toISOString()
    });

    res.status(201).json({
      success: true,
      tradingAccount: {
        id: newAccount.id,
        name: newAccount.name,
        accountIndex: newAccount.account_index,
        isActive: newAccount.is_active,
        createdAt: newAccount.created_at
      }
    });

  } catch (error) {
    logger.error('Error creating trading account:', error);
    res.status(500).json({
      error: 'Failed to create trading account',
      code: 'CREATE_ERROR'
    });
  }
}));

// Update sub-account
router.put('/sub-accounts/:id', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { id } = req.params;
  const { name } = req.body;
  const userId = req.userId;

  if (!name || name.trim().length === 0) {
    return res.status(400).json({
      error: 'Sub-account name is required',
      code: 'MISSING_NAME'
    });
  }

  try {
    // Verify ownership
    const ownershipResult = await db.select('sub_accounts', 'id', { 
      id: id, 
      main_account_id: userId 
    });

    if (ownershipResult.length === 0) {
      return res.status(404).json({
        error: 'Sub-account not found',
        code: 'NOT_FOUND'
      });
    }

    // Update sub-account
    const result = await db.update('sub_accounts', {
      name: name.trim(),
      updated_at: new Date().toISOString()
    }, { id: id });

    const updatedAccount = result[0];

    res.json({
      success: true,
      subAccount: {
        id: updatedAccount.id,
        name: updatedAccount.name,
        subAccountIndex: updatedAccount.sub_account_index,
        isActive: updatedAccount.is_active,
        updatedAt: updatedAccount.updated_at
      }
    });

  } catch (error) {
    logger.error('Error updating sub-account:', error);
    res.status(500).json({
      error: 'Failed to update sub-account',
      code: 'UPDATE_ERROR'
    });
  }
}));

// Deactivate sub-account
router.delete('/sub-accounts/:id', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { id } = req.params;
  const userId = req.userId;

  try {
    // Verify ownership
    const ownershipResult = await db.select('sub_accounts', 'id', { 
      id: id, 
      main_account_id: userId 
    });

    if (ownershipResult.length === 0) {
      return res.status(404).json({
        error: 'Sub-account not found',
        code: 'NOT_FOUND'
      });
    }

    // Check for open positions
    const positionCount = await db.count('positions', { 
      sub_account_id: id, 
      size: { gt: 0 }, 
      is_liquidated: false 
    });

    if (positionCount > 0) {
      return res.status(400).json({
        error: 'Cannot deactivate sub-account with open positions',
        code: 'HAS_POSITIONS'
      });
    }

    // Deactivate sub-account
    await db.update('sub_accounts', {
      is_active: false,
      updated_at: new Date().toISOString()
    }, { id: id });

    res.json({
      success: true,
      message: 'Sub-account deactivated successfully'
    });

  } catch (error) {
    logger.error('Error deactivating sub-account:', error);
    res.status(500).json({
      error: 'Failed to deactivate sub-account',
      code: 'DEACTIVATE_ERROR'
    });
  }
}));

// Get delegated accounts
router.get('/delegates', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const userId = req.userId;
    
    const result = await db.select('delegated_accounts', '*', { 
      main_account_id: userId, 
      is_active: true 
    });

    res.json({
      success: true,
      delegates: result.map(delegate => ({
        id: delegate.id,
        delegateWalletAddress: delegate.delegate_wallet_address,
        permissions: delegate.permissions,
        isActive: delegate.is_active,
        createdAt: delegate.created_at
      }))
    });

  } catch (error) {
    logger.error('Error fetching delegates:', error);
    res.status(500).json({
      error: 'Failed to fetch delegates',
      code: 'FETCH_ERROR'
    });
  }
}));

// Add delegate account
router.post('/delegates', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { delegateWalletAddress, permissions } = req.body;
  const userId = req.userId;

  if (!delegateWalletAddress || !delegateWalletAddress.match(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/)) {
    return res.status(400).json({
      error: 'Valid delegate wallet address is required',
      code: 'INVALID_WALLET'
    });
  }

  try {
    // Check if delegate already exists
    const existingResult = await db.select('delegated_accounts', 'id', { 
      main_account_id: userId, 
      delegate_wallet_address: delegateWalletAddress, 
      is_active: true 
    });

    if (existingResult.length > 0) {
      return res.status(400).json({
        error: 'Delegate already exists',
        code: 'DELEGATE_EXISTS'
      });
    }

    // Default permissions
    const defaultPermissions = {
      deposit: true,
      trade: true,
      cancel: true,
      withdraw: false
    };

    const finalPermissions = { ...defaultPermissions, ...permissions };

    // Create delegate
    const result = await db.insert('delegated_accounts', {
      main_account_id: userId,
      delegate_wallet_address: delegateWalletAddress,
      permissions: JSON.stringify(finalPermissions)
    });

    const newDelegate = result[0];

    res.status(201).json({
      success: true,
      delegate: {
        id: newDelegate.id,
        delegateWalletAddress: newDelegate.delegate_wallet_address,
        permissions: JSON.parse(newDelegate.permissions),
        isActive: newDelegate.is_active,
        createdAt: newDelegate.created_at
      }
    });

  } catch (error) {
    logger.error('Error creating delegate:', error);
    res.status(500).json({
      error: 'Failed to create delegate',
      code: 'CREATE_ERROR'
    });
  }
}));

// Update delegate permissions
router.put('/delegates/:id', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { id } = req.params;
  const { permissions } = req.body;
  const userId = req.userId;

  try {
    // Verify ownership
    const ownershipResult = await db.select('delegated_accounts', 'id', { 
      id: id, 
      main_account_id: userId 
    });

    if (ownershipResult.length === 0) {
      return res.status(404).json({
        error: 'Delegate not found',
        code: 'NOT_FOUND'
      });
    }

    // Update permissions
    const result = await db.update('delegated_accounts', {
      permissions: JSON.stringify(permissions),
      updated_at: new Date().toISOString()
    }, { id: id });

    const updatedDelegate = result[0];

    res.json({
      success: true,
      delegate: {
        id: updatedDelegate.id,
        delegateWalletAddress: updatedDelegate.delegate_wallet_address,
        permissions: JSON.parse(updatedDelegate.permissions),
        isActive: updatedDelegate.is_active,
        updatedAt: updatedDelegate.updated_at
      }
    });

  } catch (error) {
    logger.error('Error updating delegate:', error);
    res.status(500).json({
      error: 'Failed to update delegate',
      code: 'UPDATE_ERROR'
    });
  }
}));

// Remove delegate
router.delete('/delegates/:id', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { id } = req.params;
  const userId = req.userId;

  try {
    // Verify ownership
    const ownershipResult = await db.select('delegated_accounts', 'id', { 
      id: id, 
      main_account_id: userId 
    });

    if (ownershipResult.length === 0) {
      return res.status(404).json({
        error: 'Delegate not found',
        code: 'NOT_FOUND'
      });
    }

    // Deactivate delegate
    await db.update('delegated_accounts', {
      is_active: false,
      updated_at: new Date().toISOString()
    }, { id: id });

    res.json({
      success: true,
      message: 'Delegate removed successfully'
    });

  } catch (error) {
    logger.error('Error removing delegate:', error);
    res.status(500).json({
      error: 'Failed to remove delegate',
      code: 'REMOVE_ERROR'
    });
  }
}));

// Get all account balances (cross-collateral view)
router.get('/balances', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const userId = req.userId;
    
    // Get main account balances
    const mainBalancesResult = await db.complexQuery(
      `SELECT asset, balance, locked_balance, available_balance 
       FROM user_balances 
       WHERE user_id = $1 AND sub_account_id IS NULL`,
      [userId]
    );

    // Get sub-account balances
    const subAccountBalancesResult = await db.complexQuery(
      `SELECT 
         sa.id as sub_account_id,
         sa.name as sub_account_name,
         sa.sub_account_index,
         ub.asset,
         ub.balance,
         ub.locked_balance,
         ub.available_balance
       FROM sub_accounts sa
       LEFT JOIN user_balances ub ON sa.id = ub.sub_account_id
       WHERE sa.main_account_id = $1 AND sa.is_active = true
       ORDER BY sa.sub_account_index, ub.asset`,
      [userId]
    );

    res.json({
      success: true,
      balances: {
        mainAccount: mainBalancesResult,
        subAccounts: subAccountBalancesResult
      }
    });

  } catch (error) {
    logger.error('Error fetching account balances:', error);
    res.status(500).json({
      error: 'Failed to fetch account balances',
      code: 'FETCH_ERROR'
    });
  }
}));

// Transfer between sub-accounts
router.post('/transfer', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { fromSubAccountId, toSubAccountId, asset, amount } = req.body;
  const userId = req.userId;

  if (!fromSubAccountId || !toSubAccountId || !asset || !amount || amount <= 0) {
    return res.status(400).json({
      error: 'Invalid transfer parameters',
      code: 'INVALID_PARAMS'
    });
  }

  try {
    // Verify ownership of both sub-accounts
    const ownershipResult = await db.select('sub_accounts', 'id', { 
      id: [fromSubAccountId, toSubAccountId], 
      main_account_id: userId, 
      is_active: true 
    });

    if (ownershipResult.length !== 2) {
      return res.status(400).json({
        error: 'Invalid sub-accounts',
        code: 'INVALID_ACCOUNTS'
      });
    }

    // Check sufficient balance
    const balanceResult = await db.select('user_balances', 'available_balance', { 
      sub_account_id: fromSubAccountId, 
      asset: asset 
    });

    if (balanceResult.length === 0 || balanceResult[0].available_balance < amount) {
      return res.status(400).json({
        error: 'Insufficient balance',
        code: 'INSUFFICIENT_BALANCE'
      });
    }

    // Execute transfer operations
    try {
      // Get current balance for debit calculation
      const { data: currentBalance, error: balanceError } = await db.getClient()
        .from('user_balances')
        .select('balance')
        .eq('sub_account_id', fromSubAccountId)
        .eq('asset', asset)
        .single();

      if (balanceError || !currentBalance) {
        throw new Error('Source balance not found');
      }

      const newBalance = currentBalance.balance - amount;

      // Debit from source
      const { error: debitError } = await db.getClient()
        .from('user_balances')
        .update({ 
          balance: newBalance,
          updated_at: new Date().toISOString()
        })
        .eq('sub_account_id', fromSubAccountId)
        .eq('asset', asset);

      if (debitError) {
        throw debitError;
      }

      // Credit to destination (create if doesn't exist)
      const { error: creditError } = await db.getClient()
        .from('user_balances')
        .upsert({
          user_id: userId,
          sub_account_id: toSubAccountId,
          asset: asset,
          balance: amount,
          locked_balance: 0,
          updated_at: new Date().toISOString()
        }, {
          onConflict: 'user_id,sub_account_id,asset'
        });

      if (creditError) {
        throw creditError;
      }
    } catch (error) {
      logger.error('Error executing transfer:', error);
      return res.status(500).json({
        error: 'Transfer failed',
        code: 'TRANSFER_ERROR'
      });
    }

    res.json({
      success: true,
      message: 'Transfer completed successfully',
      transfer: {
        fromSubAccountId,
        toSubAccountId,
        asset,
        amount
      }
    });

  } catch (error) {
    logger.error('Error executing transfer:', error);
    res.status(500).json({
      error: 'Failed to execute transfer',
      code: 'TRANSFER_ERROR'
    });
  }
}));

export default router;
