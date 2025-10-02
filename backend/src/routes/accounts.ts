import express from 'express';
import { DatabaseService } from '../services/database';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandler';
import { AuthenticatedRequest } from '../middleware/auth';
import { transactionVerificationService } from '../services/transactionVerificationService';

const router = express.Router();
const logger = new Logger();
const db = DatabaseService.getInstance();

// Verify account creation transaction
router.post('/verify-creation', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { transactionSignature, accountIndex } = req.body;
  const userId = req.user!.id;

  if (!transactionSignature) {
    return res.status(400).json({
      error: 'Transaction signature is required',
      code: 'MISSING_SIGNATURE'
    });
  }

  try {
    // Get user wallet address
    const userResult = await db.query(
      `SELECT wallet_address FROM users WHERE id = $1`,
      [userId]
    );

    if (userResult.rows.length === 0) {
      return res.status(404).json({
        error: 'User not found',
        code: 'USER_NOT_FOUND'
      });
    }

    const userWallet = userResult.rows[0].wallet_address;

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

    // Update user's account state in database
    await db.query(
      `UPDATE users SET 
        account_created = true, 
        account_created_at = NOW(),
        last_activity = NOW()
       WHERE id = $1`,
      [userId]
    );

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
    const userId = req.user!.id;
    
    const result = await db.query(
      `SELECT * FROM trading_accounts 
       WHERE master_account_id = $1 AND is_active = true 
       ORDER BY account_index`,
      [userId]
    );

    res.json({
      success: true,
      tradingAccounts: result.rows.map(account => ({
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
  const userId = req.user!.id;

  if (!name || name.trim().length === 0) {
    return res.status(400).json({
      error: 'Trading account name is required',
      code: 'MISSING_NAME'
    });
  }

  try {
    // Get next trading account index
    const indexResult = await db.query(
      `SELECT COALESCE(MAX(account_index), 0) + 1 as next_index 
       FROM trading_accounts 
       WHERE master_account_id = $1`,
      [userId]
    );

    const nextIndex = indexResult.rows[0].next_index;

    // Create trading account
    const result = await db.query(
      `INSERT INTO trading_accounts (master_account_id, account_index, name) 
       VALUES ($1, $2, $3) 
       RETURNING *`,
      [userId, nextIndex, name.trim()]
    );

    const newAccount = result.rows[0];

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
  const userId = req.user!.id;

  if (!name || name.trim().length === 0) {
    return res.status(400).json({
      error: 'Sub-account name is required',
      code: 'MISSING_NAME'
    });
  }

  try {
    // Verify ownership
    const ownershipResult = await db.query(
      `SELECT id FROM sub_accounts 
       WHERE id = $1 AND main_account_id = $2`,
      [id, userId]
    );

    if (ownershipResult.rows.length === 0) {
      return res.status(404).json({
        error: 'Sub-account not found',
        code: 'NOT_FOUND'
      });
    }

    // Update sub-account
    const result = await db.query(
      `UPDATE sub_accounts 
       SET name = $1, updated_at = NOW() 
       WHERE id = $2 
       RETURNING *`,
      [name.trim(), id]
    );

    const updatedAccount = result.rows[0];

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
  const userId = req.user!.id;

  try {
    // Verify ownership
    const ownershipResult = await db.query(
      `SELECT id FROM sub_accounts 
       WHERE id = $1 AND main_account_id = $2`,
      [id, userId]
    );

    if (ownershipResult.rows.length === 0) {
      return res.status(404).json({
        error: 'Sub-account not found',
        code: 'NOT_FOUND'
      });
    }

    // Check for open positions
    const positionsResult = await db.query(
      `SELECT COUNT(*) as position_count 
       FROM positions 
       WHERE sub_account_id = $1 AND size > 0 AND NOT is_liquidated`,
      [id]
    );

    if (parseInt(positionsResult.rows[0].position_count) > 0) {
      return res.status(400).json({
        error: 'Cannot deactivate sub-account with open positions',
        code: 'HAS_POSITIONS'
      });
    }

    // Deactivate sub-account
    await db.query(
      `UPDATE sub_accounts 
       SET is_active = false, updated_at = NOW() 
       WHERE id = $1`,
      [id]
    );

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
    const userId = req.user!.id;
    
    const result = await db.query(
      `SELECT * FROM delegated_accounts 
       WHERE main_account_id = $1 AND is_active = true 
       ORDER BY created_at DESC`,
      [userId]
    );

    res.json({
      success: true,
      delegates: result.rows.map(delegate => ({
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
  const userId = req.user!.id;

  if (!delegateWalletAddress || !delegateWalletAddress.match(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/)) {
    return res.status(400).json({
      error: 'Valid delegate wallet address is required',
      code: 'INVALID_WALLET'
    });
  }

  try {
    // Check if delegate already exists
    const existingResult = await db.query(
      `SELECT id FROM delegated_accounts 
       WHERE main_account_id = $1 AND delegate_wallet_address = $2 AND is_active = true`,
      [userId, delegateWalletAddress]
    );

    if (existingResult.rows.length > 0) {
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
    const result = await db.query(
      `INSERT INTO delegated_accounts (main_account_id, delegate_wallet_address, permissions) 
       VALUES ($1, $2, $3) 
       RETURNING *`,
      [userId, delegateWalletAddress, JSON.stringify(finalPermissions)]
    );

    const newDelegate = result.rows[0];

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
  const userId = req.user!.id;

  try {
    // Verify ownership
    const ownershipResult = await db.query(
      `SELECT id FROM delegated_accounts 
       WHERE id = $1 AND main_account_id = $2`,
      [id, userId]
    );

    if (ownershipResult.rows.length === 0) {
      return res.status(404).json({
        error: 'Delegate not found',
        code: 'NOT_FOUND'
      });
    }

    // Update permissions
    const result = await db.query(
      `UPDATE delegated_accounts 
       SET permissions = $1, updated_at = NOW() 
       WHERE id = $2 
       RETURNING *`,
      [JSON.stringify(permissions), id]
    );

    const updatedDelegate = result.rows[0];

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
  const userId = req.user!.id;

  try {
    // Verify ownership
    const ownershipResult = await db.query(
      `SELECT id FROM delegated_accounts 
       WHERE id = $1 AND main_account_id = $2`,
      [id, userId]
    );

    if (ownershipResult.rows.length === 0) {
      return res.status(404).json({
        error: 'Delegate not found',
        code: 'NOT_FOUND'
      });
    }

    // Deactivate delegate
    await db.query(
      `UPDATE delegated_accounts 
       SET is_active = false, updated_at = NOW() 
       WHERE id = $1`,
      [id]
    );

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
    const userId = req.user!.id;
    
    // Get main account balances
    const mainBalancesResult = await db.query(
      `SELECT asset, balance, locked_balance, available_balance 
       FROM user_balances 
       WHERE user_id = $1 AND sub_account_id IS NULL`,
      [userId]
    );

    // Get sub-account balances
    const subAccountBalancesResult = await db.query(
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
        mainAccount: mainBalancesResult.rows,
        subAccounts: subAccountBalancesResult.rows
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
  const userId = req.user!.id;

  if (!fromSubAccountId || !toSubAccountId || !asset || !amount || amount <= 0) {
    return res.status(400).json({
      error: 'Invalid transfer parameters',
      code: 'INVALID_PARAMS'
    });
  }

  try {
    // Verify ownership of both sub-accounts
    const ownershipResult = await db.query(
      `SELECT id FROM sub_accounts 
       WHERE id IN ($1, $2) AND main_account_id = $3 AND is_active = true`,
      [fromSubAccountId, toSubAccountId, userId]
    );

    if (ownershipResult.rows.length !== 2) {
      return res.status(400).json({
        error: 'Invalid sub-accounts',
        code: 'INVALID_ACCOUNTS'
      });
    }

    // Check sufficient balance
    const balanceResult = await db.query(
      `SELECT available_balance FROM user_balances 
       WHERE sub_account_id = $1 AND asset = $2`,
      [fromSubAccountId, asset]
    );

    if (balanceResult.rows.length === 0 || balanceResult.rows[0].available_balance < amount) {
      return res.status(400).json({
        error: 'Insufficient balance',
        code: 'INSUFFICIENT_BALANCE'
      });
    }

    // Execute transfer in transaction
    await db.transaction(async (client) => {
      // Debit from source
      await client.query(
        `UPDATE user_balances 
         SET balance = balance - $1, updated_at = NOW() 
         WHERE sub_account_id = $2 AND asset = $3`,
        [amount, fromSubAccountId, asset]
      );

      // Credit to destination (create if doesn't exist)
      await client.query(
        `INSERT INTO user_balances (user_id, sub_account_id, asset, balance, locked_balance) 
         VALUES ($1, $2, $3, $4, 0) 
         ON CONFLICT (user_id, sub_account_id, asset) 
         DO UPDATE SET balance = user_balances.balance + $4, updated_at = NOW()`,
        [userId, toSubAccountId, asset, amount]
      );
    });

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
