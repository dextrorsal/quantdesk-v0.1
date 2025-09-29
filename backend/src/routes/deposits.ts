import express from 'express';
import { DatabaseService } from '../services/database';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandler';
import { AuthenticatedRequest } from '../middleware/auth';
import { Connection, PublicKey, Transaction } from '@solana/web3.js';

const router = express.Router();
const logger = new Logger();
const db = DatabaseService.getInstance();

// Supported tokens for deposits/withdrawals
const SUPPORTED_TOKENS = {
  'SOL': {
    symbol: 'SOL',
    name: 'Solana',
    decimals: 9,
    mintAddress: 'So11111111111111111111111111111111111111112' // Native SOL
  },
  'USDC': {
    symbol: 'USDC',
    name: 'USD Coin',
    decimals: 6,
    mintAddress: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
  },
  'USDT': {
    symbol: 'USDT',
    name: 'Tether USD',
    decimals: 6,
    mintAddress: 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB'
  },
  'BTC': {
    symbol: 'BTC',
    name: 'Bitcoin (Wrapped)',
    decimals: 8,
    mintAddress: '9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E'
  },
  'ETH': {
    symbol: 'ETH',
    name: 'Ethereum (Wrapped)',
    decimals: 8,
    mintAddress: '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs'
  }
};

// Get user's token balances
router.get('/balances', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const userId = req.user!.id;
    const { tradingAccountId } = req.query;

    let query = `
      SELECT asset, balance, locked_balance, available_balance, updated_at
      FROM user_balances 
      WHERE user_id = $1
    `;
    let params = [userId];

    if (tradingAccountId) {
      query += ` AND trading_account_id = $2`;
      params.push(tradingAccountId as string);
    } else {
      query += ` AND trading_account_id IS NULL`; // Master account only
    }

    query += ` ORDER BY asset`;

    const result = await db.query(query, params);

    res.json({
      success: true,
      balances: result.rows.map(balance => ({
        asset: balance.asset,
        balance: parseFloat(balance.balance),
        lockedBalance: parseFloat(balance.locked_balance),
        availableBalance: parseFloat(balance.available_balance),
        updatedAt: balance.updated_at
      })),
      supportedTokens: SUPPORTED_TOKENS
    });

  } catch (error) {
    logger.error('Error fetching balances:', error);
    res.status(500).json({
      error: 'Failed to fetch balances',
      code: 'FETCH_ERROR'
    });
  }
}));

// Initiate deposit
router.post('/deposit', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { asset, amount, tradingAccountId } = req.body;
  const userId = req.user!.id;
  const userWallet = req.user!.walletAddress;

  if (!asset || !amount || amount <= 0) {
    return res.status(400).json({
      error: 'Valid asset and amount required',
      code: 'INVALID_PARAMS'
    });
  }

  if (!SUPPORTED_TOKENS[asset as keyof typeof SUPPORTED_TOKENS]) {
    return res.status(400).json({
      error: 'Unsupported token',
      code: 'UNSUPPORTED_TOKEN'
    });
  }

  try {
    // Verify trading account ownership if specified
    if (tradingAccountId) {
      const accountCheck = await db.query(
        `SELECT id FROM trading_accounts 
         WHERE id = $1 AND master_account_id = $2 AND is_active = true`,
        [tradingAccountId, userId]
      );

      if (accountCheck.rows.length === 0) {
        return res.status(400).json({
          error: 'Invalid trading account',
          code: 'INVALID_ACCOUNT'
        });
      }
    }

    // Create deposit record
    const depositResult = await db.query(
      `INSERT INTO deposits (
        user_id, 
        trading_account_id,
        asset, 
        amount, 
        status, 
        wallet_address,
        created_at
      ) VALUES ($1, $2, $3, $4, 'pending', $5, NOW()) 
      RETURNING *`,
      [userId, tradingAccountId || null, asset, amount, userWallet]
    );

    const deposit = depositResult.rows[0];

    // TODO: Generate Solana transaction for user to sign
    // For now, we'll return the deposit record and expect frontend to handle transaction

    res.status(201).json({
      success: true,
      deposit: {
        id: deposit.id,
        asset: deposit.asset,
        amount: parseFloat(deposit.amount),
        status: deposit.status,
        tradingAccountId: deposit.trading_account_id,
        createdAt: deposit.created_at
      },
      message: 'Deposit initiated. Please sign the transaction in your wallet.'
    });

  } catch (error) {
    logger.error('Error initiating deposit:', error);
    res.status(500).json({
      error: 'Failed to initiate deposit',
      code: 'DEPOSIT_ERROR'
    });
  }
}));

// Confirm deposit (called after transaction is signed and broadcasted)
router.post('/deposit/confirm', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { depositId, transactionSignature } = req.body;
  const userId = req.user!.id;

  if (!depositId || !transactionSignature) {
    return res.status(400).json({
      error: 'Deposit ID and transaction signature required',
      code: 'MISSING_PARAMS'
    });
  }

  try {
    // Get deposit record
    const depositResult = await db.query(
      `SELECT * FROM deposits 
       WHERE id = $1 AND user_id = $2 AND status = 'pending'`,
      [depositId, userId]
    );

    if (depositResult.rows.length === 0) {
      return res.status(404).json({
        error: 'Deposit not found',
        code: 'NOT_FOUND'
      });
    }

    const deposit = depositResult.rows[0];

    // TODO: Verify transaction on Solana blockchain
    // For now, we'll assume the transaction is valid

    // Update deposit status and user balance in a transaction
    await db.transaction(async (client) => {
      // Update deposit status
      await client.query(
        `UPDATE deposits 
         SET status = 'completed', transaction_signature = $1, confirmed_at = NOW() 
         WHERE id = $2`,
        [transactionSignature, depositId]
      );

      // Update or create user balance
      await client.query(
        `INSERT INTO user_balances (
          user_id, 
          trading_account_id, 
          asset, 
          balance, 
          locked_balance, 
          available_balance
        ) VALUES ($1, $2, $3, $4, 0, $4)
        ON CONFLICT (user_id, COALESCE(trading_account_id, '00000000-0000-0000-0000-000000000000'::uuid), asset)
        DO UPDATE SET 
          balance = user_balances.balance + $4,
          available_balance = user_balances.available_balance + $4,
          updated_at = NOW()`,
        [userId, deposit.trading_account_id, deposit.asset, deposit.amount]
      );
    });

    res.json({
      success: true,
      message: 'Deposit confirmed successfully',
      deposit: {
        id: deposit.id,
        asset: deposit.asset,
        amount: parseFloat(deposit.amount),
        status: 'completed',
        transactionSignature
      }
    });

  } catch (error) {
    logger.error('Error confirming deposit:', error);
    res.status(500).json({
      error: 'Failed to confirm deposit',
      code: 'CONFIRM_ERROR'
    });
  }
}));

// Initiate withdrawal
router.post('/withdraw', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { asset, amount, tradingAccountId, destinationAddress } = req.body;
  const userId = req.user!.id;
  const userWallet = req.user!.walletAddress;

  if (!asset || !amount || amount <= 0) {
    return res.status(400).json({
      error: 'Valid asset and amount required',
      code: 'INVALID_PARAMS'
    });
  }

  if (!SUPPORTED_TOKENS[asset as keyof typeof SUPPORTED_TOKENS]) {
    return res.status(400).json({
      error: 'Unsupported token',
      code: 'UNSUPPORTED_TOKEN'
    });
  }

  // Use user's wallet as destination if not specified
  const finalDestination = destinationAddress || userWallet;

  try {
    // Verify trading account ownership if specified
    if (tradingAccountId) {
      const accountCheck = await db.query(
        `SELECT id FROM trading_accounts 
         WHERE id = $1 AND master_account_id = $2 AND is_active = true`,
        [tradingAccountId, userId]
      );

      if (accountCheck.rows.length === 0) {
        return res.status(400).json({
          error: 'Invalid trading account',
          code: 'INVALID_ACCOUNT'
        });
      }
    }

    // Check available balance
    const balanceResult = await db.query(
      `SELECT available_balance FROM user_balances 
       WHERE user_id = $1 AND asset = $2 AND 
       (trading_account_id = $3 OR (trading_account_id IS NULL AND $3 IS NULL))`,
      [userId, asset, tradingAccountId]
    );

    if (balanceResult.rows.length === 0 || parseFloat(balanceResult.rows[0].available_balance) < amount) {
      return res.status(400).json({
        error: 'Insufficient balance',
        code: 'INSUFFICIENT_BALANCE'
      });
    }

    // Create withdrawal record and lock funds
    await db.transaction(async (client) => {
      // Create withdrawal record
      const withdrawalResult = await client.query(
        `INSERT INTO withdrawals (
          user_id, 
          trading_account_id,
          asset, 
          amount, 
          destination_address,
          status, 
          created_at
        ) VALUES ($1, $2, $3, $4, $5, 'pending', NOW()) 
        RETURNING *`,
        [userId, tradingAccountId || null, asset, amount, finalDestination]
      );

      const withdrawal = withdrawalResult.rows[0];

      // Lock the funds
      await client.query(
        `UPDATE user_balances 
         SET locked_balance = locked_balance + $1,
             available_balance = available_balance - $1,
             updated_at = NOW()
         WHERE user_id = $2 AND asset = $3 AND 
         (trading_account_id = $4 OR (trading_account_id IS NULL AND $4 IS NULL))`,
        [amount, userId, asset, tradingAccountId]
      );

      res.status(201).json({
        success: true,
        withdrawal: {
          id: withdrawal.id,
          asset: withdrawal.asset,
          amount: parseFloat(withdrawal.amount),
          destinationAddress: withdrawal.destination_address,
          status: withdrawal.status,
          tradingAccountId: withdrawal.trading_account_id,
          createdAt: withdrawal.created_at
        },
        message: 'Withdrawal initiated. Funds have been locked pending processing.'
      });
    });

  } catch (error) {
    logger.error('Error initiating withdrawal:', error);
    res.status(500).json({
      error: 'Failed to initiate withdrawal',
      code: 'WITHDRAWAL_ERROR'
    });
  }
}));

// Get deposit/withdrawal history
router.get('/history', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const userId = req.user!.id;
    const { tradingAccountId, type, limit = 50, offset = 0 } = req.query;

    let deposits = [];
    let withdrawals = [];

    if (!type || type === 'deposits') {
      let depositQuery = `
        SELECT id, asset, amount, status, transaction_signature, 
               trading_account_id, created_at, confirmed_at
        FROM deposits 
        WHERE user_id = $1
      `;
      let depositParams = [userId];

      if (tradingAccountId) {
        depositQuery += ` AND trading_account_id = $2`;
        depositParams.push(tradingAccountId as string);
      }

      depositQuery += ` ORDER BY created_at DESC LIMIT $${depositParams.length + 1} OFFSET $${depositParams.length + 2}`;
      depositParams.push(limit as string, offset as string);

      const depositResult = await db.query(depositQuery, depositParams);
      deposits = depositResult.rows.map(d => ({
        ...d,
        type: 'deposit',
        amount: parseFloat(d.amount)
      }));
    }

    if (!type || type === 'withdrawals') {
      let withdrawalQuery = `
        SELECT id, asset, amount, destination_address, status, transaction_signature,
               trading_account_id, created_at, confirmed_at
        FROM withdrawals 
        WHERE user_id = $1
      `;
      let withdrawalParams = [userId];

      if (tradingAccountId) {
        withdrawalQuery += ` AND trading_account_id = $2`;
        withdrawalParams.push(tradingAccountId as string);
      }

      withdrawalQuery += ` ORDER BY created_at DESC LIMIT $${withdrawalParams.length + 1} OFFSET $${withdrawalParams.length + 2}`;
      withdrawalParams.push(limit as string, offset as string);

      const withdrawalResult = await db.query(withdrawalQuery, withdrawalParams);
      withdrawals = withdrawalResult.rows.map(w => ({
        ...w,
        type: 'withdrawal',
        amount: parseFloat(w.amount)
      }));
    }

    // Combine and sort by date
    const allTransactions = [...deposits, ...withdrawals]
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

    res.json({
      success: true,
      transactions: allTransactions,
      pagination: {
        limit: parseInt(limit as string),
        offset: parseInt(offset as string),
        total: allTransactions.length
      }
    });

  } catch (error) {
    logger.error('Error fetching transaction history:', error);
    res.status(500).json({
      error: 'Failed to fetch transaction history',
      code: 'HISTORY_ERROR'
    });
  }
}));

export default router;
