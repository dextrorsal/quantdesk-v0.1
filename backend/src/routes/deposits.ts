import express from 'express';
import { databaseService } from '../services/supabaseDatabase';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandling';
import { AuthenticatedRequest } from '../middleware/auth';
import { transactionVerificationService } from '../services/transactionVerificationService';

const router = express.Router();
const logger = new Logger();
const supabase = databaseService.getClient();

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
    const userId = req.userId!;
    const { tradingAccountId } = req.query;

    // Build Supabase query
    let query = supabase.from("user_balances").select("*").eq("user_id", userId);

    if (tradingAccountId) {
      query = query.eq("trading_account_id", tradingAccountId as string);
    } else {
      query = query.is("trading_account_id", null); // Master account only
    }

    const { data: result, error } = await query.order("asset");

    if (error) {
      logger.error('Failed to fetch balances:', error);
      return res.status(500).json({
        error: 'Failed to fetch balances',
        code: 'FETCH_ERROR'
      });
    }

    res.json({
      success: true,
      balances: (result || []).map(balance => ({
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
  const userId = req.userId!;
  const userWallet = req.walletPubkey!;

  // Enhanced validation with comprehensive error handling
  if (!asset || !amount || amount <= 0) {
    return res.status(400).json({
      error: 'Valid asset and amount required',
      code: 'INVALID_PARAMS',
      details: {
        asset: asset || 'missing',
        amount: amount || 'missing',
        validation: 'Amount must be positive'
      }
    });
  }

  // Additional validation for amount precision
  if (amount > 1000000) { // Max $1M per transaction
    return res.status(400).json({
      error: 'Amount exceeds maximum limit',
      code: 'AMOUNT_TOO_LARGE',
      details: {
        maxAmount: 1000000,
        providedAmount: amount
      }
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
      const accountCheck = await databaseService.select('trading_accounts', 'id', { 
        id: tradingAccountId, 
        master_account_id: userId, 
        is_active: true 
      });

      if (accountCheck.length === 0) {
        return res.status(400).json({
          error: 'Invalid trading account',
          code: 'INVALID_ACCOUNT'
        });
      }
    }

    // Create deposit record
    const depositResult = await databaseService.insert('deposits', {
      user_id: userId,
      trading_account_id: tradingAccountId || null,
      asset: asset,
      amount: amount,
      status: 'pending',
      wallet_address: userWallet,
      created_at: new Date().toISOString()
    });

    const deposit = depositResult[0];

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
  const userId = req.userId!;

  if (!depositId || !transactionSignature) {
    return res.status(400).json({
      error: 'Deposit ID and transaction signature required',
      code: 'MISSING_PARAMS'
    });
  }

  try {
    // Get deposit record
    const depositResult = await databaseService.select('deposits', '*', { 
      id: depositId, 
      user_id: userId, 
      status: 'pending' 
    });

    if (depositResult.length === 0) {
      return res.status(404).json({
        error: 'Deposit not found',
        code: 'NOT_FOUND'
      });
    }

    const deposit = depositResult[0];

    // Verify transaction on Solana blockchain
    logger.info(`🔍 Verifying deposit transaction: ${transactionSignature}`);
    
    const verificationResult = await transactionVerificationService.verifyDepositTransaction(
      transactionSignature,
      {
        userWallet: req.walletPubkey!,
        amount: deposit.amount,
        asset: deposit.asset,
        expectedProgramId: process.env.QUANTDESK_PROGRAM_ID // Optional: verify specific program
      }
    );

    if (!verificationResult.isValid) {
      logger.error(`❌ Deposit transaction verification failed: ${verificationResult.error}`);
      return res.status(400).json({
        error: 'Transaction verification failed',
        details: verificationResult.error,
        code: 'TRANSACTION_VERIFICATION_FAILED'
      });
    }

    logger.info(`✅ Deposit transaction verified successfully: ${transactionSignature}`);

    // Update deposit status and user balance in a transaction
    await databaseService.transaction(async (client) => {
      // Update deposit status
      await databaseService.update('deposits', {
        status: 'completed',
        transaction_signature: transactionSignature,
        confirmed_at: new Date().toISOString()
      }, { id: depositId });

      // Update or create user balance
      await databaseService.upsert('user_balances', {
        user_id: userId,
        trading_account_id: deposit.trading_account_id,
        asset: deposit.asset,
        balance: deposit.amount,
        locked_balance: 0,
        available_balance: deposit.amount
      });
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
  const userId = req.userId!;
  const userWallet = req.walletPubkey!;

  // Enhanced validation with comprehensive error handling
  if (!asset || !amount || amount <= 0) {
    return res.status(400).json({
      error: 'Valid asset and amount required',
      code: 'INVALID_PARAMS',
      details: {
        asset: asset || 'missing',
        amount: amount || 'missing',
        validation: 'Amount must be positive'
      }
    });
  }

  // Additional validation for amount precision
  if (amount > 1000000) { // Max $1M per transaction
    return res.status(400).json({
      error: 'Amount exceeds maximum limit',
      code: 'AMOUNT_TOO_LARGE',
      details: {
        maxAmount: 1000000,
        providedAmount: amount
      }
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
      const accountCheck = await databaseService.select('trading_accounts', 'id', {
        id: tradingAccountId,
        master_account_id: userId,
        is_active: true
      });

      if (accountCheck.length === 0) {
        return res.status(400).json({
          error: 'Invalid trading account',
          code: 'INVALID_ACCOUNT'
        });
      }
    }

    // Check available balance
    const balanceResult = await databaseService.complexQuery(
      `SELECT available_balance FROM user_balances 
       WHERE user_id = $1 AND asset = $2 AND 
       (trading_account_id = $3 OR (trading_account_id IS NULL AND $3 IS NULL))`,
      [userId, asset, tradingAccountId]
    );

    if (balanceResult.length === 0 || parseFloat(balanceResult[0].available_balance) < amount) {
      return res.status(400).json({
        error: 'Insufficient balance',
        code: 'INSUFFICIENT_BALANCE'
      });
    }

    // Create withdrawal record and lock funds
    await databaseService.transaction(async (client) => {
      // Create withdrawal record
      const withdrawalResult = await databaseService.insert('withdrawals', {
        user_id: userId,
        trading_account_id: tradingAccountId || null,
        asset: asset,
        amount: amount,
        destination_address: finalDestination,
        status: 'pending',
        created_at: new Date().toISOString()
      });

      const withdrawal = withdrawalResult[0];

      // Lock the funds
      await databaseService.complexQuery(
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
    const userId = req.userId!;
    const { tradingAccountId, type, limit = 50, offset = 0 } = req.query;

    // Convert limit and offset to numbers
    const limitNum = parseInt(limit as string) || 50;
    const offsetNum = parseInt(offset as string) || 0;

    let deposits = [];
    let withdrawals = [];

    if (!type || type === 'deposits') {
      // Use fluent API for deposits query
      let depositsQuery = databaseService.getClient()
        .from('deposits')
        .select('id, asset, amount, destination_address, status, transaction_signature, trading_account_id, created_at, confirmed_at')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .range(offsetNum, offsetNum + limitNum - 1);

      if (tradingAccountId) {
        depositsQuery = depositsQuery.eq('trading_account_id', tradingAccountId);
      }

      const { data: depositResult, error: depositError } = await depositsQuery;
      
      if (depositError) {
        logger.error('Error fetching deposits:', depositError);
        return res.status(500).json({ success: false, error: 'Failed to fetch deposits' });
      }

      deposits = (depositResult || []).map(d => ({
        ...d,
        type: 'deposit',
        amount: parseFloat(d.amount)
      }));
    }

    if (!type || type === 'withdrawals') {
      // Use fluent API for withdrawals query
      let withdrawalsQuery = databaseService.getClient()
        .from('withdrawals')
        .select('id, asset, amount, destination_address, status, transaction_signature, trading_account_id, created_at, confirmed_at')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .range(offsetNum, offsetNum + limitNum - 1);

      if (tradingAccountId) {
        withdrawalsQuery = withdrawalsQuery.eq('trading_account_id', tradingAccountId);
      }

      const { data: withdrawalResult, error: withdrawalError } = await withdrawalsQuery;
      
      if (withdrawalError) {
        logger.error('Error fetching withdrawals:', withdrawalError);
        return res.status(500).json({ success: false, error: 'Failed to fetch withdrawals' });
      }

      withdrawals = (withdrawalResult || []).map(w => ({
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
