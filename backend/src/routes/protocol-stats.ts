import express from 'express';
import { Connection, PublicKey } from '@solana/web3.js';
import { config } from '../config/environment';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandling';

const router = express.Router();
const logger = new Logger();

const connection = new Connection(config.SOLANA_RPC_URL, 'confirmed');
const programId = new PublicKey(config.QUANTDESK_PROGRAM_ID);

/**
 * GET /api/protocol/stats
 * Public endpoint - shows protocol health/stats
 * Safe to expose (doesn't show sensitive admin data)
 */
router.get('/stats', asyncHandler(async (req, res) => {
  try {
    // 1. Program account balance (infrastructure)
    const programBalance = await connection.getBalance(programId);

    // 2. Fee collector PDA
    const [feeCollectorPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from('fee_collector')],
      programId
    );

    let feeCollectorBalance = 0;
    let feeCollectorInitialized = false;
    
    try {
      const feeAccountInfo = await connection.getAccountInfo(feeCollectorPDA);
      if (feeAccountInfo) {
        feeCollectorBalance = await connection.getBalance(feeCollectorPDA);
        feeCollectorInitialized = true;
      }
    } catch (e) {
      // Not initialized yet
    }

    // 3. Insurance fund PDA
    const [insuranceFundPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from('insurance_fund')],
      programId
    );

    let insuranceFundBalance = 0;
    let insuranceFundInitialized = false;
    
    try {
      const insuranceInfo = await connection.getAccountInfo(insuranceFundPDA);
      if (insuranceInfo) {
        insuranceFundBalance = await connection.getBalance(insuranceFundPDA);
        insuranceFundInitialized = true;
      }
    } catch (e) {
      // Not initialized yet
    }

    res.json({
      success: true,
      network: config.SOLANA_NETWORK,
      program: {
        id: programId.toBase58(),
        balance_lamports: programBalance,
        balance_sol: programBalance / 1e9,
        deployed: programBalance > 0,
      },
      fee_collector: {
        address: feeCollectorPDA.toBase58(),
        initialized: feeCollectorInitialized,
        balance_lamports: feeCollectorBalance,
        balance_sol: feeCollectorBalance / 1e9,
      },
      insurance_fund: {
        address: insuranceFundPDA.toBase58(),
        initialized: insuranceFundInitialized,
        balance_lamports: insuranceFundBalance,
        balance_sol: insuranceFundBalance / 1e9,
      },
      explorer_links: {
        program: `https://explorer.solana.com/address/${programId.toBase58()}?cluster=${config.SOLANA_NETWORK}`,
        fee_collector: `https://explorer.solana.com/address/${feeCollectorPDA.toBase58()}?cluster=${config.SOLANA_NETWORK}`,
        insurance_fund: `https://explorer.solana.com/address/${insuranceFundPDA.toBase58()}?cluster=${config.SOLANA_NETWORK}`,
      },
    });
  } catch (error) {
    logger.error('Error fetching protocol stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch protocol stats',
    });
  }
}));

/**
 * GET /api/protocol/treasury
 * ADMIN ONLY - shows detailed treasury info
 * Should require admin authentication in production
 */
router.get('/treasury', asyncHandler(async (req, res) => {
  // TODO: Add admin auth middleware
  // if (!req.user?.isAdmin) return res.status(403).json({ error: 'Forbidden' });

  try {
    const stats = {
      total_value_locked_sol: 0,
      fee_revenue_sol: 0,
      insurance_fund_sol: 0,
      warning: 'ADMIN ONLY - Do not expose publicly in production',
    };

    // Fetch detailed treasury data
    // This would include SPL token balances, fee breakdowns, etc.

    res.json({
      success: true,
      treasury: stats,
      note: 'In production, add authentication and restrict access',
    });
  } catch (error) {
    logger.error('Error fetching treasury stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch treasury stats',
    });
  }
}));

/**
 * GET /api/protocol/user/:wallet
 * Get user account details (ADMIN ONLY in production)
 */
router.get('/user/:wallet', asyncHandler(async (req, res) => {
  // TODO: Add admin auth middleware in production
  // if (!req.user?.isAdmin) return res.status(403).json({ error: 'Forbidden' });

  const { wallet } = req.params;

  try {
    const userPubkey = new PublicKey(wallet);

    // Derive user account PDA
    const accountIndex = 0;
    const accountIndexBuffer = Buffer.alloc(2);
    accountIndexBuffer.writeUInt16LE(accountIndex, 0);

    const [userAccountPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from('user_account'), userPubkey.toBuffer(), accountIndexBuffer],
      programId
    );

    // Check if account exists
    const userAccountInfo = await connection.getAccountInfo(userAccountPDA);

    if (!userAccountInfo) {
      return res.json({
        success: true,
        exists: false,
        wallet,
        message: 'User account not created yet',
      });
    }

    // Get wallet balance
    const walletBalance = await connection.getBalance(userPubkey);

    // Decode basic info (simplified - full decode needs Anchor program)
    const data = userAccountInfo.data;
    const authority = new PublicKey(data.slice(8, 40));
    const accountIndexDecoded = data.readUInt16LE(40);
    const totalCollateral = Number(data.readBigUInt64LE(42));
    const totalPositions = data.readUInt16LE(50);
    const totalOrders = data.readUInt16LE(52);

    res.json({
      success: true,
      exists: true,
      wallet,
      account: {
        pda: userAccountPDA.toBase58(),
        authority: authority.toBase58(),
        account_index: accountIndexDecoded,
        wallet_balance_sol: walletBalance / 1e9,
        account_rent_sol: userAccountInfo.lamports / 1e9,
        total_collateral_usdc: totalCollateral / 1e6,
        total_positions: totalPositions,
        total_orders: totalOrders,
        data_size: userAccountInfo.data.length,
      },
      explorer_links: {
        wallet: `https://explorer.solana.com/address/${wallet}?cluster=${config.SOLANA_NETWORK}`,
        user_account: `https://explorer.solana.com/address/${userAccountPDA.toBase58()}?cluster=${config.SOLANA_NETWORK}`,
      },
      warning: 'ADMIN ONLY - Add authentication before production',
    });
  } catch (error) {
    logger.error('Error fetching user account:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch user account',
    });
  }
}));

export default router;

