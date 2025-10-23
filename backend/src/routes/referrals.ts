import express, { Request, Response } from 'express';
import { asyncHandler } from '../middleware/errorHandling';
import { databaseService } from '../services/supabaseDatabase';
import { ReferralPayoutService } from '../services/referralPayout';
import { referralService } from '../services/referralService';

const router = express.Router();

// GET /api/referrals/summary?wallet=<pubkey>
router.get('/summary', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const wallet = String(req.query.wallet || '');
  if (!wallet) { res.status(400).json({ error: 'wallet required' }); return; }
  const refs = await databaseService.select('referrals', '*', { referrer_pubkey: wallet });
  const earnings = await referralService.calculateReferralEarnings(wallet);
  res.json({ count: refs?.length || 0, earnings, refs });
}));

// GET /api/referrals/preview?referrer=<pubkey>
router.get('/preview', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const referrer = String(req.query.referrer || '');
  if (!referrer) { res.status(400).json({ error: 'referrer required' }); return; }
  const referralShare = await referralService.calculateReferralEarnings(referrer);
  res.json({ referrer, totalRefereeFees: referralShare / 0.25, referralShare });
}));

// POST /api/referrals/claim -> { referrer }
router.post('/claim', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const { referrer } = req.body || {};
  if (!referrer) { res.status(400).json({ error: 'referrer required' }); return; }
  const amountToClaim = await referralService.calculateReferralEarnings(referrer);
  if (amountToClaim <= 0) { res.status(400).json({ error: 'no earnings to claim' }); return; }

  const rpc = process.env.RPC_URL || 'https://api.devnet.solana.com';
  const payer = process.env.PAYOUT_PAYER_SECRET_BASE58 || '';
  if (!payer) { res.status(500).json({ error: 'missing payout payer secret' }); return; }

  const svc = new ReferralPayoutService(rpc, payer);
  const result = await svc.sendSol(referrer, amountToClaim);
  await referralService.recordPayout(referrer, amountToClaim, result.txSig);

  res.json({ success: true, tx: result.txSig, lamports: result.lamports });
}));

// POST /api/referrals/activate -> { referee_pubkey, minimum_volume }
router.post('/activate', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const { referee_pubkey, minimum_volume } = req.body || {};
  if (!referee_pubkey) { res.status(400).json({ error: 'referee_pubkey required' }); return; }
  // For now, minimum_volume is a mock value. In a real scenario, this would be determined by actual trading activity.
  await databaseService.getClient().rpc('activate_referral', { p_referee: referee_pubkey, p_min_volume: minimum_volume || 0 });
  res.json({ success: true });
}));

export default router;


