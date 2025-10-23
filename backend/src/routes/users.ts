import express, { Request, Response } from 'express';
import { asyncHandler } from '../middleware/errorHandling';
import { authMiddleware } from '../middleware/auth';
import { databaseService } from '../services/supabaseDatabase';

const router = express.Router();

// GET /api/users/profile (Protected)
router.get('/profile', authMiddleware, asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const walletPubkey = req.walletPubkey; // From authMiddleware

  if (!walletPubkey) {
    res.status(400).json({ error: 'walletPubkey not found in session' });
    return;
  }

  const user = await databaseService.getUserByWallet(walletPubkey);

  if (!user) {
    res.status(404).json({ error: 'User not found' });
    return;
  }

  // Omit sensitive fields if necessary before sending to frontend
  const { id, wallet_pubkey, username, email, referrer_pubkey, is_activated, role, created_at } = user;
  res.json({ user: { id, wallet_pubkey, username, email, referrer_pubkey, is_activated, role, created_at } });
}));

export default router;
