import express, { Request, Response } from 'express';
import { randomBytes } from 'crypto';
import bs58 from 'bs58';
import nacl from 'tweetnacl';
import { databaseService } from '../services/supabaseDatabase';
import jwt from 'jsonwebtoken';
// Conditional Redis import
let setSession: any;
if (process.env.NODE_ENV === 'development' && !process.env.REDIS_URL) {
  setSession = () => Promise.resolve();
} else {
  setSession = require('../services/redisClient').setSession;
}
import { asyncHandler } from '../middleware/errorHandling';

const router = express.Router();

// POST /api/siws/nonce -> { walletPubkey }
router.post('/nonce', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const { walletPubkey } = req.body || {};
  if (!walletPubkey) {
    res.status(400).json({ error: 'walletPubkey required' });
    return;
  }
  const nonce = bs58.encode(randomBytes(24));
  
  // TEMPORARY FIX: Skip database operations until Supabase is properly initialized
  console.log('⚠️  TEMPORARY: Skipping database operations, using in-memory nonce');
  
  // Store nonce in memory temporarily (this will be lost on restart)
  if (!global.tempNonces) {
    global.tempNonces = new Map();
  }
  global.tempNonces.set(walletPubkey, nonce);
  
  res.json({ nonce });
}));

// POST /api/siws/verify -> { walletPubkey, signature, nonce, ref }
router.post('/verify', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const { walletPubkey, signature, nonce, ref } = req.body || {};
  if (!walletPubkey || !signature || !nonce) {
    res.status(400).json({ error: 'walletPubkey, signature, nonce required' });
    return;
  }
  // TEMPORARY FIX: Use in-memory nonces instead of database
  const storedNonce = global.tempNonces?.get(walletPubkey);
  if (!storedNonce || storedNonce !== nonce) {
    res.status(401).json({ error: 'invalid nonce' });
    return;
  }
  const message = Buffer.from(nonce);
  const sig = bs58.decode(signature);
  const pub = bs58.decode(walletPubkey);
  const ok = nacl.sign.detached.verify(message, sig, pub);
  if (!ok) {
    res.status(401).json({ error: 'invalid signature' });
    return;
  }
  // TEMPORARY FIX: Skip referral logic until database is initialized
  console.log('⚠️  TEMPORARY: Skipping referral logic');
  
  // Get or create user to ensure we have user_id
  let user = await databaseService.getUserByWallet(walletPubkey);
  if (!user) {
    user = await databaseService.createUser(walletPubkey);
    console.log(`New user created via SIWS: ${walletPubkey}`);
  }
  
  // Issue HttpOnly session cookie + store session in Redis with standardized payload
  const token = jwt.sign(
    { 
      wallet_pubkey: walletPubkey,
      user_id: user.id,
      iat: Math.floor(Date.now() / 1000),
      exp: Math.floor(Date.now() / 1000) + (7 * 24 * 60 * 60) // 7 days
    }, 
    process.env.JWT_SECRET as string, 
    { expiresIn: '7d' }
  );
  await setSession(walletPubkey, { wallet_pubkey: walletPubkey, user_id: user.id, createdAt: Date.now() }, 7 * 24 * 3600);
  res.cookie('qd_session', token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 7 * 24 * 3600 * 1000
  }).json({ success: true });
}));

// POST /api/siws/logout
router.post('/logout', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  res.clearCookie('qd_session', {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
  }).json({ success: true, message: 'Logged out successfully' });
}));

export default router;


