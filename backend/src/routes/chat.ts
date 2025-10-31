import express, { Request, Response } from 'express';
import jwt from 'jsonwebtoken';
import { asyncHandler } from '../middleware/errorHandling';
import { databaseService } from '../services/supabaseDatabase';
// Conditional Redis import
let rateLimitSimple: any, setPresence: any, publishRedisMessage: any;
if (process.env.NODE_ENV === 'development' && !process.env.REDIS_URL) {
  rateLimitSimple = () => Promise.resolve({ success: true });
  setPresence = () => Promise.resolve();
  publishRedisMessage = () => Promise.resolve();
} else {
  const redisModule = require('../services/redisClient');
  rateLimitSimple = redisModule.rateLimitSimple;
  setPresence = redisModule.setPresence;
  publishRedisMessage = redisModule.publishRedisMessage;
}
import { authMiddleware } from '../middleware/auth';

const router = express.Router();

// GET /api/chat/channels (Protected)
router.get('/channels', authMiddleware, asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const channels = await databaseService.getChannels();
  res.json({ channels });
}));

// POST /api/chat/channels (Protected)
router.post('/channels', authMiddleware, asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const { name, description, isPrivate } = req.body || {};
  const userId = req.userId; // From authMiddleware
  
  if (!name || !userId) {
    res.status(400).json({ error: 'Channel name and user ID required' });
    return;
  }

  try {
    const newChannel = await databaseService.createChannel(name, description, isPrivate || false, userId);
    res.status(201).json({ success: true, channel: newChannel });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
}));

// GET /api/chat/history?channelId=<id>&limit=50 (Protected)
router.get('/history', authMiddleware, asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const channelId = typeof req.query.channelId === 'string' ? req.query.channelId : '';
  if (!channelId) {
    res.status(400).json({ error: 'channelId required' });
    return;
  }
  const limitParam = typeof req.query.limit === 'string' ? req.query.limit : '50';
  const limit = Math.min(parseInt(limitParam), 200);

  // presence heartbeat (optional, can be done via WS directly as well)
  const wallet = req.walletPubkey; // From authMiddleware
  if (wallet) await setPresence(channelId, wallet);

  const messages = await databaseService.getMessages(channelId, limit);
  const reversedMessages = [...messages].reverse();
  res.json({ messages: reversedMessages });
}));

// POST /api/chat/token -> { wallet_pubkey } (Protected)
router.post('/token', authMiddleware, asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const wallet_pubkey = req.walletPubkey; // From authMiddleware
  if (!wallet_pubkey) { res.status(400).json({ error: 'wallet_pubkey required' }); return; } // Should not happen with authMiddleware
  
  const user = await databaseService.getUserByWallet(wallet_pubkey);
  if (!user) { res.status(404).json({ error: 'user not found' }); return; }
  
  // This token is for client-side use with Supabase Realtime or other services,
  // not directly for our WebSocket server which uses qd_session cookie.
  const jwtSecret = process.env.JWT_SECRET;
  if (!jwtSecret) {
    res.status(500).json({ error: 'JWT_SECRET not configured' });
    return;
  }
  const token = jwt.sign({ wallet_pubkey, role: user.role || 'tester' }, jwtSecret, { expiresIn: '1h' });
  res.json({ token });
}));

// POST /api/chat/send -> { channelId, message } (Protected)
router.post('/send', authMiddleware, asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const { channelId, message } = req.body || {};
  const wallet_pubkey = req.walletPubkey; // From authMiddleware

  if (!wallet_pubkey || !message || !channelId) {
    res.status(400).json({ error: 'wallet_pubkey, message, and channelId required' });
    return;
  }

  // Simple rate limit: 5 msgs / 10s per user
  const ok = await rateLimitSimple(`chat:${wallet_pubkey}`, 5, 10);
  if (!ok) { res.status(429).json({ error: 'rate_limited' }); return; }

  // Parse mentions from message (@wallet_pubkey)
  const mentionRegex = /@([A-Za-z0-9]{32,44})/g;
  const mentions = [];
  let match;
  while ((match = mentionRegex.exec(message)) !== null) {
    mentions.push(match[1]);
  }

  // Save message to Supabase (for history)
  const { data, error } = await databaseService.sendMessage(channelId, wallet_pubkey, message, mentions);
  if (error) { res.status(500).json({ error: error.message }); return; }

  // Publish message to Redis Pub/Sub for real-time delivery
  const chatMessage = {
    type: 'chat_message',
    id: data?.[0].id,
    channelId,
    author_pubkey: wallet_pubkey,
    message,
    mentions,
    created_at: data?.[0].created_at,
  };
  await publishRedisMessage(channelId, JSON.stringify(chatMessage));

  res.json({ success: true, messageId: data?.[0].id });
}));

export default router;


