import express, { Request, Response } from 'express'
import { asyncHandler } from '../middleware/errorHandler'
import { jitAuctionService } from '../services/jitAuction'
import { authMiddleware } from '../middleware/auth'
import { rateLimitMiddleware } from '../middleware/rateLimit'

const router = express.Router()

// Apply auth and basic rate limiting to liquidity endpoints
router.use(authMiddleware)
router.use(rateLimitMiddleware({ windowMs: 60 * 1000, max: 120 }))

// Create a JIT auction
router.post('/auctions', asyncHandler(async (req: Request, res: Response) => {
  const { symbol, side, size, durationMs, maxSlippageBps } = req.body || {}
  if (!symbol || !side || !size) {
    return res.status(400).json({ error: 'symbol, side, size are required' })
  }
  const auction = await jitAuctionService.createAuction({ symbol, side, size, durationMs, maxSlippageBps })
  res.json({ success: true, auction })
}))

// Submit maker quote
router.post('/auctions/:id/quotes', asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params
  const { makerId, price } = req.body || {}
  if (!makerId || typeof price !== 'number') {
    return res.status(400).json({ error: 'makerId and price are required' })
  }
  const auction = jitAuctionService.submitQuote(id, makerId, price)
  res.json({ success: true, auction })
}))

// Settle auction
router.post('/auctions/:id/settle', asyncHandler(async (req: Request, res: Response) => {
  const { id } = req.params
  const result = jitAuctionService.settle(id)
  res.json({ success: true, result })
}))

export default router


