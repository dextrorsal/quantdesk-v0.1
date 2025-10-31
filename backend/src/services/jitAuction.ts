import { v4 as uuidv4 } from 'uuid'
import { Logger } from '../utils/logger'
import { pythOracleService } from './pythOracleService'
import { SupabaseDatabaseService } from './supabaseDatabase'
import { WebSocketService } from './websocket'

export type AuctionSide = 'buy' | 'sell'

export interface Auction {
  id: string
  symbol: string
  side: AuctionSide
  size: number
  referencePrice: number
  maxSlippageBps: number
  createdAt: number
  expiresAt: number
  quotes: Quote[]
  settled: boolean
}

export interface Quote {
  makerId: string
  price: number
  timestamp: number
}

export interface SettlementResult {
  filled: boolean
  fillPrice?: number
  makerId?: string
  reason?: string
}

/**
 * Minimal in-memory JIT auction service inspired by Drift's short auction window.
 * This is stateless across restarts and intended for prototype/testing only.
 */
export class JitAuctionService {
  private static instance: JitAuctionService
  private readonly logger: Logger
  private readonly auctions: Map<string, Auction>
  private timer: NodeJS.Timeout | null = null
  private readonly db: SupabaseDatabaseService
  private ws?: WebSocketService

  private constructor() {
    this.logger = new Logger()
    this.auctions = new Map()
    this.startKeeper()
    this.db = SupabaseDatabaseService.getInstance()
  }

  public static getInstance(): JitAuctionService {
    if (!JitAuctionService.instance) {
      JitAuctionService.instance = new JitAuctionService()
    }
    return JitAuctionService.instance
  }

  public attachWebSocket(ws: WebSocketService): void {
    this.ws = ws
  }

  // Keeper scans and auto-settles expired auctions every second
  private startKeeper(): void {
    if (this.timer) return
    this.timer = setInterval(() => {
      const now = Date.now()
      for (const [id, auc] of this.auctions.entries()) {
        if (!auc.settled && now > auc.expiresAt) {
          this.settle(id)
        }
      }
    }, 1000)
  }

  public getAuction(id: string): Auction | undefined {
    return this.auctions.get(id)
  }

  public async createAuction(params: {
    symbol: string
    side: AuctionSide
    size: number
    durationMs?: number
    maxSlippageBps?: number
  }): Promise<Auction> {
    const { symbol, side, size } = params
    const durationMs = params.durationMs ?? 5_000
    const maxSlippageBps = params.maxSlippageBps ?? 50 // 0.50%

    // Use latest oracle price, fallback handled inside service
    const ref = await pythOracleService.getLatestPrice(symbol)
    if (ref == null) {
      throw new Error('Reference price unavailable')
    }

    const id = uuidv4()
    const now = Date.now()
    const auction: Auction = {
      id,
      symbol,
      side,
      size,
      referencePrice: ref,
      maxSlippageBps,
      createdAt: now,
      expiresAt: now + durationMs,
      quotes: [],
      settled: false
    }

    this.auctions.set(id, auction)
    // persist
    try {
      await this.db.insert('auctions', {
        id: id,
        symbol: symbol,
        side: side,
        size: size,
        reference_price: ref,
        max_slippage_bps: maxSlippageBps,
        created_at: new Date(now).toISOString(),
        expires_at: new Date(now + durationMs).toISOString(),
        settled: false
      });
    } catch (_e) {}
    // ws event
    this.ws?.broadcast?.('auction_created', { id, symbol, side, size, referencePrice: ref, expiresAt: auction.expiresAt })
    this.logger.info(`Created JIT auction ${id} ${side} ${size} ${symbol} @ref ${ref}`)
    return auction
  }

  public async submitQuote(auctionId: string, makerId: string, price: number): Promise<Auction> {
    const auc = this.auctions.get(auctionId)
    if (!auc) throw new Error('Auction not found')
    if (auc.settled) throw new Error('Auction already settled')
    if (Date.now() > auc.expiresAt) throw new Error('Auction expired')

    const quote = { makerId, price, timestamp: Date.now() }
    auc.quotes.push(quote)
    try {
      await this.db.insert('auction_quotes', {
        auction_id: auctionId,
        maker_id: makerId,
        price: price,
        created_at: new Date().toISOString()
      });
    } catch (_e) {}
    this.ws?.broadcast?.('auction_quote', { auctionId, ...quote })
    return auc
  }

  public async settle(auctionId: string): Promise<SettlementResult> {
    const auc = this.auctions.get(auctionId)
    if (!auc) return { filled: false, reason: 'Auction not found' }
    if (auc.settled) return { filled: false, reason: 'Already settled' }

    // Filter quotes by slippage vs reference
    const maxDeviation = (auc.referencePrice * auc.maxSlippageBps) / 10_000
    const minAcceptable = auc.side === 'buy' ? 0 : auc.referencePrice - maxDeviation
    const maxAcceptable = auc.side === 'buy' ? auc.referencePrice + maxDeviation : Infinity

    const eligible = auc.quotes.filter(q => q.price >= minAcceptable && q.price <= maxAcceptable)
    if (eligible.length === 0) {
      auc.settled = true
      return { filled: false, reason: 'No eligible quotes' }
    }

    // Best quote: buyer wants lowest price; seller wants highest price
    const best = eligible.reduce<Quote | null>((bestSoFar, q) => {
      if (!bestSoFar) return q
      if (auc.side === 'buy') return q.price < bestSoFar.price ? q : bestSoFar
      return q.price > bestSoFar.price ? q : bestSoFar
    }, null)
    if (!best) {
      auc.settled = true
      return { filled: false, reason: 'No quotes' }
    }

    auc.settled = true
    try {
      await this.db.update('auctions', { settled: true }, { id: auctionId });
      await this.db.insert('auction_settlements', {
        auction_id: auctionId,
        maker_id: best.makerId,
        fill_price: best.price,
        reason: null
      });
    } catch (_e) {}
    this.logger.info(`Settled JIT auction ${auctionId} with maker ${best.makerId} @ ${best.price}`)
    this.ws?.broadcast?.('auction_settled', { auctionId, makerId: best.makerId, fillPrice: best.price })
    return { filled: true, fillPrice: best.price, makerId: best.makerId }
  }
}

export const jitAuctionService = JitAuctionService.getInstance()


