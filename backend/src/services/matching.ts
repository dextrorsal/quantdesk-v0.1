import { DatabaseService } from './database'
import { Logger } from '../utils/logger'
import { pythOracleService } from './pythOracleService'
import { WebSocketService } from './websocket'

export type OrderSide = 'buy' | 'sell'
export type PerpSide = 'long' | 'short'
export type OrderType = 'market' | 'limit'

export interface PlaceOrderInput {
  userId: string
  symbol: string // e.g. BTC-PERP
  side: OrderSide
  size: number
  orderType: OrderType
  price?: number
  leverage?: number
}

export class MatchingService {
  private static instance: MatchingService
  private readonly db: DatabaseService
  private readonly logger: Logger

  private constructor() {
    this.db = DatabaseService.getInstance()
    this.logger = new Logger()
  }

  public static getInstance(): MatchingService {
    if (!MatchingService.instance) {
      MatchingService.instance = new MatchingService()
    }
    return MatchingService.instance
  }

  public async placeOrder(input: PlaceOrderInput): Promise<{ orderId: string; filled: boolean; fills: Array<{ price: number; size: number }>; averageFillPrice?: number }>{
    const { userId, symbol, side, size, orderType } = input
    if (size <= 0) throw new Error('Size must be positive')
    if (orderType === 'limit' && (input.price == null || input.price <= 0)) throw new Error('Limit orders require positive price')

    // Resolve market
    const marketRes = await this.db.query('SELECT id FROM markets WHERE symbol = $1', [symbol])
    if (!marketRes.rows[0]) throw new Error('Market not found')
    const marketId = marketRes.rows[0].id as string

    // Create order row (pending by default)
    const orderTypeEnum = orderType === 'market' ? 'market' : 'limit'
    const perpSide: PerpSide = side === 'buy' ? 'long' : 'short'
    const price = input.price ?? null
    const leverage = input.leverage ?? 1

    const orderInsert = await this.db.query(
      `INSERT INTO orders (user_id, market_id, order_account, order_type, side, size, price, leverage, status, created_at)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'pending', NOW()) RETURNING id` ,
      [userId, marketId, 'OFFCHAIN', orderTypeEnum, perpSide, size, price, leverage]
    )
    const orderId = orderInsert.rows[0].id as string

    // Matching logic
    const fills: Array<{ price: number; size: number }> = []
    let remaining = size

    // Opposite side in DB schema uses 'long'|'short'
    const opposite = perpSide === 'long' ? 'short' : 'long'

    // Try match against resting orders
    const priceCond = orderType === 'limit'
      ? (side === 'buy' ? 'AND price <= $4' : 'AND price >= $4')
      : ''

    const bookRes = await this.db.query(
      `SELECT id, price, remaining_size
       FROM orders
       WHERE market_id = $1 AND status = 'pending' AND side = $2 ${priceCond}
       ORDER BY price ${side === 'buy' ? 'ASC' : 'DESC'}, created_at ASC
       LIMIT 100`,
      orderType === 'limit' ? [marketId, opposite, input.price] : [marketId, opposite]
    ).catch(() => ({ rows: [] }))

    for (const row of bookRes.rows || []) {
      if (remaining <= 0) break
      const levelSize = parseFloat(row.remaining_size)
      const take = Math.min(remaining, levelSize)
      const levelPrice = parseFloat(row.price)
      fills.push({ price: levelPrice, size: take })
      remaining -= take

      // Update resting order filled size
      await this.db.query(
        `UPDATE orders SET filled_size = COALESCE(filled_size,0) + $1,
                status = CASE WHEN remaining_size - $1 <= 0 THEN 'filled' ELSE status END,
                updated_at = NOW(), filled_at = CASE WHEN remaining_size - $1 <= 0 THEN NOW() ELSE filled_at END
         WHERE id = $2`,
        [take, row.id]
      ).catch(() => {})
    }

    // If no liquidity fully filled the order, use oracle price for the remainder if market order
    if (remaining > 0 && orderType === 'market') {
      const oraclePrice = await pythOracleService.getLatestPrice(symbol)
      if (oraclePrice == null) throw new Error('Price unavailable')
      fills.push({ price: oraclePrice, size: remaining })
      remaining = 0
    }

    const filled = remaining === 0
    const totalFilled = fills.reduce((s, f) => s + f.size, 0)
    const avg = totalFilled > 0 ? (fills.reduce((s, f) => s + f.price * f.size, 0) / totalFilled) : undefined

    // If not fully filled and limit order, keep pending; else mark filled
    if (filled) {
      await this.db.query(
        `UPDATE orders SET status = 'filled', filled_size = $1, average_fill_price = $2, filled_at = NOW(), updated_at = NOW() WHERE id = $3`,
        [totalFilled, avg ?? null, orderId]
      )
    } else {
      // leave as pending with partial fill
      await this.db.query(
        `UPDATE orders SET status = 'partially_filled', filled_size = $1, average_fill_price = $2, updated_at = NOW() WHERE id = $3`,
        [totalFilled, avg ?? null, orderId]
      )
    }

    // Write trade rows and update position for the taker (user)
    for (const f of fills) {
      await this.db.query(
        `INSERT INTO trades (user_id, market_id, position_id, order_id, trade_account, side, size, price, fees, created_at)
         VALUES ($1,$2,NULL,$3,'OFFCHAIN',$4,$5,$6,0,NOW())`,
        [userId, marketId, orderId, side === 'buy' ? 'buy' : 'sell', f.size, f.price]
      )
      await this.updatePositionOnFill({ userId, marketId, side: perpSide, price: f.price, size: f.size })
      WebSocketService.current?.broadcast?.('trade', { symbol, price: f.price, size: f.size })
    }

    WebSocketService.current?.broadcast?.('order', { symbol, orderId, status: filled ? 'filled' : 'partially_filled' })
    return { orderId, filled, fills, averageFillPrice: avg }
  }

  private async updatePositionOnFill(params: { userId: string; marketId: string; side: PerpSide; size: number; price: number }): Promise<void> {
    const { userId, marketId, side, size, price } = params

    // Fetch existing open position
    const posRes = await this.db.query(
      `SELECT * FROM positions WHERE user_id = $1 AND market_id = $2 AND NOT is_liquidated ORDER BY created_at DESC LIMIT 1`,
      [userId, marketId]
    )
    const pos = posRes.rows[0]

    if (!pos) {
      // Open new position
      await this.db.query(
        `INSERT INTO positions (user_id, market_id, position_account, side, size, entry_price, current_price, margin, leverage, unrealized_pnl, created_at)
         VALUES ($1,$2,'OFFCHAIN',$3,$4,$5,$5, $6, $7, 0, NOW())`,
        [userId, marketId, side, size, price, Math.max(1, size * price * 0.1), 1]
      )
      return
    }

    // Same side: increase size and adjust entry price (VWAP)
    if (pos.side === side) {
      const oldSize = parseFloat(pos.size)
      const newSize = oldSize + size
      const newEntry = (oldSize * parseFloat(pos.entry_price) + size * price) / newSize
      await this.db.query(
        `UPDATE positions SET size = $1, entry_price = $2, current_price = $3, updated_at = NOW() WHERE id = $4`,
        [newSize, newEntry, price, pos.id]
      )
      await this.updateHealthFactor(pos.id)
      return
    }

    // Opposite side: reduce/close
    const oldSize = parseFloat(pos.size)
    if (size < oldSize) {
      const remaining = oldSize - size
      await this.db.query(
        `UPDATE positions SET size = $1, current_price = $2, updated_at = NOW() WHERE id = $3`,
        [remaining, price, pos.id]
      )
      await this.updateHealthFactor(pos.id)
    } else {
      // Close and realize PnL (simplified)
      await this.db.query(
        `UPDATE positions SET size = 0, current_price = $1, realized_pnl = COALESCE(realized_pnl,0) + ($1 - entry_price) * CASE WHEN side = 'long' THEN size ELSE -size END, updated_at = NOW(), closed_at = NOW() WHERE id = $2`,
        [price, pos.id]
      )
    }
  }

  private async updateHealthFactor(positionId: string): Promise<void> {
    try {
      const res = await this.db.query(`SELECT calculate_position_health($1) AS hf`, [positionId])
      const hf = res.rows[0]?.hf ?? null
      if (hf != null) {
        await this.db.query(`UPDATE positions SET health_factor = $1 WHERE id = $2`, [hf, positionId])
        WebSocketService.current?.broadcast?.('position', { positionId, healthFactor: hf })
      }
    } catch (_e) {}
  }
}

export const matchingService = MatchingService.getInstance()


