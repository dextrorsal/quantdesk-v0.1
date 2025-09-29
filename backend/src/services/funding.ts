import { DatabaseService } from './database'
import { Logger } from '../utils/logger'
import { pythOracleService } from './pythOracleService'

export class FundingService {
  private static instance: FundingService
  private readonly db: DatabaseService
  private readonly logger: Logger
  private timer: NodeJS.Timeout | null = null

  private constructor() {
    this.db = DatabaseService.getInstance()
    this.logger = new Logger()
  }

  public static getInstance(): FundingService {
    if (!FundingService.instance) {
      FundingService.instance = new FundingService()
    }
    return FundingService.instance
  }

  public start(): void {
    if (this.timer) return
    this.logger.info('Starting funding scheduler')
    this.timer = setInterval(() => {
      this.run().catch(err => this.logger.error('Funding run error:', err))
    }, 60 * 60 * 1000) // hourly
    this.run().catch(() => {})
  }

  public stop(): void {
    if (this.timer) clearInterval(this.timer)
    this.timer = null
  }

  private async run(): Promise<void> {
    const markets = await this.db.query('SELECT id, symbol FROM markets WHERE is_active = true')
    for (const m of markets.rows) {
      try {
        const price = await pythOracleService.getLatestPrice(m.symbol)
        if (price == null) continue
        // Placeholder funding calculation
        const fundingRate = 0
        const premiumIndex = 0
        await this.db.query(
          `INSERT INTO funding_rates (market_id, funding_rate, premium_index, oracle_price, mark_price, total_funding, created_at)
           VALUES ($1,$2,$3,$4,$5,0,NOW())`,
          [m.id, fundingRate, premiumIndex, price, price]
        )
      } catch (e) {
        this.logger.warn(`Funding write skipped for ${m.symbol}`)
      }
    }
  }
}

export const fundingService = FundingService.getInstance()


