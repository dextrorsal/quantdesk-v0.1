import { SupabaseDatabaseService } from './supabaseDatabase'
import { Logger } from '../utils/logger'
import { pythOracleService } from './pythOracleService'

export class FundingService {
  private static instance: FundingService
  private readonly db: SupabaseDatabaseService
  private readonly logger: Logger
  private timer: NodeJS.Timeout | null = null

  private constructor() {
    this.db = SupabaseDatabaseService.getInstance()
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
    // Use fluent API instead of query() for security
    const markets = await this.db.select('markets', 'id, symbol', { is_active: true });
    
    for (const market of markets) {
      try {
        const price = await pythOracleService.getLatestPrice(market.symbol);
        if (price == null) continue;
        
        // Placeholder funding calculation
        const fundingRate = 0;
        const premiumIndex = 0;
        
        // Use fluent API instead of query() for security
        await this.db.insert('funding_rates', {
          market_id: market.id,
          funding_rate: fundingRate,
          premium_index: premiumIndex,
          oracle_price: price,
          mark_price: price,
          total_funding: 0,
          created_at: new Date().toISOString()
        });
      } catch (e) {
        this.logger.warn(`Funding write skipped for ${market.symbol}`);
      }
    }
  }
}

export const fundingService = FundingService.getInstance()


