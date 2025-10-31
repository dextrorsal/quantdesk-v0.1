/*
  Load Test - Conditional Orders Seeder
  - Seeds pending conditional orders across available markets (BTC/ETH/SOL if present)
  - Uses SupabaseDatabaseService (no direct Supabase client usage outside the service)
  - Prices sourced via pythOracleService to place realistic triggers
  - Adds metadata.test_run_id to enable safe cleanup
  
  Usage:
    pnpm ts-node backend/scripts/loadtest-conditional-orders.ts seed
    pnpm ts-node backend/scripts/loadtest-conditional-orders.ts cleanup <test_run_id>

  Env (optional):
    NUM_ORDERS=1000         Total orders (split across symbols)
    ORDER_TYPES=stop,take,trailing   Comma-separated
*/

import 'dotenv/config';
import { randomUUID } from 'crypto';
import { SupabaseDatabaseService } from '../src/services/supabaseDatabase';
import { pythOracleService } from '../src/services/pythOracleService';
import { Logger } from '../src/utils/logger';

const logger = new Logger();

type OrderType = 'stop_loss' | 'take_profit' | 'trailing_stop';

async function getMarketIdMap(db: SupabaseDatabaseService): Promise<Record<string, string>> {
  const { data, error } = await db.getClient()
    .from('markets')
    .select('id,symbol');
  if (error) throw error;
  const map: Record<string, string> = {};
  (data || []).forEach((m: any) => { map[m.symbol] = m.id; });
  return map;
}

function pick<T>(arr: T[]): T { return arr[Math.floor(Math.random() * arr.length)]; }

function getOrderTypePool(): OrderType[] {
  const raw = (process.env.ORDER_TYPES || 'stop,take,trailing').split(',').map(s => s.trim());
  const pool: OrderType[] = [];
  for (const t of raw) {
    if (t === 'stop') pool.push('stop_loss');
    else if (t === 'take') pool.push('take_profit');
    else if (t === 'trailing') pool.push('trailing_stop');
  }
  return pool.length ? pool : ['stop_loss','take_profit','trailing_stop'];
}

async function seed(): Promise<void> {
  const db = SupabaseDatabaseService.getInstance();
  const testRunId = randomUUID();
  const total = Number(process.env.NUM_ORDERS || 1000);
  const orderTypes = getOrderTypePool();

  logger.info(`Seeding conditional orders: total=${total}, test_run_id=${testRunId}`);

  const marketIdMap = await getMarketIdMap(db);
  const prices = await pythOracleService.getAllPrices();
  const symbols = Object.keys(prices).filter(s => ['BTC','ETH','SOL'].includes(s));
  if (symbols.length === 0) {
    throw new Error('No BTC/ETH/SOL prices available; cannot seed realistic triggers.');
  }

  const perSymbol = Math.max(1, Math.floor(total / symbols.length));
  const inserts: any[] = [];

  for (const sym of symbols) {
    const marketId = marketIdMap[`${sym}-PERP`] || marketIdMap[`${sym}/USDT`] || marketIdMap[sym] || null;
    if (!marketId) {
      logger.warn(`Skipping ${sym}: market not found in DB`);
      continue;
    }
    const basePrice = prices[sym];
    for (let i = 0; i < perSymbol; i++) {
      const side = Math.random() < 0.5 ? 'long' : 'short';
      const type = pick(orderTypes);
      const leverage = 1 + Math.floor(Math.random() * 5);
      const size = Number((Math.random() * 0.5 + 0.1).toFixed(4));
      // Place triggers within ±2% window around current price
      const pct = (Math.random() * 0.02);
      const directionFactor = side === 'long' ? -1 : 1; // stop for long below, for short above
      const triggerPrice = Number((basePrice * (1 + directionFactor * pct)).toFixed(6));
      const trailingPct = Number((1 + Math.random() * 2.5).toFixed(2)); // 1% - 3.5%

      const row: any = {
        user_id: '00000000-0000-0000-0000-000000000000',
        market_id: marketId,
        order_type: type,
        side,
        size,
        status: 'pending',
        leverage,
        metadata: { test_run_id: testRunId },
      };
      if (type === 'trailing_stop') {
        row.trailing_distance = trailingPct;
        row.stop_price = Number((basePrice * (side === 'long'
          ? (1 - trailingPct / 100)
          : (1 + trailingPct / 100))).toFixed(6));
      } else {
        row.stop_price = triggerPrice;
      }
      inserts.push(row);
    }
  }

  if (inserts.length === 0) {
    logger.warn('No rows generated. Exiting.');
    return;
  }

  // Batch insert to reduce round trips
  const chunkSize = 500;
  for (let i = 0; i < inserts.length; i += chunkSize) {
    const chunk = inserts.slice(i, i + chunkSize);
    const { error } = await db.getClient().from('orders').insert(chunk);
    if (error) throw error;
    logger.info(`Inserted ${Math.min(i + chunkSize, inserts.length)}/${inserts.length}`);
  }

  logger.info(`Seeding complete. test_run_id=${testRunId}`);
  // Output test_run_id so caller can clean up later
  console.log(testRunId);
}

async function cleanup(testRunId: string): Promise<void> {
  const db = SupabaseDatabaseService.getInstance();
  logger.info(`Cleaning up orders with test_run_id=${testRunId}`);
  const { error } = await db.getClient()
    .from('orders')
    .delete()
    .contains('metadata', { test_run_id: testRunId });
  if (error) throw error;
  logger.info('Cleanup complete.');
}

async function main() {
  const [cmd, arg] = process.argv.slice(2);
  if (cmd === 'seed') {
    await seed();
  } else if (cmd === 'cleanup') {
    if (!arg) throw new Error('Usage: cleanup <test_run_id>');
    await cleanup(arg);
  } else {
    console.log('Usage: ts-node backend/scripts/loadtest-conditional-orders.ts <seed|cleanup <id>>');
  }
}

main().catch((e) => {
  logger.error('Loadtest script error:', e);
  process.exit(1);
});


