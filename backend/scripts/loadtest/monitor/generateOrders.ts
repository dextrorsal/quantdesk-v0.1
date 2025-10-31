import fs from 'fs';
import path from 'path';

type OrderType = 'stop_loss' | 'take_profit' | 'trailing_stop';

export interface SyntheticOrder {
  id: string;
  user_id: string;
  market_id: string;
  order_type: OrderType;
  side: 'long' | 'short';
  size: number;
  stop_price?: number;
  trailing_distance?: number; // percent
  leverage: number;
}

export function generateOrders(count: number, marketId: string): SyntheticOrder[] {
  const orders: SyntheticOrder[] = [];
  for (let i = 0; i < count; i++) {
    const t = (i % 3) as 0 | 1 | 2;
    const type: OrderType = t === 0 ? 'stop_loss' : t === 1 ? 'take_profit' : 'trailing_stop';
    const side: 'long' | 'short' = i % 2 === 0 ? 'long' : 'short';
    const base = 100 + (i % 500); // base price cluster
    const stop = type === 'trailing_stop' ? undefined : base * (side === 'long' ? 0.98 : 1.02);
    const trailing = type === 'trailing_stop' ? 1 + (i % 4) : undefined; // 1-4%

    orders.push({
      id: `syn-${i}`,
      user_id: `user-${i % 1000}`,
      market_id: marketId,
      order_type: type,
      side,
      size: 0.01 + (i % 5) * 0.01,
      stop_price: stop,
      trailing_distance: trailing,
      leverage: 3,
    });
  }
  return orders;
}

if (require.main === module) {
  const outDir = path.resolve(process.cwd(), 'backend/scripts/loadtest/monitor/output');
  fs.mkdirSync(outDir, { recursive: true });
  const orders = generateOrders(1000, 'market-btc');
  fs.writeFileSync(path.join(outDir, `orders-${Date.now()}.json`), JSON.stringify(orders, null, 2));
  console.log('Generated synthetic orders:', orders.length);
}
