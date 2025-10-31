export type Scenario = 'trend' | 'spike' | 'gap';

export interface PriceTick {
  t: number; // ms
  price: number;
}

export function simulatePrices(startPrice: number, durationMs: number, scenario: Scenario): PriceTick[] {
  const ticks: PriceTick[] = [];
  const step = 200; // 200ms per tick
  let price = startPrice;
  for (let t = 0; t <= durationMs; t += step) {
    if (scenario === 'trend') {
      price *= 1 + (Math.sin(t / 5000) * 0.0005);
    } else if (scenario === 'spike') {
      if (t > durationMs / 2 && t < durationMs / 2 + 1000) {
        price *= 1.05; // 5% spike
      } else {
        price *= 1 + (Math.random() - 0.5) * 0.0008;
      }
    } else if (scenario === 'gap') {
      if (t === Math.floor(durationMs / 3)) price *= 0.95; // 5% gap down
      if (t === Math.floor((2 * durationMs) / 3)) price *= 1.06; // 6% gap up
      price *= 1 + (Math.random() - 0.5) * 0.0006;
    }
    ticks.push({ t, price });
  }
  return ticks;
}

if (require.main === module) {
  const data = simulatePrices(100, 10000, 'spike');
  console.log('Simulated ticks:', data.length, 'first=', data[0], 'last=', data[data.length - 1]);
}
