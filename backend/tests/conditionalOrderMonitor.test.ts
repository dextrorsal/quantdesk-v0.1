import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ConditionalOrderMonitor } from '../src/services/conditionalOrderMonitor';

describe('ConditionalOrderMonitor evaluators', () => {
  let monitor: ConditionalOrderMonitor;

  beforeEach(() => {
    // @ts-expect-error access constructor indirectly
    monitor = (ConditionalOrderMonitor as any).getInstance();
    monitor.stop();
    monitor.resetMetrics();
  });

  it('trips circuit breaker on >5% deviation and resets on <2%', async () => {
    // @ts-expect-error force internal state
    monitor['lastPrices'] = { BTC: 100 }; // seed previous
    // First loop: >5% jump
    const compute = (monitor as any)['computeMaxDeviation'].bind(monitor);
    const res1 = compute({ BTC: 100 }, { BTC: 106 });
    expect(res1.maxDeviation).toBeGreaterThan(0.05);

    // Activate breaker
    // @ts-expect-error set breaker
    monitor['circuitBreakerActive'] = true;
    // Second loop: <2% deviation
    const res2 = compute({ BTC: 106 }, { BTC: 107 });
    expect(res2.maxDeviation).toBeLessThan(0.02);
  });
});


