#!/usr/bin/env ts-node
import fs from 'fs';
import path from 'path';
import os from 'os';
import { performance } from 'perf_hooks';
import { generateOrders, SyntheticOrder } from './generateOrders';
import { simulatePrices, Scenario } from './priceSimulator';

interface Args {
  orders: number;
  scenario: Scenario;
  duration: number; // ms
  startPrice: number;
}

function parseArgs(): Args {
  const a = process.argv.slice(2);
  const args: any = {};
  for (let i = 0; i < a.length; i += 2) {
    const k = a[i]?.replace(/^--/, '');
    const v = a[i + 1];
    if (!k) continue;
    args[k] = v;
  }
  return {
    orders: parseInt(args.orders || '10000', 10),
    scenario: (args.scenario || 'trend') as Scenario,
    duration: parseInt(args.duration || '60000', 10), // default 60s
    startPrice: parseFloat(args.startPrice || '100'),
  };
}

function percentile(values: number[], p: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((x, y) => x - y);
  const idx = Math.floor((p / 100) * (sorted.length - 1));
  return sorted[idx];
}

function evaluateStopLoss(order: SyntheticOrder, currentPrice: number): boolean {
  if (order.stop_price == null) return false;
  if (order.side === 'long') return currentPrice <= order.stop_price;
  return currentPrice >= order.stop_price;
}

function evaluateTakeProfit(order: SyntheticOrder, currentPrice: number): boolean {
  if (order.stop_price == null) return false; // using stop_price for target
  if (order.side === 'long') return currentPrice >= order.stop_price;
  return currentPrice <= order.stop_price;
}

function evaluateTrailingStop(order: SyntheticOrder, currentPrice: number): { execute: boolean; newStop?: number } {
  if (order.trailing_distance == null || order.stop_price == null) return { execute: false };
  const trailingPercent = order.trailing_distance;
  if (order.side === 'long') {
    // Move stop up with favorable move
    const maxPrice = Math.max(currentPrice, order.stop_price + (order.stop_price * trailingPercent / 100));
    const newStop = maxPrice - (maxPrice * trailingPercent / 100);
    const execute = currentPrice <= newStop;
    return { execute, newStop: execute ? undefined : newStop };
  } else {
    const minPrice = Math.min(currentPrice, order.stop_price - (order.stop_price * trailingPercent / 100));
    const newStop = minPrice + (minPrice * trailingPercent / 100);
    const execute = currentPrice >= newStop;
    return { execute, newStop: execute ? undefined : newStop };
  }
}

async function main() {
  const { orders: orderCount, scenario, duration, startPrice } = parseArgs();

  // Generate orders
  const orders = generateOrders(orderCount, 'market-btc');

  // Simulate prices for duration
  const ticks = simulatePrices(startPrice, duration, scenario);

  // Loop cadence target
  const LOOP_MS = 1000;
  const cadenceDurations: number[] = [];
  const triggerLatencies: number[] = [];

  const processed = new Set<string>();
  let nextLoop = performance.now();
  const loopStart = nextLoop;

  // Execution start times per order for latency measurement
  const orderStart: Record<string, number> = {};
  for (const o of orders) orderStart[o.id] = loopStart;

  // Maintain mutable stop prices for trailing stops
  const trailingStopMap: Record<string, number | undefined> = {};
  for (const o of orders) trailingStopMap[o.id] = o.stop_price;

  for (let tIndex = 0; tIndex < ticks.length; tIndex++) {
    const now = performance.now();
    const drift = Math.max(0, nextLoop - now);
    if (drift > 0) await new Promise((r) => setTimeout(r, drift));
    const loopRealStart = performance.now();

    const price = ticks[tIndex].price;

    // Evaluate orders
    for (let i = 0; i < orders.length; i++) {
      const o = orders[i];
      if (processed.has(o.id)) continue;
      let shouldExecute = false;

      if (o.order_type === 'stop_loss') {
        shouldExecute = evaluateStopLoss(o, price);
      } else if (o.order_type === 'take_profit') {
        shouldExecute = evaluateTakeProfit(o, price);
      } else if (o.order_type === 'trailing_stop') {
        const stop = trailingStopMap[o.id];
        const tmp: SyntheticOrder = { ...o, stop_price: stop };
        const { execute, newStop } = evaluateTrailingStop(tmp, price);
        shouldExecute = execute;
        if (!execute && newStop != null) trailingStopMap[o.id] = newStop;
      }

      if (shouldExecute) {
        processed.add(o.id);
        const latency = loopRealStart - (orderStart[o.id] || loopStart);
        triggerLatencies.push(latency);
      }
    }

    const loopElapsed = performance.now() - loopRealStart;
    const loopCadence = performance.now() - nextLoop;
    cadenceDurations.push(loopCadence);
    nextLoop += LOOP_MS;

    // Early exit if all orders processed
    if (processed.size >= orders.length) break;
  }

  // Resource usage (simple snapshot)
  const mem = process.memoryUsage();
  const cpuCount = os.cpus().length;

  const results = {
    scenario,
    orders: orderCount,
    durationMs: duration,
    processed: processed.size,
    cadence: {
      p95: Math.round(percentile(cadenceDurations, 95)),
      p99: Math.round(percentile(cadenceDurations, 99)),
      avg: Math.round(cadenceDurations.reduce((a, b) => a + b, 0) / Math.max(1, cadenceDurations.length)),
      samples: cadenceDurations.length,
    },
    triggerLatency: {
      p95: Math.round(percentile(triggerLatencies, 95)),
      p99: Math.round(percentile(triggerLatencies, 99)),
      avg: Math.round(triggerLatencies.reduce((a, b) => a + b, 0) / Math.max(1, triggerLatencies.length)),
      samples: triggerLatencies.length,
    },
    resources: {
      rss: mem.rss,
      heapUsed: mem.heapUsed,
      cpuCount,
    },
    timestamp: new Date().toISOString(),
  };

  const outDir = path.resolve(process.cwd(), 'backend/scripts/loadtest/monitor/output');
  fs.mkdirSync(outDir, { recursive: true });
  const file = path.join(outDir, `results-${Date.now()}.json`);
  fs.writeFileSync(file, JSON.stringify(results, null, 2));
  
  console.log('\n═══════════════════════════════════════════════════════');
  console.log('✅ LOAD TEST COMPLETE');
  console.log('═══════════════════════════════════════════════════════\n');
  console.log(`Scenario:        ${scenario}`);
  console.log(`Orders:          ${orderCount}`);
  console.log(`Duration:        ${duration}ms (${(duration / 1000).toFixed(1)}s)`);
  console.log(`Processed:       ${processed.size} orders triggered\n`);
  
  console.log('📊 CADENCE METRICS:');
  console.log(`  p95:           ${results.cadence.p95}ms`);
  console.log(`  p99:           ${results.cadence.p99}ms`);
  console.log(`  Average:       ${results.cadence.avg}ms`);
  console.log(`  Samples:       ${results.cadence.samples}\n`);
  
  console.log('⚡ TRIGGER LATENCY:');
  console.log(`  p95:           ${results.triggerLatency.p95}ms`);
  console.log(`  p99:           ${results.triggerLatency.p99}ms`);
  console.log(`  Average:       ${results.triggerLatency.avg}ms`);
  console.log(`  Triggers:      ${results.triggerLatency.samples}\n`);
  
  console.log('💻 RESOURCE USAGE:');
  console.log(`  RSS Memory:    ${(results.resources.rss / 1024 / 1024).toFixed(2)} MB`);
  console.log(`  Heap Used:     ${(results.resources.heapUsed / 1024 / 1024).toFixed(2)} MB`);
  console.log(`  CPU Cores:     ${results.resources.cpuCount}\n`);
  
  console.log(`📁 Results saved to: ${file}\n`);
  console.log('═══════════════════════════════════════════════════════\n');
}

main().catch((e) => {
  console.error('Load test failed:', e);
  process.exit(1);
});
