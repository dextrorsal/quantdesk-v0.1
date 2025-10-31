/*
  Monitor Metrics Sampler
  - Samples /api/metrics/conditional-orders every 5s for a duration
  - Aggregates basic stats and writes a markdown report under bmad/docs/reports/

  Usage:
    pnpm ts-node backend/scripts/monitor-metrics-sampler.ts

  Env (optional):
    METRICS_URL=http://localhost:3002/api/metrics/conditional-orders
    SAMPLE_INTERVAL_MS=5000
    DURATION_SEC=900
*/

import 'dotenv/config';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

const METRICS_URL = process.env.METRICS_URL || 'http://localhost:3002/api/metrics/conditional-orders';
const SAMPLE_INTERVAL_MS = Number(process.env.SAMPLE_INTERVAL_MS || 5000);
const DURATION_SEC = Number(process.env.DURATION_SEC || 900);

type Metrics = {
  processedOrders: number;
  triggersFired: number;
  errors: number;
  executionLatencyP95: number;
  processedOrdersPerSec: number;
  triggersFiredPerSec: number;
  errorsPerSec: number;
  circuitBreakerActive?: boolean;
  circuitBreakerReason?: string | null;
};

function percentile(arr: number[], p: number): number {
  if (!arr.length) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.floor(sorted.length * p);
  return sorted[Math.min(idx, sorted.length - 1)];
}

async function sampleOnce(): Promise<Metrics | null> {
  try {
    const res = await fetch(METRICS_URL);
    const json = await res.json() as any;
    if (!json.success) return null;
    return json.data as Metrics;
  } catch {
    return null;
  }
}

async function run(): Promise<void> {
  const samples: Metrics[] = [];
  const start = Date.now();
  const end = start + DURATION_SEC * 1000;
  while (Date.now() < end) {
    const m = await sampleOnce();
    if (m) samples.push(m);
    await new Promise(r => setTimeout(r, SAMPLE_INTERVAL_MS));
  }

  const p95s = samples.map(s => s.executionLatencyP95 || 0);
  const p50 = percentile(p95s, 0.5);
  const p95 = percentile(p95s, 0.95);
  const last = samples.at(-1);

  const processed = last?.processedOrders ?? 0;
  const triggers = last?.triggersFired ?? 0;
  const errors = last?.errors ?? 0;
  const breaker = last?.circuitBreakerActive ? `ACTIVE (${last?.circuitBreakerReason || ''})` : 'inactive';

  const report = `# Conditional Order Monitor - Load Test Report

Duration: ${DURATION_SEC}s  
Samples: ${samples.length} (interval ${SAMPLE_INTERVAL_MS} ms)

## Summary

- Processed Orders (cumulative): ${processed}
- Triggers Fired (cumulative): ${triggers}
- Errors (cumulative): ${errors}
- Execution Latency P50 (from p95 metric): ${p50} ms
- Execution Latency P95 (from p95 metric): ${p95} ms
- Circuit Breaker: ${breaker}

## Raw Samples (last 10)

${samples.slice(-10).map(s => `- p95=${s.executionLatencyP95}ms, processed=${s.processedOrders}, triggers=${s.triggersFired}, errors=${s.errors}, breaker=${s.circuitBreakerActive ? 'on' : 'off'}`).join('\n')}
`;

  const outDir = join(process.cwd(), '..', 'bmad', 'docs', 'reports');
  mkdirSync(outDir, { recursive: true });
  const dateStr = new Date().toISOString().slice(0,10);
  const outPath = join(outDir, `monitor-loadtest-${dateStr}.md`);
  writeFileSync(outPath, report, 'utf8');
  // eslint-disable-next-line no-console
  console.log(`Report written: ${outPath}`);
}

run().catch((e) => {
  // eslint-disable-next-line no-console
  console.error('metrics sampler failed:', e);
  process.exit(1);
});


