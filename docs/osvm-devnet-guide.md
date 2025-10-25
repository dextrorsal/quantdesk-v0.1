# OSVM Devnet Guide for QuantDesk (Node/TS)

## Ports and Environment
- Frontend: 3001
- Backend: 3002
- OSVM local RPC: 8899
- Optional MCP server: 3010

Export environment:
```bash
export SOLANA_RPC_URL=http://127.0.0.1:8899
export KEYPAIR_PATH="$(solana config get | sed -n 's/^Keypair Path: //p')"
```

## Start Local Devnet (using your existing devnet wallet)
- Your Solana CLI keypair is referenced read-only via `KEYPAIR_PATH`. We do not modify `~/.config/solana`.

Start devnet:
```bash
# If osvm is installed globally
osvm rpc devnet --rpc-port 8899 --background

# Or from this repo (if you cloned osvm-cli)
cargo run --bin osvm -- rpc devnet --rpc-port 8899 --background
```

Health check:
```bash
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' \
  http://127.0.0.1:8899
```

## Backend (3002) proxy → RPC (8899)
Add a thin proxy to avoid browser CORS to 8899.

server/rpcProxy.ts
```ts
import { createProxyMiddleware } from 'http-proxy-middleware';

export const rpcProxy = createProxyMiddleware({
  target: process.env.SOLANA_RPC_URL || 'http://127.0.0.1:8899',
  changeOrigin: false,
  pathRewrite: { '^/rpc': '' },
  timeout: 30000,
});
```

server/index.ts
```ts
import express from 'express';
import cors from 'cors';
import { rpcProxy } from './rpcProxy';

const app = express();
app.use(cors({ origin: ['http://127.0.0.1:3001','http://localhost:3001'] }));
app.use('/rpc', rpcProxy);
app.listen(3002, () => console.log('Backend on 3002'));
```

Frontend calls the backend:
```ts
await fetch('http://127.0.0.1:3002/rpc', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ jsonrpc:'2.0', id:1, method:'getSlot', params:[] }),
});
```

## Node/TS Helpers
src/infra/osvm.ts
```ts
import { Connection, clusterApiUrl } from '@solana/web3.js';

export function getConnection(): Connection {
  const url = process.env.SOLANA_RPC_URL || clusterApiUrl('devnet');
  return new Connection(url, { commitment: 'confirmed' });
}
```

src/infra/keypair.ts
```ts
import { Keypair } from '@solana/web3.js';
import fs from 'fs';

export function loadKeypair(path: string): Keypair {
  const secret = JSON.parse(fs.readFileSync(path, 'utf8')) as number[];
  return Keypair.fromSecretKey(Uint8Array.from(secret));
}
```

## Tests
tests/solana.smoke.test.ts
```ts
import { getConnection } from '../src/infra/osvm';

it('connects and fetches slot', async () => {
  const conn = getConnection();
  const slot = await conn.getSlot('confirmed');
  expect(slot).toBeGreaterThan(0);
});
```

tests/account-init.test.ts
```ts
import { Keypair, SystemProgram, Transaction, Connection } from '@solana/web3.js';
import { getConnection } from '../src/infra/osvm';
import { loadKeypair } from '../src/infra/keypair';

const KEYPAIR_PATH = process.env.KEYPAIR_PATH || '';

describe('Account init', () => {
  let conn: Connection;
  let payer: Keypair;

  beforeAll(() => {
    conn = getConnection();
    payer = loadKeypair(KEYPAIR_PATH);
  });

  it('creates a system account', async () => {
    const newAccount = Keypair.generate();
    const lamports = 10000000;

    const tx = new Transaction().add(
      SystemProgram.createAccount({
        fromPubkey: payer.publicKey,
        newAccountPubkey: newAccount.publicKey,
        lamports,
        space: 0,
        programId: SystemProgram.programId,
      })
    );

    const sig = await conn.sendTransaction(tx, [payer, newAccount], { skipPreflight: false });
    await conn.confirmTransaction(sig, 'confirmed');

    const bal = await conn.getBalance(newAccount.publicKey, 'confirmed');
    expect(bal).toBeGreaterThan(0);
  });
});
```

Run tests:
```bash
export SOLANA_RPC_URL=http://127.0.0.1:8899
export KEYPAIR_PATH="$(solana config get | sed -n 's/^Keypair Path: //p')"
npm test
```

## Troubleshooting
- Validator exits with tuning error (sudo needed):
```bash
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.rmem_default=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.core.wmem_default=134217728
```
- Health check:
```bash
curl -s -X POST -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' http://127.0.0.1:8899
```

## Optional: MCP Server (port 3010)
```bash
osvm mcp start --rpc 127.0.0.1:8899 --port 3010 --keypair "$KEYPAIR_PATH"
```
