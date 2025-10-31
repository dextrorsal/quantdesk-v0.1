#!/usr/bin/env bash
set -euo pipefail

# Usage:
#  RPC_URL=https://api.devnet.solana.com \
#  PROGRAM_ID=<YourProgramId> \
#  PYTH_SOL_FEED=H6ARHf6YXhGYeQfUzQNGk6rDN1aQfwbNgBEMwLf9f5vK \
#  KEYPAIR=~/.config/solana/id.json \
#  DEPOSIT_SOL=0.001 \
#  ACCOUNT_INDEX=0 \
#  npx ts-node scripts/devnet_smoke_test.ts

export NODE_OPTIONS="--loader ts-node/esm"

if ! command -v npx >/dev/null 2>&1; then
  echo "npx not found - install Node.js >=18" >&2
  exit 1
fi

npx ts-node scripts/devnet_smoke_test.ts
