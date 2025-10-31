# QuantDesk CLI Devnet Testing Guide

A comprehensive guide for testing QuantDesk smart contracts on devnet using CLI tools.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Test Suite Structure](#test-suite-structure)
- [Running Tests](#running-tests)
- [Writing Custom Tests](#writing-custom-tests)
- [Troubleshooting](#troubleshooting)

---

## Overview

QuantDesk provides CLI testing tools that allow you to:
- Test smart contracts on **real devnet** (no mocks)
- Run isolated test scenarios
- Validate PDA derivations
- Test account initialization and state management
- Debug transaction errors with full logs

Unlike Anchor's local test suite, these tests run against actual deployed programs on devnet, giving you confidence that your contracts work in production-like conditions.

---

## Prerequisites

### 1. Solana CLI Setup

```bash
# Install Solana CLI (if not already installed)
sh -c "$(curl -sSfL https://release.solana.com/v1.18.0/install)"

# Verify installation
solana --version

# Configure for devnet
solana config set --url devnet

# Set your keypair path
solana config set --keypair ~/.config/solana/keys/id.json

# Check your wallet balance
solana balance
```

### 2. Node.js & Dependencies

```bash
# Ensure Node.js 18+ is installed
node --version

# Install workspace dependencies
cd /home/dex/Desktop/quantdesk
pnpm install

# Build contracts (generates IDL)
cd contracts
anchor build
```

### 3. Environment Variables

The test suite uses environment variables with sensible defaults:

```bash
# Defaults (can override)
export RPC_URL=https://api.devnet.solana.com
export PROGRAM_ID=C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw
export KEYPAIR=~/.config/solana/keys/id.json
export PYTH_SOL_FEED=H6ARHf6YXhGYeQfUzQNGk6rDN1aQfwbNgBEMwLf9f5vK
```

---

## Quick Start

### Single Smoke Test

Test a basic deposit flow:

```bash
cd /home/dex/Desktop/quantdesk
npm run devnet:smoke
```

Or with custom parameters:

```bash
export DEPOSIT_SOL=0.002
export ACCOUNT_INDEX=0
npx ts-node scripts/devnet_smoke_test.ts
```

### Full Test Suite

Run all tests with detailed output:

```bash
npm run devnet:test:suite
```

Run specific test categories:

```bash
npm run devnet:test:deposit      # Deposit & collateral tests
npm run devnet:test:init        # Initialization tests
npm run devnet:test:pdas        # PDA derivation tests
npm run devnet:test:security   # Security & error handling
```

---

## Test Suite Structure

```
scripts/
├── devnet_smoke_test.ts          # Quick smoke test (single scenario)
├── devnet_test_suite.ts          # Comprehensive test runner
├── tests/
│   ├── deposit_tests.ts          # Deposit & collateral tests
│   ├── initialization_tests.ts   # Account initialization tests
│   ├── pda_tests.ts              # PDA derivation & validation
│   ├── account_order_tests.ts    # Account order validation
│   └── error_handling_tests.ts   # Error scenarios
└── utils/
    ├── test_helpers.ts           # Shared utilities
    └── account_verifier.ts       # Account validation helpers
```

### Test Categories

#### 1. **Deposit Tests** (`deposit_tests.ts`)
- ✅ Native SOL deposit
- ✅ Multiple deposits
- ✅ Collateral account initialization
- ✅ User account initialization
- ✅ Protocol vault updates

#### 2. **Initialization Tests** (`initialization_tests.ts`)
- ✅ First-time user account creation
- ✅ Existing account reuse
- ✅ Collateral account initialization
- ✅ Multiple account indices

#### 3. **PDA Tests** (`pda_tests.ts`)
- ✅ User account PDA derivation
- ✅ Collateral account PDA derivation
- ✅ Protocol vault PDA derivation
- ✅ Bump seed validation
- ✅ Cross-language PDA consistency (TS ↔ Rust)

#### 4. **Account Order Tests** (`account_order_tests.ts`)
- ✅ IDL account order validation
- ✅ Transaction account order
- ✅ Signer account positioning
- ✅ System program placement

#### 5. **Error Handling Tests** (`error_handling_tests.ts`)
- ✅ AccountNotSigner scenarios
- ✅ Insufficient funds
- ✅ Invalid account order
- ✅ Duplicate initialization attempts

---

## Running Tests

### Basic Usage

```bash
# Run all tests
npm run devnet:test:suite

# Run with verbose output
npm run devnet:test:suite -- --verbose

# Run specific test file
npx ts-node scripts/tests/deposit_tests.ts

# Run with custom RPC
export RPC_URL=https://custom-devnet-rpc.com
npm run devnet:test:suite
```

### Test Output

Tests provide detailed output including:

```
✅ Test Suite: Deposit Tests
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/5] Testing initial SOL deposit...
   Authority: wgfSHTWx1woRXhsWijj1kcpCP8tmbmK2KnouFVAuoc6
   Deposit: 0.001 SOL (1000000 lamports)
   ✅ Transaction successful: 5Zx8K...
   ✅ UserAccount initialized
   ✅ CollateralAccount initialized

[2/5] Testing PDA derivation consistency...
   UserAccount PDA: 2axc25ZgPYq2pPi8muMNWsND6mDzPdACtPzgWVijyjGK
   Collateral PDA: 59bqci5hv5wPJiyCy9G1goL4GpeHpVyoJ63bqEaoSh5h
   ✅ PDAs match IDL expectations

[3/5] Testing account order...
   ✅ Account order matches IDL exactly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ All tests passed (3/3)
```

### Filtering Tests

```bash
# Run only deposit-related tests
npm run devnet:test:suite -- --filter deposit

# Run only initialization tests
npm run devnet:test:suite -- --filter init

# Skip slow tests
npm run devnet:test:suite -- --skip-slow
```

---

## Writing Custom Tests

### Test File Template

```typescript
import { TestRunner, TestContext } from '../utils/test_helpers';
import { expect } from 'chai';

export async function runMyCustomTests(ctx: TestContext) {
  const { program, provider, signerPublicKey } = ctx;
  
  await ctx.describe('My Custom Tests', async () => {
    await ctx.it('My test case', async () => {
      // Your test logic
      const result = await program.methods
        .myMethod()
        .accounts({ /* ... */ })
        .rpc();
      
      ctx.expect(result).to.not.be.null;
    });
  });
}
```

### Using Test Helpers

```typescript
import { 
  derivePDAs, 
  verifyAccountOrder,
  assertAccountExists,
  waitForConfirmation 
} from '../utils/test_helpers';

// Derive all PDAs for a user
const pdas = await derivePDAs(signerPublicKey, program.programId);

// Verify account order matches IDL
await verifyAccountOrder(program, 'depositNativeSol', accounts);

// Assert account exists
await assertAccountExists(program, 'userAccount', pdas.userAccount);

// Wait for transaction confirmation
await waitForConfirmation(provider.connection, signature);
```

---

## Troubleshooting

### Common Issues

#### 1. **AccountNotSigner Error (3010)**

**Symptom:**
```
Error Code: AccountNotSigner. Error Number: 3010
```

**Solution:**
Ensure you're using `provider.wallet.publicKey` for the signer account:

```typescript
// ✅ Correct
user: provider.wallet.publicKey

// ❌ Incorrect
user: payer.publicKey  // Different object reference
```

#### 2. **IDL Not Found**

**Symptom:**
```
Error: IDL not found. Build contracts or copy IDL...
```

**Solution:**
```bash
cd contracts
anchor build  # Generates IDL in contracts/target/idl/
```

#### 3. **Insufficient Funds**

**Symptom:**
```
Error: Insufficient funds
```

**Solution:**
```bash
# Request airdrop
solana airdrop 2 -u devnet

# Or use the test helper
await fundAccount(publicKey, 2 * LAMPORTS_PER_SOL);
```

#### 4. **Transaction Already Processed**

**Symptom:**
```
Error: Transaction has already been processed
```

**Solution:**
- Use a different account index: `export ACCOUNT_INDEX=1`
- Wait for blockhash to expire (60 seconds)
- Generate a new keypair for testing

#### 5. **Program ID Mismatch**

**Symptom:**
```
Error: DeclaredProgramIdMismatch
```

**Solution:**
Verify the program ID in `contracts/programs/quantdesk-perp-dex/src/lib.rs` matches the deployed program:

```rust
declare_id!("C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw");
```

---

## Best Practices

### 1. **Isolated Test Environments**

Each test should be independent:

```typescript
// ✅ Good: Use unique account indices
const accountIndex = Math.floor(Math.random() * 1000);

// ❌ Bad: Reusing same accounts
const accountIndex = 0; // May conflict with other tests
```

### 2. **Proper Cleanup**

Clean up test artifacts:

```typescript
after(async () => {
  // Close accounts if needed
  // Reset test state
});
```

### 3. **Meaningful Assertions**

Always verify state changes:

```typescript
// ✅ Good: Verify account state
const account = await program.account.userAccount.fetch(pda);
expect(account.totalCollateral.toNumber()).to.equal(expectedAmount);

// ❌ Bad: Only check transaction success
// (transaction can succeed but state might be wrong)
```

### 4. **Error Context**

Capture full error details:

```typescript
try {
  await program.methods.depositNativeSol(amount).rpc();
} catch (error: any) {
  console.error('Transaction logs:', error.logs);
  console.error('Error code:', error.error?.errorCode);
  throw error;
}
```

### 5. **Rate Limiting**

Respect RPC rate limits:

```typescript
// Add delays between tests
await sleep(1000); // 1 second delay
```

---

## Advanced Usage

### Custom Test Runners

Create your own test scenarios:

```typescript
import { createTestContext, runTestSuite } from '../utils/test_helpers';

async function myCustomSuite() {
  const ctx = await createTestContext({
    rpcUrl: 'https://custom-rpc.com',
    programId: 'YourProgramId',
  });
  
  await runTestSuite(ctx, [
    myTest1,
    myTest2,
    myTest3,
  ]);
}
```

### Integration with CI/CD

Add to your CI pipeline:

```yaml
# .github/workflows/test-devnet.yml
- name: Run Devnet Tests
  run: |
    npm run devnet:test:suite
  env:
    RPC_URL: ${{ secrets.DEVNET_RPC_URL }}
    PROGRAM_ID: ${{ secrets.PROGRAM_ID }}
```

---

## Comparison: CLI Tests vs Anchor Tests

| Feature | CLI Devnet Tests | Anchor Local Tests |
|---------|------------------|-------------------|
| **Environment** | Real devnet | Local validator |
| **Speed** | Slower (network) | Faster (local) |
| **Realism** | Production-like | Isolated |
| **Debugging** | Full RPC logs | Limited |
| **Cost** | Free (devnet) | Free (local) |
| **Use Case** | Integration testing | Unit testing |

**Recommendation:** Use both! Run Anchor tests for fast unit tests, CLI tests for integration validation.

---

## Additional Resources

- [Solana Devnet Explorer](https://explorer.solana.com/?cluster=devnet)
- [Anchor Testing Guide](https://www.anchor-lang.com/docs/clients/typescript)
- [Solana Web3.js Docs](https://solana-labs.github.io/solana-web3.js/)
- [QuantDesk Architecture](../contracts/ARCHITECTURE.md)

---

**Need Help?** Check the [Troubleshooting](#troubleshooting) section or review transaction logs in the Solana Explorer.

