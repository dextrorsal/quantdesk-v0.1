# QuantDesk Scripts

This directory contains utility scripts for testing and development.

## 🚀 Quick Reference

### Devnet Testing

```bash
# Quick smoke test (single deposit)
npm run devnet:smoke

# Full test suite
npm run devnet:test:suite

# Verbose output
npm run devnet:test:suite:verbose
```

See [CLI_DEVNET_TESTING_GUIDE.md](./CLI_DEVNET_TESTING_GUIDE.md) for comprehensive documentation.

## 📁 File Structure

```
scripts/
├── devnet_smoke_test.ts          # Quick smoke test
├── devnet_test_suite.ts          # Comprehensive test runner
├── CLI_DEVNET_TESTING_GUIDE.md   # Full testing documentation
├── DEVNET_SMOKE_TEST_README.md   # Smoke test quick reference
└── run-devnet-smoke.sh           # Bash wrapper for smoke test
```

## 📚 Documentation

- **[CLI Devnet Testing Guide](./CLI_DEVNET_TESTING_GUIDE.md)** - Complete guide for CLI testing
- **[Smoke Test README](./DEVNET_SMOKE_TEST_README.md)** - Quick reference for smoke tests

## 🔧 Prerequisites

- Solana CLI configured for devnet
- Node.js 18+
- Dependencies installed: `pnpm install`
- Contracts built: `cd contracts && anchor build`

## 🎯 Usage Examples

### Basic Testing

```bash
# Run all tests
npm run devnet:test:suite

# With custom RPC
export RPC_URL=https://custom-rpc.com
npm run devnet:test:suite
```

### Custom Configuration

```bash
export PROGRAM_ID=YourProgramId
export KEYPAIR=~/.config/solana/keys/my-key.json
export RPC_URL=https://api.devnet.solana.com
npm run devnet:smoke
```

## 📊 Test Output

The test suite provides:
- ✅ Pass/fail status for each test
- 📝 Transaction signatures with Explorer links
- ⏱️  Execution time for each test
- 📈 Summary statistics

## 🆘 Troubleshooting

Common issues and solutions:

1. **IDL Not Found**: Run `cd contracts && anchor build`
2. **Insufficient Funds**: Run `solana airdrop 2 -u devnet`
3. **AccountNotSigner Error**: See [CLI_DEVNET_TESTING_GUIDE.md#troubleshooting](./CLI_DEVNET_TESTING_GUIDE.md#troubleshooting)

For detailed troubleshooting, see the [full guide](./CLI_DEVNET_TESTING_GUIDE.md#troubleshooting).
