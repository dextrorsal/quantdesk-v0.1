# Mock Data Setup for Demo

## Quick Setup

1. **Import the mock service** in your deposit modal/component
2. **Replace** `smartContractService.depositNativeSOL()` with `mockDepositService.deposit()`
3. **Show mock balances** in your UI

## Why Mock Data?
- AccountNotSigner issue is beyond frontend fix
- Might require program redeployment or Anchor version changes
- Demo needs to happen TODAY
- Real blockchain integration can be finished after demo

## Next Steps After Demo
1. Redeploy Solana program with fresh build
2. Try different wallet (Solflare instead of Phantom)
3. Test with Anchor 0.28.0 (version from working examples)
4. Check if there's a bug in init_if_needed macro
