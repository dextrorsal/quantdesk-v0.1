# Multi-Asset Oracle Integration Plan

## Overview
Expand QuantDesk to support 29 collateral assets with a multi-oracle architecture following industry best practices from Drift, Zeta, and Kamino protocols.

## Current State
- ✅ SOL deposits working with Pyth integration
- ❌ Only 6/29 tokens have configured price feeds
- ❌ Frontend collateral display bug (showing $450 instead of ~$93)

## Phase 1: Fix Current Issues (URGENT - 1-2 hours)

### 1.1 Debug Collateral Display Bug
**Action**: Check browser console logs to identify if bug is on-chain or frontend
**Files**: `frontend/src/services/smartContractService.ts`, `frontend/src/components/AccountSlideOut.tsx`
**Test**: Deposit 0.45 SOL, verify it shows ~$93 USD (at $208/SOL)

### 1.2 Verify On-Chain USD Calculation
**Action**: Query collateral account directly from Solana Sandbox
**Command**:
```bash
cd solana-sandbox
npx ts-node scripts/query-collateral-account.ts
```

## Phase 2: Pyth Price Feed Expansion (2-3 hours)

### 2.1 Add Pyth Feeds for Major Assets
Update `backend/src/services/pythOracleService.ts` with additional feeds:

```typescript
private readonly PYTH_FEED_IDS = {
  // Existing
  BTC: 'e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43',
  ETH: 'ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace',
  SOL: 'ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d',
  
  // Add Major Tokens
  USDT: '2b89b9dc8fdf9f34709a5b106b472f0f39bb6ca9ce04b0fd7f2e971688e2e53b', // USDT/USD
  USDC: 'eaa020c61cc479712813461ce153894a96a6c00b21ed0cfc2798d1f9a9e9c94a', // USDC/USD
  BNB: '2f95862b045670cd22bee3114c39763a4a08beeb663b145d283c31d7d1101c4f',  // BNB/USD
  
  // Solana Ecosystem
  JUP: 'xxx',  // Need to find Jupiter price feed ID
  PYTH: 'xxx', // Pyth's own feed
  JTO: 'xxx',  // Jito price feed
  BONK: 'xxx', // BONK/USD
  MSOL: 'xxx', // mSOL/USD or mSOL/SOL
  JITOSOL: 'xxx', // jitoSOL/USD or jitoSOL/SOL
}
```

**Resource**: https://pyth.network/developers/price-feed-ids

### 2.2 Smart Contract Oracle Integration
**Files to Update**:
- `contracts/programs/src/lib.rs`
- Add `deposit_spl_token` instruction (similar to `deposit_native_sol`)
- Add token-specific Pyth price feed handling

```rust
pub fn deposit_spl_token(
    ctx: Context<DepositSplToken>, 
    amount: u64,
    token_mint: Pubkey
) -> Result<()> {
    // Get USD value from token-specific Pyth feed
    let usd_value = match token_mint {
        USDT_MINT => get_usd_value_from_token(amount, &ctx.accounts.usdt_usd_price_feed, 6)?,
        USDC_MINT => get_usd_value_from_token(amount, &ctx.accounts.usdc_usd_price_feed, 6)?,
        BTC_MINT => get_usd_value_from_token(amount, &ctx.accounts.btc_usd_price_feed, 8)?,
        _ => return Err(ErrorCode::UnsupportedToken.into()),
    };
    
    // Update collateral with USD value
    // ... rest of deposit logic
}
```

## Phase 3: Jupiter Price API Integration (3-4 hours)

### 3.1 Setup Jupiter Price API Service
**File**: `backend/src/services/jupiterPriceService.ts`

```typescript
import axios from 'axios';

export class JupiterPriceService {
  private readonly JUPITER_API = 'https://price.jup.ag/v4';
  private readonly TOKEN_MINTS = {
    WIF: 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
    PENGU: 'EPeUFDgHRxs9xxEPVaL6kfGQvCon7jmAWKVUHuux1Tpz',
    BONK: 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
    // ... more tokens
  };
  
  async getPrices(tokenSymbols: string[]): Promise<Record<string, number>> {
    const mints = tokenSymbols.map(s => this.TOKEN_MINTS[s]).filter(Boolean);
    const response = await axios.get(`${this.JUPITER_API}/price`, {
      params: { ids: mints.join(',') }
    });
    
    // Convert mint addresses to symbols
    const priceMap: Record<string, number> = {};
    for (const [mint, data] of Object.entries(response.data.data)) {
      const symbol = Object.keys(this.TOKEN_MINTS).find(
        k => this.TOKEN_MINTS[k] === mint
      );
      if (symbol) priceMap[symbol] = data.price;
    }
    return priceMap;
  }
}
```

### 3.2 Multi-Oracle Service (Combines Pyth + Jupiter)
**File**: `backend/src/services/multiOracleService.ts`

```typescript
export class MultiOracleService {
  constructor(
    private pythService: PythOracleService,
    private jupiterService: JupiterPriceService
  ) {}
  
  async getAllPrices(): Promise<Record<string, number>> {
    // Get Pyth prices first (more reliable for major assets)
    const pythPrices = await this.pythService.getAllPrices();
    
    // Get Jupiter prices for assets not in Pyth
    const missingAssets = REQUIRED_ASSETS.filter(a => !pythPrices[a]);
    const jupiterPrices = await this.jupiterService.getPrices(missingAssets);
    
    // Merge with Pyth taking priority
    return { ...jupiterPrices, ...pythPrices };
  }
  
  async getPrice(asset: string): Promise<number> {
    // Try Pyth first
    try {
      return await this.pythService.getAssetPrice(asset);
    } catch (e) {
      // Fallback to Jupiter
      return (await this.jupiterService.getPrices([asset]))[asset];
    }
  }
}
```

## Phase 4: Smart Contract Multi-Token Support (4-6 hours)

### 4.1 Add SPL Token Deposit Instructions
**Files**:
- `contracts/programs/src/token_operations.rs`
- Add `DepositSplToken` context
- Support multiple token mints

### 4.2 Create Token-Specific Collateral Accounts
Update PDA derivation to support multiple token types:

```rust
// Current: [b"collateral", user.key().as_ref(), b"SOL"]
// New: [b"collateral", user.key().as_ref(), token_mint.as_ref()]

#[account(
    init_if_needed,
    payer = user,
    space = 8 + CollateralAccount::INIT_SPACE,
    seeds = [
        b"collateral", 
        user.key().as_ref(), 
        token_mint.as_ref()
    ],
    bump
)]
pub collateral_account: Account<'info, CollateralAccount>,
```

### 4.3 Add Price Feed Mapping
Create a registry of token mint → price feed mappings:

```rust
// In lib.rs
pub const TOKEN_PRICE_FEEDS: &[(Pubkey, Pubkey)] = &[
    (SOL_MINT, SOL_USD_FEED),
    (USDT_MINT, USDT_USD_FEED),
    (USDC_MINT, USDC_USD_FEED),
    (BTC_MINT, BTC_USD_FEED),
    // ... etc
];

pub fn get_price_feed_for_token(token_mint: &Pubkey) -> Option<Pubkey> {
    TOKEN_PRICE_FEEDS.iter()
        .find(|(mint, _)| mint == token_mint)
        .map(|(_, feed)| *feed)
}
```

## Phase 5: Frontend Multi-Asset Support (3-4 hours)

### 5.1 Update Deposit Modal
**File**: `frontend/src/components/DepositModal.tsx`

- Add token selector dropdown
- Show token-specific balances
- Dynamic price feed selection
- Support both native SOL and SPL tokens

### 5.2 Update Constants File
**File**: `frontend/src/utils/constants.ts`

```typescript
export const PYTH_PRICE_FEEDS = {
  SOL_USD: new PublicKey('H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG'),
  BTC_USD: new PublicKey('HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J'),
  ETH_USD: new PublicKey('JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB'),
  USDT_USD: new PublicKey('...'), // Add once we get the IDs
  USDC_USD: new PublicKey('...'),
  // ... more feeds
};

export const TOKEN_MINTS = {
  SOL: new PublicKey('So11111111111111111111111111111111111111112'),
  USDT: new PublicKey('Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB'),
  USDC: new PublicKey('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'),
  // ... from tokens.ts
};
```

### 5.3 Update Smart Contract Service
**File**: `frontend/src/services/smartContractService.ts`

Add methods:
- `depositSplToken(wallet, tokenMint, amount)`
- `withdrawSplToken(wallet, tokenMint, amount)`
- Auto-detect price feed based on token mint

## Phase 6: Testing & Validation (2-3 hours)

### 6.1 Sandbox Tests
**File**: `solana-sandbox/tests/multi-asset-deposits.ts`

Test matrix:
- ✅ SOL deposit/withdraw with Pyth price
- ✅ USDT deposit/withdraw with Pyth price
- ✅ USDC deposit/withdraw with Pyth price
- ✅ BTC (wrapped) deposit/withdraw with Pyth price
- ✅ WIF deposit/withdraw with Jupiter price (if no Pyth)
- ✅ Verify all USD valuations are correct
- ✅ Test collateral display in frontend

### 6.2 Price Feed Validation
**Script**: `solana-sandbox/scripts/validate-all-prices.ts`

```typescript
// Verify every configured token has a working price feed
for (const token of TOKEN_CONFIGS) {
  const price = await multiOracleService.getPrice(token.symbol);
  console.log(`${token.symbol}: $${price}`);
  assert(price > 0, `${token.symbol} price is invalid`);
}
```

## Timeline & Priority

### **CRITICAL (Today - 2-3 hours)**
1. ✅ Fix $450 USD display bug
2. ✅ Verify Pyth SOL/USD integration is working correctly
3. ✅ Test deposit → collateral → display flow

### **HIGH (Next 1-2 days)**
1. Add Pyth feeds for: USDT, USDC, BTC, ETH, BNB
2. Implement Jupiter Price API fallback
3. Update smart contract to support SPL token deposits
4. Deploy to devnet and test

### **MEDIUM (Next 3-5 days)**
1. Add remaining Solana ecosystem tokens (JUP, PYTH, JTO, LSTs)
2. Add meme coin support via Jupiter
3. Frontend multi-asset deposit UI
4. Comprehensive testing suite

### **LOW (Future)**
1. Switchboard integration for custom feeds
2. TWAP/EWMA price smoothing (like Kamino)
3. Price confidence intervals
4. Oracle failure handling

## Resources & References

- **Pyth Price Feeds**: https://pyth.network/developers/price-feed-ids#solana-mainnet
- **Jupiter Price API**: https://station.jup.ag/docs/apis/price-api
- **Drift Protocol Oracle Pattern**: https://github.com/drift-labs/protocol-v2/blob/master/sdk/src/oracles/
- **Switchboard Feeds**: https://app.switchboard.xyz/solana/mainnet
- **Solana Token List**: https://github.com/solana-labs/token-list

## Security Considerations

1. **Price Staleness**: Check `price_account.timestamp` - reject if > 5 minutes old
2. **Confidence Intervals**: Use `price_account.agg.conf` - reject if confidence too wide
3. **Multi-Oracle Comparison**: If using multiple oracles, compare prices and reject if > 5% deviation
4. **Fallback Logic**: Pyth → Jupiter → Switchboard → Fail gracefully (don't allow deposits)
5. **Price Manipulation**: Consider TWAP for volatile meme coins

## Next Steps

1. **User Action**: Check browser console logs for the $450 bug
2. **Agent Action**: Fix collateral display bug once root cause identified
3. **Agent Action**: Fetch exact Pyth price feed IDs for all major tokens
4. **Agent Action**: Implement Phase 2 (Pyth expansion) for immediate hackathon needs
5. **Agent Action**: Implement Phase 3 (Jupiter) for long-tail token support

