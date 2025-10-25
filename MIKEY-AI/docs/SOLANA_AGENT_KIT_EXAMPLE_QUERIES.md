# Solana Agent Kit Example Queries

This document provides example queries that demonstrate the new capabilities added to MIKEY-AI through the Solana Agent Kit integration.

## Basic Balance Queries

### SOL Balance Queries
```
"What's my SOL balance?"
"Show me my SOL balance for wallet ABC123"
"Check my native SOL balance"
"How much SOL do I have?"
```

### Token Balance Queries
```
"What's my USDC token balance?"
"Show me my USDC balance for wallet ABC123"
"Check my token balance for EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
"How much USDC do I have?"
```

## Trading Analysis Queries

### Swap Quote Queries
```
"Get me a quote for swapping 10 USDC to SOL"
"Show me a Jupiter swap quote for 100 SOL to USDC"
"What's the price impact for swapping 50 USDC to SOL?"
"Get swap quote: EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v to So11111111111111111111111111111111111111112, amount 100"
```

### Liquidity Analysis Queries
```
"Check liquidity for SOL/USDC pair"
"What's the liquidity like for Jupiter swaps?"
"Analyze swap liquidity for my trade"
```

## Token Research Queries

### Token Information Queries
```
"What's the information for USDC token?"
"Show me details about EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
"What are the token details for USDC?"
"Get token metadata for So11111111111111111111111111111111111111112"
```

### Wallet Analysis Queries
```
"What tokens does wallet ABC123 hold?"
"Show me all tokens in my wallet"
"Give me a complete overview of wallet ABC123"
"What's in my portfolio?"
```

## Portfolio Management Queries

### Comprehensive Portfolio Queries
```
"Show me my complete wallet information"
"Give me a portfolio overview"
"What's my total wallet value?"
"Analyze my entire portfolio"
```

### Multi-Token Queries
```
"Check all my token balances"
"Show me balances for SOL, USDC, and BTC"
"What tokens do I have and their balances?"
```

## Advanced Trading Queries

### Cross-Token Analysis
```
"Compare my SOL and USDC balances"
"Show me my token distribution"
"What's my portfolio composition?"
```

### Trading Decision Support
```
"Should I swap my USDC to SOL based on current quotes?"
"Analyze the best swap route for my trade"
"What's the optimal amount to swap?"
```

## Error Handling Examples

### Invalid Input Queries
```
"What's my balance?" (missing wallet address)
"Get swap quote" (missing parameters)
"Show token data" (missing token mint)
```

### Edge Case Queries
```
"What's my balance for an invalid wallet address?"
"Get quote for non-existent token pair"
"Show data for invalid token mint"
```

## Expected Response Formats

### Wallet Balance Response
```json
{
  "wallet": "ABC123",
  "solBalance": "0.0",
  "status": "mock data",
  "note": "POC implementation - actual balance check pending"
}
```

### Token Balance Response
```json
{
  "wallet": "ABC123",
  "tokenMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
  "balance": "0.0",
  "status": "mock data",
  "note": "POC implementation - actual token balance check pending"
}
```

### Swap Quote Response
```json
{
  "inputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
  "outputMint": "So11111111111111111111111111111111111111112",
  "inputAmount": "100",
  "outputAmount": "0.0",
  "priceImpact": "0.0%",
  "fee": "0.0",
  "status": "mock data",
  "note": "POC implementation - actual Jupiter quote pending"
}
```

### Token Data Response
```json
{
  "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
  "name": "Mock Token",
  "symbol": "MOCK",
  "decimals": 6,
  "supply": "1000000",
  "status": "mock data",
  "note": "POC implementation - actual token data pending"
}
```

### Wallet Info Response
```json
{
  "wallet": "ABC123",
  "solBalance": "0.0",
  "tokenHoldings": [],
  "totalValue": "0.0",
  "status": "mock data",
  "note": "POC implementation - actual wallet info pending"
}
```

## Testing Queries

### Unit Test Queries
```
"Test wallet balance tool"
"Test token balance functionality"
"Test swap quote generation"
"Test token data retrieval"
"Test wallet info compilation"
```

### Integration Test Queries
```
"Test complete portfolio analysis"
"Test multi-tool query processing"
"Test error handling scenarios"
"Test performance under load"
```

## Production-Ready Queries (Future)

### Real Blockchain Queries
```
"What's my actual SOL balance on mainnet?"
"Get real Jupiter swap quote for 100 USDC"
"Show my actual token holdings"
"Execute swap for 50 USDC to SOL"
```

### Advanced Analytics Queries
```
"Analyze my trading patterns"
"Suggest optimal portfolio rebalancing"
"Calculate my portfolio risk metrics"
"Show my trading performance history"
```

## Query Optimization Tips

### Best Practices
1. **Be Specific**: Include wallet addresses and token mints when possible
2. **Use Standard Formats**: Follow the comma-separated format for multi-parameter queries
3. **Check Keywords**: Use the detected keywords for optimal tool routing
4. **Handle Errors**: Expect mock data responses in POC mode

### Common Patterns
- **Balance Queries**: Always include wallet address
- **Swap Queries**: Use format "inputMint,outputMint,amount"
- **Token Queries**: Include token mint address
- **Portfolio Queries**: Use comprehensive wallet analysis

---

**Note**: All queries in this POC return mock data. For production use, actual blockchain integration would be required with proper private key management and security measures.
