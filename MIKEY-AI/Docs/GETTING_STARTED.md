# Getting Started with Solana DeFi Trading Intelligence AI

## Quick Setup Guide

### 1. Environment Setup

Copy the environment template and configure your API keys:

```bash
cp env.example .env
```

Edit `.env` with your actual API keys:

```env
# Required - Get these first
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
OPENAI_API_KEY=your_openai_api_key_here
SOLANA_PRIVATE_KEY=your_base58_private_key_here

# Optional but recommended
HELIUS_API_KEY=your_helius_api_key
PYTH_API_KEY=your_pyth_api_key
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Run Tests

Verify everything is working:

```bash
npx tsx src/test.ts
```

### 4. Start Development Server

```bash
npm run dev
```

The API will be available at `http://localhost:3000`

## API Usage Examples

### Health Check
```bash
curl http://localhost:3000/health
```

### AI Query
```bash
curl -X POST http://localhost:3000/api/v1/ai/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "query": "What is the current market sentiment for SOL?",
    "context": {"symbols": ["SOL/USD"]}
  }'
```

### Wallet Analysis
```bash
curl http://localhost:3000/api/v1/wallets/YOUR_WALLET_ADDRESS/analysis \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Market Prices
```bash
curl "http://localhost:3000/api/v1/market/prices?symbols=SOL/USD,ETH/USD" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Available Tools

The AI agent has access to these trading tools:

- **Wallet Analysis** - Portfolio composition, trading patterns
- **Price Analysis** - Real-time and historical price data
- **Transaction Analysis** - Trading activity and patterns
- **Market Sentiment** - Social media and news sentiment
- **Liquidation Detection** - Cross-protocol liquidation events
- **Whale Tracking** - Large wallet monitoring
- **Technical Analysis** - Indicators, support/resistance levels

## Security Features

- ✅ **API Key Authentication** - Secure endpoint access
- ✅ **Rate Limiting** - Prevent abuse and ensure fair usage
- ✅ **Data Encryption** - Sensitive data protection
- ✅ **Input Sanitization** - Prevent injection attacks
- ✅ **Secure Logging** - Masked sensitive data in logs
- ✅ **CORS Protection** - Cross-origin request security

## Development Commands

```bash
# Development
npm run dev          # Start development server
npm run build        # Build TypeScript
npm start           # Start production server

# Testing
npm test            # Run tests
npm run test:watch  # Watch mode

# Code Quality
npm run lint        # Check code style
npm run lint:fix    # Fix code style issues
npm run format      # Format code with Prettier
```

## Next Steps

1. **Add Real Data Sources** - Integrate with Pyth, Jupiter, Drift APIs
2. **Implement WebSocket** - Real-time data streaming
3. **Add Database** - Store historical data and user preferences
4. **Build Frontend** - Web dashboard for trading insights
5. **Deploy** - Production deployment with monitoring

## Troubleshooting

### Common Issues

**"Missing required environment variables"**
- Make sure your `.env` file exists and has all required variables
- Check that API keys are valid and have proper permissions

**"Failed to initialize Solana connection"**
- Verify your RPC URL is correct and accessible
- Check if you're using a paid RPC provider for better reliability

**"Invalid private key format"**
- Ensure your private key is base58 encoded
- Use `solana-keygen grind --starts-with ai:1` to generate a new key

**"AI query failed"**
- Verify your OpenAI API key is valid and has credits
- Check the query format and parameters

### Getting Help

- Check the logs in the `logs/` directory
- Review the API documentation in `docs/API_REFERENCE.md`
- Open an issue on GitHub for bugs or feature requests

## Security Best Practices

- 🔐 **Never commit API keys** - Use `.env` files and proper `.gitignore`
- 🔐 **Use strong API keys** - Generate secure, random keys
- 🔐 **Monitor usage** - Watch for unusual activity in logs
- 🔐 **Regular updates** - Keep dependencies updated
- 🔐 **Test on devnet** - Always test on devnet before mainnet

Happy trading! 🚀
