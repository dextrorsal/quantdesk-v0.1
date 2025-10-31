# Demo Mock Tools - Quick Guide

## 🎯 What This Does

Mock tools added for **demo purposes only**. These provide realistic market analysis and mock position opening without real transactions.

## ✅ What's Added

1. **`analyze_token_market`** - Comprehensive token analysis tool
   - Returns: Price, TVL, Market Cap, Indicators (RSI, MACD), Support/Resistance, Order Book Analysis, Sentiment, Trading Recommendations

2. **`open_position_mock`** - Mock position opening tool
   - Returns: Mock success response with position details
   - **No real transactions** - Demo mode only

## 📝 Demo Flow Example

### Step 1: Analyze FARTCOIN
```
"Analyze the market and sentiment for FARTCOIN, showing TVL, market cap, price, indicators, order book levels, and help decide if bullish or bearish"
```

**MIKEY will return:**
- Current price, 24h/7d changes
- Market cap, TVL, volume
- RSI, MACD indicators
- Support/resistance levels
- Order book concentration (where money is placed)
- Sentiment analysis (BULLISH/BEARISH)
- Trading recommendation with confidence level

### Step 2: Open Position
```
"Open a position based on that analysis" 
or
"Open a buy position for FARTCOIN-PERP with 0.1 size and 5x leverage"
```

**MIKEY will return:**
- Mock position opened successfully
- Position details (entry price, size, leverage, liquidation price)
- Demo transaction signature
- **Note**: This is demo mode - no real funds moved

## 🎬 Demo Script Suggestion

**Ask MIKEY:**
1. "Analyze FARTCOIN market sentiment, TVL, market cap, price indicators, and order book levels to help me decide if I should be bullish or bearish"

**Wait for analysis, then:**
2. "Based on that analysis, open a buy position for FARTCOIN-PERP"

**Result:** 
- You'll see detailed market analysis
- Then a mock position confirmation
- Shows how MIKEY can analyze → recommend → execute

## ⚙️ How It Works

- Detection runs **first** (before real trading tools)
- Queries with "analyze market", "FARTCOIN", "TVL", "sentiment", "indicators" trigger mock analysis
- Queries with "open position" or "place trade" trigger mock position opening
- All mock responses are clearly marked with `demo: true`

## 🚀 Quick Test

1. Restart MIKEY-AI service:
```bash
cd MIKEY-AI
pnpm run dev
```

2. Open frontend and test MIKEY chat:
```
"Analyze FARTCOIN market data"
```

You should see comprehensive mock market analysis with all the metrics!

