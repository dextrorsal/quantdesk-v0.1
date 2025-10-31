# How MIKEY Tool Routing Works

## 🎯 Decision Flow

When you ask MIKEY a question, it follows this **priority order**:

### 1. **Real Data Tools** (First Priority) ✅
**Checks for:** "analyze", "market summary", "tvl", "sentiment", "indicators", "pyth", "oracle", "live price"

**What it does:**
- Calls `/api/oracle/price/:asset` - Gets REAL Pyth prices
- Calls `/api/dev/market-summary` - Gets REAL market data (volume, OI, leverage)
- Uses `RealTokenAnalysisTool` - Combines real backend data
- Formats response nicely

**Example query:** "Analyze FARTCOIN market sentiment and TVL"
→ **Uses:** Real backend data (Pyth prices + market summary)

### 2. **QuantDesk Protocol Tools**
**Checks for:** "portfolio", "positions", "place order", "risk analysis"

**What it does:**
- Calls `/api/portfolio/:wallet` - Real portfolio data
- Calls `/api/positions/:wallet` - Real positions
- Calls `/api/orders` - Real order placement

### 3. **Demo Mock Tools** (Only if Explicit)
**Checks for:** "mock", "demo", OR "open position"/"place trade" (for demo)

**What it does:**
- Returns mock data for demo purposes
- Only used if you say "mock" or "demo" OR for position opening

---

## 🔧 Current Behavior

**Your query:** "Analyze FARTCOIN and give me a market summary"

**What happens:**
1. ✅ Matches `needsRealData()` - "analyze" detected
2. ✅ Calls `RealTokenAnalysisTool.createRealTokenAnalysisTool()`
3. ✅ Tool fetches:
   - `/api/oracle/price/FARTCOIN` → Real Pyth price
   - `/api/dev/market-summary` → Real market data
4. ✅ Formats response with real prices

**Result:** You get REAL prices from your backend!

---

## 🎬 For Demo

If you want **mock data** for demo (FARTCOIN with fake metrics):
- Ask: "Analyze FARTCOIN **mock** market data"
- OR: Just say "**demo**" or "**mock**" in the query

If you want **real data** (which we have!):
- Ask: "Analyze BTC market sentiment"
- OR: "Analyze SOL and show me TVL, market cap, indicators"
- OR: "What's the current price of BTC?"

---

## 📊 Real Data Available

**What we have working:**
- ✅ `/api/oracle/prices` - Real Pyth prices for BTC, ETH, SOL, etc.
- ✅ `/api/oracle/price/:asset` - Single asset price
- ✅ `/api/dev/market-summary` - Market data (volume, OI, leverage)
- ✅ Real-time updates every 5 seconds

**So MIKEY CAN use real data!** Just needs to route to the right tools.

---

## 🔄 After Restart

After restarting MIKEY-AI, try:
- "Analyze BTC market sentiment" → **Real data** ✅
- "What's the price of SOL?" → **Real Pyth price** ✅
- "Show me market summary for ETH" → **Real market data** ✅

The system will automatically use real tools unless you explicitly ask for "mock" or "demo"!

