# News Integration - CoinDesk, CoinTelegraph, The Block

## ✅ What's Integrated

MIKEY-AI now fetches **real news** from top crypto news outlets:

1. **CoinDesk** - `https://www.coindesk.com/arc/outboundfeeds/rss/`
2. **CoinTelegraph** - `https://cointelegraph.com/rss/tag/bitcoin`
3. **The Block** - `https://www.theblock.co/feed/news`

## 🔌 How It Works

### Backend Endpoint
- **URL**: `GET /api/news`
- **Query Params**:
  - `sources` - Comma-separated: `CoinDesk,CoinTelegraph,The Block` or `all`
  - `ticker` - Filter by ticker (e.g., `BTC`, `ETH`, `SOL`)
  - `category` - Filter by category (DeFi, NFTs, Regulatory, etc.)
  - `keyword` - Search keyword
  - `limit` - Max articles (default: 20)

### MIKEY Tool
- **Tool Name**: `get_crypto_news`
- **Detects queries with**: "news", "article", "headline", "coindesk", "cointelegraph", "the block"
- **Auto-extracts ticker** from query (e.g., "BTC news" → filters by BTC)
- **Prioritizes** CoinDesk, CoinTelegraph, The Block by default

## 📝 Example Queries

### All News Sources
```
"Get me the latest crypto news"
```
Returns: Articles from CoinDesk, CoinTelegraph, The Block

### Filtered by Ticker
```
"Show me BTC news from CoinDesk"
"Get news about ETH from CoinTelegraph"
"Latest SOL news from The Block"
```
Returns: Articles filtered by ticker from specified sources

### Specific Source
```
"Show me crypto news from The Block"
"Get CoinDesk articles"
```
Returns: Articles from the requested source(s)

## 📊 Response Format

```
📰 Latest Crypto News from CoinDesk, CoinTelegraph, The Block (13 articles)

Filtered by ticker: BTC

1. **Solana Tumbles 8%, Erasing All Year-Over-Year Gains...**
   Source: CoinDesk | 2025-10-30 04:59 PM
   Ticker: BTC | Category: Market Analysis
   [Snippet...]
   URL: https://...
```

## 🔧 Configuration

All news sources are configured in:
- **Backend**: `backend/src/routes/news.ts` (lines 22-29)
- **MIKEY Tool**: `MIKEY-AI/src/services/RealDataTools.ts` (lines 168-248)
- **Tool Routing**: `MIKEY-AI/src/agents/TradingAgent.ts` (lines 296-313)

## ✅ Test Results

- ✅ News tool detects "news", "article", "coindesk", "cointelegraph", "the block"
- ✅ Fetches real articles from RSS feeds
- ✅ Filters by ticker automatically
- ✅ Formats articles nicely (not raw JSON)
- ✅ Returns headlines, sources, dates, snippets, URLs

## 🚀 Ready to Use!

Just ask MIKEY:
- "Get me the latest news from CoinDesk"
- "Show me BTC news from CoinTelegraph"
- "What's the latest from The Block?"
- "Get crypto news about SOL"

All queries will fetch **real, live articles** from these top sources!

