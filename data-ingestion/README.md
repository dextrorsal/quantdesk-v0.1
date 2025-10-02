# QuantDesk Data Ingestion Pipeline

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Redis Streams  │    │   Workers       │
│                 │    │   (Message Bus)  │    │                 │
│ • Pyth Oracle   │───▶│ • ticks.raw      │───▶│ • Price Writer  │
│ • Whale Events  │    │ • whales.raw     │    │ • Analytics     │
│ • News Feeds    │    │ • news.raw       │    │ • Alerts        │
│ • User Actions  │    │ • user.events    │    │ • ML Features   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Supabase DB    │
                       │   (Postgres)     │
                       └──────────────────┘
```

## 🚀 Components

### 1. **Collectors** (Data Sources)
- `price-collector/` - Pyth Oracle price feeds
- `whale-monitor/` - Large wallet movements
- `news-scraper/` - Crypto news & sentiment
- `user-tracker/` - User trading actions

### 2. **Message Bus** (Redis Streams)
- `ticks.raw` - Raw price data
- `whales.raw` - Whale movement events
- `news.raw` - News articles
- `user.events` - User actions

### 3. **Workers** (Data Processing)
- `price-writer/` - Batch insert to oracle_prices
- `analytics-writer/` - Update market_stats, user_stats
- `alert-processor/` - Generate alerts/notifications
- `ml-features/` - Extract features for ML models

## 📊 Data Flow

### **Real-time Price Flow**
```
Pyth Oracle → price-collector → ticks.raw → price-writer → oracle_prices
```

### **Whale Monitoring Flow**
```
Solana RPC → whale-monitor → whales.raw → analytics-writer → system_events
```

### **News Sentiment Flow**
```
News APIs → news-scraper → news.raw → ml-features → sentiment scores
```

## 🔧 Setup Instructions

1. **Install Redis**:
   ```bash
   # Ubuntu/Debian
   sudo apt install redis-server
   
   # macOS
   brew install redis
   
   # Start Redis
   redis-server
   ```

2. **Install Dependencies**:
   ```bash
   cd data-ingestion
   npm install
   ```

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your keys
   ```

4. **Start Services**:
   ```bash
   # Start all collectors
   npm run start:collectors
   
   # Start all workers
   npm run start:workers
   
   # Start monitoring
   npm run start:monitoring
   ```

## 📈 Performance Targets

- **Price Updates**: 1000+ ticks/second per market
- **Whale Events**: Real-time detection (< 1 second)
- **News Processing**: 100+ articles/hour
- **Database Writes**: Batched every 100ms
- **Latency**: End-to-end < 50ms

## 🔍 Monitoring

- Redis Stream lengths
- Worker processing rates
- Database write performance
- Error rates and alerts
- Memory usage

## 🚨 Alerting

- Price feed disconnections
- High latency warnings
- Database connection issues
- Worker failures
- Memory/CPU thresholds
