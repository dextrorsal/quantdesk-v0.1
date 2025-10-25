# MIKEY-AI to QuantDesk Integration Guide

## Overview

This integration bridge connects MIKEY-AI's intelligence capabilities with QuantDesk's ML trading system, creating a unified trading intelligence platform.

## Features

- 🧠 **AI-Powered Analysis**: Natural language queries with MIKEY-AI
- 🤖 **ML Predictions**: QuantDesk's trained models (Lorentzian Classifier, etc.)
- 📊 **Cross-Platform Data**: 100+ exchanges via MIKEY-AI's CCXT integration
- 🐋 **Whale Tracking**: Smart money monitoring with ML impact analysis
- 💥 **Liquidation Detection**: Real-time liquidation monitoring
- 🔄 **Arbitrage Detection**: Cross-platform opportunity identification
- 📈 **Unified Dashboard**: Single API for all trading intelligence

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MIKEY-AI (Intelligence Layer)             │
├─────────────────────────────────────────────────────────────┤
│  AI Agent  │  CCXT Service  │  Solana Service  │  Analysis  │
│  (GPT-4)   │  (100+ CEX)    │  (DeFi Data)     │  Engine    │
├─────────────────────────────────────────────────────────────┤
│                    Bridge Service (This Project)             │
├─────────────────────────────────────────────────────────────┤
│                    QuantDesk (Trading Layer)                 │
├─────────────────────────────────────────────────────────────┤
│  ML Models │  Data Pipeline │  Backend API  │  Frontend    │
│  (PyTorch) │  (Supabase)    │  (Express)    │  (React)     │
└─────────────────────────────────────────────────────────────┘
```

## Setup Instructions

### 1. Prerequisites

- Node.js 18+
- QuantDesk backend running on port 3000
- MIKEY-AI system running on port 3002
- Redis (optional, for caching)

### 2. Installation

```bash
cd integration/mikey-bridge
npm install
```

### 3. Configuration

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Start the Bridge Service

```bash
# Development
npm run dev

# Production
npm run build
npm start
```

## API Endpoints

### Health Check
```
GET /health
```

### Unified Market Analysis
```
POST /api/analyze
{
  "symbol": "SOL/USDT",
  "analysisType": "comprehensive",
  "includeMLPredictions": true,
  "includeArbitrage": true,
  "includeWhaleTracking": true
}
```

### ML Predictions with Market Context
```
POST /api/predict
{
  "symbol": "SOL/USDT",
  "timeframe": "15m",
  "includeMarketData": true,
  "includeSentiment": true,
  "includeWhaleActivity": true
}
```

### Arbitrage Opportunities
```
GET /api/arbitrage/SOL/USDT?minSpreadPercent=0.1
```

### Whale Tracking with ML Impact
```
GET /api/whales/SOL/USDT?threshold=100000
```

### Natural Language Queries
```
POST /api/query
{
  "query": "What are the best arbitrage opportunities for SOL right now?",
  "context": {},
  "includeMLData": true,
  "includeMarketData": true,
  "includeArbitrageData": true
}
```

### Real-time Data Stream
```
GET /api/stream/SOL/USDT
```

## Usage Examples

### 1. Comprehensive Analysis

```javascript
const response = await fetch('http://localhost:3001/api/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    symbol: 'SOL/USDT',
    analysisType: 'comprehensive',
    includeMLPredictions: true,
    includeArbitrage: true,
    includeWhaleTracking: true
  })
});

const analysis = await response.json();
console.log('Recommendation:', analysis.recommendations.action);
console.log('Confidence:', analysis.recommendations.confidence);
```

### 2. ML Predictions with Market Context

```javascript
const response = await fetch('http://localhost:3001/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    symbol: 'SOL/USDT',
    timeframe: '15m',
    includeMarketData: true,
    includeSentiment: true,
    includeWhaleActivity: true
  })
});

const prediction = await response.json();
console.log('Consensus Prediction:', prediction.consensus.prediction);
console.log('Confidence:', prediction.consensus.confidence);
```

### 3. Natural Language Queries

```javascript
const response = await fetch('http://localhost:3001/api/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "Show me whale activity for SOL and how it might impact ML predictions",
    includeMLData: true,
    includeMarketData: true
  })
});

const aiResponse = await response.json();
console.log('AI Response:', aiResponse.response);
```

## Integration Benefits

### MIKEY-AI Brings:
- 🧠 AI-powered market analysis (natural language queries)
- 📊 100+ exchange data via CCXT
- 🐋 Whale tracking and smart money intelligence
- 💥 Cross-platform liquidation detection
- 🔄 Arbitrage opportunity detection
- 📈 Technical analysis and sentiment analysis

### QuantDesk Brings:
- 🤖 Trained ML models (Lorentzian Classifier - 53.5% win rate)
- 📈 Backtesting framework with walk-forward optimization
- 🗄️ Database infrastructure (Supabase/Neon)
- 🎯 Proven trading strategies

### Combined Power:
- **Unified Intelligence**: Single API for all trading intelligence
- **Enhanced ML Predictions**: ML models enhanced with market context
- **Cross-Platform Analysis**: CEX + DEX + ML predictions
- **Real-time Insights**: Live data with AI analysis
- **Risk Assessment**: Comprehensive risk analysis across all data sources

## Development

### Project Structure

```
src/
├── index.ts                    # Main bridge service
├── clients/
│   ├── QuantDeskClient.ts     # QuantDesk API client
│   └── MikeyAIClient.ts       # MIKEY-AI API client
├── services/
│   ├── TradingIntelligenceService.ts  # Main intelligence service
│   └── ArbitrageDetectionService.ts   # Arbitrage detection
├── config/
│   └── index.ts               # Configuration
└── utils/
    └── logger.ts              # Logging utility
```

### Adding New Features

1. **New Data Sources**: Add to respective clients
2. **New Analysis Types**: Extend TradingIntelligenceService
3. **New API Endpoints**: Add routes to main index.ts
4. **New ML Models**: Integrate via QuantDeskClient

## Monitoring

The bridge service includes comprehensive logging and monitoring:

- **Health Checks**: Service status monitoring
- **Performance Metrics**: Response times and success rates
- **Error Tracking**: Detailed error logging
- **Data Quality**: Data source validation

## Security

- **API Key Authentication**: Secure communication with both systems
- **Input Validation**: Sanitized inputs and outputs
- **Rate Limiting**: Protection against abuse
- **Error Handling**: Secure error responses

## Next Steps

1. **Deploy Bridge Service**: Set up production deployment
2. **Frontend Integration**: Connect to QuantDesk frontend
3. **Real-time Dashboard**: Build unified trading dashboard
4. **Mobile App**: Extend to mobile applications
5. **Enterprise Features**: Add advanced analytics and reporting

## Support

For issues or questions:
- Check logs in `logs/` directory
- Verify service connectivity with `/health` endpoint
- Review configuration in `.env` file
- Monitor API response times and error rates
