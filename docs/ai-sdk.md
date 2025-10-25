# QuantDesk AI SDK Documentation

## 🤖 **Complete AI Integration Guide**

QuantDesk's AI SDK provides powerful AI capabilities for trading, portfolio management, and market analysis. This documentation shows you how to integrate AI features without exposing AI internals.

---

## 🚀 **Quick Start**

### **Installation**
```bash
npm install @quantdesk/ai-sdk
```

### **Basic Setup**
```typescript
import { QuantDeskAI } from '@quantdesk/ai-sdk';

const ai = new QuantDeskAI({
  apiKey: process.env.QUANTDESK_AI_KEY,
  environment: 'devnet' // or 'mainnet'
});
```

---

## 📚 **API Reference**

### **QuantDeskAI Class**

#### **Constructor Options**
```typescript
interface QuantDeskAIOptions {
  apiKey: string;
  environment?: 'devnet' | 'mainnet';
  rateLimit?: number; // requests per minute
  timeout?: number; // request timeout in ms
}
```

#### **Methods**

##### **analyzePortfolio(options)**
Analyze portfolio with AI-powered insights.

```typescript
interface PortfolioAnalysisOptions {
  wallet: string;
  includeRiskAssessment?: boolean;
  includeRecommendations?: boolean;
  includeMarketSentiment?: boolean;
  timeframe?: '1h' | '24h' | '7d' | '30d';
}

const analysis = await ai.analyzePortfolio({
  wallet: 'wallet-address',
  includeRiskAssessment: true,
  includeRecommendations: true,
  timeframe: '7d'
});
```

**Returns:**
```typescript
interface PortfolioAnalysis {
  riskScore: number; // 0-100
  recommendations: TradingRecommendation[];
  marketSentiment: MarketSentiment;
  portfolioHealth: PortfolioHealth;
  suggestedActions: SuggestedAction[];
}
```

##### **getTradingRecommendations(options)**
Get AI-powered trading recommendations.

```typescript
interface TradingRecommendationsOptions {
  wallet: string;
  riskTolerance: 'low' | 'medium' | 'high';
  includePositionSizing?: boolean;
  includeStopLoss?: boolean;
  timeframe?: '1h' | '4h' | '24h';
}

const recommendations = await ai.getTradingRecommendations({
  wallet: 'wallet-address',
  riskTolerance: 'medium',
  includePositionSizing: true,
  includeStopLoss: true
});
```

**Returns:**
```typescript
interface TradingRecommendations {
  recommendedPositions: Position[];
  positionSizing: PositionSizing[];
  stopLossLevels: StopLossLevel[];
  takeProfitLevels: TakeProfitLevel[];
  confidence: number; // 0-100
}
```

##### **getMarketSentiment(options)**
Get AI-powered market sentiment analysis.

```typescript
interface MarketSentimentOptions {
  symbols: string[];
  timeframe?: '1h' | '24h' | '7d';
  includeNewsAnalysis?: boolean;
  includeSocialSentiment?: boolean;
  includeTechnicalAnalysis?: boolean;
}

const sentiment = await ai.getMarketSentiment({
  symbols: ['SOL', 'BTC', 'ETH'],
  timeframe: '24h',
  includeNewsAnalysis: true,
  includeSocialSentiment: true
});
```

**Returns:**
```typescript
interface MarketSentiment {
  overallSentiment: 'bullish' | 'bearish' | 'neutral';
  symbolSentiments: Record<string, SymbolSentiment>;
  newsImpact: NewsImpact;
  socialSentiment: SocialSentiment;
  technicalSentiment: TechnicalSentiment;
  confidence: number; // 0-100
}
```

##### **assessRisk(options)**
Get AI-powered risk assessment.

```typescript
interface RiskAssessmentOptions {
  wallet: string;
  includePositionRisk?: boolean;
  includeMarketRisk?: boolean;
  includeLiquidityRisk?: boolean;
}

const riskAssessment = await ai.assessRisk({
  wallet: 'wallet-address',
  includePositionRisk: true,
  includeMarketRisk: true,
  includeLiquidityRisk: true
});
```

**Returns:**
```typescript
interface RiskAssessment {
  overallRisk: number; // 0-100
  positionRisks: PositionRisk[];
  marketRisks: MarketRisk[];
  liquidityRisks: LiquidityRisk[];
  riskMitigation: RiskMitigation[];
}
```

---

## 🔒 **Authentication & Security**

### **API Key Management**
```typescript
// Set API key via environment variable
process.env.QUANTDESK_AI_KEY = 'your-api-key';

// Or pass directly to constructor
const ai = new QuantDeskAI({
  apiKey: 'your-api-key'
});
```

### **Rate Limiting**
```typescript
const ai = new QuantDeskAI({
  apiKey: 'your-api-key',
  rateLimit: 100 // 100 requests per minute
});
```

### **Error Handling**
```typescript
try {
  const analysis = await ai.analyzePortfolio({
    wallet: 'wallet-address'
  });
} catch (error) {
  if (error.code === 'RATE_LIMIT_EXCEEDED') {
    // Handle rate limiting
    console.log('Rate limit exceeded, retrying...');
  } else if (error.code === 'INVALID_API_KEY') {
    // Handle authentication error
    console.log('Invalid API key');
  } else {
    // Handle other errors
    console.error('AI Analysis Error:', error.message);
  }
}
```

---

## 🏆 **Community Points Integration**

### **Points-Based Access**
```typescript
// Check available AI features based on community points
const features = await ai.getAvailableFeatures({
  wallet: 'wallet-address'
});

console.log('Available Features:', features);
// Returns: ['portfolio-analysis', 'sentiment-analysis', 'risk-assessment']
```

### **Tier-Based Access**
```typescript
// Different AI capabilities based on community tier
const tier = await ai.getCommunityTier({
  wallet: 'wallet-address'
});

console.log('Community Tier:', tier);
// Returns: 'bronze' | 'silver' | 'gold' | 'platinum'
```

---

## 📊 **Usage Examples**

### **Complete Trading Bot with AI**
```typescript
import { QuantDeskAI } from '@quantdesk/ai-sdk';
import { QuantDeskSDK } from '@quantdesk/sdk';

class AITradingBot {
  private ai: QuantDeskAI;
  private sdk: QuantDeskSDK;

  constructor(apiKey: string, wallet: string) {
    this.ai = new QuantDeskAI({ apiKey });
    this.sdk = new QuantDeskSDK({ wallet });
  }

  async executeTrade() {
    // Get AI recommendations
    const recommendations = await this.ai.getTradingRecommendations({
      wallet: this.sdk.wallet.publicKey.toString(),
      riskTolerance: 'medium'
    });

    // Execute trades based on AI recommendations
    for (const recommendation of recommendations.recommendedPositions) {
      if (recommendation.confidence > 80) {
        await this.sdk.placeOrder({
          symbol: recommendation.symbol,
          side: recommendation.side,
          amount: recommendation.amount,
          stopLoss: recommendation.stopLoss
        });
      }
    }
  }
}
```

### **Portfolio Monitoring with AI**
```typescript
class AIPortfolioMonitor {
  private ai: QuantDeskAI;

  constructor(apiKey: string) {
    this.ai = new QuantDeskAI({ apiKey });
  }

  async monitorPortfolio(wallet: string) {
    // Get portfolio analysis
    const analysis = await this.ai.analyzePortfolio({
      wallet,
      includeRiskAssessment: true,
      includeRecommendations: true
    });

    // Check risk levels
    if (analysis.riskScore > 80) {
      console.log('High risk detected, consider reducing position sizes');
    }

    // Execute recommended actions
    for (const action of analysis.suggestedActions) {
      if (action.priority === 'high') {
        console.log(`High priority action: ${action.description}`);
      }
    }
  }
}
```

---

## 🔧 **Advanced Configuration**

### **Custom Timeouts**
```typescript
const ai = new QuantDeskAI({
  apiKey: 'your-api-key',
  timeout: 30000 // 30 seconds
});
```

### **Retry Logic**
```typescript
const ai = new QuantDeskAI({
  apiKey: 'your-api-key',
  retries: 3,
  retryDelay: 1000 // 1 second
});
```

### **Environment Configuration**
```typescript
const ai = new QuantDeskAI({
  apiKey: 'your-api-key',
  environment: 'devnet', // Use devnet for testing
  baseUrl: 'https://ai-api.quantdesk.com' // Custom API endpoint
});
```

---

## 📈 **Performance Optimization**

### **Caching**
```typescript
// Enable caching for repeated requests
const ai = new QuantDeskAI({
  apiKey: 'your-api-key',
  cache: true,
  cacheTTL: 300 // 5 minutes
});
```

### **Batch Requests**
```typescript
// Batch multiple requests for better performance
const batchResults = await ai.batchRequest([
  { method: 'analyzePortfolio', params: { wallet: 'wallet1' } },
  { method: 'analyzePortfolio', params: { wallet: 'wallet2' } },
  { method: 'getMarketSentiment', params: { symbols: ['SOL', 'BTC'] } }
]);
```

---

## 🆘 **Support & Resources**

- **[GitHub Repository](https://github.com/quantdesk/ai-sdk)** - Source code and issues
- **[Discord Community](https://discord.gg/quantdesk)** - Community support
- **[Documentation](https://docs.quantdesk.com/ai)** - Complete documentation
- **[Examples](https://github.com/quantdesk/examples)** - More integration examples

---

## 🔒 **Privacy & Security**

- **No AI Model Exposure**: AI models and algorithms remain private
- **Secure API**: All AI interactions through secure API endpoints
- **Data Protection**: User data protected with enterprise-grade security
- **Rate Limiting**: Prevents abuse and ensures fair usage
- **Audit Trail**: All AI interactions logged for security and compliance
