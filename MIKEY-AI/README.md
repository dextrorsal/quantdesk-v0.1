# MIKEY-AI Service

The QuantDesk MIKEY-AI service is a LangChain-powered AI trading assistant that provides market analysis, trading strategies, and risk management.

## 🚀 Quick Start

```bash
cd MIKEY-AI
pnpm install
pnpm run dev
```

## 🏗️ Architecture

### Tech Stack
- **LangChain** - AI agent framework
- **OpenAI GPT-4** - Primary language model
- **Anthropic Claude** - Alternative language model
- **TypeScript** - Type safety
- **Express.js** - API server
- **WebSocket** - Real-time communication

### Key Features
- **Market Analysis** - AI-powered market sentiment analysis
- **Trading Strategy** - Automated strategy generation
- **Risk Management** - Portfolio risk assessment
- **Multi-LLM Routing** - Intelligent model selection
- **Real-time Updates** - Live market data integration
- **Memory Management** - Context-aware conversations

## 📁 Structure

```
MIKEY-AI/
├── src/
│   ├── agents/        # AI agent implementations
│   │   ├── common/   # Public agent utilities
│   │   └── trading/  # Trading agents (proprietary)
│   ├── services/     # AI service layer
│   ├── middleware/   # Express middleware
│   ├── routes/       # API routes
│   ├── utils/        # Helper functions
│   └── types/        # TypeScript types
├── tests/            # Test files
└── dist/            # Build output
```

## 🔧 Development

### Environment Setup
```bash
cp .env.example .env
# Configure your environment variables
```

### Available Scripts
- `pnpm run dev` - Start development server
- `pnpm run build` - Build for production
- `pnpm run start` - Start production server
- `pnpm run test` - Run tests
- `pnpm run lint` - Run ESLint

### Agent Development
```typescript
// Example agent structure
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';

export class MarketAnalysisAgent {
  private llm: ChatOpenAI;

  constructor(apiKey: string) {
    this.llm = new ChatOpenAI({
      openAIApiKey: apiKey,
      modelName: 'gpt-4',
      temperature: 0.1
    });
  }

  async analyzeMarket(symbol: string, data: any): Promise<any> {
    const prompt = PromptTemplate.fromTemplate(`
      Analyze the market data for {symbol}: {data}
    `);
    
    const chain = prompt.pipe(this.llm);
    return await chain.invoke({ symbol, data });
  }
}
```

## 🌐 Public Agents

The following agents are available for community use:

### Common Agents (`src/agents/common/`)
- **MarketAnalysisAgent** - Market sentiment analysis
- **RiskManagementAgent** - Portfolio risk assessment
- **DataAnalysisAgent** - Market data interpretation
- **UtilityAgent** - General purpose utilities

### Services (`src/services/`)
- **LLMService** - Language model management
- **MemoryService** - Conversation memory
- **ToolService** - Agent tool management
- **RoutingService** - Multi-LLM routing

### Utilities (`src/utils/`)
- **PromptTemplates** - Reusable prompt templates
- **AgentFactory** - Agent creation utilities
- **MemoryManager** - Memory management
- **ToolRegistry** - Tool registration system

## 🔒 Security

- **API Key Management** - Secure LLM API key handling
- **Input Sanitization** - Clean user inputs
- **Rate Limiting** - Prevent API abuse
- **Memory Isolation** - User session isolation
- **Audit Logging** - AI interaction logging

## 📚 Examples

See `examples/mikey-ai-agents.ts` for comprehensive agent examples.

## 🧪 Testing

```bash
pnpm run test        # Run unit tests
pnpm run test:watch  # Run tests in watch mode
pnpm run coverage    # Generate coverage report
```

## 🚀 Deployment

The MIKEY-AI service is deployed to Vercel with automatic deployments from the main branch.

### Environment Configuration
- **Development** - Local development with OpenAI API
- **Staging** - Staging environment with test models
- **Production** - Production environment with production models

## 📖 API Documentation

### Agent Endpoints
```typescript
POST /api/agents/analyze        # Market analysis
POST /api/agents/strategy       # Strategy generation
POST /api/agents/risk           # Risk assessment
POST /api/agents/chat           # General chat
```

### WebSocket Events
```typescript
// Real-time agent responses
ws.on('agent_response', (data) => {
  console.log('Agent response:', data);
});
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `AI_BASE_URL` - AI service base URL
- `PORT` - Server port (default: 3000)

### Model Configuration
- **Primary Model** - GPT-4 for complex analysis
- **Secondary Model** - Claude for alternative perspectives
- **Fallback Model** - GPT-3.5 for simple tasks
- **Temperature** - Configurable creativity levels

## 🐛 Troubleshooting

### Common Issues
1. **API Key Errors** - Check LLM API key configuration
2. **Model Errors** - Verify model availability
3. **Memory Issues** - Check memory configuration
4. **Rate Limiting** - Adjust API rate limits

### Debug Mode
```bash
DEBUG=mikey:* pnpm run dev
```

### Health Check
```bash
curl http://localhost:3000/api/health
```

## 📊 Monitoring

### Metrics
- **Agent Performance** - Response time and accuracy
- **Model Usage** - LLM API usage statistics
- **Memory Usage** - Conversation memory consumption
- **Error Rate** - Agent error frequency

### Logging
- **Agent Interactions** - Detailed conversation logs
- **Model Calls** - LLM API call logs
- **Performance Metrics** - Response time tracking
- **Error Tracking** - Detailed error logs

## 🤖 Agent Capabilities

### Market Analysis
- **Sentiment Analysis** - Bullish/bearish/neutral assessment
- **Technical Analysis** - Chart pattern recognition
- **Fundamental Analysis** - Market data interpretation
- **Confidence Scoring** - Analysis confidence levels

### Trading Strategy
- **Strategy Generation** - Custom trading strategies
- **Entry Conditions** - When to enter positions
- **Exit Conditions** - When to exit positions
- **Risk Management** - Position sizing and stop losses

### Risk Management
- **Portfolio Assessment** - Overall risk evaluation
- **Position Analysis** - Individual position risk
- **Correlation Analysis** - Asset correlation assessment
- **Liquidation Risk** - Liquidation probability

## 📄 License

This MIKEY-AI code is part of QuantDesk and is licensed under Apache License 2.0.