# MIKEY-AI Service

## 🤖 AI-Powered Trading Assistant

MIKEY-AI is QuantDesk's intelligent trading assistant, providing AI-powered market analysis, trading insights, and natural language trading commands through advanced language models.

## 🛠️ Technology Stack

- **LangChain** - LLM application framework
- **TypeScript** - Type-safe development
- **Multi-LLM Routing** - OpenAI, Anthropic, Google AI
- **Vector Databases** - Semantic search and memory
- **WebSocket** - Real-time communication

## 🎯 Key Features

### AI Trading Assistant
- Natural language trading commands
- Market analysis and insights
- Risk assessment and recommendations
- Trading strategy suggestions

### Multi-LLM Integration
- OpenAI GPT models
- Anthropic Claude models
- Google Gemini models
- Intelligent model routing

### Market Intelligence
- Real-time market analysis
- Sentiment analysis
- Technical indicator interpretation
- Risk management insights

## 📁 Project Structure

```
MIKEY-AI/
├── src/
│   ├── agents/        # AI agent implementations
│   ├── chains/        # LangChain workflows
│   ├── memory/        # Conversation memory
│   ├── tools/         # Trading tools and functions
│   ├── models/        # LLM model configurations
│   ├── services/      # Business logic services
│   └── utils/         # Utility functions
├── examples/          # Integration examples
└── package.json       # Dependencies and scripts
```

## 🚀 Getting Started

### Prerequisites
- Node.js 20+
- pnpm package manager
- API keys for LLM providers

### Installation
```bash
cd MIKEY-AI
pnpm install
```

### Environment Setup
```bash
cp .env.example .env
# Fill in your LLM API keys
```

### Development
```bash
pnpm run dev
```

## 🔧 Configuration

### Environment Variables
```bash
# LLM Providers
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Backend Integration
BACKEND_API_URL=http://localhost:3002
BACKEND_WS_URL=ws://localhost:3002

# AI Configuration
DEFAULT_MODEL=gpt-4-turbo
FALLBACK_MODEL=gpt-3.5-turbo
MAX_TOKENS=4000
TEMPERATURE=0.7
```

## 🤖 AI Capabilities

### Trading Analysis
- Market trend analysis
- Technical indicator interpretation
- Risk assessment
- Portfolio optimization suggestions

### Natural Language Processing
- Trading command interpretation
- Market question answering
- Strategy explanation
- Risk explanation

### Memory and Context
- Conversation history
- User preference learning
- Market context awareness
- Personalized recommendations

## 🔄 Integration Patterns

### Backend Communication
- REST API for data access
- WebSocket for real-time updates
- Event-driven responses
- Asynchronous processing

### Frontend Integration
- Chat interface
- Voice commands
- Real-time responses
- Context-aware suggestions

## 📚 Examples

See `examples/mikey-ai-agents.ts` for:
- Agent implementation patterns
- Tool integration examples
- Memory management
- Multi-LLM routing

## 🧠 AI Models

### Supported Models
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3, Claude-2
- **Google**: Gemini Pro, Gemini Ultra

### Model Selection
- Automatic model routing
- Fallback mechanisms
- Performance optimization
- Cost management

## 🔧 Tools and Functions

### Trading Tools
- Market data analysis
- Position calculations
- Risk metrics
- Portfolio analysis

### External Integrations
- Price feed access
- News sentiment analysis
- Social media monitoring
- Economic indicators

## 🧪 Testing

### Test Structure
```bash
tests/
├── unit/           # Unit tests for agents
├── integration/    # Integration tests
├── e2e/           # End-to-end AI tests
└── fixtures/      # Test data and prompts
```

### Running Tests
```bash
# All tests
pnpm run test

# AI-specific tests
pnpm run test:ai

# Performance tests
pnpm run test:performance
```

## 📈 Performance

### Optimization Strategies
- Model response caching
- Token usage optimization
- Parallel processing
- Memory management

### Monitoring
- Response times
- Token usage
- Model performance
- Error rates

## 🔒 Security

### API Security
- Secure API key management
- Rate limiting
- Input sanitization
- Output validation

### Data Privacy
- Conversation encryption
- User data protection
- Privacy compliance
- Audit logging

## 🚀 Deployment

### Production Setup
- Environment configuration
- Model API keys
- Scaling configuration
- Monitoring setup

### Deployment Options
- Vercel (serverless)
- Railway
- Docker containers
- Kubernetes

## 📚 Integration Examples

### Basic Agent Setup
```typescript
import { TradingAgent } from './agents/TradingAgent';

const agent = new TradingAgent({
  model: 'gpt-4-turbo',
  tools: ['market-analysis', 'risk-assessment'],
  memory: true
});

const response = await agent.analyze('What is the current market sentiment?');
```

### Custom Tools
```typescript
import { createTool } from './tools/ToolFactory';

const customTool = createTool({
  name: 'portfolio-analyzer',
  description: 'Analyzes portfolio performance',
  execute: async (portfolio) => {
    // Custom analysis logic
  }
});
```

## 🔧 Development Tools

### Debugging
- Agent conversation logs
- Tool execution tracking
- Model performance metrics
- Error analysis

### Code Quality
- TypeScript strict mode
- ESLint configuration
- Automated testing
- Documentation generation

## 🤝 Contributing

### Development Workflow
1. Create feature branch
2. Implement agent or tool
3. Add comprehensive tests
4. Update documentation
5. Submit pull request

### Code Standards
- LangChain best practices
- TypeScript patterns
- AI safety guidelines
- Performance optimization

## 📞 Support

For MIKEY-AI specific questions:
- LangChain documentation
- OpenAI API documentation
- Anthropic API documentation
- Google AI documentation

---

*MIKEY-AI provides intelligent, context-aware trading assistance that learns from user interactions and provides personalized insights for better trading decisions.*
