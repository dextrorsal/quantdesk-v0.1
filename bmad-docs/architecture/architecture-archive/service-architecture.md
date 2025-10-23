# Service Architecture

## Backend Service (Port 3002)
- **API Gateway**: Centralized API management
- **Database Service**: Supabase abstraction layer
- **Oracle Integration**: Pyth Network price feeds
- **Authentication**: Multi-factor authentication
- **Rate Limiting**: Tiered rate limits
- **Error Handling**: Custom error classes
- **Social Media Integration**: Twitter API, sentiment analysis
- **Alpha Channel Integration**: Discord/Telegram APIs

## Frontend Service (Port 3001)
- **Trading Interface**: Professional trading terminal
- **Portfolio Management**: Real-time portfolio tracking
- **User Dashboard**: Account management and analytics
- **Responsive Design**: Mobile and desktop optimized
- **Unified Data Dashboard**: News, sentiment, social media, alpha channels
- **Real-time Updates**: Live data from all integrated sources

## MIKEY-AI Service (Port 3000)
- **AI Trading Agent**: LangChain-powered assistant
- **Market Analysis**: Real-time market intelligence
- **Trading Recommendations**: AI-powered suggestions
- **Multi-LLM Routing**: Intelligent LLM selection
- **Sentiment Analysis**: News and social media sentiment
- **Alpha Channel Processing**: Discord/Telegram message analysis

## Data Ingestion Service (Port 3003)
- **Real-time Data**: Market data collection
- **Data Processing**: Data normalization and storage
- **Pipeline Management**: Data flow orchestration
- **Monitoring**: Data quality and pipeline health
- **Social Media Feeds**: Twitter, Reddit, Discord integration
- **News Feeds**: Real-time news aggregation and processing
