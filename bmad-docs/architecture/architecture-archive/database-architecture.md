# Database Architecture

## Supabase Integration
- **PostgreSQL**: Primary database
- **Real-time Subscriptions**: Live data updates
- **Row Level Security**: Data access control
- **API Integration**: RESTful and GraphQL APIs

## Database Schema

### Core Tables
- **users**: Wallet-based authentication with KYC, risk levels, referral tracking
- **markets**: Perpetual contract configuration with Pyth price feeds
- **positions**: Trading positions with health factors and liquidation tracking
- **orders**: Advanced order types (market, limit, stop-loss, take-profit, trailing-stop)
- **trades**: Trade execution history with PnL tracking
- **user_balances**: Multi-asset balance management
- **funding_payments**: Funding rate calculations and payments
- **liquidations**: Liquidation events and insurance fund usage
- **chat_channels**: Multi-channel chat system
- **chat_messages**: Message history with mentions and system announcements
- **system_events**: Monitoring and debugging events

### AI Tools Integration Tables
- **news_articles**: News articles with sentiment analysis
- **social_media_posts**: Twitter/Reddit posts with sentiment
- **alpha_channel_messages**: Discord/Telegram messages
- **sentiment_scores**: Aggregated sentiment data
- **ai_insights**: MIKEY-AI generated insights and recommendations

### Data Types
- **Custom ENUMs**: `position_side`, `order_type`, `order_status`, `trade_side`, `liquidation_type`
- **JSONB Fields**: Flexible metadata storage for extensibility
- **TimescaleDB**: Time-series data for market data and analytics
- **UUID Primary Keys**: Distributed system compatibility

### Performance Optimizations
- **Indexes**: Optimized for trading queries (user_id, market_id, created_at)
- **Partitioning**: Time-based partitioning for high-volume tables
- **Connection Pooling**: Efficient database connection management
