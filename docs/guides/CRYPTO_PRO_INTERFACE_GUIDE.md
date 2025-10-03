# QuantDesk Crypto Pro Interface Guide

## Overview

The QuantDesk Crypto Pro Interface is a comprehensive terminal-style trading platform designed specifically for cryptocurrency trading. It provides a powerful command-driven interface with multiple windows for real-time market data, trading, analysis, and portfolio management.

## Architecture

### Backup System
- **Stock Version Backup**: `frontend/src/pro/index-stock-backup.tsx` - Original stock-focused interface preserved
- **Crypto Version**: `frontend/src/pro/index.tsx` - Current crypto-focused interface

### Key Components
1. **Command System**: Terminal-style command interface with backtick (`) menu
2. **Window Management**: Multi-window system with drag, resize, and minimize capabilities
3. **Real-time Data**: Live crypto market data integration
4. **Trading Interface**: Order placement and position management
5. **Analysis Tools**: Technical analysis, correlation, and on-chain data

## Command Reference

### Core Trading & Market Data Commands

#### `QM` - Quote Monitor
- **Description**: Real-time crypto prices and market data
- **Window Size**: 400x600px
- **Features**: Live price updates, volume, change percentages
- **Usage**: Type `QM` and press Enter

#### `CHART` - Advanced Charting
- **Description**: Advanced crypto charting with technical indicators
- **Window Size**: 800x500px
- **Features**: Candlestick charts, technical indicators, multiple timeframes
- **Usage**: Type `CHART` and press Enter
- **Default Pair**: BTC/USDT

#### `ORDER` - Place Trading Order
- **Description**: Place buy/sell orders for crypto pairs
- **Window Size**: 350x400px
- **Features**: Order types (LIMIT, MARKET), position sizing, risk management
- **Usage**: Type `ORDER` and press Enter

#### `POSITIONS` - Current Positions
- **Description**: View current trading positions and P&L
- **Window Size**: 500x350px
- **Features**: Real-time P&L, position sizes, entry prices
- **Usage**: Type `POSITIONS` and press Enter

#### `PF` - Portfolio Performance
- **Description**: Portfolio analytics and performance metrics
- **Window Size**: 600x400px
- **Features**: Total value, returns, asset allocation, trade history
- **Usage**: Type `PF` and press Enter

#### `AL` - Price Alerts & Notifications
- **Description**: Set price alerts and receive notifications
- **Window Size**: 500x300px
- **Features**: Price alerts, volume alerts, news notifications
- **Usage**: Type `AL` and press Enter

#### `BT` - Strategy Backtesting
- **Description**: Backtest trading strategies
- **Window Size**: 600x400px
- **Features**: Historical data testing, performance metrics, strategy optimization
- **Usage**: Type `BT` and press Enter

### Crypto Market Analysis Commands

#### `VOLUME` - Trading Volume Analysis
- **Description**: Analyze trading volume patterns
- **Window Size**: 450x300px
- **Features**: Volume trends, top pairs by volume, volume profile
- **Usage**: Type `VOLUME` and press Enter

#### `FEAR` - Fear & Greed Index
- **Description**: Market sentiment indicator
- **Window Size**: 300x200px
- **Features**: Current sentiment, historical data, trend analysis
- **Usage**: Type `FEAR` and press Enter

#### `CORR` - Crypto Correlation Analysis
- **Description**: Analyze correlations between crypto pairs
- **Window Size**: 500x350px
- **Features**: Correlation matrix, pair analysis, timeframe selection
- **Usage**: Type `CORR` and press Enter

#### `FLOW` - On-Chain Data & Whale Movements
- **Description**: On-chain analytics and whale transaction tracking
- **Window Size**: 500x350px
- **Features**: Whale transactions, exchange flows, on-chain metrics
- **Usage**: Type `FLOW` and press Enter

#### `DEFI` - DeFi Protocols & Yield Farming
- **Description**: DeFi protocol analysis and yield farming opportunities
- **Window Size**: 500x350px
- **Features**: TVL data, APY rates, protocol rankings, risk assessment
- **Usage**: Type `DEFI` and press Enter

#### `NFT` - NFT Market Analysis
- **Description**: NFT market trends and collection analysis
- **Window Size**: 500x350px
- **Features**: Floor prices, volume data, collection rankings
- **Usage**: Type `NFT` and press Enter

### News & Research Commands

#### `N` - Real-time Crypto News
- **Description**: Live crypto news feed
- **Window Size**: 500x300px
- **Features**: Real-time updates, multiple sources, filtering
- **Usage**: Type `N` and press Enter

#### `CN` - CoinDesk News Feed
- **Description**: CoinDesk news integration
- **Window Size**: 500x300px
- **Features**: CoinDesk articles, market analysis, regulatory news
- **Usage**: Type `CN` and press Enter

#### `CT` - CoinTelegraph News
- **Description**: CoinTelegraph news integration
- **Window Size**: 500x300px
- **Features**: CoinTelegraph articles, technical analysis, market updates
- **Usage**: Type `CT` and press Enter

#### `TB` - The Block News
- **Description**: The Block news integration
- **Window Size**: 500x300px
- **Features**: The Block articles, institutional news, DeFi updates
- **Usage**: Type `TB` and press Enter

#### `RES` - Crypto Research Reports
- **Description**: Professional crypto research and analysis
- **Window Size**: 500x350px
- **Features**: Research reports, market analysis, investment insights
- **Usage**: Type `RES` and press Enter

#### `WHITEPAPER` - Token Whitepapers
- **Description**: Access to token whitepapers and technical documentation
- **Window Size**: 500x400px
- **Features**: Whitepaper library, technical specs, project details
- **Usage**: Type `WHITEPAPER` and press Enter

### Account & Settings Commands

#### `ACCT` - Account Management
- **Description**: User account settings and information
- **Window Size**: 500x300px
- **Features**: Profile management, subscription details, usage stats
- **Usage**: Type `ACCT` and press Enter

#### `WALLET` - Wallet Connections & Balances
- **Description**: Manage connected wallets and view balances
- **Window Size**: 400x300px
- **Features**: Wallet connections, balance tracking, transaction history
- **Usage**: Type `WALLET` and press Enter

#### `API` - API Key Management
- **Description**: Manage API keys for exchanges and data providers
- **Window Size**: 500x350px
- **Features**: API key storage, permissions, usage monitoring
- **Usage**: Type `API` and press Enter

#### `SETTINGS` - User Settings & Preferences
- **Description**: Configure trading and display preferences
- **Window Size**: 500x400px
- **Features**: Trading settings, notification preferences, display options
- **Usage**: Type `SETTINGS` and press Enter

### Tools & Utilities Commands

#### `CALC` - Crypto Calculator & Converters
- **Description**: Financial calculators for crypto trading
- **Window Size**: 400x300px
- **Features**: Position sizing, profit/loss, currency conversion, DCA calculator
- **Usage**: Type `CALC` and press Enter

#### `GAS` - Gas Fee Tracker
- **Description**: Monitor gas fees across different networks
- **Window Size**: 350x250px
- **Features**: Ethereum, Polygon, Arbitrum gas tracking, fee optimization
- **Usage**: Type `GAS` and press Enter

#### `STAKING` - Staking Rewards Calculator
- **Description**: Calculate staking rewards and APY
- **Window Size**: 400x300px
- **Features**: Multiple protocols, APY comparison, reward calculations
- **Usage**: Type `STAKING` and press Enter

#### `NOTE` - Rich Text Notes Editor
- **Description**: Trading notes and analysis documentation
- **Window Size**: 500x400px
- **Features**: Rich text editing, note organization, search functionality
- **Usage**: Type `NOTE` and press Enter

### Social & Community Commands

#### `CHAT` - Live Crypto Chat
- **Description**: Real-time chat with other crypto traders
- **Window Size**: 500x400px
- **Features**: Live messaging, user profiles, chat rooms
- **Usage**: Type `CHAT` and press Enter

#### `TWITTER` - Crypto Twitter Feed
- **Description**: Curated crypto Twitter feed
- **Window Size**: 500x400px
- **Features**: Influencer tweets, market sentiment, trending topics
- **Usage**: Type `TWITTER` and press Enter

#### `REDDIT` - Crypto Reddit Discussions
- **Description**: Reddit crypto community discussions
- **Window Size**: 500x400px
- **Features**: Subreddit feeds, trending discussions, community insights
- **Usage**: Type `REDDIT` and press Enter

### System Commands

#### `HELP` - Crypto Terminal Documentation
- **Description**: Complete command reference and help system
- **Window Size**: 600x500px
- **Features**: Command list, usage examples, keyboard shortcuts
- **Usage**: Type `HELP` and press Enter

#### `S` - Keyboard Shortcuts
- **Description**: Display all available keyboard shortcuts
- **Window Size**: 500x400px
- **Features**: Shortcut reference, quick actions, navigation help
- **Usage**: Type `S` and press Enter

#### `CLEAR` - Clear All Windows
- **Description**: Close all open windows
- **Usage**: Type `CLEAR` and press Enter

#### `LAYOUT` - Save/Load Window Layouts
- **Description**: Save and restore window configurations
- **Window Size**: 400x300px
- **Features**: Layout management, workspace organization
- **Usage**: Type `LAYOUT` and press Enter

#### `ERR` - Report Bugs and Get Support
- **Description**: Bug reporting and technical support
- **Window Size**: 500x400px
- **Features**: Bug reporting, system info, contact support
- **Usage**: Type `ERR` and press Enter

## Crypto Trading Pairs

The interface supports major crypto trading pairs:

### Major Pairs
- **BTC/USDT** - Bitcoin vs Tether
- **ETH/USDT** - Ethereum vs Tether
- **SOL/USDT** - Solana vs Tether
- **ADA/USDT** - Cardano vs Tether
- **DOT/USDT** - Polkadot vs Tether
- **MATIC/USDT** - Polygon vs Tether
- **AVAX/USDT** - Avalanche vs Tether
- **LINK/USDT** - Chainlink vs Tether
- **UNI/USDT** - Uniswap vs Tether
- **ATOM/USDT** - Cosmos vs Tether

### Data Structure
Each trading pair includes:
- **Symbol**: Trading pair identifier
- **Name**: Full token name
- **Price**: Current market price
- **Change**: 24h price change
- **ChangePercent**: 24h percentage change
- **Type**: SPOT (spot trading)
- **Volume**: 24h trading volume

## Window Management

### Window Operations
- **Drag**: Click and drag window title bar to move
- **Resize**: Drag window edges to resize
- **Minimize**: Click minimize button to hide window
- **Close**: Click X button to close window
- **Bring to Front**: Click anywhere on window to bring to front

### Window Sizes
Each command opens windows with optimized sizes:
- **Small**: 300-350px width (FEAR, GAS)
- **Medium**: 400-500px width (QM, WALLET, CALC)
- **Large**: 600-800px width (CHART, PF, HELP)

### Z-Index Management
- Windows automatically manage z-index for proper layering
- Clicking a window brings it to the front
- New windows appear on top of existing ones

## Keyboard Shortcuts

### Navigation
- **Backtick (`)**: Open command menu
- **Arrow Keys**: Navigate command menu
- **Enter**: Execute selected command
- **Escape**: Close command menu
- **Tab**: Auto-complete commands

### Trading Hotkeys
- **Ctrl+B**: Quick buy order
- **Ctrl+S**: Quick sell order
- **Ctrl+C**: Close position
- **Ctrl+A**: Set alert

### Window Management
- **Ctrl+W**: Close current window
- **Ctrl+M**: Minimize current window
- **Ctrl+Shift+C**: Clear all windows
- **Ctrl+L**: Save layout

## Integration Points

### API Endpoints
The interface integrates with various APIs:
- **Market Data**: `/api/data/unified/{symbol}`
- **Portfolio**: `/api/portfolio/*`
- **Alerts**: `/api/alerts/*`
- **Backtesting**: `/api/backtest/*`

### External Services
- **News Feeds**: CoinDesk, CoinTelegraph, The Block
- **Social Media**: Twitter, Reddit integration
- **On-Chain Data**: Blockchain analytics providers
- **DeFi Data**: DeFiPulse, DeFiLlama integration

## Development Notes

### File Structure
```
frontend/src/pro/
├── index.tsx                    # Main crypto pro interface
├── index-stock-backup.tsx      # Backup of stock version
└── components/                  # Additional components (if needed)
```

### Key Functions
- **executeCommand()**: Handles command execution
- **createWindow()**: Creates new windows with crypto-specific sizing
- **fetchChartData()**: Retrieves crypto market data
- **performSearch()**: Searches commands, instruments, and news

### State Management
- **windows**: Array of open windows
- **command**: Current command input
- **marketData**: Real-time market data
- **commandHistory**: Command execution history

## Migration from Stock Version

### What Changed
1. **Commands**: Replaced stock commands with crypto-specific ones
2. **Instruments**: Changed from stocks to crypto trading pairs
3. **News**: Updated to crypto news sources
4. **Window Sizes**: Optimized for crypto trading workflows
5. **Data Sources**: Switched to crypto market data APIs

### What Stayed the Same
1. **Core Architecture**: Terminal interface and window management
2. **User Experience**: Command-driven workflow
3. **Visual Design**: Dark theme and professional styling
4. **Keyboard Shortcuts**: Same navigation patterns

## Future Enhancements

### Planned Features
1. **Advanced Order Types**: Stop-loss, take-profit, trailing stops
2. **Portfolio Analytics**: More detailed performance metrics
3. **Social Trading**: Copy trading and signal sharing
4. **Mobile Support**: Responsive design for mobile devices
5. **Custom Indicators**: User-defined technical indicators
6. **Backtesting Engine**: More sophisticated strategy testing
7. **Risk Management**: Advanced risk controls and position sizing
8. **Multi-Exchange**: Support for multiple exchange connections

### Integration Roadmap
1. **More Exchanges**: Binance, Coinbase, Kraken integration
2. **DeFi Protocols**: Direct DeFi protocol interaction
3. **NFT Marketplaces**: OpenSea, Magic Eden integration
4. **Cross-Chain**: Multi-blockchain support
5. **AI Features**: Machine learning price predictions

## Troubleshooting

### Common Issues
1. **Window Not Responding**: Use `CLEAR` command to reset
2. **Data Not Loading**: Check API connections and network
3. **Command Not Found**: Use `HELP` to see available commands
4. **Performance Issues**: Close unnecessary windows

### Support
- **Bug Reports**: Use `ERR` command
- **Documentation**: Use `HELP` command
- **Community**: Use `CHAT` command

## Conclusion

The QuantDesk Crypto Pro Interface provides a comprehensive, professional-grade trading platform specifically designed for cryptocurrency markets. With its command-driven interface, multi-window system, and extensive crypto-specific features, it offers traders a powerful tool for market analysis, trading execution, and portfolio management.

The interface successfully transforms the original stock-focused terminal into a crypto-native platform while maintaining the professional user experience and powerful functionality that makes it suitable for both retail and institutional traders.
