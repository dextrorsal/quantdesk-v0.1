# Data Models

## User Model

**Purpose:** Represents traders and users of the platform with authentication, profile, and trading preferences

**Key Attributes:**
- `id`: UUID - Unique user identifier
- `wallet_address`: String - Solana wallet address (primary trading identifier)
- `email`: String - User contact information
- `created_at`: Timestamp - Account creation date
- `last_login`: Timestamp - Last authentication timestamp
- `trading_preferences`: JSON - User-specific trading settings and risk parameters
- `kyc_status`: Enum - Know Your Customer verification status
- `risk_level`: Enum - User risk assessment level (conservative, moderate, aggressive)

**Relationships:**
- One-to-Many with TradingPosition (user can have multiple positions)
- One-to-Many with TradingOrder (user can place multiple orders)
- One-to-Many with Portfolio (user has portfolio history)

## TradingPosition Model

**Purpose:** Represents active perpetual trading positions with real-time P&L and risk metrics

**Key Attributes:**
- `id`: UUID - Unique position identifier
- `user_id`: UUID - Foreign key to User
- `market_symbol`: String - Trading pair (e.g., "SOL-PERP", "BTC-PERP")
- `side`: Enum - Position side (long/short)
- `size`: Decimal - Position size in base currency
- `entry_price`: Decimal - Average entry price
- `current_price`: Decimal - Current market price (from Pyth)
- `unrealized_pnl`: Decimal - Current unrealized profit/loss
- `margin_used`: Decimal - Margin allocated to position
- `leverage`: Decimal - Position leverage multiplier
- `created_at`: Timestamp - Position opening time
- `updated_at`: Timestamp - Last position update

**Relationships:**
- Many-to-One with User (position belongs to user)
- One-to-Many with PositionHistory (position has history snapshots)

## TradingOrder Model

**Purpose:** Represents trading orders (market, limit, stop) with execution tracking

**Key Attributes:**
- `id`: UUID - Unique order identifier
- `user_id`: UUID - Foreign key to User
- `position_id`: UUID - Foreign key to TradingPosition (if applicable)
- `order_type`: Enum - Market, Limit, Stop, Stop-Limit
- `side`: Enum - Buy/Sell
- `market_symbol`: String - Trading pair
- `quantity`: Decimal - Order quantity
- `price`: Decimal - Limit price (for limit orders)
- `status`: Enum - Pending, Filled, Cancelled, Rejected
- `filled_quantity`: Decimal - Quantity filled
- `average_fill_price`: Decimal - Average execution price
- `created_at`: Timestamp - Order placement time
- `executed_at`: Timestamp - Order execution time

**Relationships:**
- Many-to-One with User (order belongs to user)
- Many-to-One with TradingPosition (order may affect position)

## MarketData Model

**Purpose:** Stores real-time and historical market data from Pyth Network oracle

**Key Attributes:**
- `id`: UUID - Unique data point identifier
- `symbol`: String - Market symbol (e.g., "SOL", "BTC")
- `price`: Decimal - Current price
- `timestamp`: Timestamp - Data timestamp
- `confidence`: Decimal - Price confidence interval
- `exponent`: Integer - Price decimal exponent
- `source`: String - Data source identifier
- `staleness_threshold`: Integer - Maximum acceptable staleness (seconds)

**Relationships:**
- Referenced by TradingPosition (for current_price)
- Referenced by TradingOrder (for execution prices)

## Portfolio Model

**Purpose:** Tracks user portfolio performance, balances, and trading history

**Key Attributes:**
- `id`: UUID - Unique portfolio snapshot identifier
- `user_id`: UUID - Foreign key to User
- `total_balance`: Decimal - Total portfolio value
- `available_balance`: Decimal - Available for trading
- `margin_used`: Decimal - Margin currently in use
- `unrealized_pnl`: Decimal - Total unrealized P&L
- `realized_pnl`: Decimal - Total realized P&L
- `snapshot_time`: Timestamp - Portfolio snapshot timestamp
- `trading_fees_paid`: Decimal - Cumulative trading fees

**Relationships:**
- Many-to-One with User (portfolio belongs to user)

## AIAnalysis Model

**Purpose:** Stores MIKEY AI trading analysis, recommendations, and market intelligence

**Key Attributes:**
- `id`: UUID - Unique analysis identifier
- `user_id`: UUID - Foreign key to User (if user-specific)
- `analysis_type`: Enum - Market analysis, trading signal, risk assessment
- `market_symbol`: String - Analyzed trading pair
- `confidence_score`: Decimal - AI confidence in analysis (0-1)
- `recommendation`: Enum - Buy, Sell, Hold, Close
- `reasoning`: Text - AI reasoning and analysis details
- `created_at`: Timestamp - Analysis generation time
- `expires_at`: Timestamp - Analysis validity period

**Relationships:**
- Many-to-One with User (analysis may be user-specific)
- Referenced by TradingOrder (for AI-assisted orders)
