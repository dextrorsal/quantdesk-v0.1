# Portfolio Manager Implementation Specification

## Overview
Professional-grade algorithmic portfolio manager for institutional trading operations. Manages profit harvesting, loss recovery, and capital allocation across multiple strategy pools with focus on capital protection and black swan event mitigation.

## Trader Profile & Requirements
- **Trading Volume**: 20M+ annually (professional trader level)
- **Operation Scale**: Institutional-grade multi-strategy deployment
- **Capital Allocation**: $500-1K initial deployment, scalable architecture
- **Risk Philosophy**: Profit protection over complex risk limits
- **Architecture Goal**: Strategy pool isolation with emergency reserves

## Architecture

### Professional Strategy Pool Structure
```
Bitget Professional Setup (Phase 1 Focus)
├── Manual Trading Account (Discretionary trading)
├── Emergency Reserve Pool (20% - $100-200)
├── Low Frequency Pool (50% - $250-500)
│   ├── 1h/4h Trend Strategies (Cross margin)
│   ├── Daily Swing Strategies (Isolated margin)
│   └── LF Pool Reserve (40% of pool)
├── High Frequency Pool (30% - $150-300)
│   ├── 1m/5m Scalping Bots (Isolated margin)
│   ├── 15m Momentum Bots (Cross margin)
│   └── HF Pool Reserve (30% of pool)
└── Future Expansion Pools (Spot, other exchanges)
```

### Capital Allocation Philosophy
- **20% Emergency Reserves**: Black swan event protection
- **50% Low Frequency**: Proven strategies (1h SOL Lorentzian)
- **30% High Frequency**: Development/testing strategies
- **Profit Protection**: Regular harvesting to prevent total loss
- **No Complex Risk Limits**: Focus on profit preservation over restrictions

### Exchange Integration Strategy
- **Primary**: Bitget (professional volume, sub-account support)
- **Secondary**: Other exchanges as strategies mature
- **Account Types**: Sub-accounts per strategy for isolation
- **API Requirements**: Balance, transfers, position management

## Implementation Requirements

### 1. Database Schema
```sql
-- Portfolio state tracking
CREATE TABLE portfolio_state (
    id INT PRIMARY KEY AUTO_INCREMENT,
    pot_balance DECIMAL(16,8),
    total_allocated DECIMAL(16,8),
    total_pnl DECIMAL(16,8),
    active_bots INT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bot performance tracking
CREATE TABLE bot_performance (
    id INT PRIMARY KEY AUTO_INCREMENT,
    bot_id VARCHAR(50),
    exchange VARCHAR(20),
    current_balance DECIMAL(16,8),
    initial_investment DECIMAL(16,8),
    total_pnl DECIMAL(16,8),
    win_rate DECIMAL(5,2),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(5,2),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transaction log
CREATE TABLE portfolio_transactions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    bot_id VARCHAR(50),
    exchange VARCHAR(20),
    transaction_type ENUM('harvest', 'refill', 'rebalance', 'allocation'),
    amount DECIMAL(16,8),
    pot_balance_before DECIMAL(16,8),
    pot_balance_after DECIMAL(16,8),
    reason TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2. Core Classes

#### PortfolioManager
```python
class PortfolioManager:
    def __init__(self, config):
        self.config = config
        self.exchanges = self._initialize_exchanges()
        self.bots = {}
        self.pot_balance = 0.0
        self.monitoring_interval = 300  # 5 minutes
        
    async def start_monitoring(self):
        """Main monitoring loop"""
        
    async def register_bot(self, bot_id, exchange, initial_investment):
        """Register new bot for monitoring"""
        
    async def get_portfolio_status(self):
        """Get current portfolio state"""
```

#### ProfitHarvester
```python
class ProfitHarvester:
    def __init__(self, portfolio_manager):
        self.pm = portfolio_manager
        self.harvest_threshold = 1.25  # 25% profit
        self.harvest_percentage = 0.80  # Take 80% of excess
        
    async def check_harvest_opportunities(self):
        """Check all bots for harvest opportunities"""
        
    async def harvest_profits(self, bot_id, current_balance):
        """Execute profit harvesting"""
```

#### LossRecoveryManager
```python
class LossRecoveryManager:
    def __init__(self, portfolio_manager):
        self.pm = portfolio_manager
        self.intervention_threshold = 0.25  # 75% loss
        self.refill_target = 0.75  # Refill to 75% of original
        self.minimum_pot_reserve = 200.0  # Always keep reserve
        
    async def check_recovery_needs(self):
        """Check bots needing capital injection"""
        
    async def refill_bot(self, bot_id, target_amount):
        """Execute capital injection"""
```

#### ExchangeCoordinator
```python
class ExchangeCoordinator:
    def __init__(self):
        self.exchange_clients = {}
        self.supported_exchanges = [
            'bitget', 'coinbase', 'kucoin', 'kraken', 
            'jupiter', 'drift', 'binance', 'mexc'
        ]
        
    async def get_bot_balance(self, bot_id, exchange):
        """Get bot balance from specific exchange"""
        
    async def transfer_funds(self, from_account, to_account, amount, exchange):
        """Execute internal transfer"""
        
    async def get_transfer_limits(self, exchange):
        """Get exchange-specific transfer limits"""
```

### 3. Professional Configuration Structure

```python
PORTFOLIO_CONFIG = {
    'capital_allocation': {
        'total_capital': 1000,  # $1K initial deployment
        'emergency_reserve': 0.20,  # 20% - $200
        'lf_pool': 0.50,  # 50% - $500 (proven strategies)
        'hf_pool': 0.30,  # 30% - $300 (development)
    },
    'profit_protection': {
        'lf_harvest_threshold': 1.25,  # 25% profit for LF strategies
        'lf_harvest_percentage': 0.60,  # Take 60% of excess
        'hf_harvest_threshold': 1.15,  # 15% profit for HF strategies
        'hf_harvest_percentage': 0.80,  # Take 80% of excess
        'compound_threshold': 2.0,  # 100% profit = compound mode
    },
    'black_swan_protection': {
        'drawdown_threshold': 0.60,  # 40% loss triggers protection
        'safety_allocation': 0.20,  # Move 20% to reserves
        'no_refill_policy': True,  # Don't throw good money after bad
    },
    'strategy_pools': {
        'low_frequency': {
            'proven_strategies': ['1h_lorentzian_sol', '4h_swing_btc'],
            'margin_mode': 'cross',
            'max_leverage': 75,
            'allocation_per_strategy': 150,  # $150 per strategy
        },
        'high_frequency': {
            'development_strategies': ['5m_scalping_sol', '15m_momentum_eth'],
            'margin_mode': 'isolated',
            'max_leverage': 25,
            'allocation_per_strategy': 100,  # $100 per strategy
        }
    },
    'monitoring': {
        'check_interval': 900,  # 15 minutes
        'profit_harvest_frequency': 'continuous',
        'black_swan_checks': 'every_cycle',
    },
    'exchanges': {
        'bitget': {
            'primary': True,
            'sub_account_support': True,
            'transfer_fee': 0.001,
            'min_transfer': 1.0,
            'api_limits': 'professional_tier'
        }
    }
}
```

### 4. Key Methods to Implement

#### Monitoring Loop
```python
async def portfolio_monitoring_cycle(self):
    """Main monitoring cycle - run every 5 minutes"""
    try:
        # Update all bot balances
        await self.update_bot_balances()
        
        # Check harvest opportunities
        await self.profit_harvester.check_harvest_opportunities()
        
        # Check recovery needs
        await self.loss_recovery.check_recovery_needs()
        
        # Update performance metrics
        await self.update_performance_metrics()
        
        # Log portfolio state
        await self.log_portfolio_state()
        
    except Exception as e:
        logger.error(f"Portfolio monitoring error: {e}")
```

#### Performance-Based Rebalancing
```python
async def rebalance_based_on_performance(self):
    """Reallocate capital based on bot performance"""
    performance_data = await self.get_bot_performance_metrics()
    
    # Calculate performance scores
    for bot_id, metrics in performance_data.items():
        score = self.calculate_performance_score(metrics)
        
        if score > self.config['performance_rebalance_threshold']:
            # Increase allocation
            await self.increase_bot_allocation(bot_id)
        elif score < -self.config['performance_rebalance_threshold']:
            # Decrease allocation
            await self.decrease_bot_allocation(bot_id)
```

### 5. Exchange-Specific Considerations

#### API Rate Limits
- Implement per-exchange rate limiting
- Queue transfers to avoid API limits
- Use exchange-specific error handling

#### Transfer Mechanisms
- **Centralized Exchanges**: Use sub-account transfers
- **DEX (Jupiter/Drift)**: Handle wallet-to-wallet transfers
- **Fee Optimization**: Calculate optimal transfer timing

#### Error Handling
```python
async def safe_transfer(self, from_account, to_account, amount, exchange):
    """Transfer with comprehensive error handling"""
    try:
        # Check balance
        # Validate transfer limits
        # Execute transfer
        # Verify completion
        # Log transaction
    except ExchangeAPIError as e:
        # Handle exchange-specific errors
    except InsufficientFundsError as e:
        # Handle insufficient funds
    except RateLimitError as e:
        # Handle rate limits
```

### 6. Monitoring & Alerting

#### Key Metrics to Track
- Portfolio PnL
- Individual bot performance
- Pot balance utilization
- Transfer success rates
- API error rates

#### Alert Conditions
- Pot balance below minimum reserve
- Bot performance degradation
- Transfer failures
- Unusual loss patterns

### 7. Testing Strategy

#### Unit Tests
- Test profit harvesting logic
- Test loss recovery mechanisms
- Test performance calculations
- Test exchange integrations

#### Integration Tests
- End-to-end portfolio rebalancing
- Multi-exchange coordination
- Error recovery scenarios

#### Simulation Testing
- Backtest with historical bot performance
- Stress test during market volatility
- Test various parameter configurations

### 8. Deployment Considerations

#### Process Management
- Run as separate service from trading bots
- Implement health checks
- Set up automatic restarts

#### Monitoring
- Real-time dashboard for portfolio status
- Log aggregation for debugging
- Performance metrics collection

#### Security
- Secure API key management
- Encrypted database connections
- Transfer amount validation

## Implementation Priority

### Professional Deployment Strategy
1. **Phase 1**: Deploy proven 1h SOL Lorentzian ($250 from LF pool)
2. **Phase 2**: Implement profit harvesting and black swan protection
3. **Phase 3**: Scale proven strategy to multiple symbols (BTC, ETH)
4. **Phase 4**: Deploy HF strategies as development completes
5. **Phase 5**: Cross-pool rebalancing and performance optimization
6. **Phase 6**: Multi-exchange expansion

### Key Architectural Decisions
- **Start Small**: $500-1K total capital for proven concept validation
- **Profit First**: Focus on protecting gains over complex risk management
- **Strategy Isolation**: Sub-accounts prevent strategy interference
- **Emergency Reserves**: Always maintain 20% for black swan events
- **Scalable Design**: Architecture supports growth to larger capital

## Success Metrics

- **Profit Retention**: % of profits successfully harvested
- **Loss Recovery**: % of losing bots successfully recovered
- **System Uptime**: 99.9% availability target
- **Transfer Success Rate**: 99.5% success rate
- **Portfolio Growth**: Consistent positive returns

## Notes

- Prioritize robust error handling due to exchange API volatility
- Consider implementing circuit breakers for problematic exchanges
- Build comprehensive logging for debugging and compliance
- Design for horizontal scaling as bot count increases
- Implement proper async/await patterns for concurrent operations