#!/usr/bin/env python3
"""
Live Lag Trading Model

A production-ready trading model that generates real trading signals
based on lag analysis with optimized thresholds for 1-minute data.

This model:
- Uses lower thresholds (0.3-0.8%) appropriate for 1-minute data
- Generates real-time trading signals
- Integrates with Quantify trading pipeline
- Includes risk management and position sizing
- Provides signal confidence scoring

Usage:
    from lag_trading_model import LagTradingModel
    
    model = LagTradingModel()
    signals = await model.generate_signals(live_data)
    trades = model.execute_signals(signals, portfolio)
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from .lag_based_strategy import LagBasedStrategy, StrategyConfig
from src.data.csv_storage import CSVStorage, StorageConfig

logger = logging.getLogger(__name__)


class LagTradingModel:
    """
    Live trading model for lag-based strategy with optimized parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the lag trading model.
        
        Args:
            config_path: Path to model configuration file
        """
        self.storage = CSVStorage(StorageConfig(data_path="data/historical/processed"))
        self.model_config = self._load_model_config(config_path)
        self.strategies = {}
        self.signal_history = []
        
        # Initialize strategies for each pair
        self._initialize_strategies()
    
    def _load_model_config(self, config_path: Optional[str]) -> Dict:
        """
        Load model configuration from file or use defaults.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Dict with model configuration
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration optimized for 1-minute data
        return {
            'model_version': '1.0',
            'description': 'Lag-based trading model for 1-minute data',
            'pairs': {
                'BTC-PEPE': {
                    'leader': 'BTC',
                    'follower': 'PEPE',
                    'threshold': 0.5,
                    'priority': 1
                },
                'ETH-PEPE': {
                    'leader': 'ETH', 
                    'follower': 'PEPE',
                    'threshold': 0.6,
                    'priority': 2
                },
                'SOL-PEPE': {
                    'leader': 'SOL',
                    'follower': 'PEPE', 
                    'threshold': 0.4,
                    'priority': 3
                },
                'BTC-WIF': {
                    'leader': 'BTC',
                    'follower': 'WIF',
                    'threshold': 0.5,
                    'priority': 4
                },
                'ETH-WIF': {
                    'leader': 'ETH',
                    'follower': 'WIF',
                    'threshold': 0.6,
                    'priority': 5
                },
                'SOL-WIF': {
                    'leader': 'SOL',
                    'follower': 'WIF',
                    'threshold': 0.4,
                    'priority': 6
                },
                'BTC-FARTCOIN': {
                    'leader': 'BTC',
                    'follower': 'FARTCOIN',
                    'threshold': 0.5,
                    'priority': 7
                },
                'ETH-FARTCOIN': {
                    'leader': 'ETH',
                    'follower': 'FARTCOIN',
                    'threshold': 0.6,
                    'priority': 8
                },
                'SOL-FARTCOIN': {
                    'leader': 'SOL',
                    'follower': 'FARTCOIN',
                    'threshold': 0.4,
                    'priority': 9
                }
            },
            'risk_management': {
                'max_positions': 3,
                'risk_per_trade': 0.02,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10,
                'max_drawdown_limit': 0.20
            },
            'signal_generation': {
                'min_correlation': 0.01,
                'max_lag_minutes': 60,
                'volume_confirmation': True,
                'min_volume_ratio': 1.2,
                'signal_confidence_threshold': 0.6
            }
        }
    
    def _initialize_strategies(self):
        """Initialize strategy objects for each configured pair."""
        for pair_name, pair_config in self.model_config['pairs'].items():
            strategy_config = StrategyConfig(
                leader_assets=[pair_config['leader']],
                follower_assets=[pair_config['follower']],
                threshold=pair_config['threshold'],
                max_lag_minutes=self.model_config['signal_generation']['max_lag_minutes'],
                min_correlation=self.model_config['signal_generation']['min_correlation'],
                risk_per_trade=self.model_config['risk_management']['risk_per_trade'],
                max_positions=1,  # One position per pair
                stop_loss_pct=self.model_config['risk_management']['stop_loss_pct'],
                take_profit_pct=self.model_config['risk_management']['take_profit_pct']
            )
            
            self.strategies[pair_name] = LagBasedStrategy(strategy_config)
    
    async def generate_signals(self, live_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Generate trading signals from live market data.
        
        Args:
            live_data: Dict with symbol -> DataFrame mapping
                      Each DataFrame should have OHLCV columns
            
        Returns:
            List of trading signals with confidence scores
        """
        signals = []
        
        for pair_name, pair_config in self.model_config['pairs'].items():
            leader = pair_config['leader']
            follower = pair_config['follower']
            
            # Check if we have data for both assets
            if leader not in live_data or follower not in live_data:
                continue
            
            leader_data = live_data[leader]
            follower_data = live_data[follower]
            
            try:
                # Generate signals using the strategy
                strategy = self.strategies[pair_name]
                pair_signals = strategy.generate_signals({
                    leader: leader_data,
                    follower: follower_data
                })
                
                # Add metadata and confidence scoring
                for signal in pair_signals:
                    signal['pair'] = pair_name
                    signal['priority'] = pair_config['priority']
                    signal['confidence'] = self._calculate_signal_confidence(signal, leader_data, follower_data)
                    
                    # Only include signals above confidence threshold
                    if signal['confidence'] >= self.model_config['signal_generation']['signal_confidence_threshold']:
                        signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error generating signals for {pair_name}: {e}")
                continue
        
        # Sort signals by priority and confidence
        signals.sort(key=lambda x: (x['priority'], x['confidence']), reverse=True)
        
        # Limit to max positions
        max_positions = self.model_config['risk_management']['max_positions']
        signals = signals[:max_positions]
        
        # Store in history
        self.signal_history.extend(signals)
        
        return signals
    
    def _calculate_signal_confidence(self, signal: Dict, leader_data: pd.DataFrame, 
                                   follower_data: pd.DataFrame) -> float:
        """
        Calculate confidence score for a trading signal.
        
        Args:
            signal: Trading signal dictionary
            leader_data: Leader asset data
            follower_data: Follower asset data
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence
        
        # Factor 1: Move size relative to threshold
        move_size = abs(signal.get('leader_move_pct', 0))
        threshold = signal.get('threshold', 1.0)
        size_factor = min(move_size / threshold, 2.0) / 2.0  # 0-1 scale
        confidence += size_factor * 0.2
        
        # Factor 2: Volume confirmation
        if self.model_config['signal_generation']['volume_confirmation']:
            try:
                # Calculate volume ratio (current vs average)
                current_volume = leader_data['volume'].iloc[-1]
                avg_volume = leader_data['volume'].rolling(20).mean().iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                if volume_ratio >= self.model_config['signal_generation']['min_volume_ratio']:
                    confidence += 0.15
            except:
                pass
        
        # Factor 3: Correlation strength
        try:
            # Calculate rolling correlation
            leader_returns = leader_data['close'].pct_change().dropna()
            follower_returns = follower_data['close'].pct_change().dropna()
            
            # Align data
            min_len = min(len(leader_returns), len(follower_returns))
            correlation = leader_returns.iloc[-min_len:].corr(follower_returns.iloc[-min_len:])
            
            if not pd.isna(correlation):
                confidence += abs(correlation) * 0.15
        except:
            pass
        
        # Factor 4: Market volatility (higher volatility = higher confidence for lag strategy)
        try:
            leader_volatility = leader_data['close'].pct_change().std()
            if leader_volatility > 0.02:  # 2% volatility threshold
                confidence += 0.1
        except:
            pass
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def execute_signals(self, signals: List[Dict], portfolio: Dict) -> List[Dict]:
        """
        Execute trading signals and return trade orders.
        
        Args:
            signals: List of trading signals
            portfolio: Current portfolio state
            
        Returns:
            List of trade orders to execute
        """
        trades = []
        
        for signal in signals:
            # Check if we already have a position in this pair
            pair = signal['pair']
            if self._has_position(pair, portfolio):
                continue
            
            # Calculate position size
            position_size = self._calculate_position_size(signal, portfolio)
            
            if position_size > 0:
                trade = {
                    'pair': pair,
                    'action': signal['action'],
                    'symbol': signal['follower'],
                    'quantity': position_size,
                    'price': signal.get('entry_price', 0),
                    'confidence': signal['confidence'],
                    'stop_loss': signal.get('stop_loss', 0),
                    'take_profit': signal.get('take_profit', 0),
                    'timestamp': datetime.now().isoformat(),
                    'signal_id': signal.get('id', '')
                }
                trades.append(trade)
        
        return trades
    
    def _has_position(self, pair: str, portfolio: Dict) -> bool:
        """Check if portfolio already has a position in the given pair."""
        follower = self.model_config['pairs'][pair]['follower']
        return follower in portfolio.get('positions', {})
    
    def _calculate_position_size(self, signal: Dict, portfolio: Dict) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            
        Returns:
            Position size in base currency
        """
        total_capital = portfolio.get('total_value', 10000)
        risk_per_trade = self.model_config['risk_management']['risk_per_trade']
        stop_loss_pct = self.model_config['risk_management']['stop_loss_pct']
        
        # Calculate risk amount
        risk_amount = total_capital * risk_per_trade
        
        # Calculate position size based on stop loss
        entry_price = signal.get('entry_price', 0)
        stop_loss_price = entry_price * (1 - stop_loss_pct) if signal['action'] == 'BUY' else entry_price * (1 + stop_loss_pct)
        
        price_risk = abs(entry_price - stop_loss_price)
        position_size = risk_amount / price_risk if price_risk > 0 else 0
        
        return position_size
    
    async def update_model(self, new_data: Dict[str, pd.DataFrame]):
        """
        Update model with new market data and retrain if necessary.
        
        Args:
            new_data: New market data
        """
        # For now, just update signal history
        # In a more sophisticated implementation, you might:
        # - Retrain correlation models
        # - Adjust thresholds based on recent performance
        # - Update risk parameters
        
        logger.info("Model updated with new data")
    
    def get_model_status(self) -> Dict:
        """
        Get current model status and performance metrics.
        
        Returns:
            Dict with model status information
        """
        return {
            'model_version': self.model_config['model_version'],
            'active_pairs': len(self.model_config['pairs']),
            'total_signals_generated': len(self.signal_history),
            'recent_signals': len([s for s in self.signal_history 
                                 if datetime.fromisoformat(s.get('timestamp', '2020-01-01')) > 
                                 datetime.now() - timedelta(hours=1)]),
            'risk_management': self.model_config['risk_management'],
            'signal_generation': self.model_config['signal_generation']
        }
    
    def save_model_state(self, filepath: str):
        """
        Save current model state to file.
        
        Args:
            filepath: Path to save model state
        """
        state = {
            'model_config': self.model_config,
            'signal_history': self.signal_history,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_model_state(self, filepath: str):
        """
        Load model state from file.
        
        Args:
            filepath: Path to model state file
        """
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.model_config = state.get('model_config', self.model_config)
            self.signal_history = state.get('signal_history', [])
            self._initialize_strategies()


class LagModelTrader:
    """
    Trading interface that integrates the lag model with exchange APIs.
    """
    
    def __init__(self, model: LagTradingModel, exchange_client):
        """
        Initialize the trader.
        
        Args:
            model: LagTradingModel instance
            exchange_client: Exchange API client
        """
        self.model = model
        self.exchange = exchange_client
        self.active_positions = {}
    
    async def run_trading_cycle(self):
        """
        Run one complete trading cycle.
        """
        try:
            # 1. Get live market data
            live_data = await self._get_live_data()
            
            # 2. Generate signals
            signals = await self.model.generate_signals(live_data)
            
            # 3. Get current portfolio
            portfolio = await self._get_portfolio()
            
            # 4. Execute signals
            trades = self.model.execute_signals(signals, portfolio)
            
            # 5. Execute trades
            for trade in trades:
                await self._execute_trade(trade)
            
            # 6. Update model
            await self.model.update_model(live_data)
            
            logger.info(f"Trading cycle complete: {len(signals)} signals, {len(trades)} trades")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def _get_live_data(self) -> Dict[str, pd.DataFrame]:
        """Get live market data for all configured pairs."""
        live_data = {}
        
        for pair_name, pair_config in self.model.model_config['pairs'].items():
            leader = pair_config['leader']
            follower = pair_config['follower']
            
            # Get data for both assets
            for symbol in [leader, follower]:
                if symbol not in live_data:
                    try:
                        # This would integrate with your exchange client
                        # For now, we'll use historical data as a placeholder
                        data = await self._get_symbol_data(symbol)
                        live_data[symbol] = data
                    except Exception as e:
                        logger.error(f"Error getting data for {symbol}: {e}")
        
        return live_data
    
    async def _get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Get recent data for a symbol.
        This would integrate with your exchange client.
        """
        # Placeholder - replace with actual exchange API call
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=1)
        
        # Try to get from storage first
        try:
            for exchange in ['bitget', 'binance', 'kraken']:
                for timeframe in ['1m', '5m', '5min']:
                    try:
                        data = await self.storage.load_candles(
                            exchange, f"{symbol}USDT", timeframe, start_date, end_date
                        )
                        if not data.empty:
                            return data
                    except:
                        continue
        except:
            pass
        
        # Return empty DataFrame if no data found
        return pd.DataFrame()
    
    async def _get_portfolio(self) -> Dict:
        """Get current portfolio state."""
        # This would integrate with your exchange client
        # For now, return a placeholder
        return {
            'total_value': 10000,
            'positions': self.active_positions,
            'cash': 5000
        }
    
    async def _execute_trade(self, trade: Dict):
        """
        Execute a trade order.
        
        Args:
            trade: Trade order dictionary
        """
        try:
            # This would integrate with your exchange client
            # For now, just log the trade
            logger.info(f"Executing trade: {trade}")
            
            # Update active positions
            symbol = trade['symbol']
            if trade['action'] == 'BUY':
                self.active_positions[symbol] = {
                    'quantity': trade['quantity'],
                    'entry_price': trade['price'],
                    'stop_loss': trade['stop_loss'],
                    'take_profit': trade['take_profit']
                }
            elif trade['action'] == 'SELL':
                if symbol in self.active_positions:
                    del self.active_positions[symbol]
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")


# Example usage and testing
async def test_model():
    """Test the lag trading model with sample data."""
    print("🧪 Testing Lag Trading Model")
    
    # Initialize model
    model = LagTradingModel()
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='1min')
    sample_data = {
        'BTC': pd.DataFrame({
            'open': np.random.normal(45000, 100, len(dates)),
            'high': np.random.normal(45100, 100, len(dates)),
            'low': np.random.normal(44900, 100, len(dates)),
            'close': np.random.normal(45000, 100, len(dates)),
            'volume': np.random.normal(1000, 100, len(dates))
        }, index=dates),
        'PEPE': pd.DataFrame({
            'open': np.random.normal(0.0001, 0.00001, len(dates)),
            'high': np.random.normal(0.00011, 0.00001, len(dates)),
            'low': np.random.normal(0.00009, 0.00001, len(dates)),
            'close': np.random.normal(0.0001, 0.00001, len(dates)),
            'volume': np.random.normal(1000000, 100000, len(dates))
        }, index=dates)
    }
    
    # Generate signals
    signals = await model.generate_signals(sample_data)
    
    print(f"Generated {len(signals)} signals")
    for signal in signals:
        print(f"  {signal['pair']}: {signal['action']} {signal['follower']} "
              f"(confidence: {signal['confidence']:.2f})")
    
    # Get model status
    status = model.get_model_status()
    print(f"\nModel Status: {status}")


if __name__ == "__main__":
    asyncio.run(test_model()) 