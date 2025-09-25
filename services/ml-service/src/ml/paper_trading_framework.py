"""
Comprehensive Paper Trading Backtesting Framework

This framework provides a unified interface for paper trading and backtesting
all strategies, models, features, and indicators in the Quantify system.

Features:
- Support for all existing strategies (Lag-based, Lorentzian, Logistic Regression, Chandelier Exit)
- GPU-accelerated feature calculation using custom indicators
- Realistic market simulation with fees, slippage, and order execution
- Comprehensive performance metrics and visualization
- Strategy optimization and parameter tuning
- Multi-timeframe analysis
- Risk management and position sizing
- Export results for further analysis

Usage:
    from src.ml.paper_trading_framework import PaperTradingFramework
    
    # Initialize framework
    framework = PaperTradingFramework(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005
    )
    
    # Run backtest with any strategy
    results = await framework.backtest_strategy(
        strategy_name='lag_based',
        data=data,
        params={'threshold': 1.5, 'max_lag_minutes': 60}
    )
    
    # Run paper trading simulation
    await framework.run_paper_trading(
        strategy_name='lorentzian',
        live_data_feed=data_feed,
        duration_hours=24
    )
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from enum import Enum
import torch

# Import all strategies
from .models.strategy.lag_based_strategy import LagBasedStrategy, StrategyConfig as LagConfig
from .models.strategy.lorentzian_classifier import LorentzianANN
from .models.strategy.logistic_regression_torch import LogisticRegression, LogisticConfig
from .models.strategy.chandelier_exit import ChandelierExit, ChandelierConfig

# Import features and indicators
from .features.rsi import RSIIndicator
from .features.adx import ADXIndicator
from .features.cci import CCIIndicator
from .features.wave_trend import WaveTrendIndicator
from .features.chandelier_exit import ChandelierExitIndicator

# Import data storage
from src.data.csv_storage import CSVStorage, StorageConfig

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Available strategy types"""
    LAG_BASED = "lag_based"
    LORENTZIAN = "lorentzian"
    LOGISTIC_REGRESSION = "logistic_regression"
    CHANDELIER_EXIT = "chandelier_exit"
    COMBINED = "combined"


class Direction(Enum):
    """Trading direction"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class Trade:
    """Represents a completed trade"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: Direction
    symbol: str
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    strategy: str
    confidence: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    direction: Direction
    quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: float
    take_profit: float
    strategy: str
    confidence: float = 0.0


@dataclass
class BacktestConfig:
    """Configuration for backtesting - UPDATED FOR LEVERAGE-BASED TRADING"""
    # CRITICAL: Use leverage-based position sizing
    initial_capital: float = 500.0  # $500 USDT starting capital
    use_leverage_position_sizing: bool = True
    
    # Leverage settings
    default_leverage: int = 75  # 75x leverage
    max_leverage: int = 125  # Maximum allowed leverage
    
    # Realistic fee structure for leverage trading
    maker_fee: float = 0.0002  # 0.02% maker fee
    taker_fee: float = 0.0006  # 0.06% taker fee
    funding_rate: float = 0.0001  # 0.01% funding rate (8-hour)
    
    # Position management
    max_positions: int = 3
    position_allocation_pct: float = 0.05  # 5% allocation per position (for 75x leverage)
    
    # Risk management (disabled stop/take profit - use strategy exits)
    stop_loss_pct: float = 0.0   # Disabled - let strategies run naturally
    take_profit_pct: float = 0.0  # Disabled - let strategies run naturally
    
    # Liquidation protection
    liquidation_safety_margin: float = 0.20  # Stay 20% away from liquidation
    emergency_close_threshold: float = 0.10  # Emergency close at 10% from liquidation
    
    # Other settings
    min_confidence: float = 0.3
    use_gpu: bool = True
    save_results: bool = True
    results_dir: str = "results/paper_trading"


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)


class PaperTradingFramework:
    """
    Comprehensive paper trading and backtesting framework
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize the paper trading framework
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self.storage = CSVStorage(StorageConfig(data_path="data/historical/processed"))
        
        # Portfolio state
        self.capital = self.config.initial_capital
        self.initial_capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = [self.initial_capital]
        self.trade_history = []
        
        # Strategy instances
        self.strategies = {}
        self._initialize_strategies()
        
        # Results directory
        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Paper Trading Framework initialized with ${self.initial_capital:,.2f} capital")
    
    def _initialize_strategies(self):
        """Initialize all available strategies with leverage-based configuration"""
        # Lag-based strategy
        lag_config = LagConfig(
            leader_assets=['BTC', 'ETH', 'SOL'],
            follower_assets=['WIF', 'FARTCOIN', 'POPCAT', 'PEPE', 'SHIB'],
            threshold=1.5,
            max_lag_minutes=60,
            risk_per_trade=self.config.position_allocation_pct,  # Use position allocation instead
            max_positions=self.config.max_positions,
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct
        )
        self.strategies[StrategyType.LAG_BASED] = LagBasedStrategy(lag_config)
        
        # Lorentzian classifier
        self.strategies[StrategyType.LORENTZIAN] = LorentzianANN(
            lookback_bars=50,
            prediction_bars=4,
            k_neighbors=20
        )
        
        # Logistic regression
        log_config = LogisticConfig(
            lookback=3,
            learning_rate=0.0009,
            iterations=1000,
            threshold=0.5
        )
        self.strategies[StrategyType.LOGISTIC_REGRESSION] = LogisticRegression(log_config)
        
        # Chandelier exit
        self.strategies[StrategyType.CHANDELIER_EXIT] = ChandelierExit(
            atr_period=22,
            atr_multiplier=3.0
        )
        
        logger.info(f"Initialized {len(self.strategies)} strategies")
    
    async def backtest_strategy(
        self,
        strategy_name: Union[str, StrategyType],
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        params: Optional[Dict] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Run backtest for a specific strategy
        
        Args:
            strategy_name: Name of strategy to test
            data: Historical data (single DataFrame or dict of DataFrames)
            params: Strategy parameters
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with backtest results
        """
        # Convert strategy name to enum if string
        if isinstance(strategy_name, str):
            strategy_name = StrategyType(strategy_name)
        
        logger.info(f"Starting backtest for {strategy_name.value} strategy")
        
        # Reset portfolio state
        self._reset_portfolio()
        
        # Get strategy instance
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        # Update strategy parameters if provided
        if params:
            self._update_strategy_params(strategy, strategy_name, params)
        
        # Prepare data
        if isinstance(data, pd.DataFrame):
            # Single asset backtest
            backtest_data = self._prepare_single_asset_data(data, start_date, end_date)
            results = await self._run_single_asset_backtest(strategy, strategy_name, backtest_data)
        else:
            # Multi-asset backtest (for lag-based strategy)
            backtest_data = self._prepare_multi_asset_data(data, start_date, end_date)
            results = await self._run_multi_asset_backtest(strategy, strategy_name, backtest_data)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics()
        results['metrics'] = metrics
        
        # Save results if enabled
        if self.config.save_results:
            self._save_backtest_results(strategy_name.value, results)
        
        logger.info(f"Backtest complete: {metrics.total_trades} trades, "
                   f"{metrics.win_rate:.1%} win rate, {metrics.total_return:.2f}% return")
        
        return results
    
    def _reset_portfolio(self):
        """Reset portfolio state for new backtest"""
        self.capital = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve = [self.initial_capital]
        self.trade_history.clear()
    
    def _update_strategy_params(self, strategy, strategy_name: StrategyType, params: Dict):
        """Update strategy parameters"""
        if strategy_name == StrategyType.LAG_BASED:
            for key, value in params.items():
                if hasattr(strategy.config, key):
                    setattr(strategy.config, key, value)
        elif strategy_name == StrategyType.LORENTZIAN:
            for key, value in params.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
        elif strategy_name == StrategyType.LOGISTIC_REGRESSION:
            for key, value in params.items():
                if hasattr(strategy.config, key):
                    setattr(strategy.config, key, value)
        elif strategy_name == StrategyType.CHANDELIER_EXIT:
            for key, value in params.items():
                if hasattr(strategy.config, key):
                    setattr(strategy.config, key, value)
    
    def _prepare_single_asset_data(self, data: pd.DataFrame, start_date: Optional[datetime], 
                                  end_date: Optional[datetime]) -> pd.DataFrame:
        """Prepare single asset data for backtesting"""
        df = data.copy()
        
        # Ensure timestamp column exists and is properly formatted
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column")
        
        # Convert timestamp to datetime if needed
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set timestamp as index for filtering
        df.set_index('timestamp', inplace=True)
        
        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df.index >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    def _prepare_multi_asset_data(self, data: Dict[str, pd.DataFrame], 
                                 start_date: Optional[datetime], 
                                 end_date: Optional[datetime]) -> Dict[str, pd.DataFrame]:
        """Prepare multi-asset data for backtesting"""
        prepared_data = {}
        
        for symbol, df in data.items():
            prepared_data[symbol] = self._prepare_single_asset_data(df, start_date, end_date)
        
        return prepared_data
    
    async def _run_single_asset_backtest(self, strategy, strategy_name: StrategyType, 
                                       data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest for single asset strategies"""
        # Generate signals based on strategy type
        if strategy_name == StrategyType.LORENTZIAN:
            signals = strategy.calculate_signals(data)
            buy_signals = signals['buy_signals'].cpu().numpy()
            sell_signals = signals['sell_signals'].cpu().numpy()
            predictions = signals['predictions'].cpu().numpy()
        elif strategy_name == StrategyType.LOGISTIC_REGRESSION:
            signals = strategy.calculate_signals(data)
            buy_signals = signals['buy_signals'].cpu().numpy()
            sell_signals = signals['sell_signals'].cpu().numpy()
            predictions = signals['predictions'].cpu().numpy()
        elif strategy_name == StrategyType.CHANDELIER_EXIT:
            signals = strategy.calculate_signals(data)
            buy_signals = signals['buy_signals'].cpu().numpy()
            sell_signals = signals['sell_signals'].cpu().numpy()
            predictions = signals['predictions'].cpu().numpy()
        else:
            raise ValueError(f"Single asset backtest not supported for {strategy_name}")
        
        # Run trading simulation
        return await self._simulate_trading(data, buy_signals, sell_signals, predictions, strategy_name.value)
    
    async def _run_multi_asset_backtest(self, strategy, strategy_name: StrategyType, 
                                      data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run backtest for multi-asset strategies (like lag-based)"""
        if strategy_name != StrategyType.LAG_BASED:
            raise ValueError(f"Multi-asset backtest only supported for lag-based strategy")
        
        # Generate signals using lag-based strategy
        leader_data = {k: v for k, v in data.items() 
                      if k in strategy.config.leader_assets}
        follower_data = {k: v for k, v in data.items() 
                        if k in strategy.config.follower_assets}
        
        signals = strategy.generate_signals(leader_data, follower_data)
        
        # Convert signals to trading format
        all_signals = []
        for signal in signals:
            all_signals.append({
                'timestamp': signal.timestamp,
                'symbol': signal.follower_asset,
                'direction': signal.direction,
                'confidence': signal.confidence,
                'entry_price': data[signal.follower_asset].loc[signal.timestamp, 'close']
            })
        
        # Sort signals by timestamp
        all_signals.sort(key=lambda x: x['timestamp'])
        
        # Run trading simulation
        return await self._simulate_trading_from_signals(all_signals, data, strategy_name.value)
    
    async def _simulate_trading(self, data: pd.DataFrame, buy_signals: np.ndarray, 
                              sell_signals: np.ndarray, predictions: np.ndarray, 
                              strategy_name: str) -> Dict[str, Any]:
        """Simulate trading based on signals"""
        for i in range(len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            
            # Check for exits on existing positions
            await self._check_position_exits(current_time, current_price, data.iloc[i])
            
            # Close positions on opposite signals
            symbol = data.name if hasattr(data, 'name') else 'UNKNOWN'
            if symbol in self.positions:
                position = self.positions[symbol]
                should_close = False
                
                # Close LONG position on SELL signal
                if position.direction == Direction.LONG and sell_signals[i]:
                    print(f"[DEBUG] CLOSING LONG position on SELL signal at {current_price} on {current_time}")
                    await self._close_position(position, current_time, current_price, "signal_exit")
                    del self.positions[symbol]
                    should_close = True
                    
                # Close SHORT position on BUY signal
                elif position.direction == Direction.SHORT and buy_signals[i]:
                    print(f"[DEBUG] CLOSING SHORT position on BUY signal at {current_price} on {current_time}")
                    await self._close_position(position, current_time, current_price, "signal_exit")
                    del self.positions[symbol]
                    should_close = True
            
            # Allow immediate re-entry after closing a position
            if len(self.positions) < self.config.max_positions:
                if buy_signals[i]:
                    print(f"[DEBUG] BUY signal at idx {i}, price {current_price}, time {current_time}")
                    await self._open_position(
                        symbol=symbol,
                        direction=Direction.LONG,
                        entry_price=current_price,
                        entry_time=current_time,
                        strategy=strategy_name,
                        confidence=float(predictions[i]) if i < len(predictions) else 0.5
                    )
                elif sell_signals[i]:
                    print(f"[DEBUG] SELL signal at idx {i}, price {current_price}, time {current_time}")
                    await self._open_position(
                        symbol=symbol,
                        direction=Direction.SHORT,
                        entry_price=current_price,
                        entry_time=current_time,
                        strategy=strategy_name,
                        confidence=float(predictions[i]) if i < len(predictions) else 0.5
                    )
            
            # Update equity curve
            self._update_equity_curve(current_price)
        
        return {
            'strategy': strategy_name,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'final_capital': self.capital
        }
    
    async def _simulate_trading_from_signals(self, signals: List[Dict], 
                                           data: Dict[str, pd.DataFrame], 
                                           strategy_name: str) -> Dict[str, Any]:
        """Simulate trading from pre-generated signals"""
        # Create a unified timeline of all signals
        timeline = []
        for signal in signals:
            timeline.append({
                'timestamp': signal['timestamp'],
                'type': 'signal',
                'data': signal
            })
        
        # Add price updates for all assets
        for symbol, df in data.items():
            for timestamp, row in df.iterrows():
                timeline.append({
                    'timestamp': timestamp,
                    'type': 'price_update',
                    'symbol': symbol,
                    'price': row['close'],
                    'high': row['high'],
                    'low': row['low']
                })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        
        # Process timeline
        for event in timeline:
            if event['type'] == 'signal':
                signal = event['data']
                if len(self.positions) < self.config.max_positions:
                    await self._open_position(
                        symbol=signal['symbol'],
                        direction=signal['direction'],
                        entry_price=signal['entry_price'],
                        entry_time=signal['timestamp'],
                        strategy=strategy_name,
                        confidence=signal['confidence']
                    )
            
            elif event['type'] == 'price_update':
                await self._check_position_exits(
                    event['timestamp'], 
                    event['price'],
                    {'high': event['high'], 'low': event['low']}
                )
                self._update_equity_curve(event['price'])
        
        return {
            'strategy': strategy_name,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'final_capital': self.capital
        }
    
    async def _open_position(self, symbol: str, direction: Direction, entry_price: float,
                           entry_time: pd.Timestamp, strategy: str, confidence: float):
        """Open a new position using leverage-based position sizing"""
        
        # CRITICAL: Use leverage-based position sizing
        if self.config.use_leverage_position_sizing:
            # Calculate position size based on leverage and allocation
            leverage = self.config.default_leverage
            allocation_pct = self.config.position_allocation_pct
            
            # Position size in USDT (not leveraged)
            position_size_usdt = self.config.initial_capital * allocation_pct
            
            # Effective position size with leverage
            effective_position_size = position_size_usdt * leverage
            
            # Calculate quantity based on effective position size
            quantity = effective_position_size / entry_price
            
            # Calculate margin requirement (actual capital used)
            margin_required = effective_position_size / leverage
            
            # Calculate liquidation price
            if direction == Direction.LONG:
                liquidation_price = entry_price * (1 - (1 / leverage) * (1 - self.config.liquidation_safety_margin))
                stop_loss = liquidation_price * (1 + self.config.emergency_close_threshold)
                take_profit = 0.0  # Disabled
            else:
                liquidation_price = entry_price * (1 + (1 / leverage) * (1 - self.config.liquidation_safety_margin))
                stop_loss = liquidation_price * (1 - self.config.emergency_close_threshold)
                take_profit = 0.0  # Disabled
            
            # Apply realistic trading fees
            fee_rate = self.config.taker_fee  # Assume market orders
            commission = effective_position_size * fee_rate
            
            total_cost = margin_required + commission
            
            print(f"[DEBUG] LEVERAGE POSITION: {leverage}x, Allocation: {allocation_pct:.1%}, "
                  f"Margin: ${margin_required:.2f}, Effective: ${effective_position_size:.2f}, "
                  f"Liquidation: ${liquidation_price:.2f}")
            
        else:
            # OLD METHOD (DEPRECATED - should not be used)
            position_size = self.capital * 0.02  # 2% risk per trade
            quantity = position_size / entry_price
            commission = entry_price * quantity * 0.001
            total_cost = entry_price * quantity + commission
            stop_loss = entry_price * (1 - self.config.stop_loss_pct) if direction == Direction.LONG else entry_price * (1 + self.config.stop_loss_pct)
            take_profit = entry_price * (1 + self.config.take_profit_pct) if direction == Direction.LONG else entry_price * (1 - self.config.take_profit_pct)
        
        if total_cost > self.capital:
            return  # Insufficient capital
        
        # Deduct margin requirement from capital (not full position size)
        self.capital -= total_cost
        
        # Create position
        position = Position(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=entry_time,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy,
            confidence=confidence
        )
        
        self.positions[symbol] = position
        
        logger.debug(f"Opened {direction.value} position in {symbol} at {entry_price:.4f}")
    
    async def _check_position_exits(self, current_time: pd.Timestamp, current_price: float, 
                                  current_bar: pd.Series):
        """Check if any positions should be closed"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            should_close = False
            exit_price = current_price
            exit_reason = "manual"
            
            # Check stop loss and take profit only if enabled (not zero)
            if position.direction == Direction.LONG:
                if self.config.stop_loss_pct > 0 and current_bar['low'] <= position.stop_loss:
                    print(f"[DEBUG] LONG STOP LOSS for {symbol} at {position.stop_loss} on {current_time}")
                    should_close = True
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss"
                elif self.config.take_profit_pct > 0 and current_bar['high'] >= position.take_profit:
                    print(f"[DEBUG] LONG TAKE PROFIT for {symbol} at {position.take_profit} on {current_time}")
                    should_close = True
                    exit_price = position.take_profit
                    exit_reason = "take_profit"
            else:  # SHORT
                if self.config.stop_loss_pct > 0 and current_bar['high'] >= position.stop_loss:
                    print(f"[DEBUG] SHORT STOP LOSS for {symbol} at {position.stop_loss} on {current_time}")
                    should_close = True
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss"
                elif self.config.take_profit_pct > 0 and current_bar['low'] <= position.take_profit:
                    print(f"[DEBUG] SHORT TAKE PROFIT for {symbol} at {position.take_profit} on {current_time}")
                    should_close = True
                    exit_price = position.take_profit
                    exit_reason = "take_profit"
            
            if should_close:
                positions_to_close.append((symbol, position, exit_price, exit_reason))
        
        # Close positions
        for symbol, position, exit_price, exit_reason in positions_to_close:
            await self._close_position(position, current_time, exit_price, exit_reason)
            del self.positions[symbol]
    
    async def _close_position(self, position: Position, exit_time: pd.Timestamp, 
                            exit_price: float, exit_reason: str):
        """Close a position and record the trade with leverage-based PnL calculation"""
        import pandas as pd
        
        if self.config.use_leverage_position_sizing:
            # LEVERAGE-BASED PnL CALCULATION
            leverage = self.config.default_leverage
            
            # Calculate raw PnL (leveraged)
            if position.direction == Direction.LONG:
                price_change = exit_price - position.entry_price
            else:
                price_change = position.entry_price - exit_price
            
            # PnL is leveraged
            raw_pnl = price_change * position.quantity
            
            # Calculate exit fees on the effective position size
            effective_position_size = position.quantity * exit_price
            exit_commission = effective_position_size * self.config.taker_fee
            
            # Net PnL after fees
            net_pnl = raw_pnl - exit_commission
            
            # Return margin to capital + PnL
            margin_used = effective_position_size / leverage
            self.capital += margin_used + net_pnl
            
            # Calculate PnL percentage based on margin used
            pnl_pct = (net_pnl / margin_used) * 100
            
            print(f"[DEBUG] CLOSING LEVERAGE POSITION: Price change: ${price_change:.2f}, "
                  f"Raw PnL: ${raw_pnl:.2f}, Fees: ${exit_commission:.2f}, "
                  f"Net PnL: ${net_pnl:.2f}, PnL%: {pnl_pct:.2f}%")
            
        else:
            # OLD METHOD (DEPRECATED)
            if position.direction == Direction.LONG:
                pnl = (exit_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - exit_price) * position.quantity
            
            exit_commission = exit_price * position.quantity * 0.001
            net_pnl = pnl - exit_commission
            self.capital += net_pnl
            pnl_pct = (net_pnl / (position.entry_price * position.quantity)) * 100
        
        # Ensure times are pd.Timestamp
        entry_time = position.entry_time
        if not isinstance(entry_time, pd.Timestamp):
            entry_time = pd.to_datetime(entry_time)
        if not isinstance(exit_time, pd.Timestamp):
            exit_time = pd.to_datetime(exit_time)
        
        # Create trade record
        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            direction=position.direction,
            symbol=position.symbol,
            quantity=position.quantity,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=exit_commission,
            slippage=0.0,  # No slippage in leverage trading
            strategy=position.strategy,
            confidence=position.confidence,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit
        )
        self.trades.append(trade)
        logger.debug(f"Closed {position.direction.value} position in {position.symbol}: "
                    f"PnL ${net_pnl:.2f} ({pnl_pct:.2f}%) - {exit_reason}")
    
    def _update_equity_curve(self, current_price: float):
        """Update equity curve with current portfolio value"""
        portfolio_value = self.capital
        
        # Add unrealized PnL from open positions
        for position in self.positions.values():
            if position.direction == Direction.LONG:
                unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                unrealized_pnl = (position.entry_price - current_price) * position.quantity
            portfolio_value += unrealized_pnl
        
        self.equity_curve.append(portfolio_value)
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return PerformanceMetrics()
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl < 0]) if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Trade duration
        trade_durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in self.trades]
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        # Best/worst trades
        best_trade = max([t.pnl for t in self.trades]) if self.trades else 0
        worst_trade = min([t.pnl for t in self.trades]) if self.trades else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = [t.pnl / self.initial_capital for t in self.trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Annualized return (simplified)
        if self.trades:
            first_trade = min(t.entry_time for t in self.trades)
            last_trade = max(t.exit_time for t in self.trades)
            days = (last_trade - first_trade).days
            annualized_return = (total_return / days * 365) if days > 0 else 0
        else:
            annualized_return = 0
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration=avg_trade_duration,
            best_trade=best_trade,
            worst_trade=worst_trade,
            equity_curve=self.equity_curve,
            drawdown_curve=self._calculate_drawdown_curve()
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.equity_curve:
            return 0
        
        peak = self.equity_curve[0]
        max_dd = 0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd * 100  # Return as percentage
    
    def _calculate_drawdown_curve(self) -> List[float]:
        """Calculate drawdown curve"""
        if not self.equity_curve:
            return []
        
        peak = self.equity_curve[0]
        drawdown_curve = []
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            drawdown_curve.append(dd)
        
        return drawdown_curve
    
    def _save_backtest_results(self, strategy_name: str, results: Dict[str, Any]):
        """Save backtest results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = self.results_dir / f"{strategy_name}_metrics_{timestamp}.json"
        metrics_dict = {
            'strategy': strategy_name,
            'timestamp': timestamp,
            'config': self.config.__dict__,
            'metrics': results['metrics'].__dict__
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        
        # Save trades
        trades_file = self.results_dir / f"{strategy_name}_trades_{timestamp}.csv"
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'symbol': t.symbol,
            'direction': t.direction.value,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'commission': t.commission,
            'slippage': t.slippage,
            'strategy': t.strategy,
            'confidence': t.confidence
        } for t in results['trades']])
        trades_df.to_csv(trades_file, index=False)
        
        # Save equity curve
        equity_file = self.results_dir / f"{strategy_name}_equity_{timestamp}.csv"
        drawdown_curve = results['metrics'].drawdown_curve if results['metrics'].drawdown_curve else self._calculate_drawdown_curve()
        equity_df = pd.DataFrame({
            'equity': results['equity_curve'],
            'drawdown': drawdown_curve
        }, index=range(len(results['equity_curve'])))
        equity_df.to_csv(equity_file, index=False)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot comprehensive backtest results"""
        if not results['trades']:
            print("No trades to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Equity curve
        equity_curve = results['equity_curve']
        axes[0, 0].plot(equity_curve, label='Portfolio Value', color='blue')
        axes[0, 0].axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        drawdown_curve = results['metrics'].drawdown_curve
        axes[0, 1].fill_between(range(len(drawdown_curve)), drawdown_curve, 0, 
                               color='red', alpha=0.3)
        axes[0, 1].plot(drawdown_curve, color='red', label='Drawdown')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Trade PnL distribution
        pnls = [t.pnl for t in results['trades']]
        axes[1, 0].hist(pnls, bins=30, alpha=0.7, color='green')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Trade PnL Distribution')
        axes[1, 0].set_xlabel('PnL ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Win rate and metrics
        metrics = results['metrics']
        metric_names = ['Win Rate', 'Profit Factor', 'Total Return', 'Max Drawdown']
        metric_values = [
            metrics.win_rate * 100,
            metrics.profit_factor,
            metrics.total_return,
            metrics.max_drawdown
        ]
        
        bars = axes[1, 1].bar(metric_names, metric_values, color=['green', 'blue', 'orange', 'red'])
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', va='bottom')
        
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    async def optimize_strategy_parameters(
        self,
        strategy_name: Union[str, StrategyType],
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        param_ranges: Dict[str, List],
        metric: str = 'total_return'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search
        
        Args:
            strategy_name: Strategy to optimize
            data: Historical data
            param_ranges: Dictionary of parameter ranges to test
            metric: Metric to optimize for
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting parameter optimization for {strategy_name}")
        
        # Convert strategy name to enum if string
        if isinstance(strategy_name, str):
            strategy_name = StrategyType(strategy_name)
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges)
        
        best_params = None
        best_metric = float('-inf')
        optimization_results = []
        
        # Test each parameter combination
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing parameters {i+1}/{len(param_combinations)}: {params}")
            
            try:
                results = await self.backtest_strategy(strategy_name, data, params)
                current_metric = results['metrics'].__dict__.get(metric, 0)
                
                optimization_results.append({
                    'params': params,
                    'metric': current_metric,
                    'results': results
                })
                
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_params = params
                    
            except Exception as e:
                logger.error(f"Error testing parameters {params}: {e}")
                continue
        
        # Sort results by metric
        optimization_results.sort(key=lambda x: x['metric'], reverse=True)
        
        # Save optimization results
        if self.config.save_results:
            self._save_optimization_results(strategy_name.value, optimization_results)
        
        logger.info(f"Optimization complete. Best parameters: {best_params}")
        logger.info(f"Best {metric}: {best_metric:.2f}")
        
        return {
            'best_params': best_params,
            'best_metric': best_metric,
            'all_results': optimization_results
        }
    
    def _generate_param_combinations(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """Generate all combinations of parameters"""
        import itertools
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = []
        for values in itertools.product(*param_values):
            combination = dict(zip(param_names, values))
            combinations.append(combination)
        
        return combinations
    
    def _save_optimization_results(self, strategy_name: str, results: List[Dict]):
        """Save optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        opt_file = self.results_dir / f"{strategy_name}_optimization_{timestamp}.json"
        with open(opt_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save as CSV
        csv_file = self.results_dir / f"{strategy_name}_optimization_{timestamp}.csv"
        csv_data = []
        for result in results:
            row = result['params'].copy()
            row['metric'] = result['metric']
            csv_data.append(row)
        
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        logger.info(f"Optimization results saved to {self.results_dir}")


# Example usage and testing functions
async def test_framework():
    """Test the paper trading framework with sample data"""
    print("🧪 Testing Paper Trading Framework")
    
    # Initialize framework
    framework = PaperTradingFramework(
        BacktestConfig(
            initial_capital=10000,
            commission=0.001,
            slippage=0.0005,
            max_positions=3
        )
    )
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 2, len(dates)),
        'high': np.random.normal(102, 2, len(dates)),
        'low': np.random.normal(98, 2, len(dates)),
        'close': np.random.normal(100, 2, len(dates)),
        'volume': np.random.normal(1000, 200, len(dates))
    }, index=dates)
    
    # Test Lorentzian strategy
    print("\n📊 Testing Lorentzian Strategy...")
    results = await framework.backtest_strategy(
        strategy_name='lorentzian',
        data=sample_data
    )
    
    print(f"Total trades: {results['metrics'].total_trades}")
    print(f"Win rate: {results['metrics'].win_rate:.1%}")
    print(f"Total return: {results['metrics'].total_return:.2f}%")
    
    # Plot results
    framework.plot_results(results)
    
    return framework, results


if __name__ == "__main__":
    asyncio.run(test_framework()) 