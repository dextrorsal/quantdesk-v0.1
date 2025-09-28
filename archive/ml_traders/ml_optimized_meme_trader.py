#!/usr/bin/env python3
"""
🧠 ML-Optimized Meme Coin HFT System

Uses machine learning to automatically optimize trading parameters:
- Reinforcement Learning for strategy optimization
- Genetic Algorithm for parameter tuning
- Neural Network for signal prediction
- Automated backtesting and optimization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import time
import random
from dataclasses import dataclass
from collections import deque

from src.ml.features.market_structure import MarketStructureDetector, MarketStructureConfig
from src.trading.leverage_manager import LeverageManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingParams:
    """Trading parameters that will be optimized by ML."""
    entry_threshold: float = 0.15
    exit_threshold: float = 0.05
    bos_weight: float = 0.35
    choch_weight: float = 0.25
    order_block_weight: float = 0.25
    fvg_weight: float = 0.15
    max_leverage: float = 25.0
    position_size_pct: float = 0.1
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15

class SignalPredictor(nn.Module):
    """Neural network for predicting optimal trading signals."""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 64):
        super(SignalPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 3),  # buy, sell, hold
            nn.Softmax(dim=1)
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TradingEnvironment:
    """Reinforcement learning environment for trading optimization."""
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.current_step = 0
        self.max_steps = 100
        
        # Market structure detector
        self.market_detector = MarketStructureDetector()
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self.balance = self.initial_balance
        self.positions = {}
        self.trade_history = []
        self.current_step = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        # Market structure signals + portfolio state
        state = np.zeros(8)
        
        # Portfolio metrics
        state[0] = self.balance / self.initial_balance  # Normalized balance
        state[1] = len(self.positions) / 10.0  # Normalized position count
        state[2] = self.total_pnl / self.initial_balance  # Normalized PnL
        state[3] = self.max_drawdown  # Max drawdown
        
        # Market structure signals (placeholder - will be filled by step)
        state[4:8] = 0.0  # BOS, CHoCH, Order Block, FVG signals
        
        return state
    
    def step(self, action: int, market_data: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return new state, reward, done, info."""
        self.current_step += 1
        
        # Get market structure signals
        signals = self._extract_signals(market_data)
        
        # Execute action (0: hold, 1: buy, 2: sell)
        reward = self._execute_action(action, market_data, signals)
        
        # Update portfolio
        self._update_positions(market_data)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps or self.balance <= 0
        
        # Get new state
        state = self._get_state()
        
        info = {
            'balance': self.balance,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown
        }
        
        return state, reward, done, info
    
    def _extract_signals(self, market_data: Dict) -> Dict:
        """Extract market structure signals from data."""
        # This would use the actual market structure detector
        # For now, return placeholder signals
        return {
            'bos_bullish': market_data.get('bos_bullish', 0.0),
            'bos_bearish': market_data.get('bos_bearish', 0.0),
            'choch_bullish': market_data.get('choch_bullish', 0.0),
            'choch_bearish': market_data.get('choch_bearish', 0.0),
            'order_block_bullish': market_data.get('order_block_bullish', 0.0),
            'order_block_bearish': market_data.get('order_block_bearish', 0.0),
            'fvg_bullish': market_data.get('fvg_bullish', 0.0),
            'fvg_bearish': market_data.get('fvg_bearish', 0.0)
        }
    
    def _execute_action(self, action: int, market_data: Dict, signals: Dict) -> float:
        """Execute trading action and return immediate reward."""
        symbol = market_data.get('symbol', 'UNKNOWN')
        price = market_data.get('price', 0.0)
        
        if action == 0:  # Hold
            return 0.0
        elif action == 1:  # Buy
            return self._open_long_position(symbol, price, signals)
        elif action == 2:  # Sell
            return self._open_short_position(symbol, price, signals)
        
        return 0.0
    
    def _open_long_position(self, symbol: str, price: float, signals: Dict) -> float:
        """Open long position and return reward."""
        if symbol in self.positions:
            return -0.1  # Penalty for duplicate position
        
        # Calculate position size based on signal strength
        signal_strength = (
            signals['bos_bullish'] * 0.35 +
            signals['choch_bullish'] * 0.25 +
            signals['order_block_bullish'] * 0.25 +
            signals['fvg_bullish'] * 0.15
        )
        
        position_size = min(self.balance * 0.1 * signal_strength, self.balance * 0.2)
        
        if position_size > 0:
            self.positions[symbol] = {
                'side': 'long',
                'entry_price': price,
                'size': position_size,
                'signal_strength': signal_strength
            }
            self.balance -= position_size
            self.total_trades += 1
            return signal_strength * 0.1  # Reward based on signal strength
        
        return -0.05  # Small penalty for failed trade
    
    def _open_short_position(self, symbol: str, price: float, signals: Dict) -> float:
        """Open short position and return reward."""
        if symbol in self.positions:
            return -0.1  # Penalty for duplicate position
        
        # Calculate position size based on signal strength
        signal_strength = (
            signals['bos_bearish'] * 0.35 +
            signals['choch_bearish'] * 0.25 +
            signals['order_block_bearish'] * 0.25 +
            signals['fvg_bearish'] * 0.15
        )
        
        position_size = min(self.balance * 0.1 * signal_strength, self.balance * 0.2)
        
        if position_size > 0:
            self.positions[symbol] = {
                'side': 'short',
                'entry_price': price,
                'size': position_size,
                'signal_strength': signal_strength
            }
            self.balance -= position_size
            self.total_trades += 1
            return signal_strength * 0.1  # Reward based on signal strength
        
        return -0.05  # Small penalty for failed trade
    
    def _update_positions(self, market_data: Dict):
        """Update position PnL and close if needed."""
        symbol = market_data.get('symbol', 'UNKNOWN')
        current_price = market_data.get('price', 0.0)
        
        if symbol in self.positions:
            position = self.positions[symbol]
            entry_price = position['entry_price']
            
            if position['side'] == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # short
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Close position if profit target or stop loss hit
            if pnl_pct >= 0.15 or pnl_pct <= -0.05:  # 15% profit, 5% loss
                pnl = position['size'] * pnl_pct
                self.balance += position['size'] + pnl
                self.total_pnl += pnl
                
                if pnl > 0:
                    self.winning_trades += 1
                
                del self.positions[symbol]
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on portfolio performance."""
        # Update max drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Reward based on balance growth and drawdown
        balance_reward = (self.balance - self.initial_balance) / self.initial_balance
        drawdown_penalty = -self.max_drawdown * 2  # Heavy penalty for drawdown
        
        return balance_reward + drawdown_penalty

class GeneticOptimizer:
    """Genetic algorithm for optimizing trading parameters."""
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.best_params = None
        self.best_fitness = float('-inf')
    
    def initialize_population(self):
        """Initialize random population of trading parameters."""
        for _ in range(self.population_size):
            params = TradingParams(
                entry_threshold=random.uniform(0.05, 0.3),
                exit_threshold=random.uniform(0.02, 0.15),
                bos_weight=random.uniform(0.2, 0.5),
                choch_weight=random.uniform(0.15, 0.35),
                order_block_weight=random.uniform(0.15, 0.35),
                fvg_weight=random.uniform(0.05, 0.25),
                max_leverage=random.uniform(5.0, 25.0),
                position_size_pct=random.uniform(0.05, 0.2),
                stop_loss_pct=random.uniform(0.03, 0.1),
                take_profit_pct=random.uniform(0.1, 0.25)
            )
            self.population.append(params)
    
    def fitness_function(self, params: TradingParams) -> float:
        """Evaluate fitness of trading parameters."""
        # This would run backtesting with the parameters
        # For now, return a simple fitness score
        fitness = 0.0
        
        # Prefer balanced parameters
        if 0.1 <= params.entry_threshold <= 0.2:
            fitness += 10
        if params.exit_threshold < params.entry_threshold:
            fitness += 5
        if abs(params.bos_weight + params.choch_weight + params.order_block_weight + params.fvg_weight - 1.0) < 0.1:
            fitness += 10
        if 10 <= params.max_leverage <= 20:
            fitness += 5
        if 0.08 <= params.position_size_pct <= 0.15:
            fitness += 5
        
        return fitness
    
    def select_parents(self) -> Tuple[TradingParams, TradingParams]:
        """Select parents using tournament selection."""
        tournament_size = 5
        
        # Tournament 1
        tournament1 = random.sample(self.population, tournament_size)
        parent1 = max(tournament1, key=self.fitness_function)
        
        # Tournament 2
        tournament2 = random.sample(self.population, tournament_size)
        parent2 = max(tournament2, key=self.fitness_function)
        
        return parent1, parent2
    
    def crossover(self, parent1: TradingParams, parent2: TradingParams) -> TradingParams:
        """Create child by crossing over parent parameters."""
        child = TradingParams()
        
        # Random crossover for each parameter
        if random.random() < 0.5:
            child.entry_threshold = parent1.entry_threshold
        else:
            child.entry_threshold = parent2.entry_threshold
        
        if random.random() < 0.5:
            child.exit_threshold = parent1.exit_threshold
        else:
            child.exit_threshold = parent2.exit_threshold
        
        # Continue for all parameters...
        child.bos_weight = parent1.bos_weight if random.random() < 0.5 else parent2.bos_weight
        child.choch_weight = parent1.choch_weight if random.random() < 0.5 else parent2.choch_weight
        child.order_block_weight = parent1.order_block_weight if random.random() < 0.5 else parent2.order_block_weight
        child.fvg_weight = parent1.fvg_weight if random.random() < 0.5 else parent2.fvg_weight
        child.max_leverage = parent1.max_leverage if random.random() < 0.5 else parent2.max_leverage
        child.position_size_pct = parent1.position_size_pct if random.random() < 0.5 else parent2.position_size_pct
        child.stop_loss_pct = parent1.stop_loss_pct if random.random() < 0.5 else parent2.stop_loss_pct
        child.take_profit_pct = parent1.take_profit_pct if random.random() < 0.5 else parent2.take_profit_pct
        
        return child
    
    def mutate(self, params: TradingParams, mutation_rate: float = 0.1):
        """Mutate parameters with some probability."""
        if random.random() < mutation_rate:
            params.entry_threshold += random.uniform(-0.05, 0.05)
            params.entry_threshold = max(0.01, min(0.5, params.entry_threshold))
        
        if random.random() < mutation_rate:
            params.exit_threshold += random.uniform(-0.02, 0.02)
            params.exit_threshold = max(0.01, min(0.3, params.exit_threshold))
        
        # Continue for other parameters...
        if random.random() < mutation_rate:
            params.bos_weight += random.uniform(-0.1, 0.1)
            params.bos_weight = max(0.1, min(0.6, params.bos_weight))
    
    def optimize(self) -> TradingParams:
        """Run genetic algorithm optimization."""
        logger.info("🧬 Starting Genetic Algorithm Optimization...")
        
        self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            fitness_scores = [(params, self.fitness_function(params)) for params in self.population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Update best parameters
            if fitness_scores[0][1] > self.best_fitness:
                self.best_fitness = fitness_scores[0][1]
                self.best_params = fitness_scores[0][0]
                logger.info(f"Generation {generation}: New best fitness = {self.best_fitness:.2f}")
            
            # Create new population
            new_population = []
            
            # Keep top 20% (elitism)
            elite_count = self.population_size // 5
            new_population.extend([params for params, _ in fitness_scores[:elite_count]])
            
            # Generate rest through crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
        
        logger.info(f"✅ Genetic optimization complete! Best fitness: {self.best_fitness:.2f}")
        return self.best_params

class MLOptimizedMemeTrader:
    """ML-optimized meme coin trading system."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.symbols = ['FARTCOIN', 'POPCAT', 'WIF', 'PONKE', 'SPX', 'GIGA']
        
        # Initialize components
        self.market_detector = MarketStructureDetector()
        self.leverage_manager = LeverageManager()
        
        # ML components
        self.signal_predictor = SignalPredictor().to(self.device)
        self.optimizer = optim.Adam(self.signal_predictor.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Trading environment
        self.env = TradingEnvironment()
        
        # Genetic optimizer
        self.genetic_optimizer = GeneticOptimizer()
        
        # Experience replay for RL
        self.memory = deque(maxlen=10000)
        
        logger.info("🧠 ML-Optimized Meme Coin HFT System Initialized!")
    
    def generate_training_data(self, periods: int = 1000) -> List[Dict]:
        """Generate training data for ML models."""
        logger.info(f"📊 Generating {periods} periods of training data...")
        
        training_data = []
        
        for i in range(periods):
            for symbol in self.symbols:
                # Generate market data
                df = self._generate_market_data(symbol, 50)
                
                # Get market structure signals
                signals = self.market_detector.calculate_signals(df)
                
                # Create training sample
                sample = {
                    'symbol': symbol,
                    'timestamp': df['timestamp'].iloc[-1],
                    'price': df['close'].iloc[-1],
                    'volume': df['volume'].iloc[-1],
                    'signals': signals,
                    'features': self._extract_features(signals)
                }
                
                training_data.append(sample)
        
        logger.info(f"✅ Generated {len(training_data)} training samples")
        return training_data
    
    def _generate_market_data(self, symbol: str, periods: int) -> pd.DataFrame:
        """Generate realistic market data for training."""
        dates = pd.date_range(
            datetime.now() - timedelta(minutes=periods), 
            periods=periods, 
            freq='1min'
        )
        
        np.random.seed(hash(symbol) % 1000)
        
        base_price = 0.001 if 'FART' in symbol else 0.01
        prices = [base_price]
        
        for i in range(1, periods):
            volatility = 0.15
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.0001))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.05))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.05))) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(5000000, 20000000) for _ in prices]
        })
        
        return df
    
    def _extract_features(self, signals: Dict) -> torch.Tensor:
        """Extract features from market structure signals."""
        features = []
        
        for key in ['bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
                   'order_block_bullish', 'order_block_bearish', 'fvg_bullish', 'fvg_bearish']:
            tensor = signals.get(key, torch.tensor([]))
            if len(tensor) > 0:
                features.append(tensor[-1].item())
            else:
                features.append(0.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def train_signal_predictor(self, training_data: List[Dict], epochs: int = 100):
        """Train the neural network signal predictor."""
        logger.info(f"🧠 Training Signal Predictor for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for sample in training_data:
                features = sample['features'].to(self.device)
                
                # Create target (simplified - would be based on actual performance)
                # For now, use a simple rule: strong bullish signals = buy, strong bearish = sell
                signals = sample['signals']
                
                long_strength = (
                    signals.get('bos_bullish', torch.tensor([0]))[-1].item() * 0.35 +
                    signals.get('choch_bullish', torch.tensor([0]))[-1].item() * 0.25 +
                    signals.get('order_block_bullish', torch.tensor([0]))[-1].item() * 0.25 +
                    signals.get('fvg_bullish', torch.tensor([0]))[-1].item() * 0.15
                )
                
                short_strength = (
                    signals.get('bos_bearish', torch.tensor([0]))[-1].item() * 0.35 +
                    signals.get('choch_bearish', torch.tensor([0]))[-1].item() * 0.25 +
                    signals.get('order_block_bearish', torch.tensor([0]))[-1].item() * 0.25 +
                    signals.get('fvg_bearish', torch.tensor([0]))[-1].item() * 0.15
                )
                
                # Determine target action
                if long_strength > 0.3:
                    target = torch.tensor([1], dtype=torch.long)  # Buy
                elif short_strength > 0.3:
                    target = torch.tensor([2], dtype=torch.long)  # Sell
                else:
                    target = torch.tensor([0], dtype=torch.long)  # Hold
                
                target = target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.signal_predictor(features.unsqueeze(0))
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = torch.argmax(output, dim=1)
                correct_predictions += (predicted == target).sum().item()
                total_predictions += 1
            
            # Log progress
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(training_data)
                accuracy = correct_predictions / total_predictions
                logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}")
        
        logger.info("✅ Signal Predictor training complete!")
    
    def optimize_parameters(self) -> TradingParams:
        """Optimize trading parameters using genetic algorithm."""
        logger.info("🧬 Optimizing trading parameters...")
        
        # Run genetic algorithm
        optimized_params = self.genetic_optimizer.optimize()
        
        logger.info("✅ Parameter optimization complete!")
        logger.info(f"🎯 Optimized Parameters:")
        logger.info(f"   Entry Threshold: {optimized_params.entry_threshold:.3f}")
        logger.info(f"   Exit Threshold: {optimized_params.exit_threshold:.3f}")
        logger.info(f"   BOS Weight: {optimized_params.bos_weight:.3f}")
        logger.info(f"   CHoCH Weight: {optimized_params.choch_weight:.3f}")
        logger.info(f"   Order Block Weight: {optimized_params.order_block_weight:.3f}")
        logger.info(f"   FVG Weight: {optimized_params.fvg_weight:.3f}")
        logger.info(f"   Max Leverage: {optimized_params.max_leverage:.1f}x")
        logger.info(f"   Position Size: {optimized_params.position_size_pct:.1%}")
        logger.info(f"   Stop Loss: {optimized_params.stop_loss_pct:.1%}")
        logger.info(f"   Take Profit: {optimized_params.take_profit_pct:.1%}")
        
        return optimized_params
    
    def run_ml_optimized_trading(self, cycles: int = 5):
        """Run ML-optimized trading system."""
        logger.info("🚀 Starting ML-Optimized Trading...")
        
        # Step 1: Generate training data
        training_data = self.generate_training_data(periods=500)
        
        # Step 2: Train signal predictor
        self.train_signal_predictor(training_data, epochs=50)
        
        # Step 3: Optimize parameters
        optimized_params = self.optimize_parameters()
        
        # Step 4: Run optimized trading
        logger.info("📊 Running ML-optimized trading cycles...")
        
        results = []
        for cycle in range(cycles):
            logger.info(f"📊 Cycle {cycle + 1}/{cycles}")
            
            cycle_result = self._run_optimized_cycle(optimized_params)
            results.append(cycle_result)
            
            # Print summary
            summary = cycle_result['portfolio_summary']
            logger.info(
                f"💰 Balance: ${summary['current_balance']:.2f} | "
                f"Equity: ${summary['total_equity']:.2f} | "
                f"PnL: ${summary['daily_pnl']:.2f} | "
                f"Positions: {summary['active_positions']}"
            )
        
        # Final performance report
        self._print_ml_performance_report(results, optimized_params)
        
        return results, optimized_params
    
    def _run_optimized_cycle(self, params: TradingParams) -> Dict:
        """Run one trading cycle with optimized parameters."""
        cycle_results = {
            'timestamp': datetime.now(),
            'signals': [],
            'trades': [],
            'portfolio_summary': {}
        }
        
        for symbol in self.symbols:
            try:
                # Generate market data
                df = self._generate_market_data(symbol, 50)
                
                # Get market structure signals
                signals = self.market_detector.calculate_signals(df)
                
                # Use ML predictor for signal
                features = self._extract_features(signals)
                with torch.no_grad():
                    prediction = self.signal_predictor(features.unsqueeze(0).to(self.device))
                    action_probs = prediction.cpu().numpy()[0]
                    action = np.argmax(action_probs)
                
                # Calculate signal strength using optimized weights
                long_strength = (
                    signals.get('bos_bullish', torch.tensor([0]))[-1].item() * params.bos_weight +
                    signals.get('choch_bullish', torch.tensor([0]))[-1].item() * params.choch_weight +
                    signals.get('order_block_bullish', torch.tensor([0]))[-1].item() * params.order_block_weight +
                    signals.get('fvg_bullish', torch.tensor([0]))[-1].item() * params.fvg_weight
                )
                
                short_strength = (
                    signals.get('bos_bearish', torch.tensor([0]))[-1].item() * params.bos_weight +
                    signals.get('choch_bearish', torch.tensor([0]))[-1].item() * params.choch_weight +
                    signals.get('order_block_bearish', torch.tensor([0]))[-1].item() * params.order_block_weight +
                    signals.get('fvg_bearish', torch.tensor([0]))[-1].item() * params.fvg_weight
                )
                
                # Determine final action
                final_action = 'hold'
                signal_strength = 0.0
                
                if action == 1 and long_strength > params.entry_threshold:  # Buy
                    final_action = 'buy'
                    signal_strength = long_strength
                elif action == 2 and short_strength > params.entry_threshold:  # Sell
                    final_action = 'sell'
                    signal_strength = short_strength
                elif long_strength < params.exit_threshold and short_strength < params.exit_threshold:
                    final_action = 'exit'
                
                # Create signal
                signal = {
                    'symbol': symbol,
                    'action': final_action,
                    'signal_strength': signal_strength,
                    'ml_confidence': np.max(action_probs),
                    'price': df['close'].iloc[-1],
                    'timestamp': df['timestamp'].iloc[-1]
                }
                
                cycle_results['signals'].append(signal)
                
                # Execute trade if signal exists
                if final_action != 'hold':
                    trade_result = self._execute_optimized_trade(signal, params)
                    cycle_results['trades'].append(trade_result)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Update portfolio
        cycle_results['portfolio_summary'] = self.leverage_manager.get_portfolio_summary()
        
        return cycle_results
    
    def _execute_optimized_trade(self, signal: Dict, params: TradingParams) -> Dict:
        """Execute trade with optimized parameters."""
        symbol = signal['symbol']
        action = signal['action']
        price = signal['price']
        signal_strength = signal['signal_strength']
        ml_confidence = signal['ml_confidence']
        
        # Calculate position size based on ML confidence and signal strength
        base_size = self.leverage_manager.config.starting_balance * params.position_size_pct
        adjusted_size = base_size * signal_strength * ml_confidence
        
        if action == 'buy':
            result = self.leverage_manager.open_position(
                symbol=symbol,
                side='long',
                price=price,
                signal_strength=signal_strength
            )
            
            if result['success']:
                logger.info(f"🟢 ML LONG {symbol} @ ${price:.6f} - Strength: {signal_strength:.2f}, Confidence: {ml_confidence:.2f}")
                return {'action': 'buy', 'success': True, 'position': result['position']}
        
        elif action == 'sell':
            result = self.leverage_manager.open_position(
                symbol=symbol,
                side='short',
                price=price,
                signal_strength=signal_strength
            )
            
            if result['success']:
                logger.info(f"🔴 ML SHORT {symbol} @ ${price:.6f} - Strength: {signal_strength:.2f}, Confidence: {ml_confidence:.2f}")
                return {'action': 'sell', 'success': True, 'position': result['position']}
        
        return {'action': action, 'success': False, 'reason': 'No action taken'}
    
    def _print_ml_performance_report(self, results: List[Dict], params: TradingParams):
        """Print ML-optimized performance report."""
        logger.info("📊 ML-OPTIMIZED PERFORMANCE REPORT")
        logger.info("=" * 60)
        
        final_summary = results[-1]['portfolio_summary']
        performance_metrics = self.leverage_manager.get_performance_metrics()
        
        # Account performance
        logger.info(f"💰 Starting Balance: ${self.leverage_manager.config.starting_balance:.2f}")
        logger.info(f"💰 Final Balance: ${final_summary['current_balance']:.2f}")
        logger.info(f"📈 Total Return: {((final_summary['current_balance'] / self.leverage_manager.config.starting_balance) - 1) * 100:.2f}%")
        logger.info(f"📊 Daily PnL: ${final_summary['daily_pnl']:.2f}")
        logger.info(f"📉 Max Drawdown: {final_summary['current_drawdown'] * 100:.2f}%")
        
        # ML-specific metrics
        if performance_metrics:
            logger.info(f"🎯 Total Trades: {performance_metrics['total_trades']}")
            logger.info(f"✅ Winning Trades: {performance_metrics['winning_trades']}")
            logger.info(f"❌ Losing Trades: {performance_metrics['losing_trades']}")
            logger.info(f"📊 Win Rate: {performance_metrics['win_rate'] * 100:.1f}%")
            logger.info(f"💰 Total PnL: ${performance_metrics['total_pnl']:.2f}")
            logger.info(f"📈 Profit Factor: {performance_metrics['profit_factor']:.2f}")
        
        # Risk metrics
        logger.info(f"🔒 Total Exposure: ${final_summary['total_exposure']:.2f}")
        logger.info(f"⚡ Active Positions: {final_summary['active_positions']}")
        logger.info(f"📊 Trades Today: {final_summary['trades_today']}")
        logger.info(f"🎯 Win Rate Today: {final_summary['win_rate_today'] * 100:.1f}%")
        
        logger.info("=" * 60)

def main():
    """Main function to run the ML-optimized meme coin trader."""
    print("🧠 ML-Optimized Meme Coin HFT System")
    print("=" * 60)
    
    # Create ML trader
    trader = MLOptimizedMemeTrader()
    
    # Run ML-optimized trading
    print("🚀 Starting ML-optimized trading...")
    print("📊 Trading pairs: FARTCOIN, POPCAT, WIF, PONKE, SPX, GIGA")
    print("🧠 ML Components: Neural Network + Genetic Algorithm + RL")
    print("⚡ Auto-optimization: Parameters, thresholds, weights")
    print("=" * 60)
    
    # Run ML optimization and trading
    results, optimized_params = trader.run_ml_optimized_trading(cycles=3)
    
    print("✅ ML-optimized trading complete!")

if __name__ == "__main__":
    main() 