"""
Performance Metrics for Trading Strategies

This module calculates various trading performance metrics to evaluate strategy effectiveness.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, List, Tuple


def calculate_trading_metrics(actual_returns: np.ndarray, signals: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive trading performance metrics
    
    Parameters:
    -----------
    actual_returns : np.ndarray
        Array of actual price returns for each period
    signals : np.ndarray
        Array of trading signals: 1 (buy), -1 (sell), 0 (hold)
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of performance metrics
    """
    # Convert inputs to numpy arrays if they aren't already
    actual_returns = np.array(actual_returns)
    signals = np.array(signals)
    
    # Create a series of strategy returns based on signals
    strategy_returns = np.zeros_like(actual_returns, dtype=float)
    
    # For simplicity, we'll assume signals are applied to next period's returns
    # Shift signals to align with future returns they're predicting
    shifted_signals = np.zeros_like(signals)
    shifted_signals[:-1] = signals[1:]
    
    # Calculate strategy returns (signal * next period return)
    strategy_returns = shifted_signals * actual_returns
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    
    # Calculate metrics
    metrics = {}
    
    # Core Performance Metrics
    metrics['total_return'] = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
    
    # Count trades (when signal changes)
    trades = np.diff(signals)
    trades = trades[trades != 0]
    metrics['total_trades'] = len(trades)
    
    # Calculate win rate
    winning_trades = np.sum(strategy_returns > 0)
    if metrics['total_trades'] > 0:
        metrics['win_rate'] = winning_trades / metrics['total_trades']
    else:
        metrics['win_rate'] = 0
    
    # Calculate percentage profitable
    metrics['percent_profitable'] = np.sum(strategy_returns > 0) / len(strategy_returns)
    
    # Calculate drawdown
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / (peak + 1)
    metrics['max_drawdown'] = drawdown.min()
    
    # Calculate Sharpe Ratio (assuming risk-free rate of 0)
    if np.std(strategy_returns) > 0:
        metrics['sharpe_ratio'] = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)  # Annualized
    else:
        metrics['sharpe_ratio'] = 0
    
    # Calculate Sortino Ratio (only using downside deviation)
    downside_returns = strategy_returns[strategy_returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        metrics['sortino_ratio'] = np.mean(strategy_returns) / np.std(downside_returns) * np.sqrt(252)
    else:
        metrics['sortino_ratio'] = 0
    
    # Calculate Calmar Ratio
    if metrics['max_drawdown'] != 0:
        metrics['calmar_ratio'] = metrics['total_return'] / abs(metrics['max_drawdown'])
    else:
        metrics['calmar_ratio'] = 0
    
    # Calculate Profit Factor
    gross_profits = np.sum(strategy_returns[strategy_returns > 0])
    gross_losses = abs(np.sum(strategy_returns[strategy_returns < 0]))
    
    if gross_losses > 0:
        metrics['profit_factor'] = gross_profits / gross_losses
    else:
        metrics['profit_factor'] = float('inf') if gross_profits > 0 else 0
    
    # Calculate Expected Value (average profit per trade)
    if metrics['total_trades'] > 0:
        metrics['expected_value'] = np.sum(strategy_returns) / metrics['total_trades']
    else:
        metrics['expected_value'] = 0
    
    # Risk Management Metrics
    avg_win = np.mean(strategy_returns[strategy_returns > 0]) if np.any(strategy_returns > 0) else 0
    avg_loss = np.mean(strategy_returns[strategy_returns < 0]) if np.any(strategy_returns < 0) else 0
    
    if avg_loss != 0:
        metrics['risk_reward_ratio'] = abs(avg_win / avg_loss)
    else:
        metrics['risk_reward_ratio'] = float('inf') if avg_win > 0 else 0
    
    metrics['avg_win'] = avg_win
    metrics['avg_loss'] = avg_loss
    
    # Calculate Maximum Consecutive Losses
    consecutive_losses = 0
    max_consecutive_losses = 0
    
    for ret in strategy_returns:
        if ret < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    
    metrics['max_consecutive_losses'] = max_consecutive_losses
    
    # Recovery Factor
    if metrics['max_drawdown'] != 0:
        metrics['recovery_factor'] = metrics['total_return'] / abs(metrics['max_drawdown'])
    else:
        metrics['recovery_factor'] = float('inf') if metrics['total_return'] > 0 else 0
    
    return metrics


def calculate_portfolio_metrics(portfolio_value: List[float]) -> Dict[str, float]:
    """
    Calculate metrics based on portfolio value over time
    
    Parameters:
    -----------
    portfolio_value : List[float]
        List of portfolio values over time
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of portfolio performance metrics
    """
    portfolio_value = np.array(portfolio_value)
    
    # Calculate returns
    returns = np.diff(portfolio_value) / portfolio_value[:-1]
    
    # Calculate metrics
    metrics = {}
    
    # Total return
    metrics['total_return'] = (portfolio_value[-1] / portfolio_value[0]) - 1
    
    # Calculate drawdown
    peak = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value - peak) / peak
    metrics['max_drawdown'] = drawdown.min()
    
    # Sharpe Ratio (assuming risk-free rate of 0)
    if np.std(returns) > 0:
        metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    else:
        metrics['sharpe_ratio'] = 0
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        metrics['sortino_ratio'] = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
    else:
        metrics['sortino_ratio'] = 0
    
    # Calmar Ratio
    if metrics['max_drawdown'] != 0:
        metrics['calmar_ratio'] = metrics['total_return'] / abs(metrics['max_drawdown'])
    else:
        metrics['calmar_ratio'] = 0
    
    return metrics


def backtest_strategy(price_data: pd.DataFrame, signals: np.ndarray, 
                      initial_capital: float = 10000.0) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Backtest a trading strategy and calculate performance metrics
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        DataFrame with price data (must include 'close' column)
    signals : np.ndarray
        Array of trading signals: 1 (buy), -1 (sell), 0 (hold)
    initial_capital : float, default=10000.0
        Initial capital for the backtest
    
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, float]]
        Portfolio DataFrame and performance metrics
    """
    # Create a copy of the price data
    portfolio = price_data.copy()
    
    # Add signals to the portfolio
    portfolio['signal'] = signals
    
    # Calculate returns
    portfolio['returns'] = portfolio['close'].pct_change()
    
    # Calculate strategy returns
    portfolio['strategy_returns'] = portfolio['signal'].shift(1) * portfolio['returns']
    
    # Calculate cumulative returns
    portfolio['cumulative_returns'] = (1 + portfolio['strategy_returns']).cumprod()
    
    # Calculate portfolio value
    portfolio['portfolio_value'] = initial_capital * portfolio['cumulative_returns']
    
    # Fill NaN values
    portfolio = portfolio.fillna(method='ffill').fillna(method='bfill')
    
    # Calculate performance metrics
    metrics = calculate_portfolio_metrics(portfolio['portfolio_value'].values)
    
    return portfolio, metrics


def generate_performance_report(portfolio: pd.DataFrame, metrics: Dict[str, float]) -> str:
    """
    Generate a performance report as a string
    
    Parameters:
    -----------
    portfolio : pd.DataFrame
        Portfolio DataFrame from backtest_strategy
    metrics : Dict[str, float]
        Performance metrics dictionary
    
    Returns:
    --------
    str
        Performance report as a string
    """
    report = []
    report.append("# Trading Strategy Performance Report\n")
    
    report.append("## Core Performance Metrics")
    report.append(f"- Return on Investment (ROI): {metrics['total_return']*100:.2f}%")
    report.append(f"- Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
    report.append(f"- Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
    
    report.append("\n## Risk-Adjusted Return Metrics")
    report.append(f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    report.append(f"- Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    report.append(f"- Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    
    if 'profit_factor' in metrics:
        report.append("\n## Additional Metrics")
        report.append(f"- Profit Factor: {metrics['profit_factor']:.2f}")
        report.append(f"- Expected Value: {metrics['expected_value']*100:.2f}%")
        report.append(f"- Risk-Reward Ratio: {metrics['risk_reward_ratio']:.2f}")
        report.append(f"- Recovery Factor: {metrics['recovery_factor']:.2f}")
        report.append(f"- Maximum Consecutive Losses: {metrics['max_consecutive_losses']}")
    
    return "\n".join(report) 