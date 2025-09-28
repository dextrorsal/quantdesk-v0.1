"""
Rule-Based Strategy Search - Find optimal indicator combinations

This script searches for the best combinations of technical indicators
using rule-based logic (AND/OR combinations) to find profitable trading strategies.

USE CASES:
- **Strategy discovery**: Find profitable combinations of technical indicators
- **Indicator optimization**: Test different indicator combinations systematically
- **Rule-based trading**: Develop simple rule-based trading strategies
- **Feature selection**: Identify which indicators work best together
- **Strategy research**: Explore different trading logic combinations
- **Performance screening**: Screen for high-performing indicator combinations

DIFFERENCES FROM OTHER STRATEGY SCRIPTS:
- rule_based_strategy_search.py: Rule-based indicator combination search
- run_comparison.py: Strategy comparison with multiple models
- final_comparison.py: Final evaluation with optimized parameters
- combined_model_trader.py: Multi-timeframe model combination and trading
- start_trading_system.py: Complete system orchestration

WHEN TO USE:
- When you want to find profitable indicator combinations
- For systematic strategy discovery and research
- When developing rule-based trading systems
- For feature selection in ML models
- When exploring different trading logics

FEATURES:
- Multiple technical indicators (RSI, ADX, CCI, WaveTrend, ChandelierExit)
- AND/OR logic combinations
- Systematic strategy testing
- Performance metrics calculation
- Results ranking and summary
- Automated strategy discovery

EXAMPLES:
    # Run rule-based strategy search
    python scripts/rule_based_strategy_search.py
    
    # This will test all 2-indicator combinations with AND/OR logic
    # and rank them by ROI, Sharpe ratio, and max drawdown
"""
import asyncio
import numpy as np
import pandas as pd
import torch
from itertools import combinations

from src.data.database_loader import load_trading_data
from src.ml.features.rsi import RSIIndicator
from src.ml.features.adx import ADXIndicator
from src.ml.features.cci import CCIIndicator
from src.ml.features.wave_trend import WaveTrendIndicator
from src.ml.features.chandelier_exit import ChandelierExitIndicator
from src.utils.performance_metrics import backtest_strategy


# --- Config ---
SYMBOL = "SOL/USDT"
RESOLUTION = "15m"
EXCHANGE = "binance"
DAYS = 90
INITIAL_CAPITAL = 10000.0

INDICATOR_CLASSES = [
    ("RSI", RSIIndicator),
    ("ADX", ADXIndicator),
    ("CCI", CCIIndicator),
    ("WaveTrend", WaveTrendIndicator),
    ("ChandelierExit", ChandelierExitIndicator),
]


# --- Helper: Convert torch signals to numpy ---
def to_np(signal):
    if isinstance(signal, torch.Tensor):
        return signal.cpu().numpy()
    return np.asarray(signal)


# --- Main Pipeline ---
async def main():
    print(
        f"Loading data for {SYMBOL} {RESOLUTION} {EXCHANGE} "
        f"({DAYS} days)..."
    )
    df = await load_trading_data(SYMBOL, RESOLUTION, EXCHANGE, DAYS)
    if df.empty:
        print("No data loaded. Exiting.")
        return

    # --- Generate signals for all indicators ---
    print("Generating indicator signals...")
    indicator_signals = {}
    for name, cls in INDICATOR_CLASSES:
        try:
            ind = cls()
            signals = ind.calculate_signals(df)
            indicator_signals[name] = {
                'buy': to_np(signals['buy_signals']),
                'sell': to_np(signals['sell_signals'])
            }
            print(f"  {name}: signals generated.")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # --- Generate all 2-indicator combos ---
    print("\nTesting all 2-indicator rule-based strategies...")
    results = []
    for (name1, _), (name2, _) in combinations(INDICATOR_CLASSES, 2):
        for logic in ["AND", "OR"]:
            # Buy: both/any buy; Sell: both/any sell
            if logic == "AND":
                buy = (
                    indicator_signals[name1]['buy'] &
                    indicator_signals[name2]['buy']
                )
                sell = (
                    indicator_signals[name1]['sell'] &
                    indicator_signals[name2]['sell']
                )
            else:
                buy = (
                    indicator_signals[name1]['buy'] |
                    indicator_signals[name2]['buy']
                )
                sell = (
                    indicator_signals[name1]['sell'] |
                    indicator_signals[name2]['sell']
                )
            # Convert to trading signal: 1=buy, -1=sell, 0=hold
            signals = np.zeros(len(df))
            signals[buy == 1] = 1
            signals[sell == 1] = -1
            # Backtest
            _, metrics = backtest_strategy(
                df, signals, initial_capital=INITIAL_CAPITAL
            )
            results.append({
                'ind1': name1,
                'ind2': name2,
                'logic': logic,
                'roi': metrics.get('total_return', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'sharpe': metrics.get('sharpe_ratio', 0),
            })
            print(
                f"  {name1} {logic} {name2}: "
                f"ROI={metrics.get('total_return', 0):.2%}, "
                f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
                f"MaxDD={metrics.get('max_drawdown', 0):.2%}"
            )

    # --- Output summary table ---
    print("\n=== Top 2-Indicator Strategies (by ROI) ===")
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='roi', ascending=False)
    print(
        df_results[
            ['ind1', 'logic', 'ind2', 'roi', 'sharpe', 'max_drawdown']
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    asyncio.run(main()) 