# SOL Trading Dashboard

## How to Run the Trading System

1. Make sure your ML model environment is activated:
   ```bash
   conda activate ML-torch
   ```

2. Run the all-in-one trading system script:
   ```bash
   python scripts/start_trading_system.py
   ```

3. Open your browser and go to:
   ```
   http://127.0.0.1:5000
   ```

## Dashboard Features

The dashboard provides a visual representation of your trading system with:

1. **Real-time Price Chart**: Displays the SOL price with real-time updates

2. **Signal Alerts**: Shows trading signals as they occur, with:
   - Color-coded signal types:
     - Purple: TradingView-style signals
     - Blue: Combined signals (weighted approach)
     - Orange: Strong signals (both models agree)
   - Confidence level bars for both models
   - Timestamp and price at signal generation

3. **Performance Statistics**: Shows:
   - Total number of signals
   - Win rate
   - Average return
   - Last signal time

## Filtering Signals

You can filter signals by type using the dropdown menu in the Trading Signals section:
- All Signals
- TradingView Style
- Combined
- Strong

## Customizing Thresholds

If you want to adjust the confidence thresholds, run the script with custom parameters:

```bash
python scripts/start_trading_system.py --confidence-threshold 0.28 --combined-threshold 0.22
```

## Stopping the System

To stop both the trading system and dashboard, press `Ctrl+C` in the terminal where you started the system.

## Troubleshooting

If the dashboard isn't showing any data:
1. Check the logs in the `logs` directory
2. Make sure the trading system is running
3. Try refreshing the dashboard by clicking the 🔄 icon

If trading signals aren't appearing:
1. The model might not have detected favorable trading conditions yet
2. Try lowering the confidence thresholds
3. Make sure the models are loaded correctly 