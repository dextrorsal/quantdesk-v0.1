# 🧠 Model Training Guide

*What is this doc?*  
This guide walks you through training and evaluating ML models for the trading system. It's for anyone who wants to run, tune, or understand the model training process.

[ML Model](ML_MODEL.md) | [Technical Strategy](TECHNICAL_STRATEGY.md) | [Project README](../README.md)

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Basic Model Training](#basic-model-training)
3. [Walk-Forward Optimization](#walk-forward-optimization)
4. [Performance Metrics](#performance-metrics)
5. [Visualization Dashboard](#visualization-dashboard)
6. [Tips for Improving Performance](#tips-for-improving-performance)

## Prerequisites

Before training your model, make sure you have:

- **Data Collection System**: Ensure your Binance data collection is running and your database has sufficient historical data
- **Required Packages**: Install the required Python packages
  ```bash
  pip install -r requirements.txt
  ```
- **GPU Setup (Optional)**: For faster training, configure PyTorch to use your GPU
  ```bash
  # Check if GPU is available
  python -c "import torch; print(torch.cuda.is_available())"
  ```

## Basic Model Training

The basic training script trains a model on a fixed train/test split:

```bash
python scripts/train_model.py --data-days 90 --epochs 50
```

### Parameters:

- `--data-days`: Number of days of historical data to use (default: 90)
- `--test-size`: Fraction of data to use for testing (default: 0.2)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--batch-size`: Batch size for training (default: 64)
- `--save-path`: Directory to save the trained model (default: 'models/trained')

### Outputs:

- Trained model saved to `models/trained/lorentzian_classifier_YYYYMMDD_HHMMSS.pt`
- Performance metrics saved to `models/trained/metrics_YYYYMMDD_HHMMSS.json`
- Loss curve plot saved to `models/trained/loss_curve_YYYYMMDD_HHMMSS.png`

## Walk-Forward Optimization

For more realistic performance assessment, use walk-forward optimization:

```bash
python scripts/train_model_walkforward.py --data-days 180 --window-size 30 --test-size 7 --optimize-metric sharpe_ratio
```

### Parameters:

- `--data-days`: Number of days of historical data to use (default: 180)
- `--window-size`: Size of each training window in days (default: 30)
- `--test-size`: Size of each test window in days (default: 7)
- `--epochs`: Number of training epochs per window (default: 30)
- `--optimize-metric`: Metric to optimize (choices: 'total_return', 'win_rate', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio'; default: 'sharpe_ratio')

### Outputs:

- Best model from each window saved to `models/trained/best_model_window_N.pt`
- Final best model saved to `models/trained/best_model_final_YYYYMMDD_HHMMSS.pt`
- Window metrics saved to `models/trained/window_metrics_YYYYMMDD_HHMMSS.csv`
- Visualization outputs saved to `models/trained/visualizations/`

## Performance Metrics

The system calculates and tracks these key performance metrics:

### Core Performance Metrics
- **Return on Investment (ROI)**: Total percentage return
- **Win Rate**: Percentage of winning trades
- **Percentage Profitable**: Percentage of profitable periods
- **Maximum Drawdown**: Largest peak-to-trough decline

### Risk-Adjusted Metrics
- **Sharpe Ratio**: Risk-adjusted returns (higher is better, >1.5 is good)
- **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
- **Calmar Ratio**: Annual return divided by maximum drawdown

### Advanced Metrics
- **Profit Factor**: Gross profits divided by gross losses (>2 is strong)
- **Expected Value**: Average profit per trade
- **Risk-Reward Ratio**: Potential profit relative to potential loss
- **Recovery Factor**: Total net profit divided by max drawdown
- **Maximum Consecutive Losses**: Resilience during losing streaks

## Visualization Dashboard

To visualize model performance, use the dashboard:

```bash
# Install dashboard requirements
pip install -r scripts/dashboard/requirements-dashboard.txt

# Run the dashboard
python scripts/dashboard/model_dashboard.py
```

The dashboard will be available at http://127.0.0.1:8050/ and includes:
- Model performance metrics
- Training session comparison
- Window-by-window performance visualization
- Metrics correlation analysis

## Tips for Improving Performance

### 1. Feature Engineering
- Add additional technical indicators
- Experiment with different timeframes
- Include market sentiment data
- Consider inter-market relationships (BTC/ETH influence on SOL)

### 2. Model Tuning
- Adjust threshold values for signal generation
- Experiment with different model architectures
- Use learning rate schedulers for better convergence
- Implement early stopping to prevent overfitting

### 3. Dataset Optimization
- Ensure balanced class distribution
- Increase historical data timespan
- Use more granular data for training
- Experiment with different labeling strategies

### 4. Hyperparameter Optimization
Try these hyperparameters:
- Learning rates: [0.01, 0.001, 0.0001]
- Batch sizes: [32, 64, 128]
- Hidden layer sizes: [32, 64, 128, 256]
- Dropout rates: [0.1, 0.2, 0.3, 0.5]

### 5. Risk Management
- Implement position sizing based on volatility
- Use asymmetric thresholds (higher for buy signals)
- Add stop-loss and take-profit logic
- Consider time-based exit strategies

## Advanced Techniques

For even better results, consider these advanced techniques:

1. **Ensemble Methods**: Combine multiple models for more robust predictions
2. **Adversarial Validation**: Ensure your validation set truly represents live conditions
3. **Monte Carlo Simulation**: Assess strategy robustness under different market conditions
4. **Reinforcement Learning**: Train an agent to learn optimal trading policies
5. **Transfer Learning**: Pretrain on related markets, then fine-tune on Solana

---

Remember that the goal is to build a robust model that performs well across various market conditions, not just one that fits historical data perfectly. Focus on risk-adjusted metrics like Sharpe and Sortino ratios rather than just total returns. 

## See Also
- [Project README](../README.md) — Project overview and structure
- [ML Model Architecture](ML_MODEL.md) — Model structure and integration
- [Technical Strategy](TECHNICAL_STRATEGY.md) — How training fits into the trading system
- [Neon Data Pipeline](NEON_PIPELINE.md) — Data ingestion and feature engineering
- [scripts/train_model.py](../scripts/train_model.py) — Main training script
- [scripts/train_model_walkforward.py](../scripts/train_model_walkforward.py) — Walk-forward optimization script
- [scripts/dashboard/model_dashboard.py](../scripts/dashboard/model_dashboard.py) — Visualization dashboard 