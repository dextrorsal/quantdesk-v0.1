# 📁 Project Organization Guide

## Overview
This document outlines the organized structure of the QuantDesk trading system project. All files have been categorized based on their purpose and use case.

## 📂 Directory Structure

### 🏠 Root Level
```
QuantDesk-1.0.1/
├── README.md              # Main project documentation
├── setup.py               # Package installation script
├── environment.yml        # Conda environment configuration
├── .env                   # Environment variables
└── .gitignore            # Git ignore rules
```

### 📜 Documentation (`docs/`)
All project documentation and guides:
- `AGENT_HANDOFF_SUMMARY.md` - AI agent handoff documentation
- `ai-agent-integration.md` - AI agent integration guide
- `AUTOMATED_LAG_TRADING_SYSTEM.md` - Lag trading system documentation
- `CLEANUP_SUMMARY.md` - Project cleanup summary
- `HFT_PROGRESS_SUMMARY.md` - High-frequency trading progress
- `HIGH_LEVERAGE_MEME_COIN_STRATEGY.md` - Meme coin strategy guide
- `IMPLEMENTATION_CHECKLIST.md` - Implementation checklist
- `LIVE_TRADING_READINESS_PLAN.md` - Live trading preparation
- `OPTIMIZATION_STATUS.md` - Optimization status tracking
- `PHASE_1_COMPLETE_SUMMARY.md` - Phase 1 completion summary
- `Project_Overview.md` - Project overview
- `TODO.md` - Project todo list
- `TRADING_SYSTEM_OVERVIEW.md` - Trading system overview
- `summary-update.md` - Summary updates

### 🚀 Scripts (`scripts/`)
Main executable scripts for trading, analysis, and system management:

#### Core Trading Scripts
- `automated_lag_trading_system.py` - Main automated trading system
- `systematic_multi_timeframe_test.py` - Multi-timeframe testing
- `ultra_aggressive_returns_trader.py` - High-performance ML trader
- `paper_trading_cli.py` - Paper trading command interface

#### Analysis & Testing Scripts
- `comprehensive_backtest_suite.py` - Complete backtesting framework
- `conservative_validation_test.py` - Conservative strategy validation
- `enhanced_multi_timeframe_optimizer.py` - Multi-timeframe optimization
- `quick_validation_test.py` - Quick strategy validation
- `final_comparison.py` - Final strategy comparisons
- `run_comparison.py` - Strategy comparison runner

#### Data & Utility Scripts
- `check_data.py` - Data availability checker
- `fetch.py` - Data fetching utility
- `setup_database.py` - Database setup
- `test_db_connection.py` - Database connection testing
- `fetch_all_ohlcv_to_neon.py` - OHLCV data fetching
- `migrate_configs.py` - Configuration migration
- `cleanup_and_refetch.py` - Data cleanup and refetch
- `monitor_fetch_progress.py` - Fetch progress monitoring
- `fix_folder_structure.py` - Folder structure fixes

#### Subdirectories
- `debug/` - Debug scripts (moved from root)
- `training/` - Model training scripts
- `dashboard/` - Dashboard applications
- `deployment/` - Deployment scripts

### 🐛 Debug (`debug/`)
Debug and analysis scripts for troubleshooting:
- `debug_chandelier_signals.py` - Chandelier exit signal debugging
- `debug_logistic_signals.py` - Logistic regression signal debugging

### 🧪 Tests (`tests/`)
All testing files organized by type:

#### Unit Tests (`tests/unit/`)
- `test_chandelier_exit.py` - Chandelier exit unit tests
- `test_fixed_logistic.py` - Fixed logistic regression tests
- `test_logistic_chandelier.py` - Logistic chandelier tests
- `test_updated_backtest.py` - Updated backtest tests

#### Integration Tests (`tests/integration/`)
- `test_fetch_bitget_to_supabase.py` - Bitget to Supabase integration
- `test_lorentzian.py` - Lorentzian strategy integration

#### Strategy Tests (`tests/strategy_tests/`)
- `data_loader.py` - Strategy data loading tests
- `quick_strategy_test.py` - Quick strategy testing
- `simple_test_runner.py` - Simple test runner
- `test_all_strategies_leverage.py` - Leverage strategy tests
- `test_fixed_strategy.py` - Fixed strategy tests
- `test_lag_analysis.py` - Lag analysis tests
- `test_logistic_chandelier.py` - Logistic chandelier tests
- `test_runner.py` - Main test runner

### 📦 Source Code (`src/`)
Core application source code (well-organized):

#### Core (`src/core/`)
- `config.py` - Configuration management
- `exceptions.py` - Custom exceptions
- `logging.py` - Logging setup
- `models.py` - Data models
- `symbol_mapper.py` - Symbol mapping utilities
- `time_utils.py` - Time utilities

#### Data (`src/data/`)
- `csv_storage.py` - CSV data storage
- `data_loader.py` - Data loading utilities
- `database_loader.py` - Database loading
- `bitget_ws_to_supabase.py` - Bitget WebSocket to Supabase
- `providers/` - Data providers
- `pipeline/` - Data processing pipeline
- `processors/` - Data processors

#### Exchanges (`src/exchanges/`)
- `base.py` - Base exchange handler
- `binance/` - Binance exchange integration
- `bitget/` - Bitget exchange integration
- `coinbase/` - Coinbase exchange integration
- `jupiter/` - Jupiter exchange integration
- `auth/` - Authentication modules

#### Machine Learning (`src/ml/`)
- `features/` - Feature engineering
- `indicators/` - Technical indicators
- `models/` - ML models and strategies
- `paper_trading_framework.py` - Paper trading framework
- `strategy_backtest.py` - Strategy backtesting

#### Trading (`src/trading/`)
- `bitget/` - Bitget trading
- `coinbase/` - Coinbase trading
- `devnet/` - Devnet trading
- `jup/` - Jupiter trading
- `mainnet/` - Mainnet trading
- `leverage_manager.py` - Leverage management
- `security/` - Security modules

#### Utilities (`src/utils/`)
- `db_connector.py` - Database connector
- `log_setup.py` - Logging setup
- `performance_metrics.py` - Performance metrics
- `position_sizing.py` - Position sizing
- `wallet/` - Wallet management

#### CLI (`src/cli/`)
- `base.py` - CLI base
- `data/` - Data CLI commands
- `trading/` - Trading CLI commands
- `utils/` - CLI utilities
- `wallet/` - Wallet CLI commands

### 📊 Data (`data/`)
- `historical/` - Historical data storage
- `ohlcv/` - OHLCV data
- `hft_consolidated/` - High-frequency trading consolidated data
- `live.py` - Live data handling
- `processed.py` - Processed data
- `raw.py` - Raw data

### 🤖 Models (`models/`)
- `configs/` - Model configurations
- `hft_trained/` - High-frequency trading trained models

### 📓 Notebooks (`notebooks/`)
- `analysis/` - Analysis notebooks
- `experiments/` - Experimental notebooks

### 📁 Archive (`archive/`)
Archived experimental and redundant scripts:
- 40+ archived scripts from previous development phases
- Includes various ML traders, optimizers, and experimental scripts

## 🎯 File Organization Principles

### 1. **Purpose-Based Organization**
- Files are organized by their primary function
- Clear separation between scripts, tests, debug tools, and documentation

### 2. **Maintainability**
- Related files are grouped together
- Easy to find specific functionality
- Clear naming conventions

### 3. **Developer Experience**
- New developers can quickly understand the project structure
- Clear paths for different types of work (testing, debugging, development)

### 4. **Scalability**
- Structure supports future growth
- Easy to add new categories without disrupting existing organization

## 🔍 Finding Files

### For Development:
- **Core logic**: `src/` directory
- **Scripts**: `scripts/` directory
- **Tests**: `tests/` directory

### For Debugging:
- **Debug tools**: `debug/` directory
- **Debug scripts**: `scripts/debug/` directory

### For Documentation:
- **All docs**: `docs/` directory
- **Main README**: Root level `README.md`

### For Configuration:
- **Model configs**: `models/configs/`
- **Project configs**: `configs/`
- **Environment**: Root level `.env`

## 📝 Adding New Files

When adding new files, follow these guidelines:

1. **Scripts** → `scripts/` (with appropriate subdirectory if needed)
2. **Tests** → `tests/` (unit, integration, or strategy_tests)
3. **Debug tools** → `debug/` or `scripts/debug/`
4. **Documentation** → `docs/`
5. **Core functionality** → `src/` (appropriate subdirectory)
6. **Experimental work** → `archive/` (when no longer needed)

## 🧹 Maintenance

- Regularly review and archive experimental scripts
- Keep documentation updated with new features
- Maintain clear separation between different types of files
- Use consistent naming conventions 