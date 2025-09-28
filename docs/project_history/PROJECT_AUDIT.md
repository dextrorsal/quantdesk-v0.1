# ML-MODEL Project Audit Summary

## Top-Level Structure
- **archive/**: Contains all files and folders moved for reference but not active use, including:
  - model-evaluation/ (old comparison module, now archived)
  - tests/archived/ (old or legacy test files)
- **src/**: Main source code directory, organized by feature and function.
- **scripts/**: Automation, training, deployment, and dashboard scripts. See scripts/README.md for details.
- **tests/**: Unit, integration, and data pipeline tests. See tests/README.md for details.

---

## Directory-by-Directory Summary

### src/comparison/
- **Kept:** All main scripts and configs (advanced_backtester.py, analysis_implementation.py, compare_all_implementations.py, compare_with_backtester.py, comparison_config.py, data_fetcher.py, modern_pytorch_implementation.py, simple_test.py, standalone_implementation.py, your_implementation.py, README.md, __init__.py)
  - **Reason:** Core to model comparison, benchmarking, and backtesting. Many are GPU-accelerated or modular.
- **Archived:** All PNG image files (equity_curves.png, model_comparison.png, model_performance_comparison.png) to archive/comparison/.
- **models/** and **results/** subfolders: Empty or not reviewed in detail.

---

### src/data/
- **Kept:** All collectors, pipeline, and processor scripts.
  - **Reason:** Essential for data ingestion, processing, and feature engineering. Includes GPU-aware batch loaders and Neon/Postgres integration.

---

### src/examples/
- **Status:** Empty.

---

### src/features/
- **Kept:** All indicator scripts (adx.py, base_torch_indicator.py, cci.py, chandelier_exit.py, rsi.py, wave_trend.py).
  - **Reason:** Hand-made, GPU-accelerated features essential for model training and strategy development.

---

### src/indicators/
- **Kept:** base_torch_indicator.py, technical_indicators.py.
  - **Reason:** Core GPU-accelerated indicator logic.
- **Archived:** smart_money_concepts.py, market_structure.py, trend_levels.py to archive/indicators/.
  - **Reason:** Useful for reference but not currently prioritized.

---

### src/models/
- **Kept:** strategy/ and training/ subdirectories.
  - **Reason:** Core to model logic and training routines.
- **Archived:** archived/ and states/ subdirectories to archive/models/.
  - **Reason:** Legacy or state management code, not currently active.

---

### src/pattern-recognition/
- **Archived:** Entire directory to archive/.
  - **Reason:** Not currently prioritized, but kept for reference.

---

### src/utils/
- **Kept:** All utility scripts (db_connector.py, performance_metrics.py, position_sizing.py, neon_visualizer.py).
  - **Reason:** Essential for database, metrics, risk management, and visualization.

---

### src/
- **Kept:** strategy_backtest.py, __init__.py.
  - **Reason:** Core backtesting logic and package structure.
- **Archived:** model_comparison.png, model_performance_comparison.png to archive/.
  - **Reason:** Visualizations for reference.

---

### scripts/
- **Kept:** All scripts and subdirectories. See scripts/README.md for detailed descriptions and usage.
- **Archived:** Old scripts documentation moved to docs/OLD_SCRIPTS_README.md.

---

### tests/
- **Kept:** All current test modules and subdirectories. See tests/README.md for detailed descriptions and structure.
- **Archived:** tests/archived/ moved to archive/tests/.

---

## Key Project Themes & Priorities
- **GPU Acceleration:** Priority given to scripts and modules that leverage PyTorch and CUDA for speed and scalability.
- **Custom Features:** Hand-made indicators and feature engineering modules are central to the project's value.
- **Data Pipeline:** Robust ingestion, processing, and database integration for both historical and real-time data.
- **Modularity:** Preference for well-structured, extensible code that can be easily updated or swapped out.
- **Archival:** Legacy, experimental, or less relevant scripts are archived (not deleted) for future reference.

---

## Next Steps
- Continue the audit for any remaining directories or files.
- Use this summary to update `README.md` and/or create a new `docs/PROJECT_AUDIT.md`.
- Add any additional notes or clarifications as you revisit or refactor modules.
- Reference scripts/README.md and tests/README.md for details on scripts and tests.
