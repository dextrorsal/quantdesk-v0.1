# 🏗️ Indicators Directory

*What is this folder?*  
This directory contains base classes and utilities for building GPU-accelerated technical indicators. These are used as foundations for custom features in the trading models.

## What's Inside
- `base_torch_indicator.py`: PyTorch base class for all indicators
- `technical_indicators.py`: Shared logic and utilities

## How to Use
- Extend the base classes here to create new indicators
- Used by scripts in `src/features/` and `src/models/strategy/`

---

## See Also
- [Technical Indicators Documentation](../../docs/INDICATORS.md)
- [Features Directory](../features/) — Custom indicator implementations
- [ML Model Architecture](../../docs/ML_MODEL.md) 