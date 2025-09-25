#!/usr/bin/env python3
"""
Test script to check import issues in the ml-service
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print(f"Python path: {sys.path[:3]}")

# Test basic imports
try:
    from src.core.exceptions import ValidationError, DataFetcherError
    print("✅ Core exceptions imported successfully")
except ImportError as e:
    print(f"❌ Core exceptions import failed: {e}")

try:
    from src.core.models import StandardizedCandle, TimeRange
    print("✅ Core models imported successfully")
except ImportError as e:
    print(f"❌ Core models import failed: {e}")

try:
    from src.ml.models.strategy.lorentzian_classifier import LorentzianANN
    print("✅ Lorentzian classifier imported successfully")
except ImportError as e:
    print(f"❌ Lorentzian classifier import failed: {e}")

try:
    import torch
    print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

print("\n🎯 Import test completed!")
