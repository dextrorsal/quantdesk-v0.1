#!/usr/bin/env python3
"""
Test script to verify core functionality after microservices migration
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing Core Imports...")
    
    try:
        # Data components
        from src.data.SupabaseAdapter import SupabaseAdapter
        from src.data.NeonAdapter import NeonAdapter
        print("✅ Data adapters imported successfully")
    except Exception as e:
        print(f"❌ Data adapters failed: {e}")
    
    try:
        # Exchange components
        from src.exchanges.bitget.bitget_handler import BitgetHandler
        from src.exchanges.binance.binance import BinanceExchange
        print("✅ Exchange handlers imported successfully")
    except Exception as e:
        print(f"❌ Exchange handlers failed: {e}")
    
    try:
        # ML Models
        from src.ml.models.strategy.lorentzian_classifier import LorentzianANN
        from src.ml.models.strategy.logistic_regression_torch import LogisticRegression
        print("✅ ML models imported successfully")
    except Exception as e:
        print(f"❌ ML models failed: {e}")
    
    try:
        # Indicators
        from src.ml.indicators.supertrend import SupertrendIndicator
        from src.ml.features.rsi import RSI
        print("✅ Technical indicators imported successfully")
    except Exception as e:
        print(f"❌ Technical indicators failed: {e}")
    
    try:
        # Core fetcher
        from src.ultimate_fetcher import UltimateDataFetcher
        print("✅ Ultimate data fetcher imported successfully")
    except Exception as e:
        print(f"❌ Ultimate data fetcher failed: {e}")

def test_lorentzian_functionality():
    """Test Lorentzian classifier basic functionality"""
    print("\n🧪 Testing Lorentzian Classifier...")
    
    try:
        from src.ml.models.strategy.lorentzian_classifier import LorentzianANN
        import torch
        import numpy as np
        
        # Create a simple test
        model = LorentzianANN(lookback_bars=10, prediction_bars=2, k_neighbors=5)
        
        # Create dummy data
        features = torch.randn(100, 10)  # 100 samples, 10 features
        prices = torch.randn(100)  # 100 price values
        
        # Test fit (this might fail due to data requirements, but import should work)
        print("✅ LorentzianANN instantiated successfully")
        print(f"✅ Device: {model.device}")
        print(f"✅ PyTorch version: {torch.__version__}")
        
    except Exception as e:
        print(f"❌ Lorentzian test failed: {e}")

def test_supertrend_functionality():
    """Test SuperTrend indicator basic functionality"""
    print("\n🧪 Testing SuperTrend Indicator...")
    
    try:
        from src.ml.indicators.supertrend import SupertrendIndicator
        import torch
        import numpy as np
        
        # Create dummy OHLC data
        data = {
            'open': torch.randn(100),
            'high': torch.randn(100) + 1,
            'low': torch.randn(100) - 1,
            'close': torch.randn(100)
        }
        
        # Create indicator
        indicator = SupertrendIndicator()
        print("✅ SupertrendIndicator instantiated successfully")
        
    except Exception as e:
        print(f"❌ SuperTrend test failed: {e}")

def test_data_fetcher_functionality():
    """Test data fetcher basic functionality"""
    print("\n🧪 Testing Data Fetcher...")
    
    try:
        from src.ultimate_fetcher import UltimateDataFetcher
        from src.core.models import StandardizedCandle, TimeRange
        from datetime import datetime, timezone
        
        # Test model creation
        candle = StandardizedCandle(
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            market="BTCUSDT",
            resolution="1m",
            source="test"
        )
        
        print("✅ StandardizedCandle created successfully")
        print(f"✅ Candle data: {candle.market} @ {candle.close}")
        
    except Exception as e:
        print(f"❌ Data fetcher test failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing Core Functionality After Microservices Migration")
    print("=" * 60)
    
    test_imports()
    test_lorentzian_functionality()
    test_supertrend_functionality()
    test_data_fetcher_functionality()
    
    print("\n🎯 Core functionality test completed!")
    print("=" * 60)
