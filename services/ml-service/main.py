#!/usr/bin/env python3
"""
ML Service - Main entry point for the Quantify ML microservice
"""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import your existing modules (these will need import fixes)
try:
    from src.core.models import StandardizedCandle, TimeRange
    from src.core.exceptions import ValidationError, DataFetcherError
    from src.ml.models.strategy.lorentzian_classifier import LorentzianANN  # Fixed class name
    print("✅ Successfully imported existing modules")
except ImportError as e:
    print(f"⚠️  Import issues detected: {e}")
    print("This is normal - we'll fix imports in Phase 3")

app = FastAPI(
    title="Quantify ML Service",
    description="Machine Learning and Analytics Service for Quantify Trading System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"service": "Quantify ML Service", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-service"}

@app.get("/models")
async def list_models():
    """List available ML models"""
    # TODO: Implement model listing
    return {"models": ["lorentzian", "logistic_regression", "neural_network"]}

@app.post("/predict")
async def predict(data: dict):
    """Make predictions using ML models"""
    # TODO: Implement prediction endpoint
    return {"prediction": "not_implemented_yet", "data": data}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
