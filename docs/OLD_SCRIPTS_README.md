# 📜 Scripts

This directory contains executable scripts for training, deployment, and maintenance.

*For current scripts and usage, see [scripts/README.md](../scripts/README.md).*

## Directory Structure

### 📊 Training
Scripts for model training and evaluation:
- [`train_pattern_recognition.py`](../scripts/training/train_pattern_recognition.py): Train the pattern recognition model
- [`data_pipeline_example.py`](../scripts/training/data_pipeline_example.py): Example of using the data pipeline

### 🚀 Deployment
Scripts for deployment and monitoring:
- [`monitor_gpu.py`](../scripts/deployment/monitor_gpu.py): Monitor GPU usage during training/inference
- [`check_gpu.py`](../scripts/deployment/check_gpu.py): Verify GPU setup and compatibility

## Usage

### Training Scripts
```bash
# Train pattern recognition model
python scripts/training/train_pattern_recognition.py

# Run data pipeline example
python scripts/training/data_pipeline_example.py
```

### Deployment Scripts
```bash
# Monitor GPU usage
python scripts/deployment/monitor_gpu.py

# Check GPU setup
python scripts/deployment/check_gpu.py
``` 

---

## See Also
- [scripts/README.md](../scripts/README.md) — Main scripts and usage guide
- [Model Training Guide](MODEL_TRAINING.md) — How to train and evaluate models
- [Neon Data Pipeline](NEON_PIPELINE.md) — Data ingestion and processing 