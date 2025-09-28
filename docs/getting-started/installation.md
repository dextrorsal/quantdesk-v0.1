# 💻 Installation Guide

This guide covers installing QuantDesk on different operating systems.

## 📋 Prerequisites

### Required Software
- **Python 3.11+** - Download from [python.org](https://python.org)
- **Git** - Download from [git-scm.com](https://git-scm.com)
- **Conda** (recommended) - Download from [anaconda.com](https://anaconda.com)

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **GPU**: Optional but recommended for ML strategies

## 🖥️ Installation by Operating System

### Windows

#### Option 1: Using Conda (Recommended)
```bash
# Install Anaconda from anaconda.com
# Open Anaconda Prompt

# Clone QuantDesk
git clone https://github.com/dextrorsal/quantdesk.git
cd QuantDesk

# Create environment
conda create -n QuantDesk python=3.11
conda activate QuantDesk

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using Python Only
```bash
# Install Python 3.11+ from python.org
# Open Command Prompt

# Clone QuantDesk
git clone https://github.com/dextrorsal/quantdesk.git
cd QuantDesk

# Create virtual environment
python -m venv quantdesk-env
quantdesk-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python@3.11 git

# Clone QuantDesk
git clone https://github.com/dextrorsal/quantdesk.git
cd QuantDesk

# Create environment
python3.11 -m venv quantdesk-env
source quantdesk-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)

```bash
# Update system
sudo apt update

# Install Python and Git
sudo apt install python3.11 python3.11-venv python3-pip git

# Clone QuantDesk
git clone https://github.com/dextrorsal/quantdesk.git
cd QuantDesk

# Create environment
python3.11 -m venv quantdesk-env
source quantdesk-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🔧 GPU Setup (Optional)

### NVIDIA GPU (CUDA)
```bash
# Install CUDA toolkit
# Download from nvidia.com/cuda

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### AMD GPU (ROCm)
```bash
# Install ROCm (Linux only)
# Follow AMD ROCm installation guide

# Install PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

## ✅ Verify Installation

### Test Python Environment
```bash
# Activate environment
conda activate QuantDesk  # or source quantdesk-env/bin/activate

# Test Python
python --version
# Should show: Python 3.11.x

# Test imports
python -c "import pandas, numpy, torch; print('✅ All packages installed')"
```

### Test QuantDesk Installation
```bash
# Test QuantDesk imports
python -c "
import sys
sys.path.append('src')
from data.csv_storage import CSVStorage
from ml.paper_trading_framework import PaperTradingFramework
print('✅ QuantDesk installed successfully')
"
```

## 🚨 Common Installation Issues

### "Python not found"
- Make sure Python 3.11+ is installed
- Check PATH environment variable
- Try using `python3` instead of `python`

### "Permission denied"
- Use `sudo` on Linux/macOS
- Run Command Prompt as Administrator on Windows
- Check file permissions

### "Package installation failed"
```bash
# Upgrade pip
pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v

# Try installing packages individually
pip install pandas numpy torch
```

### "CUDA/ROCm not working"
- Verify GPU drivers are installed
- Check CUDA/ROCm installation
- Test with: `python -c "import torch; print(torch.cuda.is_available())"`

## 🎯 Next Steps

After successful installation:

1. **Configure Environment**: [Configuration Guide](configuration.md)
2. **Get Started**: [Quick Start Guide](quick-start.md)
3. **Web Interface**: [Web UI Guide](../user-guide/web-ui.md)

## 🆘 Still Having Issues?

- **Check System Requirements**: Make sure you meet minimum requirements
- **Update Dependencies**: Try updating pip and packages
- **Clean Installation**: Remove and reinstall everything
- **Get Help**: Contact support or check GitHub issues

---

*Previous: [Getting Started](README.md) | Next: [Configuration Guide](configuration.md)*
