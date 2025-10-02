# WSL Development Environment for QuantDesk

**Complete Windows Subsystem for Linux (WSL2) setup for QuantDesk development**

---

## 🎯 Quick Start

For Windows users who want to collaborate on QuantDesk development:

1. **Follow the setup guide**: [WSL_SETUP_GUIDE.md](./WSL_SETUP_GUIDE.md)
2. **Run the automated setup**: `./setup-wsl-dev.sh`
3. **Validate your setup**: `./test-wsl-setup.sh`

---

## 📁 Files in this Directory

| File | Description |
|------|-------------|
| `WSL_SETUP_GUIDE.md` | Complete step-by-step setup guide |
| `setup-wsl-dev.sh` | Automated setup script |
| `test-wsl-setup.sh` | Validation script to test your setup |
| `README.md` | This overview file |

---

## 🚀 What You'll Get

After completing the setup, you'll have:

- ✅ **WSL2 Ubuntu 24.04** running
- ✅ **Node.js 20.x** installed
- ✅ **Docker Desktop** integrated
- ✅ **QuantDesk project** cloned and configured
- ✅ **Cursor IDE** connected to WSL
- ✅ **All development tools** ready to use

---

## 🔧 Prerequisites

- Windows 10 version 2004+ or Windows 11
- 8GB RAM minimum (16GB recommended)
- 50GB free disk space
- Administrator access
- Stable internet connection

---

## 📖 Detailed Instructions

### 1. Install WSL2 and Ubuntu 24.04

```powershell
# Run as Administrator in PowerShell
wsl --install -d Ubuntu-24.04
```

### 2. Restart Your Computer

Restart when prompted to complete the installation.

### 3. Launch Ubuntu and Complete Setup

- Create your user account when prompted
- Update the system: `sudo apt update && sudo apt upgrade -y`

### 4. Run the Automated Setup

```bash
# Navigate to the WSL directory
cd ~/quantdesk/WSL

# Make the setup script executable
chmod +x setup-wsl-dev.sh

# Run the automated setup
./setup-wsl-dev.sh
```

### 5. Validate Your Setup

```bash
# Run the validation script
chmod +x test-wsl-setup.sh
./test-wsl-setup.sh
```

### 6. Start Development

```bash
# Navigate to the project
cd ~/quantdesk

# Start all services
./quick-start.sh

# Open in Cursor IDE
cursor .
```

---

## 🎯 Key Features

### Automated Setup Script
- Installs all required dependencies
- Configures Git for WSL
- Sets up Node.js 20.x
- Configures shell environment
- Creates useful aliases and scripts

### Validation Script
- Checks system requirements
- Verifies all installations
- Tests service health
- Provides detailed feedback
- Generates summary report

### Useful Scripts Created
- `quick-start.sh` - Start all services
- `quick-stop.sh` - Stop all services
- `health-check.sh` - Check service health

---

## 🚨 Troubleshooting

### Common Issues

1. **WSL won't start**
   ```powershell
   wsl --shutdown
   wsl
   ```

2. **Node.js not found**
   ```bash
   curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
   sudo apt-get install -y nodejs
   ```

3. **Docker not working**
   - Check Docker Desktop is running
   - Verify WSL integration is enabled
   - Restart Docker Desktop

4. **Services won't start**
   ```bash
   # Check if ports are in use
   sudo lsof -i :3000
   sudo lsof -i :3001
   sudo lsof -i :5173
   
   # Kill processes if needed
   sudo kill -9 <PID>
   ```

### Performance Tips

1. **Store project in WSL filesystem** (not Windows filesystem)
2. **Use WSL2** (not WSL1)
3. **Allocate sufficient RAM** to WSL in `.wslconfig`

Create `C:\Users\YourUsername\.wslconfig`:
```ini
[wsl2]
memory=8GB
processors=4
swap=2GB
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check** the troubleshooting section above
2. **Run** the validation script: `./test-wsl-setup.sh`
3. **Review** the detailed guide: [WSL_SETUP_GUIDE.md](./WSL_SETUP_GUIDE.md)
4. **Ask** for help in the project communication channel
5. **Share** error messages for specific assistance

---

## 🔗 Useful Commands

```bash
# Quick commands (after setup)
qd              # Navigate to QuantDesk project
qd-start        # Start all services
qd-stop         # Stop all services
qd-logs         # View service logs
qd-health       # Check service health

# Manual commands
cd ~/quantdesk && ./quick-start.sh    # Start services
cd ~/quantdesk && ./quick-stop.sh     # Stop services
cd ~/quantdesk && ./health-check.sh   # Check health
cd ~/quantdesk && cursor .            # Open in Cursor
```

---

## 🎉 Success!

Once your setup is complete, you'll be ready to:

- Collaborate on QuantDesk development
- Use all Linux development tools
- Run the complete QuantDesk stack
- Debug and test in a Linux environment
- Contribute to the project effectively

**Happy coding!** 🚀