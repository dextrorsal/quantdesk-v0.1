# WSL2 Ubuntu 24.04 Development Setup for QuantDesk

**Complete guide for Windows users to set up WSL2 Ubuntu 24.04 for QuantDesk development**

---

## 🎯 Overview

This guide provides a streamlined approach to setting up WSL2 Ubuntu 24.04 on Windows for QuantDesk development. It combines the best practices from Microsoft's official documentation with project-specific requirements.

**Estimated time:** 1-2 hours  
**Difficulty:** Beginner to Intermediate  
**Prerequisites:** Windows 10 version 2004+ or Windows 11 with administrator access

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Windows 10 version 2004+** or **Windows 11**
- [ ] **8GB RAM minimum** (16GB recommended)
- [ ] **50GB free disk space**
- [ ] **Administrator access**
- [ ] **Stable internet connection**
- [ ] **64-bit processor** with virtualization support

---

## 🚀 Step 1: Install WSL2 and Ubuntu 24.04

### 1.1 Quick Installation (Recommended)

Open **PowerShell as Administrator** and run:

```powershell
# Install WSL2 and Ubuntu 24.04 in one command
wsl --install -d Ubuntu-24.04
```

This single command will:
- Enable WSL2 feature
- Enable Virtual Machine Platform
- Download and install Ubuntu 24.04 LTS
- Set WSL2 as the default version

### 1.2 Restart Your Computer

**Important:** Restart your computer when prompted to complete the installation.

```powershell
# Restart immediately
Restart-Computer
```

### 1.3 Launch Ubuntu and Complete Setup

1. **Launch Ubuntu 24.04** from Start menu
2. **Wait** for initial setup (may take a few minutes)
3. **Create user account** when prompted:

```
Installing, this may take a few minutes...
Please create a default UNIX user account.
Enter new UNIX username: [your-username]
New password:
Retype new password:
```

**Choose a username and password you'll remember!**

---

## ⚙️ Step 2: Configure WSL2

### 2.1 Verify WSL2 Installation

Open PowerShell and verify:

```powershell
# Check WSL version and status
wsl --list --verbose
```

You should see:
```
  NAME            STATE           VERSION
* Ubuntu-24.04    Running         2
```

### 2.2 Update Ubuntu System

In your Ubuntu terminal:

```bash
# Update package list and upgrade system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget git build-essential python3 python3-pip lsof libssl-dev
```

---

## 🔧 Step 3: Install Node.js 20.x

### 3.1 Install Node.js via NodeSource (Recommended)

```bash
# Install Node.js 20.x using NodeSource repository
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version  # Should show v20.x.x
npm --version   # Should show 10.x.x
```

### 3.2 Install Yarn (Optional but Recommended)

```bash
# Install Yarn globally
npm install -g yarn

# Verify installation
yarn --version
```

### 3.3 Alternative: Install via NVM (If you need multiple Node.js versions)

```bash
# Install NVM
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# Reload shell configuration
source ~/.bashrc

# Install and use Node.js 20.x
nvm install 20
nvm use 20
nvm alias default 20

# Verify installation
node --version
npm --version
```

---

## 🐳 Step 4: Install Docker Desktop

### 4.1 Download and Install Docker Desktop

1. **Go to** [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. **Download** Docker Desktop for Windows
3. **Run** the installer as Administrator
4. **Enable WSL2 integration** during installation

### 4.2 Configure WSL Integration

1. **Start Docker Desktop**
2. **Go to Settings** → **Resources** → **WSL Integration**
3. **Enable** integration with Ubuntu-24.04
4. **Apply & Restart**

### 4.3 Test Docker Installation

In your Ubuntu terminal:

```bash
# Test Docker
docker --version
docker run hello-world
```

---

## 📁 Step 5: Set Up QuantDesk Project

### 5.1 Clone the Repository

```bash
# Navigate to home directory
cd ~

# Clone the QuantDesk repository
git clone https://github.com/dextrorsal/quantdesk.git

# Navigate to project directory
cd quantdesk
```

### 5.2 Configure Git

```bash
# Configure Git for WSL (important for line endings)
git config --global core.autocrlf input
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global init.defaultBranch main
```

### 5.3 Install Project Dependencies

```bash
# Install all project dependencies
npm run install:all

# This installs dependencies for:
# - Backend
# - Frontend  
# - Admin Dashboard
# - Data Ingestion (if present)
```

### 5.4 Configure Environment Variables

```bash
# Copy environment template
cp env.example .env

# Edit the environment file
nano .env
```

**Essential configuration:**
```env
# Development Environment
NODE_ENV=development

# Database (using Docker)
DATABASE_URL=postgresql://username:password@localhost:5432/quantdesk

# Redis (using Docker)
REDIS_URL=redis://localhost:6379

# JWT Secret (generate a secure key)
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production-minimum-32-chars

# Solana Network
SOLANA_NETWORK=devnet
RPC_URL=https://api.devnet.solana.com

# Pyth Network
PYTH_NETWORK_URL=https://hermes.pyth.network/v2/updates/price/latest

# Ports
BACKEND_PORT=3001
FRONTEND_PORT=5173
ADMIN_PORT=3000
```

---

## 🧪 Step 6: Start Services

### 6.1 Make Scripts Executable

```bash
# Make all startup scripts executable
chmod +x start-all-services.sh
chmod +x backend/start-backend.sh
chmod +x admin-dashboard/start-admin.sh
chmod +x frontend/start-frontend.sh

# Check if data-ingestion exists and make executable
[ -f "data-ingestion/start-pipeline.sh" ] && chmod +x data-ingestion/start-pipeline.sh
```

### 6.2 Start All Services

```bash
# Start all QuantDesk services
./start-all-services.sh
```

### 6.3 Verify Services Are Running

Open new terminal windows and check:

```bash
# Check backend health
curl http://localhost:3001/health

# Check frontend (should return HTML)
curl http://localhost:5173

# Check admin dashboard
curl http://localhost:3000

# Check running processes
ps aux | grep node
```

---

## 🎯 Step 7: Configure Cursor IDE

### 7.1 Install Cursor IDE

1. **Download** Cursor IDE from [cursor.sh](https://cursor.sh)
2. **Install** Cursor IDE on Windows
3. **Launch** Cursor IDE

### 7.2 Install WSL Extension

1. **Open** Cursor IDE
2. **Go to** Extensions (Ctrl+Shift+X)
3. **Search for** "WSL"
4. **Install** "WSL" extension by Microsoft

### 7.3 Connect to WSL

1. **Click** the green button in bottom-left corner
2. **Select** "Connect to WSL"
3. **Choose** Ubuntu-24.04
4. **Wait** for connection to establish

### 7.4 Open QuantDesk Project

1. **File** → **Open Folder**
2. **Navigate to** `/home/your-username/quantdesk`
3. **Click** "OK"

### 7.5 Configure Terminal

1. **Open** Settings (Ctrl+,)
2. **Search for** "terminal shell"
3. **Set** "Terminal > Integrated > Default Profile: Linux" to "bash"
4. **Verify** Node.js path: `which node` should show the correct path

---

## ✅ Step 8: Validation

### 8.1 Run Validation Script

```bash
# Navigate to WSL directory
cd ~/quantdesk/WSL

# Make validation script executable
chmod +x test-wsl-setup.sh

# Run validation
./test-wsl-setup.sh
```

### 8.2 Manual Verification Checklist

Check these items manually:

- [ ] **WSL2 Ubuntu 24.04** starts successfully
- [ ] **Node.js 20.x** is installed and working
- [ ] **npm/yarn** are working
- [ ] **Docker Desktop** shows WSL integration
- [ ] **QuantDesk repository** is cloned
- [ ] **All dependencies** are installed
- [ ] **Environment variables** are configured
- [ ] **All services** start successfully
- [ ] **Cursor IDE** connects to WSL
- [ ] **Files are accessible** from Windows

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### WSL Won't Start
```powershell
# Check WSL status
wsl --status

# Restart WSL
wsl --shutdown
wsl
```

#### Node.js Not Found
```bash
# Check if Node.js is installed
which node
node --version

# If not found, reinstall
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

#### Docker Not Working
1. **Check** Docker Desktop is running
2. **Verify** WSL integration is enabled in Docker Desktop settings
3. **Restart** Docker Desktop and WSL

#### Services Won't Start
```bash
# Check if ports are in use
sudo lsof -i :3000  # Admin dashboard
sudo lsof -i :3001  # Backend
sudo lsof -i :5173  # Frontend

# Kill processes if needed
sudo kill -9 <PID>
```

#### Cursor IDE Can't Connect
1. **Restart** Cursor IDE
2. **Check** WSL extension is installed
3. **Verify** Ubuntu is running (`wsl --list --running`)
4. **Try** connecting from Command Palette (Ctrl+Shift+P → "WSL: Connect to WSL")

#### Permission Issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER ~/quantdesk
chmod -R 755 ~/quantdesk
```

#### Performance Issues
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

## 🎉 Success!

If you've completed all steps successfully, you should now have:

- ✅ **WSL2 Ubuntu 24.04** running
- ✅ **Node.js 20.x** installed
- ✅ **Docker Desktop** integrated
- ✅ **QuantDesk project** cloned and configured
- ✅ **All services** running
- ✅ **Cursor IDE** connected to WSL

You're now ready to collaborate on the QuantDesk project!

---

## 📞 Getting Help

If you encounter issues:

1. **Check** the troubleshooting section above
2. **Run** the validation script
3. **Check** service logs in the `logs/` directory
4. **Ask** for help in the project communication channel
5. **Share** error messages for specific assistance

---

## 🔗 Useful Commands

```bash
# Quick start services
cd ~/quantdesk && ./start-all-services.sh

# Stop services
pkill -f 'node.*backend'
pkill -f 'node.*frontend'
pkill -f 'node.*admin'

# Check service health
curl http://localhost:3001/health
curl http://localhost:5173
curl http://localhost:3000

# View logs
tail -f logs/*.log

# Update project
cd ~/quantdesk && git pull origin main && npm run install:all

# Open project in Cursor
cd ~/quantdesk && cursor .
```

---

**Next Steps:** Once your setup is working, you can start contributing to the QuantDesk project. Check out the main project README for development guidelines and contribution instructions.

**Happy coding!** 🚀