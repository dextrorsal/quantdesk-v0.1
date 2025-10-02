# Cursor IDE WSL Configuration Guide

**Configure Cursor IDE to work seamlessly with WSL2 Ubuntu 24.04 for QuantDesk development**

---

## 🎯 Overview

This guide will help you configure Cursor IDE to work with WSL2 Ubuntu 24.04, enabling seamless development of the QuantDesk project with AI agents working in the Linux environment.

**Prerequisites:**
- WSL2 Ubuntu 24.04 installed and configured
- QuantDesk project cloned in WSL
- Cursor IDE installed on Windows

---

## 📥 Step 1: Install Cursor IDE

### 1.1 Download Cursor IDE

1. **Go to** [cursor.sh](https://cursor.sh)
2. **Download** Cursor IDE for Windows
3. **Run** the installer
4. **Follow** the installation wizard

### 1.2 Launch Cursor IDE

1. **Open** Cursor IDE from Start menu
2. **Complete** the initial setup
3. **Sign in** to your account (optional)

---

## 🔌 Step 2: Install WSL Extension

### 2.1 Open Extensions Panel

1. **Click** the Extensions icon in the sidebar (Ctrl+Shift+X)
2. **Search for** "WSL"
3. **Install** "WSL" extension by Microsoft

### 2.2 Verify Installation

1. **Check** that the WSL extension is installed
2. **Look for** the WSL icon in the status bar
3. **Verify** the extension is enabled

---

## 🔗 Step 3: Connect to WSL

### 3.1 Connect to WSL Environment

1. **Click** the green button in the bottom-left corner of Cursor
2. **Select** "Connect to WSL"
3. **Choose** "Ubuntu-24.04" from the list
4. **Wait** for the connection to establish

### 3.2 Verify Connection

You should see:
- **WSL: Ubuntu-24.04** in the status bar
- **Terminal** opens in WSL environment
- **File explorer** shows WSL file system

---

## 📁 Step 4: Open QuantDesk Project

### 4.1 Open Project Folder

1. **File** → **Open Folder**
2. **Navigate to** `/home/your-username/quantdesk`
3. **Click** "OK"

### 4.2 Verify Project Structure

You should see the QuantDesk project structure:
```
quantdesk/
├── backend/
├── frontend/
├── admin-dashboard/
├── WSL/
├── package.json
└── README.md
```

---

## ⚙️ Step 5: Configure Terminal

### 5.1 Set Default Terminal

1. **Open** Command Palette (Ctrl+Shift+P)
2. **Type** "Terminal: Select Default Profile"
3. **Choose** "WSL" or "Ubuntu-24.04"

### 5.2 Configure Terminal Settings

1. **Go to** Settings (Ctrl+,)
2. **Search for** "terminal.integrated.defaultProfile.windows"
3. **Set** to "WSL"

---

## 🤖 Step 6: Configure AI Agents

### 6.1 Verify AI Agent Access

1. **Open** a terminal in Cursor
2. **Type** `pwd` - should show `/home/your-username/quantdesk`
3. **Type** `node --version` - should show Node.js version
4. **Type** `npm --version` - should show npm version

### 6.2 Test AI Agent Commands

Try these commands in the terminal:
```bash
# Check Node.js
node --version

# Check npm
npm --version

# Check project structure
ls -la

# Start services (if configured)
./start-all-services.sh
```

---

## 🔧 Step 7: Configure Development Environment

### 7.1 Install Recommended Extensions

Install these extensions for better development experience:

1. **JavaScript (ES6) code snippets**
2. **TypeScript Importer**
3. **Auto Rename Tag**
4. **Bracket Pair Colorizer**
5. **GitLens**
6. **Prettier**
7. **ESLint**

### 7.2 Configure Settings

Create or update `.vscode/settings.json` in your project:

```json
{
  "terminal.integrated.defaultProfile.windows": "WSL",
  "files.associations": {
    "*.env": "dotenv"
  },
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "typescript.preferences.importModuleSpecifier": "relative",
  "javascript.preferences.importModuleSpecifier": "relative"
}
```

---

## 🧪 Step 8: Test the Setup

### 8.1 Test File Access

1. **Open** `package.json` from the file explorer
2. **Edit** the file
3. **Save** the file
4. **Verify** changes are reflected in WSL

### 8.2 Test Terminal Commands

Run these commands in the Cursor terminal:

```bash
# Navigate to project
cd ~/quantdesk

# Check Node.js
node --version

# Install dependencies (if not done)
npm install

# Start services
./start-all-services.sh
```

### 8.3 Test AI Agent Integration

1. **Open** a TypeScript/JavaScript file
2. **Use** AI features (Ctrl+K, Ctrl+L)
3. **Verify** AI agents can access WSL files
4. **Test** code generation and editing

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### Cursor Can't Connect to WSL
1. **Check** WSL is running: `wsl --list --verbose`
2. **Restart** WSL: `wsl --shutdown` then `wsl`
3. **Restart** Cursor IDE
4. **Check** WSL extension is installed

#### Files Not Accessible
1. **Verify** you're connected to WSL (check status bar)
2. **Check** file path is correct
3. **Restart** Cursor IDE
4. **Reconnect** to WSL

#### Terminal Not Working
1. **Check** default terminal profile is set to WSL
2. **Open** new terminal (Ctrl+Shift+`)
3. **Verify** WSL connection
4. **Check** Ubuntu is running

#### AI Agents Not Working
1. **Verify** you're in the correct directory
2. **Check** Node.js and npm are accessible
3. **Test** basic commands in terminal
4. **Restart** Cursor IDE

#### Performance Issues
1. **Close** unnecessary files
2. **Disable** unused extensions
3. **Check** WSL memory allocation
4. **Restart** WSL and Cursor

---

## ✅ Success Criteria

You'll know the setup is working when:

1. **Cursor IDE** connects to WSL successfully
2. **QuantDesk project** opens from WSL file system
3. **Terminal** runs in WSL environment
4. **File editing** works seamlessly
5. **AI agents** can access and modify files
6. **Node.js commands** work in terminal
7. **Project services** can be started from Cursor

---

## 🎯 Best Practices

### Development Workflow

1. **Always** work from WSL terminal in Cursor
2. **Use** WSL file system for all project files
3. **Run** all commands in WSL environment
4. **Use** AI agents for code generation and editing
5. **Commit** changes from WSL terminal

### File Management

1. **Keep** all project files in WSL
2. **Don't** edit files from Windows file explorer
3. **Use** Cursor's file explorer for navigation
4. **Backup** important files regularly

### Performance Tips

1. **Close** unused files and tabs
2. **Disable** unnecessary extensions
3. **Use** WSL2 for better performance
4. **Allocate** sufficient memory to WSL

---

## 📞 Getting Help

### If You Get Stuck

1. **Check** this troubleshooting guide
2. **Verify** WSL is running correctly
3. **Test** basic WSL commands
4. **Restart** Cursor IDE and WSL
5. **Ask** for help in the project chat

### Useful Commands

```bash
# Check WSL status
wsl --list --verbose

# Restart WSL
wsl --shutdown
wsl

# Check Node.js
node --version
npm --version

# Navigate to project
cd ~/quantdesk

# Check project status
ls -la
```

---

## 🎉 Next Steps

Once Cursor IDE is configured with WSL:

1. **Start** developing on QuantDesk
2. **Use** AI agents for code assistance
3. **Run** services from Cursor terminal
4. **Commit** changes using Git in WSL
5. **Collaborate** with the team

**Happy coding with Cursor IDE and WSL!** 🚀
