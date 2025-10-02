# 🏗️ QuantDesk Project Organization

This document outlines the organized structure of the QuantDesk trading platform project.

## 📁 Project Structure Overview

```
quantdesk/
├── 📁 archive/              # Backup files and old versions
├── 📁 backend/              # Backend server and API
├── 📁 contracts/            # Smart contracts and blockchain code
├── 📁 database/             # Database schemas and migrations
├── 📁 docs/                 # 📚 All documentation (organized)
├── 📁 examples/             # Code examples and demos
├── 📁 frontend/             # React frontend application
├── 📁 scripts/              # 🛠️ All shell scripts (organized)
├── 📁 sdk/                  # Software development kit
├── 📁 tests/                # 🧪 All test files (organized)
├── 📁 test-ledger/          # Test ledger data
├── 📄 README.md             # Main project README
├── 📄 LICENSE               # Project license
└── 📄 TODO.md               # Project todo list
```

## 🗂️ Organized Directories

### 📚 Documentation (`docs/`)
```
docs/
├── api/                     # API documentation
├── architecture/            # System architecture
├── deployment/             # Deployment guides
├── guides/                 # User and developer guides
├── analytics/              # Analytics documentation
├── security/               # Security documentation
├── support/                # Support documentation
├── trading/                # Trading system docs
└── getting-started/        # Getting started guides
```

### 🛠️ Scripts (`scripts/`)
```
scripts/
├── dev/                    # Development scripts
├── deploy/                 # Deployment scripts
├── maintenance/            # Maintenance scripts
└── README.md              # Scripts documentation
```

### 🧪 Tests (`tests/`)
```
tests/
├── integration/            # Integration tests
├── unit/                   # Unit tests
├── e2e/                    # End-to-end tests
├── performance/            # Performance tests
└── README.md              # Tests documentation
```

## 📋 File Organization Summary

### ✅ **Organized Files**

#### Test Scripts → `tests/integration/`
- `test-advanced-orders.js`
- `test-advanced-risk-management.js`
- `test-api-improvements.js`
- `test-backend-websocket.js`
- `test-cross-collateralization.js`
- `test-frontend-price-system.js`
- `test-jit-liquidity.js`
- `test-new-markets.js`
- `test-portfolio-analytics.js`
- `test-pyth-fix.js`
- `debug-pyth-connection.js`
- `scrape-drift-orderbook.js`
- All backend test files (`test-hermes-*.js`, `test-oracle-*.js`)

#### Shell Scripts → `scripts/dev/`
- `run-frontend.sh`
- `run-tests.sh`
- `run-all-debug-tests.sh`
- `kill-all.sh`
- `kill-frontend.sh`
- `kill-backend.sh`
- `start-backend.sh`
- `security-check.sh`
- `setup-demo.sh`
- All smart contract scripts (`auto-test.sh`, `fix-and-test.sh`, etc.)

#### Deployment Scripts → `scripts/deploy/`
- `deploy.sh`

#### Documentation → `docs/`
- **API Docs**: `API.md` → `docs/api/`
- **Deployment**: `DEPLOYMENT_GUIDE.md`, `FRONTEND_DEPLOYMENT.md` → `docs/deployment/`
- **Guides**: `ENVIRONMENT_SETUP.md`, `GETTING_STARTED.md`, `SECURITY_CHECKLIST.md`, `FEATURES.md` → `docs/guides/`
- **Design**: `MOBILE_STRATEGY.md`, `UI_UX_DESIGN_SYSTEM.md`, `LITE_MODE_COLOR_SCHEME.md` → `docs/guides/`
- **Security**: `SECURITY_GUIDE.md` → `docs/guides/`

## 🎯 Benefits of This Organization

### ✅ **Improved Maintainability**
- **Clear separation** of concerns
- **Easy to find** specific files
- **Consistent structure** across project
- **Reduced clutter** in root directory

### ✅ **Better Development Experience**
- **Logical grouping** of related files
- **Clear documentation** for each directory
- **Standardized naming** conventions
- **Easy navigation** for new developers

### ✅ **Enhanced Testing**
- **Organized test suites** by type
- **Clear test documentation**
- **Easy test execution** with proper paths
- **Better test coverage** tracking

### ✅ **Streamlined Deployment**
- **Separate deployment** scripts
- **Environment-specific** configurations
- **Clear deployment** documentation
- **Automated deployment** processes

## 🚀 Quick Start Commands

### Development
```bash
# Start frontend
./scripts/dev/run-frontend.sh

# Start backend
./scripts/dev/start-backend.sh

# Run tests
./scripts/dev/run-tests.sh

# Kill all processes
./scripts/dev/kill-all.sh
```

### Testing
```bash
# Run integration tests
node tests/integration/test-hermes-client.js

# Run specific test category
npm run test:integration

# Debug connections
node tests/integration/debug-pyth-connection.js
```

### Deployment
```bash
# Deploy frontend
./scripts/deploy/deploy.sh

# Security check
./scripts/maintenance/security-check.sh
```

## 📖 Documentation Access

### Quick Reference
- **API Documentation**: `docs/api/API.md`
- **Getting Started**: `docs/guides/GETTING_STARTED.md`
- **Deployment Guide**: `docs/deployment/DEPLOYMENT_GUIDE.md`
- **Mobile Strategy**: `docs/guides/MOBILE_STRATEGY.md`
- **Security Guide**: `docs/guides/SECURITY_GUIDE.md`

### Directory-Specific Docs
- **Scripts**: `scripts/README.md`
- **Tests**: `tests/README.md`
- **Documentation**: `docs/README.md`

## 🔄 Maintenance Guidelines

### Adding New Files
1. **Choose appropriate directory** based on file type
2. **Follow naming conventions** for consistency
3. **Update relevant README** files
4. **Add to this organization** document if needed

### File Naming Conventions
- **Scripts**: `action-purpose.sh` (e.g., `run-frontend.sh`)
- **Tests**: `test-feature-name.js` (e.g., `test-hermes-client.js`)
- **Docs**: `UPPERCASE.md` for main docs, `lowercase.md` for subsections
- **Configs**: `purpose.config.js` (e.g., `vite.config.ts`)

### Directory Updates
- **Keep README files** up-to-date
- **Document new directories** when created
- **Maintain consistent** structure
- **Review organization** quarterly

## 🎉 Project Organization Complete!

The QuantDesk project is now well-organized with:

✅ **Clear directory structure**  
✅ **Organized test files**  
✅ **Consolidated shell scripts**  
✅ **Structured documentation**  
✅ **Comprehensive README files**  
✅ **Easy navigation and maintenance**  

This organization will make development, testing, and maintenance much more efficient and enjoyable! 🚀
