# 📜 QuantDesk Scripts Directory

This directory contains all shell scripts organized by purpose and environment.

## 📁 Directory Structure

```
scripts/
├── dev/           # Development scripts
├── deploy/        # Deployment scripts  
├── maintenance/   # Maintenance and utility scripts
└── README.md      # This file
```

## 🛠️ Development Scripts (`dev/`)

### Core Development
- `run-frontend.sh` - Start frontend development server
- `run-tests.sh` - Run all test suites
- `run-all-debug-tests.sh` - Run comprehensive debug tests
- `kill-all.sh` - Kill all running processes
- `kill-frontend.sh` - Kill frontend processes
- `kill-backend.sh` - Kill backend processes

### Backend Development
- `start-backend.sh` - Start backend server
- `test-each-feed.js` - Test individual price feeds
- `test-hermes-*.js` - Hermes service tests
- `test-oracle-*.js` - Oracle service tests
- `test-single-feed.js` - Single feed testing

### Smart Contracts
- `auto-test.sh` - Automated contract testing
- `fix-and-test.sh` - Fix and test contracts
- `quick-test.sh` - Quick contract tests
- `run-comprehensive-tests.sh` - Comprehensive contract tests
- `setup-test.sh` - Setup test environment

## 🚀 Deployment Scripts (`deploy/`)

- `deploy.sh` - Frontend deployment script

## 🔧 Maintenance Scripts (`maintenance/`)

- `security-check.sh` - Security audit script
- `setup-demo.sh` - Demo environment setup

## 📋 Usage Examples

### Start Development Environment
```bash
# Start frontend
./scripts/dev/run-frontend.sh

# Start backend  
./scripts/dev/start-backend.sh

# Run tests
./scripts/dev/run-tests.sh
```

### Deploy Application
```bash
# Deploy frontend
./scripts/deploy/deploy.sh
```

### Maintenance Tasks
```bash
# Security check
./scripts/maintenance/security-check.sh

# Setup demo
./scripts/maintenance/setup-demo.sh
```

## 🔒 Security Notes

- All scripts are executable and should be run with appropriate permissions
- Some scripts may require environment variables to be set
- Always review scripts before execution in production environments

## 📝 Adding New Scripts

When adding new scripts:

1. **Choose the right directory** based on purpose
2. **Make scripts executable**: `chmod +x script-name.sh`
3. **Add documentation** to this README
4. **Test thoroughly** before committing
5. **Follow naming conventions**: `action-purpose.sh`

## 🐛 Troubleshooting

### Common Issues
- **Permission denied**: Run `chmod +x script-name.sh`
- **Command not found**: Check PATH or use full path
- **Environment variables**: Ensure required vars are set

### Getting Help
- Check individual script comments
- Review logs in `backend/logs/`
- Consult main project README
