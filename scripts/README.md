# Scripts Directory

This directory contains utility scripts for development, deployment, testing, and maintenance of the QuantDesk platform.

## 📁 Structure

```
scripts/
├── dev/              # Development scripts
├── deploy/           # Deployment scripts
├── ci-cd/            # CI/CD pipeline scripts
├── security/         # Security-related scripts
├── maintenance/      # Maintenance scripts
├── git/              # Git utility scripts
├── seed/             # Database seeding scripts
└── README.md         # This file
```

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have the required tools
node --version  # Node.js 20+
pnpm --version  # pnpm package manager
```

### Running Scripts
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run a script
./scripts/env-scanner.sh
```

## 📚 Script Categories

### 1. Development Scripts (`dev/`)
- **Environment Setup** - Development environment configuration
- **Database Setup** - Local database initialization
- **Service Management** - Start/stop development services
- **Code Generation** - Generate boilerplate code

### 2. Deployment Scripts (`deploy/`)
- **Production Deployment** - Deploy to production
- **Staging Deployment** - Deploy to staging
- **Database Migration** - Run database migrations
- **Health Checks** - Verify deployment health

### 3. CI/CD Scripts (`ci-cd/`)
- **Build Pipeline** - Automated build process
- **Test Pipeline** - Automated testing
- **Deploy Pipeline** - Automated deployment
- **Rollback Pipeline** - Automated rollback

### 4. Security Scripts (`security/`)
- **Security Scan** - Security vulnerability scanning
- **Audit Scripts** - Security audit utilities
- **Key Management** - API key management
- **Access Control** - Permission management

### 5. Maintenance Scripts (`maintenance/`)
- **Database Cleanup** - Database maintenance
- **Log Rotation** - Log file management
- **Backup Scripts** - Data backup utilities
- **Performance Monitoring** - Performance tracking

### 6. Git Scripts (`git/`)
- **Branch Management** - Git branch utilities
- **Commit Hooks** - Git commit hooks
- **Release Management** - Release automation
- **Changelog Generation** - Automated changelogs

### 7. Seed Scripts (`seed/`)
- **Database Seeding** - Populate database with test data
- **Market Data** - Seed market data
- **User Data** - Seed user accounts
- **Test Data** - Generate test data

## 🔧 Common Scripts

### Environment Scanner
```bash
# Scan for environment variable issues
./scripts/env-scanner.sh

# Validate environment configuration
./scripts/validate-environment-migration.js
```

### Database Management
```bash
# Seed essential markets
./scripts/seed-markets.sh

# Run database migrations
./scripts/migrate-environment.sh
```

### Security Audits
```bash
# Run security audit
./scripts/security/audit.sh

# Check for vulnerabilities
./scripts/security/vulnerability-scan.sh
```

### Deployment
```bash
# Deploy to staging
./scripts/deploy/staging.sh

# Deploy to production
./scripts/deploy/production.sh
```

## 🧪 Testing Scripts

### Unit Tests
```bash
# Run all unit tests
./scripts/test/unit-tests.sh

# Run specific service tests
./scripts/test/backend-tests.sh
./scripts/test/frontend-tests.sh
```

### Integration Tests
```bash
# Run integration tests
./scripts/test/integration-tests.sh

# Run end-to-end tests
./scripts/test/e2e-tests.sh
```

### Performance Tests
```bash
# Run performance tests
./scripts/test/performance-tests.sh

# Load testing
./scripts/test/load-tests.sh
```

## 🔒 Security Scripts

### Vulnerability Scanning
```bash
# Scan for security vulnerabilities
./scripts/security/vulnerability-scan.sh

# Audit dependencies
./scripts/security/dependency-audit.sh
```

### Access Control
```bash
# Manage API keys
./scripts/security/manage-api-keys.sh

# Update permissions
./scripts/security/update-permissions.sh
```

## 📊 Monitoring Scripts

### Health Checks
```bash
# Check service health
./scripts/monitor/service-health.sh

# Check database health
./scripts/monitor/database-health.sh
```

### Performance Monitoring
```bash
# Monitor performance metrics
./scripts/monitor/performance-metrics.sh

# Generate performance reports
./scripts/monitor/performance-report.sh
```

## 🗄️ Database Scripts

### Migration Scripts
```bash
# Run database migrations
./scripts/database/migrate.sh

# Rollback migrations
./scripts/database/rollback.sh
```

### Backup Scripts
```bash
# Backup database
./scripts/database/backup.sh

# Restore database
./scripts/database/restore.sh
```

### Seeding Scripts
```bash
# Seed test data
./scripts/seed/test-data.sh

# Seed market data
./scripts/seed/market-data.sh
```

## 🚀 Deployment Scripts

### Environment Setup
```bash
# Setup development environment
./scripts/deploy/setup-dev.sh

# Setup production environment
./scripts/deploy/setup-prod.sh
```

### Service Deployment
```bash
# Deploy frontend
./scripts/deploy/frontend.sh

# Deploy backend
./scripts/deploy/backend.sh

# Deploy all services
./scripts/deploy/all-services.sh
```

## 🔄 CI/CD Scripts

### Build Pipeline
```bash
# Build all services
./scripts/ci-cd/build-all.sh

# Build specific service
./scripts/ci-cd/build-service.sh frontend
```

### Test Pipeline
```bash
# Run all tests
./scripts/ci-cd/test-all.sh

# Run tests for specific service
./scripts/ci-cd/test-service.sh backend
```

### Deploy Pipeline
```bash
# Deploy to staging
./scripts/ci-cd/deploy-staging.sh

# Deploy to production
./scripts/ci-cd/deploy-production.sh
```

## 🛠️ Utility Scripts

### Code Quality
```bash
# Run linting
./scripts/utils/lint.sh

# Format code
./scripts/utils/format.sh

# Type checking
./scripts/utils/type-check.sh
```

### Documentation
```bash
# Generate documentation
./scripts/utils/generate-docs.sh

# Update README files
./scripts/utils/update-readmes.sh
```

## 📖 Script Documentation

### Script Parameters
Most scripts support common parameters:
- `--help` - Show help information
- `--verbose` - Enable verbose output
- `--dry-run` - Show what would be done without executing
- `--force` - Force execution without confirmation

### Environment Variables
Scripts use environment variables for configuration:
- `NODE_ENV` - Environment (development/staging/production)
- `LOG_LEVEL` - Logging level (debug/info/warn/error)
- `DRY_RUN` - Enable dry run mode
- `VERBOSE` - Enable verbose output

### Error Handling
All scripts include proper error handling:
- **Exit Codes** - Proper exit codes for success/failure
- **Error Messages** - Clear error messages
- **Logging** - Structured logging output
- **Cleanup** - Proper cleanup on failure

## 🔧 Custom Scripts

### Creating New Scripts
```bash
#!/bin/bash
# Script template

set -e  # Exit on error
set -u  # Exit on undefined variable

# Script configuration
SCRIPT_NAME="$(basename "$0")"
LOG_FILE="/tmp/${SCRIPT_NAME}.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Main script logic
main() {
    log "Starting $SCRIPT_NAME"
    
    # Your script logic here
    
    log "Completed $SCRIPT_NAME"
}

# Run main function
main "$@"
```

### Script Testing
```bash
# Test script with dry run
./scripts/your-script.sh --dry-run

# Test script with verbose output
./scripts/your-script.sh --verbose
```

## 📄 License

All scripts are part of QuantDesk and are licensed under Apache License 2.0.

## 🤝 Contributing

We welcome contributions to improve scripts:
- **New Scripts** - Additional utility scripts
- **Bug Fixes** - Fix issues in existing scripts
- **Documentation** - Improve script documentation
- **Testing** - Add test coverage for scripts

## 📞 Support

For script support:
- **GitHub Issues** - Report bugs or ask questions
- **Documentation** - Check script-specific documentation
- **Community** - Join our Discord community