# 🚀 QuantDesk CI/CD Quick Reference

## 🎯 Quick Commands

### Local Testing
```bash
# Test all workflows
./test-workflows.sh

# Simulate workflow execution
./dry-run-test.sh

# Check workflow status
./check-workflow-status.sh
```

### Development Workflow
```bash
# Start development
npm run dev

# Run tests
npm run test

# Build all services
npm run build

# Docker operations
npm run docker:up
npm run docker:down
npm run docker:logs
```

## 📊 Workflow Overview

| Workflow | Purpose | Triggers | Features |
|----------|---------|----------|----------|
| `testing.yml` | Comprehensive testing | Push/PR | Unit, integration, coverage |
| `code-quality.yml` | Code quality checks | Push/PR | ESLint, TypeScript, formatting |
| `docker-build-push.yml` | Docker builds | Push/PR/Manual | Multi-platform, registry |
| `ci-cd.yml` | Main pipeline | Push/PR/Manual | Quality → Build → Deploy |
| `docker-security-scanning.yml` | Security scanning | Push/PR/Schedule | Vulnerability scanning |
| `postman-api-testing.yml` | API testing | Push/PR/Schedule | Smoke, integration, performance |

## 🔧 Testing Without Deployment

### 1. Manual Workflow Dispatch
1. Go to GitHub Actions tab
2. Select workflow
3. Click "Run workflow"
4. Choose branch and run

### 2. Pull Request Testing
```bash
git checkout -b test-ci-cd
git add .
git commit -m "Test CI/CD workflows"
git push origin test-ci-cd
# Create PR to trigger workflows
```

### 3. Test Branch Strategy
```bash
# Push to develop for staging
git push origin develop

# Push to main for production
git push origin main
```

## 🚀 Deployment Triggers

| Branch | Environment | Triggers |
|--------|-------------|----------|
| `develop` | Staging | All checks + staging deployment |
| `main` | Production | All checks + production deployment |
| `feature/*` | Testing | Code quality + testing only |

## 🔒 Required Secrets

Set these in GitHub repository settings:

```bash
POSTMAN_API_KEY=your_postman_api_key
RAILWAY_TOKEN=your_railway_token
RAILWAY_PROJECT_ID=your_project_id
RAILWAY_SERVICE_ID=your_service_id
API_BASE_URL=https://your-api-url.com
SOCKET_API_TOKEN=your_socket_token
```

## 📈 Workflow Statistics

- **Total Workflows**: 17
- **Active Workflows**: 17
- **Scheduled Workflows**: 6
- **Manual Workflows**: 13

## 🎯 Workflow Categories

### 🧪 Testing & Quality (3)
- `testing.yml` - Comprehensive testing
- `code-quality.yml` - Code quality checks
- `postman-api-testing.yml` - API testing

### 🐳 Docker & Build (6)
- `docker-build-push.yml` - Docker builds
- `docker-compose.yml` - Service orchestration
- `docker-deployment.yml` - Deployment strategies
- `docker-monitoring.yml` - Container monitoring
- `docker-security-scanning.yml` - Security scanning
- `build-deploy.yml` - Build coordination

### 🚀 Deployment (4)
- `ci-cd.yml` - Main CI/CD pipeline
- `railway-deployment.yml` - Railway deployment
- `vercel-deployment.yml` - Vercel deployment
- `build-deploy.yml` - Build and deploy

### 🔒 Security (2)
- `dependency-audit.yml` - Dependency scanning
- `docker-security-scanning.yml` - Docker security

### 📊 Monitoring (3)
- `docker-monitoring.yml` - Docker monitoring
- `redis-monitoring.yml` - Redis monitoring
- `supabase-migration.yml` - Database migrations

## 🛠️ Troubleshooting

### Common Issues

**Workflow Syntax Errors:**
```bash
# Check YAML syntax
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/workflow.yml'))"
```

**Missing Secrets:**
- Go to GitHub Settings > Secrets and variables > Actions
- Add required secrets

**Docker Build Failures:**
```bash
# Test locally
docker build -f Dockerfile.backend .
```

**Test Failures:**
```bash
# Run tests locally
npm run test
```

### Debug Commands

```bash
# Validate workflows
./test-workflows.sh

# Check status
./check-workflow-status.sh

# Simulate execution
./dry-run-test.sh
```

## 📚 Best Practices

### Before Pushing
1. Run local tests: `npm run test`
2. Check code quality: `npm run lint`
3. Validate workflows: `./test-workflows.sh`
4. Test Docker builds: `npm run docker:build`

### Deployment Strategy
1. **Feature Branch**: Test code quality and basic tests
2. **Develop Branch**: Full testing + staging deployment
3. **Main Branch**: All checks + production deployment

### Monitoring
- Check GitHub Actions tab for workflow status
- Monitor deployment logs
- Set up alerts for failures
- Regular security scanning

## 🎉 Quick Start

1. **Setup**: Configure GitHub secrets
2. **Test**: Run `./test-workflows.sh`
3. **Develop**: Push to feature branch
4. **Staging**: Merge to develop branch
5. **Production**: Merge to main branch

---

**🚀 Your CI/CD pipeline is ready!**

For detailed information, see `docs/CI_CD_COMPREHENSIVE_GUIDE.md`
