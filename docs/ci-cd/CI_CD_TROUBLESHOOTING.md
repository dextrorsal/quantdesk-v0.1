# 🔧 QuantDesk CI/CD Troubleshooting Guide

## 🚨 Common Issues & Solutions

### 1. Workflow Syntax Errors

**Problem**: YAML syntax errors in workflow files

**Symptoms**:
- Workflow fails to start
- "Invalid YAML" error messages
- GitHub Actions shows syntax errors

**Solutions**:
```bash
# Check YAML syntax
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/workflow.yml'))"

# Use online YAML validator
# Copy workflow content to yamlvalidator.com

# Check indentation (must be spaces, not tabs)
# Use 2-space indentation consistently
```

**Prevention**:
- Use YAML linter in your editor
- Test workflows locally with `./test-workflows.sh`
- Validate before pushing

### 2. Missing GitHub Secrets

**Problem**: Workflows fail due to missing environment variables

**Symptoms**:
- "Secret not found" errors
- Authentication failures
- Deployment failures

**Solutions**:
```bash
# Required secrets in GitHub repository settings:
POSTMAN_API_KEY=your_postman_api_key
RAILWAY_TOKEN=your_railway_token
RAILWAY_PROJECT_ID=your_project_id
RAILWAY_SERVICE_ID=your_service_id
API_BASE_URL=https://your-api-url.com
SOCKET_API_TOKEN=your_socket_token
```

**Setup Steps**:
1. Go to GitHub repository
2. Settings > Secrets and variables > Actions
3. Click "New repository secret"
4. Add each secret with proper values

### 3. Docker Build Failures

**Problem**: Docker images fail to build

**Symptoms**:
- Build timeout errors
- "Dockerfile not found" errors
- Build context issues

**Solutions**:
```bash
# Test Dockerfile locally
docker build -f Dockerfile.backend .

# Check Docker context
docker build --no-cache -f Dockerfile.backend .

# Verify .dockerignore
cat .dockerignore

# Check build context size
du -sh .
```

**Common Fixes**:
- Ensure Dockerfile exists in correct location
- Check .dockerignore excludes unnecessary files
- Verify base image availability
- Use multi-stage builds for optimization

### 4. Test Failures

**Problem**: Tests fail in CI/CD pipeline

**Symptoms**:
- Test suite failures
- Coverage below threshold
- Timeout errors

**Solutions**:
```bash
# Run tests locally
npm run test

# Run specific test suite
npm run test:backend
npm run test:frontend

# Check test configuration
cat jest.config.js
cat package.json | grep -A 10 "scripts"
```

**Debugging Steps**:
1. Run tests locally first
2. Check test environment variables
3. Verify database connections
4. Check test data setup
5. Review test logs in GitHub Actions

### 5. Deployment Failures

**Problem**: Deployment to staging/production fails

**Symptoms**:
- Deployment timeout
- Service not starting
- Health check failures

**Solutions**:
```bash
# Check deployment logs
# Go to GitHub Actions > Workflow run > Job logs

# Verify environment variables
# Check deployment platform settings

# Test deployment locally
docker-compose up -d
curl http://localhost:3002/health
```

**Platform-Specific Issues**:

**Railway Deployment**:
- Check Railway project settings
- Verify service configuration
- Check resource limits
- Review Railway logs

**Vercel Deployment**:
- Check Vercel project settings
- Verify build configuration
- Check environment variables
- Review Vercel logs

### 6. Security Scan Failures

**Problem**: Security scans report vulnerabilities

**Symptoms**:
- High/Critical vulnerabilities found
- Security scan failures
- Compliance issues

**Solutions**:
```bash
# Run security audit locally
npm audit

# Fix vulnerabilities
npm audit fix

# Check specific packages
npm audit --audit-level=high

# Update vulnerable packages
npm update package-name
```

**Docker Security**:
```bash
# Scan Docker images locally
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image your-image:latest

# Check base image vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image node:20-bookworm-slim
```

### 7. Performance Issues

**Problem**: Workflows run slowly or timeout

**Symptoms**:
- Long execution times
- Timeout errors
- Resource exhaustion

**Solutions**:
```bash
# Optimize Docker builds
# Use multi-stage builds
# Implement proper caching
# Reduce build context

# Optimize tests
# Run tests in parallel
# Use test sharding
# Mock external dependencies

# Optimize workflows
# Use matrix builds
# Implement job dependencies
# Cache dependencies
```

### 8. Environment Issues

**Problem**: Environment-specific failures

**Symptoms**:
- Staging works, production fails
- Environment variable issues
- Service discovery problems

**Solutions**:
```bash
# Check environment configuration
# Verify environment variables
# Check service endpoints
# Review environment-specific settings

# Test environments locally
# Use docker-compose for local testing
# Verify service connectivity
# Check network configuration
```

## 🔍 Debugging Workflows

### 1. Enable Debug Logging

**Add to workflow file**:
```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

### 2. Check Workflow Runs

**Steps**:
1. Go to GitHub Actions tab
2. Click on failed workflow run
3. Click on failed job
4. Review step logs
5. Download logs if needed

### 3. Local Debugging

**Test Scripts**:
```bash
# Validate workflow configuration
./test-workflows.sh

# Check workflow status
./check-workflow-status.sh

# Simulate execution
./dry-run-test.sh
```

**Manual Testing**:
```bash
# Test package.json scripts
npm run build
npm run test
npm run lint

# Test Docker configuration
docker-compose config
docker-compose build
```

### 4. Workflow Debugging

**Common Debug Steps**:
1. Check workflow syntax
2. Verify trigger conditions
3. Check job dependencies
4. Review step configurations
5. Test individual steps

## 📊 Monitoring & Alerting

### 1. Workflow Monitoring

**GitHub Actions**:
- Check workflow run status
- Monitor execution time
- Review failure rates
- Track success metrics

**Custom Monitoring**:
```bash
# Check service health
curl http://localhost:3002/health

# Monitor resource usage
docker stats

# Check logs
docker-compose logs -f
```

### 2. Alerting Setup

**GitHub Notifications**:
- Enable email notifications
- Set up webhook notifications
- Configure team notifications

**External Monitoring**:
- Set up monitoring tools
- Configure alerting rules
- Monitor key metrics

## 🛠️ Maintenance

### 1. Regular Updates

**Dependencies**:
```bash
# Update npm packages
npm update

# Update Docker images
docker pull node:20-bookworm-slim

# Update GitHub Actions
# Check for action updates in workflow files
```

**Security**:
```bash
# Regular security audits
npm audit
npm audit fix

# Update vulnerable packages
npm update package-name

# Scan Docker images
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image your-image:latest
```

### 2. Performance Optimization

**Workflow Optimization**:
- Use matrix builds for parallel execution
- Implement proper caching
- Optimize job dependencies
- Reduce build context

**Docker Optimization**:
- Use multi-stage builds
- Implement proper .dockerignore
- Use specific base image versions
- Optimize layer caching

### 3. Documentation Updates

**Keep Updated**:
- Workflow documentation
- Troubleshooting guides
- Best practices
- Configuration changes

## 📞 Getting Help

### 1. Internal Resources

**Documentation**:
- `docs/CI_CD_COMPREHENSIVE_GUIDE.md`
- `docs/CI_CD_QUICK_REFERENCE.md`
- This troubleshooting guide

**Testing Scripts**:
- `./test-workflows.sh`
- `./dry-run-test.sh`
- `./check-workflow-status.sh`

### 2. External Resources

**GitHub Actions**:
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Best Practices](https://docs.github.com/en/actions/learn-github-actions)

**Docker**:
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Multi-stage Builds](https://docs.docker.com/develop/dev-best-practices/)

**Security**:
- [Security Best Practices](https://docs.github.com/en/actions/security-guides)
- [Secrets Management](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Vulnerability Scanning](https://docs.github.com/en/code-security/supply-chain-security)

### 3. Community Support

**GitHub Community**:
- GitHub Actions discussions
- Docker community forums
- Stack Overflow

**Professional Support**:
- GitHub Support
- Platform-specific support
- Consulting services

## 🎯 Prevention Strategies

### 1. Proactive Testing

**Before Pushing**:
```bash
# Run local tests
npm run test

# Check code quality
npm run lint

# Validate workflows
./test-workflows.sh

# Test Docker builds
npm run docker:build
```

### 2. Continuous Monitoring

**Regular Checks**:
- Monitor workflow success rates
- Check security scan results
- Review performance metrics
- Update dependencies regularly

### 3. Documentation Maintenance

**Keep Updated**:
- Workflow documentation
- Troubleshooting guides
- Configuration changes
- Best practices

---

**🔧 This troubleshooting guide helps you resolve CI/CD issues quickly and efficiently.**

For additional help, refer to the comprehensive documentation or contact the development team.
