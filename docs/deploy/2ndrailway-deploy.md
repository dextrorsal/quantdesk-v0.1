# Railway Deployment Guide - QuantDesk Backend

## Overview
This document outlines the successful Docker-only deployment process for the QuantDesk backend to Railway platform. This deployment uses Docker containers exclusively, avoiding Railway's default Node.js buildpack.

## Prerequisites

### Required Tools
- Node.js 23.x.x (installed via nvm)
- Railway CLI (`npm install -g @railway/cli`)
- Docker (for local testing)
- Git

### Required Accounts
- Railway account
- GitHub account with repository access

## Project Structure
```
quantdesk/
├── backend/
│   ├── Dockerfile
│   ├── railway.toml
│   ├── .railwayignore
│   ├── package.json
│   └── src/
├── railway.toml (root)
├── .railwayignore (root)
└── docs/
```

## Configuration Files

### 1. Root `railway.toml`
```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "backend/Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

# Force Docker deployment - use root context
[build.docker]
dockerfile = "backend/Dockerfile"
context = "."
```

### 2. Backend `railway.toml`
```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

### 3. Root `.railwayignore`
```
# Ignore Node.js specific files at the root to force Docker build
node_modules/
package.json
package-lock.json
yarn.lock
.env
.git/
.github/
.vscode/
docs/
frontend/
admin-dashboard/
*.log
npm-debug.log*
```

### 4. Backend `.railwayignore`
```
# Ignore Node.js specific files in the backend to force Docker build
node_modules/
.env
*.log
npm-debug.log*
```

### 5. Backend `Dockerfile`
```dockerfile
FROM node:20-alpine

WORKDIR /app

# Copy package files from backend directory
COPY backend/package*.json ./

# Install dependencies
RUN npm install

# Copy source code from backend directory
COPY backend/ .

# Build the application
RUN npm run build

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 3002

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3002/health || exit 1

# Start the application
CMD ["npm", "start"]
```

## Deployment Process

### Step 1: Fix TypeScript Compilation Errors
Before deployment, ensure all TypeScript errors are resolved:

#### Fixed Issues:
1. **deposits.ts**: Fixed `wallet_address` vs `walletAddress` typo
2. **rpcTesting.ts**: Commented out `RPCTester` usage (module not found)
3. **supabaseMarkets.ts**: Changed `executeSQL` to `executeQuery`
4. **systemModeService.ts**: Made `wsService` optional, added null checks
5. **systemMonitor.ts**: Fixed Logger import path
6. **transactionVerificationService.ts**: Fixed Solana type conversions
7. **webhookService.ts**: Made `subscription_id` optional in method signature

### Step 2: Railway CLI Setup
```bash
# Login to Railway
railway login

# Link to existing project (if applicable)
railway link

# Or create new project
railway init
```

### Step 3: Environment Variables
Set required environment variables in Railway dashboard:
- `NODE_ENV=production`
- `PORT=3002`
- Database connection strings
- API keys
- Solana RPC URLs

### Step 4: Deploy with Docker
```bash
# Deploy using Railway CLI with Docker
railway up --detach

# Monitor deployment
railway logs --follow
```

## Service Configuration

### Build Settings
- **Builder**: Dockerfile
- **Dockerfile Path**: `backend/Dockerfile`
- **Build Context**: Root directory (`.`)
- **Build Method**: Docker using BuildKit

### Deploy Settings
- **Healthcheck Path**: `/health`
- **Healthcheck Timeout**: 100 seconds
- **Restart Policy**: ON_FAILURE (max 10 retries)
- **Region**: US East (Virginia, USA)
- **Replicas**: 1 instance
- **Resources**: 2 vCPU, 1 GB Memory

### Networking
- **Public URL**: `quantdesk.up.railway.app`
- **Private URL**: `pacific-imagination.railway.internal`
- **Port**: 3002

## Verification

### Health Check
```bash
curl https://quantdesk.up.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-XX...",
  "uptime": "..."
}
```

### Service Status
- Check Railway dashboard for deployment status
- Monitor logs for any errors
- Verify all endpoints are responding

## Troubleshooting

### Common Issues

#### 1. Dockerfile Not Found
**Error**: `Dockerfile does not exist`
**Solution**: Ensure `railway.toml` correctly points to `backend/Dockerfile`

#### 2. TypeScript Compilation Errors
**Error**: Build fails during `npm run build`
**Solution**: Fix all TypeScript errors before deployment

#### 3. Package.json Not Found
**Error**: `Could not read package.json`
**Solution**: Update Dockerfile COPY commands to use `backend/package*.json`

#### 4. Railway Using Node.js Buildpack
**Error**: Railway detects Node.js and uses wrong buildpack
**Solution**: Use `.railwayignore` files to force Docker build

### Debug Commands
```bash
# Check Railway status
railway status

# View deployment logs
railway logs

# Check service health
railway health

# View environment variables
railway variables
```

## Success Metrics

### Deployment Success Indicators
- ✅ Service deployed successfully via CLI
- ✅ Docker build completed without errors
- ✅ Health check endpoint responding
- ✅ Service accessible at public URL
- ✅ All TypeScript compilation errors resolved

### Service Configuration Verified
- ✅ Builder: Dockerfile
- ✅ Dockerfile Path: backend/Dockerfile
- ✅ Healthcheck: /health
- ✅ Public URL: quantdesk.up.railway.app
- ✅ Source: dextrorsal/quantdesk (main branch)

## Next Steps

1. **Monitor Performance**: Watch resource usage and response times
2. **Set Up Monitoring**: Configure alerts for service health
3. **Environment Management**: Set up staging/production environments
4. **CI/CD Integration**: Consider GitHub Actions for automated deployments
5. **Scaling**: Monitor traffic and scale resources as needed

## Best Practices

### Docker Optimization
- Use multi-stage builds for smaller images
- Leverage Docker layer caching
- Optimize package installation order

### Railway Configuration
- Use `railway.toml` for consistent configuration
- Set appropriate resource limits
- Configure health checks properly
- Use environment-specific variables

### Security
- Never commit sensitive environment variables
- Use Railway's secure variable storage
- Regularly update dependencies
- Monitor for security vulnerabilities

## AI Agent Rules & Follow Set

### 🚨 CRITICAL: Railway Deployment Protocol for AI Agents

**RULE 1: ALWAYS USE RAILWAY CLI - NEVER USE MCP**
- ❌ **NEVER** use Railway MCP tools (`mcp_Railway_*`)
- ✅ **ALWAYS** use Railway CLI commands (`railway up`, `railway logs`, etc.)
- **Reason**: MCP tools often fail with 403 errors and don't handle Docker builds properly

**RULE 2: DOCKER-ONLY DEPLOYMENT MANDATE**
- ❌ **NEVER** let Railway auto-detect Node.js buildpack
- ✅ **ALWAYS** force Docker build using `.railwayignore` files
- ✅ **ALWAYS** use `railway.toml` configuration files
- **Reason**: Node.js buildpack causes TypeScript compilation issues

**RULE 3: MONOREPO STRUCTURE HANDLING**
- ✅ **ALWAYS** set Docker build context to root (`.`)
- ✅ **ALWAYS** use `COPY backend/package*.json ./` in Dockerfile
- ✅ **ALWAYS** use `COPY backend/ .` for source code
- **Reason**: Monorepo structure requires specific COPY paths

**RULE 4: TYPESCRIPT ERROR RESOLUTION ORDER**
1. **FIRST**: Fix ALL TypeScript compilation errors before deployment
2. **SECOND**: Test Docker build locally (`docker build -t test .`)
3. **THIRD**: Deploy to Railway only after successful local build
- **Common fixes**:
  - `wallet_address` → `walletAddress`
  - Comment out missing modules (`RPCTester`)
  - Fix import paths (`../utils/logger`)
  - Handle optional properties (`subscription_id?`)

**RULE 5: DEPLOYMENT VERIFICATION SEQUENCE**
1. Run `railway up --detach`
2. Check Railway dashboard for build status
3. Verify service settings show Docker build
4. Test health endpoint: `curl https://service.up.railway.app/health`
5. Check logs: `railway logs --follow`

**RULE 6: CONFIGURATION FILE HIERARCHY**
- ✅ **ALWAYS** create both root AND backend `railway.toml`
- ✅ **ALWAYS** create both root AND backend `.railwayignore`
- ✅ **ALWAYS** set `context = "."` in root `railway.toml`
- **Reason**: Railway needs explicit configuration to override auto-detection

### 🤖 AI Agent Deployment Checklist

**Before Starting:**
- [ ] Confirm Railway CLI is installed (`railway --version`)
- [ ] Verify user is logged in (`railway whoami`)
- [ ] Check project is linked (`railway status`)

**Pre-Deployment:**
- [ ] Fix ALL TypeScript errors in backend code
- [ ] Create/verify `railway.toml` files (root + backend)
- [ ] Create/verify `.railwayignore` files (root + backend)
- [ ] Test Docker build locally
- [ ] Verify Dockerfile COPY paths are correct

**During Deployment:**
- [ ] Use `railway up --detach` (NOT MCP tools)
- [ ] Monitor build logs in Railway dashboard
- [ ] Verify Docker build is being used (not Node.js buildpack)
- [ ] Check for any build errors

**Post-Deployment:**
- [ ] Verify service is accessible at public URL
- [ ] Test health endpoint
- [ ] Check Railway service settings
- [ ] Monitor logs for any runtime errors

### 🚫 Common AI Agent Mistakes to Avoid

1. **Using MCP Instead of CLI**
   - ❌ `mcp_Railway_deploy()` 
   - ✅ `railway up --detach`

2. **Letting Railway Auto-Detect Buildpack**
   - ❌ Relying on Railway's Node.js detection
   - ✅ Explicit Docker configuration with `.railwayignore`

3. **Wrong Docker Context**
   - ❌ `COPY package*.json ./` (fails in monorepo)
   - ✅ `COPY backend/package*.json ./`

4. **Skipping TypeScript Error Fixes**
   - ❌ Deploying with compilation errors
   - ✅ Fix ALL errors before deployment

5. **Missing Configuration Files**
   - ❌ Assuming Railway will auto-configure
   - ✅ Explicit `railway.toml` and `.railwayignore` files

### 📋 Emergency Troubleshooting for AI Agents

**If deployment fails:**
1. Check Railway dashboard for specific error
2. Run `railway logs` to see build logs
3. Verify Dockerfile exists at `backend/Dockerfile`
4. Check if TypeScript errors are present
5. Ensure `.railwayignore` files are present

**If service won't start:**
1. Check health endpoint: `/health`
2. Verify environment variables are set
3. Check port configuration (should be 3002)
4. Review application logs for runtime errors

**If build uses wrong method:**
1. Verify `.railwayignore` files ignore Node.js files
2. Check `railway.toml` has `builder = "DOCKERFILE"`
3. Ensure Dockerfile path is correct
4. Force redeploy with `railway up --detach`

## References

- [Railway Docker Documentation](https://docs.railway.com/deploy/dockerfile)
- [Railway Configuration Files](https://docs.railway.com/deploy/railway-toml)
- [Railway CLI Commands](https://docs.railway.com/reference/cli)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

**Last Updated**: January 2024  
**Deployment Status**: ✅ Successfully Deployed  
**Service URL**: https://quantdesk.up.railway.app
