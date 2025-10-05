# AI Agent Memory: Railway Deployment Protocol

## 🚨 CRITICAL RULES FOR AI AGENTS

### RULE 1: CLI OVER MCP - ALWAYS
- ❌ **NEVER** use Railway MCP tools (`mcp_Railway_*`)
- ✅ **ALWAYS** use Railway CLI commands (`railway up --detach`)
- **Reason**: MCP tools fail with 403 errors and don't handle Docker builds properly

### RULE 2: FORCE DOCKER-ONLY DEPLOYMENT
- ❌ **NEVER** let Railway auto-detect Node.js buildpack
- ✅ **ALWAYS** force Docker build using `.railwayignore` files
- ✅ **ALWAYS** use `railway.toml` configuration files
- **Reason**: Node.js buildpack causes TypeScript compilation issues

### RULE 3: MONOREPO STRUCTURE HANDLING
- ✅ **ALWAYS** set Docker build context to root (`.`)
- ✅ **ALWAYS** use `COPY backend/package*.json ./` in Dockerfile
- ✅ **ALWAYS** use `COPY backend/ .` for source code
- **Reason**: Monorepo structure requires specific COPY paths

### RULE 4: TYPESCRIPT ERROR RESOLUTION ORDER
1. **FIRST**: Fix ALL TypeScript compilation errors before deployment
2. **SECOND**: Test Docker build locally (`docker build -t test .`)
3. **THIRD**: Deploy to Railway only after successful local build

**Common TypeScript fixes**:
- `wallet_address` → `walletAddress`
- Comment out missing modules (`RPCTester`)
- Fix import paths (`../utils/logger`)
- Handle optional properties (`subscription_id?`)

### RULE 5: DEPLOYMENT VERIFICATION SEQUENCE
1. Run `railway up --detach`
2. Check Railway dashboard for build status
3. Verify service settings show Docker build
4. Test health endpoint: `curl https://service.up.railway.app/health`
5. Check logs: `railway logs --follow`

### RULE 6: CONFIGURATION FILE HIERARCHY
- ✅ **ALWAYS** create both root AND backend `railway.toml`
- ✅ **ALWAYS** create both root AND backend `.railwayignore`
- ✅ **ALWAYS** set `context = "."` in root `railway.toml`
- **Reason**: Railway needs explicit configuration to override auto-detection

## 🚫 COMMON MISTAKES TO AVOID

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

## 📋 EMERGENCY TROUBLESHOOTING

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

## ✅ SUCCESSFUL DEPLOYMENT EXAMPLE

**Project**: QuantDesk Backend  
**Service**: pacific-imagination  
**URL**: https://quantdesk.up.railway.app  
**Method**: Docker-only deployment via Railway CLI  
**Status**: ✅ Successfully deployed and verified

**Key Configuration Files Used**:
- Root `railway.toml` with `context = "."`
- Backend `railway.toml` with Dockerfile path
- Root `.railwayignore` to force Docker build
- Backend `.railwayignore` for additional Docker enforcement
- `backend/Dockerfile` with correct monorepo COPY paths

---

**Last Updated**: January 2024  
**Protocol Status**: ✅ Verified and Working  
**Deployment Method**: Railway CLI + Docker-only
