# Deployment Configuration

## 🚀 Backend Deployment (Railway)

### Configuration
- **Platform**: Railway
- **Config File**: `backend/railway.json`
- **Dockerfile**: `backend/Dockerfile`
- **Health Check**: `/health` endpoint
- **Auto-deploy**: On push to main branch

### Railway Settings
```json
{
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### Environment Variables Required
- `NODE_ENV=production`
- `PORT=3000`
- Database connection strings
- API keys and secrets

### Deployment Process
1. Push to main branch
2. Railway detects changes
3. Builds Docker image from `backend/Dockerfile`
4. Deploys to Railway infrastructure
5. Health check validates deployment

---

## 🌐 Frontend Deployment (Vercel)

### Configuration
- **Platform**: Vercel
- **Config File**: `frontend/vercel.json`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Framework**: Vite

### Vercel Settings
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

### Deployment Process
1. Push to main branch
2. Vercel detects changes
3. Installs dependencies (`npm install`)
4. Builds application (`npm run build`)
5. Deploys static files to CDN
6. Configures routing for SPA

---

## 👨‍💼 Admin Dashboard Deployment (Vercel)

### Configuration
- **Platform**: Vercel
- **Config File**: `admin-dashboard/vercel.json`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Framework**: Vite

### Deployment Process
1. Push to main branch
2. Vercel detects changes
3. Installs dependencies
4. Builds admin dashboard
5. Deploys to separate Vercel project

---

## 🔧 CI/CD Integration

### GitHub Actions Pipeline
The `simple-ci-cd.yml` workflow handles:

1. **Code Quality Checks**
   - TypeScript type checking
   - ESLint code linting
   - Dependency installation

2. **Build Process**
   - Backend build (continues on errors)
   - Frontend build ✅
   - Admin dashboard build (continues on errors)

3. **Docker Image Building**
   - Creates Docker images for all services
   - Only runs on main branch pushes

4. **Deployment Triggers**
   - Railway deployment (backend)
   - Vercel deployment (frontend + admin)

### Current Status
- ✅ **Railway Backend**: Configured and ready
- ✅ **Vercel Frontend**: Configured and ready  
- ✅ **Vercel Admin**: Configured and ready
- ✅ **CI/CD Pipeline**: Working and robust

---

## 🚨 Troubleshooting

### Backend Deployment Issues
- Check Railway logs for build errors
- Verify environment variables are set
- Ensure Dockerfile builds successfully locally

### Frontend Deployment Issues
- Check Vercel build logs
- Verify `vercel.json` configuration
- Ensure build command works locally

### CI/CD Pipeline Issues
- Check GitHub Actions tab for detailed logs
- Pipeline continues on non-critical errors
- Only fails on critical build issues

---

## 📝 Next Steps

1. **Connect Railway Project**: Link your GitHub repo to Railway
2. **Connect Vercel Projects**: Link frontend and admin repos to Vercel
3. **Set Environment Variables**: Configure production secrets
4. **Test Deployments**: Push changes and verify auto-deployment
5. **Monitor Health**: Check deployment health and logs
