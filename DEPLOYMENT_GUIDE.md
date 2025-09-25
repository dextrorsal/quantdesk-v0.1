# 🚀 QuantDesk Deployment Guide

## Pre-Deployment Setup

### 1. Security Configuration
```bash
# Copy environment template
cp env.example .env

# Edit .env with your actual values
nano .env
```

### 2. Required Environment Variables
```bash
# Essential for deployment
JWT_SECRET=your-super-secret-jwt-key-minimum-32-chars
DATABASE_URL=postgresql://user:password@host:port/database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
NODE_ENV=production
PORT=3002
```

### 3. Verify Security
```bash
# Check no sensitive files are tracked
git status
git check-ignore test-ledger/
git check-ignore *.env

# Verify .gitignore is working
git check-ignore node_modules/
git check-ignore __pycache__/
```

## 🐳 Docker Deployment

### Option 1: Docker Compose (Recommended)
```bash
# Build and start all services
docker-compose up --build -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 2: Individual Containers
```bash
# Build backend
cd backend
docker build -t quantdesk-backend .

# Build frontend
cd ../frontend
docker build -t quantdesk-frontend .

# Run with environment
docker run -d --env-file ../.env -p 3002:3002 quantdesk-backend
docker run -d -p 3000:3000 quantdesk-frontend
```

## ☁️ Cloud Deployment

### Vercel (Frontend)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy frontend
cd frontend
vercel --prod
```

### Railway (Backend)
```bash
# Install Railway CLI
npm i -g @railway/cli

# Deploy backend
cd backend
railway login
railway init
railway up
```

### DigitalOcean App Platform
```yaml
# .do/app.yaml
name: quantdesk
services:
- name: backend
  source_dir: backend
  github:
    repo: your-username/quantdesk
    branch: main
  run_command: npm start
  environment_slug: node-js
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: NODE_ENV
    value: production
  - key: DATABASE_URL
    value: ${db.DATABASE_URL}
    type: SECRET
```

## 🗄️ Database Setup

### PostgreSQL (Production)
```sql
-- Create database
CREATE DATABASE quantdesk;

-- Create user
CREATE USER quantdesk_user WITH PASSWORD 'secure_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE quantdesk TO quantdesk_user;
```

### Supabase (Recommended)
1. Create new project at [supabase.com](https://supabase.com)
2. Get connection details from Settings > Database
3. Update `.env` with Supabase credentials

## 🔧 Environment-Specific Configuration

### Development
```bash
NODE_ENV=development
DEBUG=true
LOG_LEVEL=debug
```

### Staging
```bash
NODE_ENV=staging
DEBUG=false
LOG_LEVEL=info
```

### Production
```bash
NODE_ENV=production
DEBUG=false
LOG_LEVEL=warn
```

## 📊 Monitoring & Health Checks

### Health Endpoints
```bash
# Backend health
curl http://localhost:3002/health

# Database health
curl http://localhost:3002/api/health/db

# WebSocket health
curl http://localhost:3002/api/health/ws
```

### Logging
```bash
# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Log levels
LOG_LEVEL=error    # Only errors
LOG_LEVEL=warn     # Warnings and errors
LOG_LEVEL=info     # Info, warnings, errors
LOG_LEVEL=debug    # All logs
```

## 🔒 Security Checklist

- [ ] Environment variables configured
- [ ] No secrets in code
- [ ] HTTPS enabled (production)
- [ ] Rate limiting configured
- [ ] CORS properly set
- [ ] Database SSL enabled
- [ ] JWT secrets are strong
- [ ] API keys are secure
- [ ] Logs don't expose secrets

## 🚀 Quick Deploy Commands

### Local Development
```bash
# Backend
cd backend && npm install && npm run dev

# Frontend
cd frontend && npm install && npm run dev
```

### Production Build
```bash
# Backend
cd backend && npm install && npm run build && npm start

# Frontend
cd frontend && npm install && npm run build
```

### Docker Production
```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up --build -d

# Scale services
docker-compose up --scale backend=3 -d
```

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check DATABASE_URL format
   echo $DATABASE_URL
   # Should be: postgresql://user:pass@host:port/db
   ```

2. **JWT Secret Missing**
   ```bash
   # Generate new secret
   node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
   ```

3. **Port Already in Use**
   ```bash
   # Kill process on port 3002
   lsof -ti:3002 | xargs kill -9
   ```

4. **Environment Variables Not Loading**
   ```bash
   # Check .env file exists
   ls -la .env
   
   # Verify format
   cat .env | grep -v '^#' | grep -v '^$'
   ```

### Health Checks
```bash
# Backend health
curl -f http://localhost:3002/health || echo "Backend down"

# Database health
curl -f http://localhost:3002/api/health/db || echo "Database down"

# WebSocket health
curl -f http://localhost:3002/api/health/ws || echo "WebSocket down"
```

## 📈 Performance Optimization

### Backend
- Enable gzip compression
- Use Redis for caching
- Implement connection pooling
- Enable request logging

### Frontend
- Enable code splitting
- Use CDN for assets
- Implement service workers
- Optimize bundle size

### Database
- Create proper indexes
- Use connection pooling
- Enable query optimization
- Set up read replicas

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to production
      run: |
        docker-compose up --build -d
```

---

**Ready to deploy! 🚀**

Remember to:
1. Test in staging first
2. Monitor logs after deployment
3. Set up alerts for errors
4. Keep backups of your database
5. Update documentation