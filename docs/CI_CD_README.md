# 🚀 QuantDesk CI/CD & Docker Setup

This document provides comprehensive information about the CI/CD pipeline and Docker configuration for the QuantDesk trading platform.

## 📋 Table of Contents

- [Overview](#overview)
- [Docker Setup](#docker-setup)
- [CI/CD Pipeline](#cicd-pipeline)
- [Local Development](#local-development)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

QuantDesk is a multi-service trading platform with the following components:

- **Backend API** (Port 3002) - Express.js/TypeScript API server
- **Frontend** (Port 3001) - React/Vite trading interface
- **Admin Dashboard** (Port 3000) - React admin interface
- **Data Ingestion** (Port 3003) - High-throughput data pipeline
- **MIKEY-AI** (Port 3000) - AI trading assistant
- **PostgreSQL** (Port 5432) - Primary database
- **Redis** (Port 6379) - Caching and session management

## 🐳 Docker Setup

### Quick Start

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Individual Services

```bash
# Build specific service
docker-compose build backend
docker-compose build frontend
docker-compose build admin

# Start specific service
docker-compose up -d backend
docker-compose up -d frontend
docker-compose up -d admin
```

### Docker Commands

```bash
# Build all services
npm run docker:build

# Start all services
npm run docker:up

# Stop all services
npm run docker:down

# View logs
npm run docker:logs

# Clean up
npm run docker:clean
```

## 🔄 CI/CD Pipeline

### Pipeline Overview

The CI/CD pipeline consists of several stages:

1. **Code Quality Check** - TypeScript, linting, formatting
2. **Security Audit** - Dependency vulnerability scanning
3. **Build & Test** - Compilation, testing, coverage
4. **Docker Build** - Container image creation
5. **Deploy** - Staging and production deployment

### Pipeline Triggers

- **Push to `main`** - Full pipeline + production deployment
- **Push to `develop`** - Full pipeline + staging deployment
- **Pull Request** - Quality checks + build + test
- **Manual Trigger** - Workflow dispatch

### Pipeline Jobs

#### 🔍 Code Quality Check
- TypeScript compilation
- ESLint code quality
- Code formatting validation

#### 🛡️ Security Audit
- `npm audit` for all services
- Dependency vulnerability scanning
- Security policy compliance

#### 🏗️ Build & Test
- Service compilation
- Unit and integration tests
- Code coverage reporting
- Build verification

#### 🐳 Docker Build
- Multi-service container builds
- Image tagging and pushing
- Registry management

#### 🚀 Deploy
- **Staging**: Automatic deployment on `develop` branch
- **Production**: Automatic deployment on `main` branch

## 💻 Local Development

### Prerequisites

- Node.js 20+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+

### Setup

```bash
# Clone repository
git clone <repository-url>
cd quantdesk

# Install dependencies
npm run install:all

# Start development services
npm run dev

# Or start with Docker
docker-compose up -d
```

### Development Commands

```bash
# Install dependencies
npm run install:all

# Start all services in development
npm run dev

# Start individual services
npm run start:backend
npm run start:frontend
npm run start:admin

# Run tests
npm run test

# Run linting
npm run lint

# Fix linting issues
npm run lint:fix

# Type checking
npm run type-check

# Build all services
npm run build
```

### Environment Variables

Create `.env` files in each service directory:

```bash
# Backend .env
DATABASE_URL=postgresql://postgres:password@localhost:5432/quantdesk
REDIS_URL=redis://localhost:6379
JWT_SECRET=your-jwt-secret
PORT=3002

# Frontend .env
VITE_API_URL=http://localhost:3002
VITE_WS_URL=ws://localhost:3002

# Admin Dashboard .env
VITE_API_URL=http://localhost:3002
VITE_WS_URL=ws://localhost:3002
```

## 🚀 Deployment

### Staging Deployment

Staging deployments are triggered automatically on pushes to the `develop` branch.

```bash
# Manual staging deployment
git push origin develop
```

### Production Deployment

Production deployments are triggered automatically on pushes to the `main` branch.

```bash
# Manual production deployment
git push origin main
```

### Manual Deployment

```bash
# Deploy specific service
docker-compose up -d backend
docker-compose up -d frontend
docker-compose up -d admin

# Deploy all services
docker-compose up -d
```

### Environment Configuration

#### Staging Environment
- Database: `quantdesk_staging`
- Redis: `redis_staging`
- API URL: `https://api-staging.quantdesk.app`

#### Production Environment
- Database: `quantdesk_production`
- Redis: `redis_production`
- API URL: `https://api.quantdesk.app`

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Conflicts
```bash
# Check port usage
lsof -i :3000
lsof -i :3001
lsof -i :3002

# Kill processes
sudo kill -9 <PID>
```

#### 2. Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres
```

#### 3. Redis Connection Issues
```bash
# Check Redis status
docker-compose ps redis

# Check Redis logs
docker-compose logs redis

# Restart Redis
docker-compose restart redis
```

#### 4. Build Failures
```bash
# Clean Docker cache
docker system prune -f

# Rebuild without cache
docker-compose build --no-cache

# Check service logs
docker-compose logs <service-name>
```

#### 5. CI/CD Pipeline Failures

**TypeScript Errors:**
```bash
# Fix TypeScript issues
npm run type-check
npm run lint:fix
```

**Test Failures:**
```bash
# Run tests locally
npm run test

# Check test coverage
npm run test:coverage
```

**Docker Build Failures:**
```bash
# Check Dockerfile syntax
docker build -f Dockerfile.backend .

# Check build context
docker build --no-cache -f Dockerfile.backend .
```

### Health Checks

```bash
# Backend health
curl http://localhost:3002/health

# Frontend health
curl http://localhost:3001

# Admin dashboard health
curl http://localhost:3000

# Database health
curl http://localhost:3002/api/health/db

# Redis health
curl http://localhost:3002/api/health/redis
```

### Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f admin

# View last 100 lines
docker-compose logs --tail=100 backend
```

## 📊 Monitoring

### Service Status

```bash
# Check service status
docker-compose ps

# Check resource usage
docker stats

# Check service health
docker-compose exec backend curl http://localhost:3002/health
```

### Performance Monitoring

- **Backend**: Response times, error rates, memory usage
- **Frontend**: Bundle size, load times, user interactions
- **Database**: Query performance, connection pool, storage
- **Redis**: Memory usage, hit rate, connection count

## 🔒 Security

### Security Measures

- **Dependency Scanning**: Automated vulnerability detection
- **Container Security**: Non-root users, minimal base images
- **Network Security**: Internal networking, port restrictions
- **Secret Management**: Environment variables, secure storage

### Security Commands

```bash
# Run security audit
npm audit

# Fix security issues
npm audit fix

# Check for vulnerabilities
npm audit --audit-level=moderate
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Node.js Best Practices](https://github.com/goldbergyoni/nodebestpractices)
- [TypeScript Configuration](https://www.typescriptlang.org/docs/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the logs for error details
- Contact the development team

---

**Ready to deploy! 🚀**

Remember to:
1. Test in staging first
2. Monitor logs after deployment
3. Set up alerts for errors
4. Keep backups of your database
5. Update documentation
