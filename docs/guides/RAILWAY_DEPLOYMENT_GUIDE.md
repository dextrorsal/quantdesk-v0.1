# Railway Deployment Guide for QuantDesk Backend

## Overview
This guide will help you deploy your QuantDesk backend to Railway with a private GitHub repository.

## Prerequisites
- Private GitHub repository with QuantDesk code
- Railway account (free tier available)
- Environment variables prepared

## Step 1: Railway Account Setup

### 1.1 Create Railway Account
1. Go to [Railway.app](https://railway.app)
2. Sign up with GitHub (recommended for private repos)
3. Authorize Railway to access your GitHub repositories

### 1.2 Install Railway CLI (Optional but Recommended)
```bash
npm install -g @railway/cli
railway login
```

## Step 2: Project Setup

### 2.1 Connect Repository
1. In Railway dashboard, click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your `quantdesk` repository
4. Railway will automatically detect the Dockerfile

### 2.2 Configure Build Settings
Railway should automatically detect:
- **Builder**: Dockerfile
- **Dockerfile Path**: `Dockerfile` (root directory)
- **Build Context**: Root directory

## Step 3: Environment Variables

### 3.1 Required Environment Variables
Set these in Railway dashboard under "Variables" tab:

```bash
# Core Configuration
NODE_ENV=production
PORT=3002
ENVIRONMENT=production

# Database Configuration
DATABASE_URL=postgresql://username:password@host:port/database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# JWT Configuration
JWT_SECRET=your-super-secret-jwt-key-minimum-32-characters-long
JWT_EXPIRES_IN=7d

# Solana Configuration
SOLANA_NETWORK=devnet
RPC_URL=https://api.devnet.solana.com
WS_URL=wss://api.devnet.solana.com
PROGRAM_ID=your_program_id_here

# RPC Provider URLs (Primary Set)
HELIUS_RPC_1_URL=https://your-helius-rpc-url.com/api-key/YOUR_KEY
QUICKNODE_1_RPC_URL=https://your-quicknode-rpc-url.com/api-key/YOUR_KEY
ALCHEMY_1_RPC_URL=https://your-alchemy-rpc-url.com/api-key/YOUR_KEY

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Oracle Configuration
PYTH_NETWORK_URL=https://hermes.pyth.network/v2/updates/price/latest
PYTH_PRICE_FEED_BTC=your_pyth_btc_feed_id
PYTH_PRICE_FEED_ETH=your_pyth_eth_feed_id
PYTH_PRICE_FEED_SOL=your_pyth_sol_feed_id

# Rate Limiting
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=1000

# WebSocket Configuration
WS_TOKEN=your_websocket_token_here
```

### 3.2 Optional Environment Variables
```bash
# External Services (if using)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Exchange API Keys (if using live trading)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# Trading Configuration
DEFAULT_STAKE_AMOUNT=1000
MAX_OPEN_TRADES=3
DEFAULT_TIMEFRAME=5m
MAX_DRAWDOWN_PERCENT=10
STOP_LOSS_PERCENT=5
TAKE_PROFIT_PERCENT=10
```

## Step 4: Database Setup

### 4.1 Railway PostgreSQL (Recommended)
1. In Railway dashboard, click "New" → "Database" → "PostgreSQL"
2. Railway will automatically provide `DATABASE_URL`
3. Copy the connection string to your environment variables

### 4.2 Alternative: External Database
If using external database (Supabase, etc.):
- Ensure your database is accessible from Railway's IP ranges
- Use the external connection string in `DATABASE_URL`

## Step 5: Deployment Process

### 5.1 Automatic Deployment
1. Railway will automatically build and deploy when you push to your main branch
2. Monitor the build logs in Railway dashboard
3. Check deployment logs for any errors

### 5.2 Manual Deployment (if needed)
```bash
# Using Railway CLI
railway up

# Or trigger deployment from Railway dashboard
```

## Step 6: Post-Deployment Configuration

### 6.1 Health Check
Your application should be accessible at:
- **Health endpoint**: `https://your-app.railway.app/health`
- **API base**: `https://your-app.railway.app/api/`

### 6.2 Domain Configuration (Optional)
1. In Railway dashboard, go to "Settings" → "Domains"
2. Add your custom domain
3. Configure DNS records as instructed

### 6.3 SSL/HTTPS
Railway automatically provides SSL certificates for:
- `*.railway.app` domains
- Custom domains (with proper DNS configuration)

## Step 7: Monitoring and Logs

### 7.1 View Logs
- Railway dashboard → Your project → "Deployments" → Click deployment
- Real-time logs are available during deployment
- Historical logs are stored for debugging

### 7.2 Health Monitoring
Railway automatically monitors:
- Application health via `/health` endpoint
- Container restarts
- Resource usage

## Step 8: Troubleshooting

### 8.1 Common Issues

#### Build Failures
```bash
# Check Dockerfile syntax
docker build -t quantdesk-test .

# Test locally
docker run -p 3002:3002 quantdesk-test
```

#### Environment Variable Issues
- Ensure all required variables are set
- Check variable names match exactly (case-sensitive)
- Verify database connection strings

#### Port Configuration
- Railway automatically maps PORT environment variable
- Ensure your app listens on the PORT environment variable
- Default Railway port is dynamic, don't hardcode

### 8.2 Debug Commands
```bash
# Check Railway CLI status
railway status

# View environment variables
railway variables

# Connect to Railway shell
railway shell
```

## Step 9: Production Optimizations

### 9.1 Performance
- Enable Railway's auto-scaling
- Configure resource limits
- Monitor memory and CPU usage

### 9.2 Security
- Use Railway's built-in secrets management
- Enable rate limiting
- Configure CORS properly
- Use HTTPS only

### 9.3 Backup Strategy
- Regular database backups
- Environment variable backups
- Code repository backups

## Step 10: CI/CD Integration

### 10.1 GitHub Actions (Optional)
Create `.github/workflows/railway-deploy.yml`:
```yaml
name: Deploy to Railway
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '20'
      - run: npm install -g @railway/cli
      - run: railway up --service backend
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
```

## Step 11: Scaling and Maintenance

### 11.1 Horizontal Scaling
- Railway supports horizontal scaling
- Configure in "Settings" → "Scaling"
- Monitor costs and performance

### 11.2 Updates and Rollbacks
- Railway maintains deployment history
- Easy rollback to previous versions
- Blue-green deployments supported

## Security Considerations

1. **Private Repository**: Ensure your GitHub repo remains private
2. **Environment Variables**: Never commit sensitive data
3. **Database Security**: Use strong passwords and restrict access
4. **API Keys**: Rotate keys regularly
5. **Monitoring**: Set up alerts for unusual activity

## Cost Optimization

1. **Resource Limits**: Set appropriate CPU/memory limits
2. **Auto-scaling**: Configure based on actual usage
3. **Database**: Use appropriate database tier
4. **Monitoring**: Track usage and optimize accordingly

## Support and Resources

- **Railway Documentation**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **GitHub Issues**: Use Railway's GitHub integration for support

## Next Steps

1. Deploy your backend to Railway
2. Test all endpoints thoroughly
3. Set up monitoring and alerts
4. Configure custom domain (optional)
5. Set up CI/CD pipeline (optional)
6. Plan for scaling as your user base grows

---

**Note**: This guide assumes you're deploying the backend service. Smart contracts should be deployed separately to Solana devnet/mainnet using Anchor CLI.
