# 🚀 QuantDesk Complete Platform - Postman Setup Guide

## 📋 **What We've Created**

### **1. QuantDesk Complete Platform Collection** (`QuantDesk-Complete-Platform.json`)
- **🏥 Health & Status** - All service health checks
- **🔐 Authentication** - Login, refresh, profile
- **💰 Account Management** - Balances, trading accounts
- **📊 Markets & Prices** - Backend + Data Pipeline prices
- **📈 Data Pipeline APIs** - Signals, alerts, reports
- **🤖 MIKEY-AI Integration** - AI queries, wallet analysis
- **📋 Orders & Trading** - Order management
- **💳 Deposits & Withdrawals** - Financial operations
- **🔗 Solana RPC** - Direct blockchain interactions
- **📊 External Data Sources** - Pyth, Jupiter, etc.
- **🛠️ Admin & Monitoring** - Admin panel, RPC stats

### **2. Complete Environment** (`QuantDesk-Complete-Environment.json`)
- **Backend URLs** - Express API, WebSocket
- **Solana Configuration** - RPC, Program ID
- **Supabase** - Database URLs and keys
- **External APIs** - Pyth, Jupiter, data sources
- **Data Pipeline** - MIKEY-AI, Redis, Grafana
- **API Keys** - All 9 data service keys
- **RPC Providers** - Load balancer options
- **Social/AI** - Twitter, Telegram, Discord, AI providers

## 🎯 **How to Use This**

### **Step 1: Import into Postman**
1. **Open Postman**
2. **Import** → **File** → Select `QuantDesk-Complete-Platform.json`
3. **Import** → **File** → Select `QuantDesk-Complete-Environment.json`

### **Step 2: Configure Environment Variables**
1. **Select Environment** → "QuantDesk Complete Platform Environment"
2. **Update these values:**
   ```bash
   # Backend
   BACKEND_BASE_URL = http://localhost:3002
   
   # Solana
   SOLANA_RPC_URL = https://api.devnet.solana.com
   PROGRAM_ID = GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a
   WALLET_PUBKEY = YOUR_DEV_WALLET_PUBKEY
   
   # Supabase
   SUPABASE_URL = https://your-project.supabase.co
   SUPABASE_ANON_KEY = YOUR_SUPABASE_ANON_KEY
   
   # Data Pipeline
   AI_BASE_URL = http://localhost:3000
   
   # API Keys (from your .env)
   BIRDEYE_API_KEY = YOUR_BIRDEYE_API_KEY
   COINGECKO_API_KEY = YOUR_COINGECKO_API_KEY
   DUNE_API_KEY = YOUR_DUNE_API_KEY
   ARTEMIS_API_KEY = YOUR_ARTEMIS_API_KEY
   COINALYZE_API_KEY = YOUR_COINALYZE_API_KEY
   ```

### **Step 3: Test the Integration**

#### **🔍 Health Check Flow:**
1. **Backend Health** → `GET {{BACKEND_BASE_URL}}/health`
2. **MIKEY-AI Health** → `GET {{AI_BASE_URL}}/health`
3. **Redis Health** → `GET {{BACKEND_BASE_URL}}/api/rpc-stats/health`

#### **🔐 Authentication Flow:**
1. **Authenticate** → `POST {{BACKEND_BASE_URL}}/api/auth/authenticate`
2. **Get Profile** → `GET {{BACKEND_BASE_URL}}/api/auth/profile`
3. **Refresh Token** → `POST {{BACKEND_BASE_URL}}/api/auth/refresh`

#### **📊 Data Pipeline Flow:**
1. **Get Signals** → `GET {{AI_BASE_URL}}/api/v1/signals`
2. **Get Alerts** → `GET {{AI_BASE_URL}}/api/v1/alerts`
3. **AI Query** → `POST {{AI_BASE_URL}}/api/v1/ai/query`

#### **🤖 MIKEY-AI Flow:**
1. **Analyze Wallet** → `GET {{AI_BASE_URL}}/api/v1/wallets/{{WALLET_PUBKEY}}/analysis`
2. **Get Whale Activities** → `GET {{AI_BASE_URL}}/api/v1/trading/whales`
3. **Technical Analysis** → `GET {{AI_BASE_URL}}/api/v1/trading/analysis`

## 🎉 **What This Gives You**

### **🔄 Complete API Management**
- **Single Collection** for entire QuantDesk platform
- **Environment Variables** for easy switching between dev/prod
- **Auto-token Management** - tokens saved automatically
- **Performance Monitoring** - response time tracking
- **Error Handling** - comprehensive test scripts

### **🚀 Advanced Features**
- **Pre-request Scripts** - Auto-token refresh, logging
- **Test Scripts** - Response validation, performance monitoring
- **Auto-extraction** - Tokens, IDs, and data from responses
- **Health Monitoring** - All services in one place
- **Documentation** - Built-in API documentation

### **📊 Monitoring & Analytics**
- **Response Time Tracking** - Performance monitoring
- **Error Rate Monitoring** - API health tracking
- **Usage Analytics** - API call patterns
- **Integration Testing** - End-to-end workflows

## 🔥 **Next Steps**

### **1. Start Your Services**
```bash
# Backend (port 3002)
cd backend && npm start

# Data Pipeline (port 3000)
cd data-ingestion && npm run start:collectors

# MIKEY-AI (port 3000)
cd MIKEY-AI && npm run dev
```

### **2. Test the Complete Flow**
1. **Health Check** all services
2. **Authenticate** with backend
3. **Get Market Data** from both systems
4. **Query MIKEY-AI** for analysis
5. **Place Test Order** through backend
6. **Monitor Signals** from data pipeline

### **3. Set Up Monitoring**
- **Postman Monitors** for automated testing
- **Grafana Dashboard** for real-time metrics
- **Alert System** for critical failures

## 🎯 **Benefits of This Setup**

### **🏗️ Unified Platform**
- **Single Source of Truth** for all QuantDesk APIs
- **Consistent Environment** across all services
- **Easy Testing** of complete workflows

### **🔧 Developer Experience**
- **Auto-completion** with environment variables
- **Built-in Documentation** for each endpoint
- **Test Scripts** for validation
- **Performance Monitoring** built-in

### **📈 Production Ready**
- **Environment Switching** (dev/staging/prod)
- **API Key Management** with secrets
- **Monitoring & Alerting** capabilities
- **Documentation Generation** for teams

## 🚀 **Ready to Test!**

Your QuantDesk platform is now **fully integrated** with Postman! 

**Start with:** Health checks → Authentication → Data pipeline → MIKEY-AI queries

**This gives you:** Complete API management, monitoring, and testing for your entire QuantDesk trading platform! 🎉
