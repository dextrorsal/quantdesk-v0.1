# 🌐 QuantDesk Open Source Repository Structure

## 🎯 Balanced Approach: Open Source Value + Proprietary Protection

This repository is designed to provide **maximum value to the open source community** while protecting **proprietary trading algorithms** and **sensitive business logic**.

## 🔒 Protected Components (Hidden from Public)

### **Proprietary Trading Logic**
```
/frontend/src/components/trading/     # Trading UI components
/frontend/src/hooks/trading/          # Trading-specific hooks
/frontend/src/services/trading/       # Trading business logic
/frontend/src/utils/trading/          # Trading utilities
/frontend/src/store/trading/          # Trading state management

/backend/src/services/trading/        # Trading API services
/backend/src/controllers/trading/     # Trading controllers
/backend/src/middleware/trading/      # Trading middleware
/backend/src/utils/trading/           # Trading utilities
/backend/src/strategies/              # Trading strategies

/MIKEY-AI/src/agents/trading/        # AI trading agents
/MIKEY-AI/src/strategies/            # AI trading strategies
/MIKEY-AI/src/models/                # AI models
/MIKEY-AI/src/algorithms/            # AI algorithms

/data-ingestion/src/processors/trading/ # Trading data processors
/data-ingestion/src/analyzers/        # Data analyzers
/data-ingestion/src/strategies/       # Data strategies

/admin-dashboard/src/components/trading/ # Admin trading components
/admin-dashboard/src/services/trading/   # Admin trading services

/contracts/smart-contracts/programs/  # Smart contract source
/contracts/programs/                  # Solana programs
```

## 🌐 Open Source Components (Visible to Community)

### **Frontend - UI & Common Components**
```
/frontend/src/components/ui/          # ✅ Reusable UI components
/frontend/src/components/common/      # ✅ Common components
/frontend/src/hooks/common/           # ✅ Common React hooks
/frontend/src/services/api/           # ✅ API service layer
/frontend/src/utils/common/            # ✅ Common utilities
/frontend/src/types/                  # ✅ TypeScript types
/frontend/src/constants/              # ✅ Constants
/frontend/public/                     # ✅ Public assets
/frontend/src/styles/                 # ✅ Styling system
/frontend/src/layouts/                # ✅ Layout components
```

### **Backend - API & Common Services**
```
/backend/src/services/api/            # ✅ API services
/backend/src/controllers/api/         # ✅ API controllers
/backend/src/middleware/common/       # ✅ Common middleware
/backend/src/utils/common/             # ✅ Common utilities
/backend/src/types/                   # ✅ TypeScript types
/backend/src/constants/                # ✅ Constants
/backend/src/routes/                  # ✅ API routes
/backend/src/validators/              # ✅ Input validators
/backend/src/errors/                  # ✅ Error handling
/backend/src/config/                 # ✅ Configuration
```

### **MIKEY-AI - Common AI Components**
```
/MIKEY-AI/src/agents/common/          # ✅ Common AI agents
/MIKEY-AI/src/utils/                  # ✅ Common utilities
/MIKEY-AI/src/types/                  # ✅ TypeScript types
/MIKEY-AI/src/constants/              # ✅ Constants
/MIKEY-AI/src/interfaces/             # ✅ AI interfaces
/MIKEY-AI/src/config/                 # ✅ AI configuration
```

### **Data Ingestion - Common Processors**
```
/data-ingestion/src/processors/common/ # ✅ Common data processors
/data-ingestion/src/utils/             # ✅ Common utilities
/data-ingestion/src/types/             # ✅ TypeScript types
/data-ingestion/src/config/            # ✅ Configuration
/data-ingestion/src/connectors/        # ✅ Data connectors
```

### **Admin Dashboard - Common Components**
```
/admin-dashboard/src/components/ui/    # ✅ UI components
/admin-dashboard/src/components/common/ # ✅ Common components
/admin-dashboard/src/services/api/    # ✅ API services
/admin-dashboard/src/utils/           # ✅ Common utilities
/admin-dashboard/src/types/           # ✅ TypeScript types
```

## 📚 Always Public Components

### **Documentation & Guides**
```
/docs/                                # ✅ Complete documentation
/examples/                            # ✅ Code examples
/scripts/                             # ✅ Utility scripts
/sdk/                                 # ✅ Public SDK
/database/schema.sql                  # ✅ Database schema
/contracts/docs/                      # ✅ Smart contract docs
/README.md                            # ✅ Main readme
/LICENSE                              # ✅ License file
/SECURITY.md                          # ✅ Security guide
```

### **Configuration & Setup**
```
package.json                          # ✅ Package configuration
pnpm-lock.yaml                        # ✅ Dependency lock
tsconfig.json                         # ✅ TypeScript config
docker-compose.yml                    # ✅ Docker configuration
.env.example                          # ✅ Environment template
```

## 🎯 Open Source Value Proposition

### **What Contributors Get:**
- 🎨 **UI Components** - Reusable React components
- 🔌 **API Services** - Complete API service layer
- 🛠️ **Utilities** - Common utility functions
- 📊 **Data Processors** - Data ingestion patterns
- 🤖 **AI Interfaces** - AI agent interfaces
- 📚 **Documentation** - Comprehensive guides
- 🧪 **Examples** - Working code samples
- 🔧 **Scripts** - Development utilities

### **What Stays Private:**
- 🎯 **Trading Algorithms** - Proprietary strategies
- 🧠 **AI Models** - Trained models and weights
- 💰 **Business Logic** - Revenue-generating code
- 🔐 **Smart Contracts** - Trading contract logic
- 📈 **Analytics** - Performance metrics

## 🚀 Community Benefits

### **For Developers:**
- Learn modern React/TypeScript patterns
- Study API design and architecture
- Understand data processing pipelines
- Explore AI agent interfaces
- Access comprehensive documentation

### **For Contributors:**
- Contribute to UI components
- Improve API services
- Add utility functions
- Enhance documentation
- Create examples and demos

### **For Integrators:**
- Use public SDK components
- Follow API patterns
- Implement data connectors
- Build on common utilities
- Reference architecture docs

## 🔍 File Structure Examples

### **Frontend Structure (Partial)**
```
frontend/src/
├── components/
│   ├── ui/                    # ✅ Public UI components
│   ├── common/               # ✅ Common components
│   └── trading/              # 🔒 Trading components (hidden)
├── hooks/
│   ├── common/               # ✅ Common hooks
│   └── trading/              # 🔒 Trading hooks (hidden)
├── services/
│   ├── api/                  # ✅ API services
│   └── trading/              # 🔒 Trading services (hidden)
└── utils/
    ├── common/               # ✅ Common utilities
    └── trading/              # 🔒 Trading utilities (hidden)
```

### **Backend Structure (Partial)**
```
backend/src/
├── services/
│   ├── api/                  # ✅ API services
│   └── trading/              # 🔒 Trading services (hidden)
├── controllers/
│   ├── api/                  # ✅ API controllers
│   └── trading/              # 🔒 Trading controllers (hidden)
├── middleware/
│   ├── common/               # ✅ Common middleware
│   └── trading/              # 🔒 Trading middleware (hidden)
└── utils/
    ├── common/               # ✅ Common utilities
    └── trading/              # 🔒 Trading utilities (hidden)
```

## 📋 Implementation Checklist

- [ ] ✅ UI components are public
- [ ] ✅ API services are public
- [ ] ✅ Common utilities are public
- [ ] ✅ Documentation is complete
- [ ] ✅ Examples are provided
- [ ] 🔒 Trading algorithms are private
- [ ] 🔒 AI models are private
- [ ] 🔒 Business logic is private
- [ ] 🔒 Smart contracts are private
- [ ] 🔒 Sensitive data is protected

---

**Result**: A valuable open source repository that provides real value to the community while protecting proprietary trading technology! 🎉
