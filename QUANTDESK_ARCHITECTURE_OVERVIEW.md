# 🚀 QuantDesk Architecture Overview

## 📊 **Complete System Layout & Navigation**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           🏠 QUANTDESK PROJECT ROOT                            │
│                              /home/dex/Desktop/quantdesk                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            ┌───────▼───────┐ ┌─────▼─────┐ ┌──────▼──────┐
            │   🌐 FRONTEND │ │ 🔧 BACKEND│ │ 📚 DOCS-SITE│
            │   Port: 5173  │ │ Port: 3002│ │ Port: 8080  │
            └───────────────┘ └───────────┘ └─────────────┘
                    │               │               │
            ┌───────▼───────┐ ┌─────▼─────┐ ┌──────▼──────┐
            │  🎛️ ADMIN-DASH│ │   📊 API  │ │  📖 HTML    │
            │   Port: 5173  │ │  Routes   │ │  Content    │
            └───────────────┘ └───────────┘ └─────────────┘
```

## 🎯 **Main Applications & Ports**

### 1. **🌐 Frontend (Port 5173)**
**Location**: `/frontend/`
**Start**: `./scripts/dev/run-frontend.sh`
**Routes**:
```
/                    → Landing Page
/lite               → QuantDesk Lite (Chrome + Lite content)
/pro                → QuantDesk Pro (Terminal + taskbar shell)
/trading            → Standalone Trading Interface
/portfolio          → Portfolio Page (P&L analytics)
/markets            → Markets Page (Market overview)
/theme-demo         → Theme Demo
```

### 2. **🎛️ Admin Dashboard (Port 5173)**
**Location**: `/admin-dashboard/`
**Start**: `./admin-dashboard/start-admin.sh`
**Routes**:
```
/                   → Admin Dashboard (Main terminal interface)
```
**Features**:
- System Mode Control (Demo/Live toggle)
- Trading Operations
- Risk Management
- User Management
- System Health
- Market Data
- Compliance
- Analytics
- QuantDesk Core
- Exchange Status
- Cross-Chain Monitoring

### 3. **🔧 Backend API (Port 3002)**
**Location**: `/backend/`
**Start**: `./scripts/dev/start-backend.sh`
**API Routes**:
```
/api/auth           → Authentication
/api/markets        → Market data
/api/positions      → Position management
/api/orders         → Order management
/api/trades         → Trade history
/api/users          → User management
/api/admin          → Admin functions
/api/liquidity      → Liquidity management
/api/oracle         → Oracle data
/api/metrics        → System metrics
/api/grafana        → Grafana integration
/api/advanced-orders → Advanced order types
/api/cross-collateral → Cross-collateral
/api/portfolio-analytics → Portfolio analytics
/api/risk-management → Risk management
/api/jit-liquidity  → JIT liquidity
```

### 4. **📚 Documentation Site (Port 8080)**
**Location**: `/docs-site/`
**Start**: `./start-docs.sh`
**Content**:
```
/                   → Main documentation index
/html/              → Converted HTML files
├── docs_*.html     → Technical documentation
├── history_*.html  → Project history
└── guides/         → User guides
```

## 🔗 **Navigation Flow**

### **User Journey**:
```
1. Landing Page (/) 
   ↓
2. Choose Mode:
   ├── /lite     → QuantDesk Lite
   ├── /pro      → QuantDesk Pro  
   └── /trading  → Trading Interface
   ↓
3. Access Features:
   ├── /portfolio → Portfolio Management
   ├── /markets   → Market Overview
   └── Admin      → /admin-dashboard/
```

### **Admin Journey**:
```
1. Admin Dashboard (/admin-dashboard/)
   ↓
2. System Control:
   ├── Mode Toggle (Demo/Live)
   ├── User Management
   ├── System Health
   └── Risk Management
   ↓
3. Monitoring:
   ├── Exchange Status
   ├── Cross-Chain
   ├── Analytics
   └── Compliance
```

## 🛠️ **Development Workflow**

### **Start All Services**:
```bash
# Terminal 1: Backend
./scripts/dev/start-backend.sh

# Terminal 2: Frontend  
./scripts/dev/run-frontend.sh

# Terminal 3: Admin Dashboard
./admin-dashboard/start-admin.sh

# Terminal 4: Documentation
./start-docs.sh
```

### **Access Points**:
- **Frontend**: http://localhost:5173
- **Admin**: http://localhost:5173 (admin-dashboard)
- **Backend API**: http://localhost:3002
- **Documentation**: http://localhost:8080

## 📁 **Key Directories**

```
quantdesk/
├── frontend/           # Main React application
├── admin-dashboard/    # Admin terminal interface
├── backend/           # Express.js API server
├── docs-site/         # Documentation website
├── docs/              # Markdown documentation
├── contracts/         # Solana smart contracts
├── scripts/           # Development scripts
├── archive/           # Legacy code
└── examples/          # Code examples
```

## 🎨 **Theming & Branding**

All applications use **QuantDesk branding**:
- **Icon**: `quantdesk-icon.png` (used in all browser tabs)
- **Logo**: `quantdesk-logo.png`
- **Theme**: Dark terminal aesthetic with blue accents
- **Font**: JetBrains Mono (monospace)

## 🔄 **Data Flow**

```
Frontend ←→ Backend API ←→ Database
    ↓           ↓
Admin Dashboard  WebSocket
    ↓           ↓
Documentation   External APIs
```

## 🚀 **Quick Start Commands**

```bash
# Start everything
./scripts/dev/start-backend.sh &    # Backend
./scripts/dev/run-frontend.sh &     # Frontend  
./admin-dashboard/start-admin.sh &  # Admin
./start-docs.sh &                   # Docs

# Access
open http://localhost:5173          # Frontend
open http://localhost:5173          # Admin (same port)
open http://localhost:3002          # Backend API
open http://localhost:8080          # Documentation
```

## 📊 **Current Status**

✅ **Frontend**: Complete with landing, trading, portfolio, markets
✅ **Admin Dashboard**: Complete with 11 tabs, system control
✅ **Backend**: Complete with 15+ API route groups
✅ **Documentation**: Complete with 96+ HTML files
✅ **Branding**: QuantDesk icons on all sites
✅ **Theming**: Consistent dark terminal aesthetic

## 🎯 **Next Steps**

1. **Integration**: Connect frontend to real backend data
2. **Authentication**: Add user login system
3. **Real-time**: WebSocket integration for live data
4. **Deployment**: Production deployment setup
5. **Testing**: Comprehensive testing suite
