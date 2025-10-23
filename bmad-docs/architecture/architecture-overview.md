# Architecture Overview

## Multi-Service Architecture

| Service | Port | Technology | Purpose |
|---------|------|------------|---------|
| **Backend** | 3002 | Node.js/Express/TypeScript | API Gateway, Database, Oracle |
| **Frontend** | 3001 | React/Vite/TypeScript | Trading Interface, Portfolio |
| **MIKEY-AI** | 3000 | LangChain/TypeScript | AI Trading Agent |
| **Data Ingestion** | 3003 | Node.js/Pipeline | Real-time Data Collection |

## Core Design Principles

1. **Backend-Centric Oracle**: Pyth prices fetched by backend, normalized and cached
2. **Consolidated Database Service**: Single abstraction layer prevents direct Supabase usage
3. **Multi-Service Coordination**: Services communicate via backend API gateway
4. **Enterprise-Grade Security**: Multi-layer security with comprehensive monitoring
5. **Unified Trading Experience**: All trading tools in one interface
