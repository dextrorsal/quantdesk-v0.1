# BMAD-CLI Quick Reference for QuantDesk

## BMAD Commands for CLI Tools (Claude Code, Gemini CLI, OpenAI Codex)

### Core Command Structure
```bash
# Use natural language with role-based requests
"As a Frontend React developer, implement the portfolio rebalancing feature across components"
"As a Backend Node.js engineer, add the risk management for liquidation system" 
"As a Smart Contracts developer, implement the oracle integration for new price feeds"
"As a Full-stack architect, design the cross-department integration for ML strategy deployment"

## Department-Specific Context Commands

### Frontend (React + TypeScript + Vite)
```bash
"Frontend dev: Add real-time portfolio dashboards using React + Zustand state management + Lightweight charts integration"
"React developer: Implement WebSocket integration for real-time trading data with error handling"
"Frontend architect: Design component structure for the DE trading interface with 45+ existing components"
```

### Backend (Node.js + Express + TypeScript)
```bash
"Backend engineer: Create 34+ API route handlers for trading engine with comprehensive error handling"
"Node.js developer: Implement JWT + SIWS authentication with Solana wallet integration"
"Express architect: Design microservice architecture for risk management, JIT liquidity, and portfolio analytics"
```

### MIKEY-AI (LangChain + Multi-LLM)
```bash
"AI engineer: Implement multi-LLM routing between OpenAI, Anthropic, Google with cost optimization"
"LangChain developer: Add CCXT integration for exchange market data analysis"
"AI architect: Design SlackChain agents for trading intelligence with QuantDesk API bridges"
```

### Data Ingestion (Node.js + Redis + Streaming)
```bash
"Data engineer: Implement 11 specialized data collectors for price feeds, whale monitoring, DeFi analytics"
"Redis developer: Set up Redis streaming architecture for real-time data processing"
"Pipeline architect: Design data flow from Pyth Network → processing → MIKEY-AI → trading systems"
```

### Smart Contracts (Rust + Anchor + Solana)
```bash" 
"Rust developer: Implement perpetual trading DEX with Anchor 0.32.1 and multi-oracle support"
"Solana architect: Design smart contract integration with program ID HsSzXVuSwiqGQvocT2zMX8zc86KQ9TYfZFZcfmmejzso"
"Blockchain engineer: Add Pyth Network oracl integration with proper confidence intervals"
```

## Cross-Department Integration Commands

### Multi-Department Features
```bash
"Full-stack architect: Design cross-department API integration between Frontend, Backend, MIKEY-AI, and Smart Contracts"
"System designer: Implement data flow: Data Ingestion → MIKEY-AI → Backend → Frontend with proper error handling"
"Integration specialist: Create real-time WebSocket communication for portfolio updates across all services"
```

### Cross-Department Testing
```bash
"QA engineer: Implement comprehensive testing strategy for cross-department trading features with P0/P1/P2 prioritization"
"Test architect: Set up integration testing for Frontend + Backend + Smart Contracts + MIKEY-AI interactions"
"Quality specialist: Design test scenarios for high-risk trading logic with proper financial risk assessment"
```

## Quality Assurance (Test Architect Commands)

The BMAD QA agent commands work in CLI too:
```bash
"QA risk assessment: Identify implementation risks for automated liquidation system"
"QA test design: Create comprehensive test strategy for real-time portfolio rebalancing"
"QA trace requirements: Ensure all acceptance criteria have test coverage for trading features"
"QA NFR validation: Verify security, performance, and reliability requirements for critical trading operations"
```

## Project-Specific Context

### QuantDesk Trading Platform Context
- **Technology Stack**: React, Node.js, TypeScript, Solana, PostgreSQL, Redis
- **Architecture**: 9-department system with 34+ backend routes, 25+ services
- **Key Features**: Perpetual trading, AI-powered analytics, real-time data ingestion
- **Risk Level**: High-stakes financial operations with comprehensive risk management

### BMAD Integration Benefits
- **Context Preservation**: Access to all department architecture automatically
- **Cross-Department Coordination**: Automatic dependency mapping between services
- **Quality Assurance**: Built-in test architecture and quality gates
- **Development Efficiency**: Targeted development with full system understanding

## Usage Examples

### Quick Bug Fixes (Claude CLI)
```bash
claude /dev "Fix the WebSocket connection issue in frontend portfolio updates"
claude /dev "Resolve the Oracle price feed integration error in backend"
claude /dev "Fix the MIKEY-AI agent context loading issue"
```

### Feature Development (Gemini CLI)
```bash
gemini /dev "Implement the real-time chart updates using Lightweight-charts library"
gemini /architect "Design the cross-department API for ML strategy deployment"
gemini /qa *risk "Assess risks for the automated rebalancing system"
```

### Code Refactoring (OpenAI Codex)
```bash
codex "Refactor the portfolio analytics service for better performance"
codex "Optimize the data ingestion pipeline for lower latency"
codex "Improve the smart contract gas efficiency for trading operations"
```

## Department Architecture References

All CLI tools can reference your department architecture documents:
- `/docs/departments/frontend-architecture.md` - React/TypeScript/Vite details
- `/docs/departments/backend-architecture.md` - Node.js/Express/34 routes  
- `/docs/departments/mikey-ai-architecture.md` - LangChain/Multi-LLM/AI integration
- `/docs/departments/data-ingestion-architecture.md` - 11 specialized collectors/Redis streaming
- `/docs/departments/smart-contracts-architecture.md` - Anchor/Solana/Program details

## Quick Start Commands

```bash
# Start with cross-department context
" architect: Design the complete system architecture for new cryptocurrency listing feature including all department integration"

# Use department-specific expertise
" frontend: Implement the UI components for cryptocurrency listing with real-time updates"
"backend: Create the API endpoints for token validation, price feeds, and listing management" 
"smart contracts: Implement the smart contract logic for token listing with oracle price validation"
"data-ingestion: Set up real-time data collectors for new token price monitoring"

# Comprehensive development workflow
"dev: Implement cross-department cryptocurrency listing feature with proper error handling and testing"
"qa: *risk cryptocurrency-listing-story  # Assess risks before implementation"
"qa: *design cryptocurrency-listing-story  # Create comprehensive test strategy"
"qa *review cryptocurrency-listing-story    # Full quality assessment with refactoring"
```

---

**Your CLI tools now have access to the complete QuantDesk BMAD context!** 🚀
