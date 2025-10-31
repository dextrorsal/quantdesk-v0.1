# MIKEY-AI Configuration Summary

## ✅ What's Fixed

1. **Port Configuration**: MIKEY-AI runs on port **3000**
2. **Backend Connection**: All tools point to backend on port **3002** (`QUANTDESK_URL=http://localhost:3002`)
3. **Tool Routing**: Fixed priority order - simple price queries route to `QuantDeskProtocolTools` first
4. **TypeScript Errors**: Fixed syntax errors in `TradingAgent.ts` and type annotations in `RealTokenAnalysisTool.ts`

## 🔧 Configuration

### Environment Variables Needed:
```bash
PORT=3000                    # MIKEY-AI port
QUANTDESK_URL=http://localhost:3002  # Backend API URL
```

### Service Ports:
- **MIKEY-AI**: Port 3000 ✅ (Running)
- **Backend**: Port 3002 (Needs to be running for tools to work)
- **Frontend**: Port 3001
- **Data Ingestion**: Port 3003

## 🔌 Tool Endpoints (All call backend on port 3002)

All MIKEY tools connect to backend on port 3002:
- `/api/oracle/price/:asset` - Single asset price
- `/api/oracle/prices` - All prices
- `/api/dev/market-summary` - Market data
- `/api/dev/user-portfolio/:wallet` - Portfolio data

## ⚠️ Current Status

- ✅ MIKEY-AI is running on port 3000
- ✅ Tool routing is fixed (simple price queries → QuantDeskProtocolTools)
- ❌ Backend needs to be running on port 3002 for tools to work

## 🚀 To Test

1. Start backend: `cd backend && pnpm run start:dev` (should start on port 3002)
2. Test query: `curl -X POST http://localhost:3000/api/v1/ai/query -H "Content-Type: application/json" -d '{"query":"What is the live price of ETH?"}'`
3. Run full test suite: `cd MIKEY-AI && node test-tool-routing.js`

