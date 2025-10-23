# Integration Points and External Dependencies

## External Services

| Service  | Purpose  | Integration Type | Key Files                      |
| -------- | -------- | ---------------- | ------------------------------ |
| Pyth Network | Oracle prices | WebSocket + REST API | `backend/src/services/pythOracleService.ts` |
| Supabase | Database | REST API | `backend/src/services/supabaseDatabase.ts` |
| Solana RPC | Blockchain | Web3.js | `frontend/src/App.tsx` |
| Twitter API | Social media | REST API | **NOT IMPLEMENTED YET** |
| Discord API | Alpha channels | Bot API | **NOT IMPLEMENTED YET** |
| Telegram API | Alpha channels | Bot API | **NOT IMPLEMENTED YET** |
| News APIs | News aggregation | REST API | **NOT IMPLEMENTED YET** |

## Internal Integration Points

- **Frontend-Backend Communication**: REST API on port 3002, expects specific headers
- **AI Service Integration**: Backend calls MIKEY-AI service on port 3000
- **Data Ingestion**: Independent service on port 3003, communicates via Redis
- **Smart Contract Integration**: Frontend uses Anchor framework to interact with Solana program
