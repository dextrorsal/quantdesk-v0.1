# Technical Debt and Known Issues

## Critical Technical Debt

1. **Missing Social Media Integration**: No Twitter API, Discord, or Telegram integration yet - needs implementation
2. **Missing News Integration**: No real-time news aggregation or sentiment analysis - needs implementation
3. **Missing Unified Dashboard**: No single interface combining all data sources - needs implementation
4. **Data Ingestion Service**: Currently minimal implementation, needs expansion for social media feeds
5. **AI Service**: Basic implementation, needs enhancement for sentiment analysis and alpha channel processing

## Workarounds and Gotchas

- **Package Manager**: ALWAYS use pnpm, never npm - this is critical for the project
- **Database Access**: Always use `databaseService` from `backend/src/services/supabaseDatabase.ts`, never direct Supabase calls
- **Oracle Prices**: Prices are in scientific notation and already normalized by backend - don't apply exponent again
- **Environment Variables**: Backend loads from `backend/.env`, not root `.env`
- **Smart Contract**: Program ID is `C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw` on devnet
- **WebSocket**: Uses both Socket.IO and native WebSocket for different purposes
