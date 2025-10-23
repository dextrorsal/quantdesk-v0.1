# Appendix - Useful Commands and Scripts

## Frequently Used Commands

```bash
# Start all services
npm run dev

# Start individual services
cd backend && pnpm run dev
cd frontend && pnpm run dev
cd MIKEY-AI && pnpm run dev
cd data-ingestion && pnpm run dev

# Build all services
npm run build

# Run tests
cd contracts && anchor test
cd backend && pnpm test

# Database operations
cd backend && pnpm run init:devnet
```

## Debugging and Troubleshooting

- **Logs**: Check `backend/logs/` for application logs
- **Debug Mode**: Set `DEBUG=app:*` for verbose logging
- **API Testing**: Use `/api/dev/*` endpoints for system introspection
- **Common Issues**: 
  - Always use pnpm, not npm
  - Check environment variables in `backend/.env`
  - Verify Solana program deployment on devnet
  - Check Supabase connection and RLS policies

## AI Development Endpoints

The backend provides special endpoints optimized for AI assistants:

```bash
# Get system architecture
curl http://localhost:3002/api/dev/codebase-structure

# Get market data structure
curl http://localhost:3002/api/dev/market-summary

# Get API documentation
curl http://localhost:3002/api/docs/swagger
```

---

**Document Status**: Complete - Ready for Enhancement Implementation  
**Last Updated**: January 27, 2025  
**Next Review**: After Phase 2 Implementation  
**Implementation**: Core Platform Ready (85%), AI Tools Integration Phase 2 (0%)

This brownfield architecture document provides a comprehensive understanding of the current QuantDesk system state, enabling AI agents to effectively implement the social media integration, news sentiment analysis, alpha channel integration, and unified dashboard features described in the PRD.