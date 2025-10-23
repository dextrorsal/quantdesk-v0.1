# Data Models and APIs

## Data Models

Instead of duplicating, reference actual model files:

- **User Model**: See `backend/src/services/supabaseDatabase.ts` - User interface with wallet authentication
- **Market Model**: See `backend/src/services/supabaseDatabase.ts` - Market interface with Pyth integration
- **Position Model**: See `backend/src/services/supabaseDatabase.ts` - Position interface with health factors
- **Order Model**: See `backend/src/services/supabaseDatabase.ts` - Order interface with advanced types
- **Trade Model**: See `backend/src/services/supabaseDatabase.ts` - Trade interface with PnL tracking
- **Database Schema**: See `database/schema.sql` - Complete PostgreSQL schema

## API Specifications

- **OpenAPI Spec**: Available at `/api/docs/swagger` endpoint
- **Development Endpoints**: `/api/dev/*` - Architecture introspection for AI assistants
- **Core Trading APIs**: `/api/positions/*`, `/api/orders/*`, `/api/trades/*` - Position and order management
- **Oracle APIs**: `/api/oracle/*` - Pyth Network price feeds
- **AI APIs**: `/api/ai/*`, `/api/chat/*` - MIKEY-AI integration
- **Admin APIs**: `/api/admin/*` - Administrative functions
