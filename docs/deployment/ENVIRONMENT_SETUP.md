# Environment Setup Guide

## Required Environment Variables

### Backend (.env)

```bash
# Database Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
DATABASE_URL=your_postgresql_connection_string

# Solana Configuration
SOLANA_RPC_URL=https://api.devnet.solana.com
SOLANA_PRIVATE_KEY=your_base58_encoded_private_key
SOLANA_PROGRAM_ID=your_program_id

# Backend Configuration
PORT=3002
NODE_ENV=development
BACKEND_URL=http://localhost:3002
FRONTEND_URL=http://localhost:3001

# JWT Secrets
JWT_SECRET=your_jwt_secret_key
ADMIN_JWT_SECRET=your_admin_jwt_secret_key

# OAuth Configuration (Admin)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret

# AI Service Configuration
MIKEY_AI_URL=http://localhost:3000

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your_redis_password

# Oracle Configuration
PYTH_NETWORK_URL=https://hermes.pyth.network/v2/updates/price
PYTH_NETWORK_WS_URL=wss://hermes.pyth.network/v2/ws

# Security Configuration
ADMIN_IP_WHITELIST=false
ADMIN_MFA_REQUIRED=false
RATE_LIMIT_ENABLED=true
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100

# Logging Configuration
LOG_LEVEL=info
LOG_FILE_PATH=./logs/backend.log

# Development Configuration
ENABLE_CHAT=true
ENABLE_AI_FEATURES=true
ENABLE_ADVANCED_ORDERS=true
ENABLE_CROSS_COLLATERAL=true
DEBUG_MODE=false
VERBOSE_LOGGING=false
```

### Frontend (.env)

```bash
VITE_BACKEND_URL=http://localhost:3002
VITE_ADMIN_GOOGLE_AUTH_URL=http://localhost:3002/api/admin/auth/google
VITE_ADMIN_GITHUB_AUTH_URL=http://localhost:3002/api/admin/auth/github
```

## OAuth Setup Instructions

### Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google+ API
4. Go to Credentials → Create Credentials → OAuth 2.0 Client ID
5. Set application type to "Web application"
6. Add authorized redirect URIs:
   - `http://localhost:3002/api/admin/auth/google/callback` (development)
   - `https://your-domain.com/api/admin/auth/google/callback` (production)
7. Copy Client ID and Client Secret to environment variables

### GitHub OAuth Setup

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click "New OAuth App"
3. Fill in application details:
   - Application name: QuantDesk Admin
   - Homepage URL: `http://localhost:3001` (development)
   - Authorization callback URL: `http://localhost:3002/api/admin/auth/github/callback`
4. Copy Client ID and Client Secret to environment variables

## Database Setup

### Apply Security Fixes

```bash
# Apply RLS policies and security fixes
psql -d your_database -f database/security-audit-fixes.sql

# Run security verification tests
psql -d your_database -f database/security-verification-tests.sql
```

### Add OAuth Columns to Admin Users

```sql
-- Add OAuth columns to admin_users table
ALTER TABLE admin_users 
ADD COLUMN IF NOT EXISTS google_id TEXT UNIQUE,
ADD COLUMN IF NOT EXISTS github_id TEXT UNIQUE,
ADD COLUMN IF NOT EXISTS avatar_url TEXT,
ADD COLUMN IF NOT EXISTS oauth_provider TEXT;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_admin_users_google_id ON admin_users(google_id);
CREATE INDEX IF NOT EXISTS idx_admin_users_github_id ON admin_users(github_id);
```

## Installation Commands

### Install Dependencies

```bash
# Install all dependencies using pnpm workspace
pnpm install

# Or install individually
cd backend && pnpm install
cd frontend && pnpm install
cd MIKEY-AI && pnpm install
cd data-ingestion && pnpm install
```

### Start Services

```bash
# Start all services (development)
pnpm run dev

# Or start individually
cd backend && pnpm run dev
cd frontend && pnpm run dev
cd MIKEY-AI && pnpm run dev
cd data-ingestion && pnpm start
```

## Admin Dashboard Access

1. Navigate to `http://localhost:3001/admin`
2. Use Google or GitHub OAuth to login
3. Ensure your email is added to the `admin_users` table with appropriate role

## Security Notes

- Never commit `.env` files to version control
- Use strong, unique JWT secrets in production
- Enable MFA for admin accounts in production
- Use IP whitelisting for admin access in production
- Regularly rotate OAuth client secrets
- Monitor admin access logs for suspicious activity
