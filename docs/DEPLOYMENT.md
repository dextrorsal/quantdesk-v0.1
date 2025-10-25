# QuantDesk Deployment Configuration

## Environment Management

### Development Environment
- **Network**: Solana Devnet
- **RPC URL**: https://api.devnet.solana.com
- **Program ID**: C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw
- **Database**: Supabase Development
- **Oracle**: Pyth Network Devnet

### Staging Environment
- **Network**: Solana Testnet
- **RPC URL**: https://api.testnet.solana.com
- **Program ID**: [To be deployed]
- **Database**: Supabase Staging
- **Oracle**: Pyth Network Testnet

### Production Environment
- **Network**: Solana Mainnet
- **RPC URL**: https://api.mainnet-beta.solana.com
- **Program ID**: [To be deployed]
- **Database**: Supabase Production
- **Oracle**: Pyth Network Mainnet

## Environment Variables

### Frontend (.env)
```bash
# API Configuration
VITE_API_URL=https://api.quantdesk.app

# Solana Configuration
VITE_SOLANA_RPC_URL=https://api.devnet.solana.com
VITE_SOLANA_WS_URL=wss://api.devnet.solana.com
VITE_WALLET_ADAPTER_NETWORK=devnet
VITE_QUANTDESK_PROGRAM_ID=C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw

# Oracle Configuration
VITE_PYTH_NETWORK_URL=https://hermes.pyth.network/v2/updates/price/latest

# Database Configuration
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### Backend (.env)
```bash
# Server Configuration
NODE_ENV=development
PORT=3002

# Database Configuration
DATABASE_URL=your_database_url
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# Authentication
JWT_SECRET=your_jwt_secret
JWT_EXPIRES_IN=24h

# Solana Configuration
SOLANA_RPC_URL=https://api.devnet.solana.com
SOLANA_WS_URL=wss://api.devnet.solana.com
QUANTDESK_PROGRAM_ID=C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw

# Oracle Configuration
PYTH_NETWORK_URL=https://hermes.pyth.network/v2/updates/price/latest

# AI Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### Contracts (.env)
```bash
# Solana Configuration
SOLANA_RPC_URL=https://api.devnet.solana.com
SOLANA_WS_URL=wss://api.devnet.solana.com
ANCHOR_PROVIDER_URL=https://api.devnet.solana.com
ANCHOR_WALLET=~/.config/solana/id.json

# Program Configuration
QUANTDESK_PROGRAM_ID=C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw

# Oracle Configuration
PYTH_PROGRAM_ID=FsJ3A3u2vn5cTVofAjvy6y5kwABJAqYWpe4975bi2epH
SWITCHBOARD_PROGRAM_ID=SW1TCH7qEPTdLsDHRgPuMQjbQxKdH2aBStViMFnt64f
```

## Deployment Commands

### Frontend Deployment
```bash
cd frontend
pnpm run build
vercel --prod
```

### Backend Deployment
```bash
cd backend
pnpm run build
vercel --prod
```

### Smart Contract Deployment
```bash
cd contracts
anchor build
anchor deploy --provider.cluster devnet
```

## Build Process

### Frontend Build
1. Install dependencies: `pnpm install`
2. Type check: `pnpm run type-check`
3. Build: `pnpm run build`
4. Deploy: `vercel --prod`

### Backend Build
1. Install dependencies: `pnpm install`
2. Type check: `pnpm run type-check`
3. Build: `pnpm run build`
4. Deploy: `vercel --prod`

### Smart Contract Build
1. Install Rust dependencies: `cargo build`
2. Build program: `anchor build`
3. Deploy: `anchor deploy --provider.cluster devnet`

## Validation Checks

### Pre-deployment
- [ ] All tests pass
- [ ] TypeScript compilation successful
- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] Smart contract deployed

### Post-deployment
- [ ] Health checks pass
- [ ] API endpoints accessible
- [ ] Database connectivity verified
- [ ] Oracle feeds working
- [ ] Smart contract interactions functional

## Rollback Procedures

### Frontend Rollback
```bash
vercel rollback [deployment-url]
```

### Backend Rollback
```bash
vercel rollback [deployment-url]
```

### Smart Contract Rollback
- Deploy previous version
- Update program ID references
- Restart services

## Monitoring

### Health Checks
- Frontend: `https://quantdesk.app/health`
- Backend: `https://api.quantdesk.app/api/health`
- Smart Contract: Check program balance and status

### Performance Metrics
- Response times
- Error rates
- Transaction success rates
- Oracle feed latency

## Security Considerations

### Environment Variables
- Never commit `.env` files
- Use Vercel environment variables for production
- Rotate secrets regularly

### Smart Contract Security
- Verify program deployment
- Monitor for unauthorized changes
- Implement circuit breakers

### Database Security
- Use connection pooling
- Implement rate limiting
- Monitor for suspicious activity
