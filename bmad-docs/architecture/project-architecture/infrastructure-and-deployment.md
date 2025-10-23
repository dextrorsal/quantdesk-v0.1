# Infrastructure and Deployment

## Infrastructure as Code

- **Tool:** Vercel CLI (Latest)
- **Location:** `vercel.json`, `vercel.toml`
- **Approach:** Serverless deployment with automatic scaling

## Deployment Strategy

- **Strategy:** Continuous deployment with Vercel
- **CI/CD Platform:** Vercel Git integration
- **Pipeline Configuration:** `vercel.json` in project root

## Environments

- **Development:** `dev-api.quantdesk.com` - Development server with devnet Solana
- **Production:** `api.quantdesk.com` - Production server with mainnet Solana
- **Staging:** `staging-api.quantdesk.com` - Staging environment for testing

## Environment Promotion Flow

```
Development → Staging → Production
     ↓           ↓         ↓
   Devnet    Devnet    Mainnet
   Testing   QA Tests   Live Users
```

## Rollback Strategy

- **Primary Method:** Vercel instant rollback to previous deployment
- **Trigger Conditions:** Error rate >5%, Response time >500ms, Critical bugs
- **Recovery Time Objective:** <2 minutes
