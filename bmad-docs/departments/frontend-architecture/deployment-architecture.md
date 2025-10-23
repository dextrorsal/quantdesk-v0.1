# Deployment Architecture

## Vercel Deployment Configuration
```json
// vercel.json - Deployment settings
{
  "buildCommand": "cd frontend && pnpm run build",
  "outputDirectory": "frontend/dist",
  "installCommand": "pnpm install",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/api/(.*)",
      "destination": "https://api.quantdesk.app/api/$1"
    }
  ],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        },
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        },
        {
          "key": "X-XSS-Protection",
          "value": "1; mode=block"
        }
      ]
    }
  ]
}
```

## Build Process
```bash
# Development
pnpm run dev          # Start development server on port 3001
pnpm run build        # Build for production
pnpm run preview      # Preview production build
pnpm run type-check   # TypeScript type checking
pnpm run lint         # ESLint code analysis
pnpm run test         # Run Vitest tests
pnpm run test:e2e     # Run Playwright E2E tests
```

## Environment-Specific Configurations
```typescript
// Environment variables
VITE_API_URL=http://localhost:3002          # Development
VITE_API_URL=https://api.quantdesk.app     # Production
VITE_WS_URL=ws://localhost:3002            # Development WebSocket
VITE_WS_URL=wss://api.quantdesk.app        # Production WebSocket
VITE_SOLANA_RPC_URL=https://api.devnet.solana.com  # Solana RPC
VITE_SOLANA_NETWORK=devnet                 # Network environment
VITE_DEBUG=true                            # Debug mode
```
