# Frontend Service

The QuantDesk frontend is a modern React-based trading interface built with TypeScript, Vite, and Tailwind CSS.

## 🚀 Quick Start

```bash
cd frontend
pnpm install
pnpm run dev
```

## 🏗️ Architecture

### Tech Stack
- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Solana Web3.js** - Blockchain integration
- **React Query** - Data fetching and caching

### Key Features
- **Trading Interface** - Professional trading terminal
- **Portfolio Management** - Real-time portfolio tracking
- **User Dashboard** - Account management and analytics
- **Responsive Design** - Mobile and desktop optimized
- **Wallet Integration** - Solana wallet connectivity

## 📁 Structure

```
frontend/
├── src/
│   ├── components/     # Reusable UI components
│   │   ├── ui/        # Basic UI components (public)
│   │   └── trading/   # Trading-specific components (proprietary)
│   ├── hooks/         # Custom React hooks
│   ├── services/      # API and blockchain services
│   ├── utils/         # Helper functions
│   ├── types/         # TypeScript type definitions
│   └── pages/         # Page components
├── public/            # Static assets
└── dist/             # Build output
```

## 🔧 Development

### Environment Setup
```bash
cp .env.example .env
# Configure your environment variables
```

### Available Scripts
- `pnpm run dev` - Start development server
- `pnpm run build` - Build for production
- `pnpm run preview` - Preview production build
- `pnpm run lint` - Run ESLint
- `pnpm run type-check` - Run TypeScript compiler

### Component Development
```typescript
// Example component structure
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

export const ExampleComponent: React.FC<{
  title: string;
  children: React.ReactNode;
}> = ({ title, children }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {children}
      </CardContent>
    </Card>
  );
};
```

## 🌐 Public Components

The following components are available for community use:

### UI Components (`src/components/ui/`)
- **Card** - Container component
- **Button** - Interactive button component
- **Input** - Form input component
- **Badge** - Status indicator component
- **Modal** - Overlay dialog component

### Hooks (`src/hooks/`)
- **useWallet** - Solana wallet integration
- **useMarketData** - Real-time market data
- **usePortfolio** - Portfolio management
- **useOrders** - Order management

### Services (`src/services/`)
- **apiService** - Backend API communication
- **walletService** - Wallet operations
- **marketService** - Market data operations

## 🔒 Security

- **Environment Variables** - Sensitive data in `.env` files
- **API Keys** - Never exposed in client-side code
- **Wallet Security** - Private keys never stored
- **Input Validation** - All user inputs validated

## 📚 Examples

See `examples/frontend-ui-components.tsx` for comprehensive component examples.

## 🧪 Testing

```bash
pnpm run test        # Run unit tests
pnpm run test:e2e    # Run end-to-end tests
pnpm run coverage    # Generate coverage report
```

## 🚀 Deployment

The frontend is deployed to Vercel with automatic deployments from the main branch.

### Build Configuration
- **Target**: Modern browsers
- **Bundle Size**: Optimized with tree shaking
- **Assets**: Optimized images and fonts
- **Caching**: Aggressive caching for static assets

## 📖 API Integration

The frontend communicates with the backend API:

```typescript
// Example API service usage
import { apiService } from './services/apiService';

const marketData = await apiService.getMarketData('SOL');
const portfolio = await apiService.getPortfolio();
const orders = await apiService.getOrders();
```

## 🔧 Configuration

### Environment Variables
- `VITE_API_URL` - Backend API URL
- `VITE_SOLANA_RPC_URL` - Solana RPC endpoint
- `VITE_WALLET_ADAPTER_NETWORK` - Solana network
- `VITE_SUPABASE_URL` - Supabase project URL
- `VITE_SUPABASE_ANON_KEY` - Supabase anonymous key

### Build Configuration
- **Vite** configuration in `vite.config.ts`
- **TypeScript** configuration in `tsconfig.json`
- **Tailwind** configuration in `tailwind.config.js`

## 🐛 Troubleshooting

### Common Issues
1. **Wallet Connection** - Ensure Solana wallet is installed
2. **API Errors** - Check backend service status
3. **Build Errors** - Clear node_modules and reinstall
4. **Type Errors** - Run `pnpm run type-check`

### Debug Mode
```bash
VITE_DEBUG=true pnpm run dev
```

## 📄 License

This frontend code is part of QuantDesk and is licensed under Apache License 2.0.
