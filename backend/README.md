# Backend Service

The QuantDesk backend is a Node.js/Express API gateway that provides trading services, database management, and oracle integration.

## 🚀 Quick Start

```bash
cd backend
pnpm install
pnpm run dev
```

## 🏗️ Architecture

### Tech Stack
- **Node.js 20+** - Runtime environment
- **Express.js** - Web framework
- **TypeScript** - Type safety
- **Supabase** - Database and authentication
- **Pyth Network** - Price oracle
- **JWT** - Authentication tokens

### Key Features
- **API Gateway** - Centralized API management
- **Database Service** - Supabase abstraction layer
- **Oracle Integration** - Pyth Network price feeds
- **Authentication** - Multi-factor authentication
- **Rate Limiting** - Tiered rate limits
- **Error Handling** - Custom error classes

## 📁 Structure

```
backend/
├── src/
│   ├── controllers/   # Request handlers
│   ├── services/      # Business logic
│   │   ├── api/      # Public API services
│   │   └── trading/  # Trading services (proprietary)
│   ├── middleware/    # Express middleware
│   ├── routes/        # API routes
│   ├── models/        # Data models
│   ├── utils/         # Helper functions
│   └── types/         # TypeScript types
├── tests/             # Test files
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
- `pnpm run start` - Start production server
- `pnpm run test` - Run tests
- `pnpm run lint` - Run ESLint

### Service Development
```typescript
// Example service structure
import { DatabaseService } from './database';
import { OracleService } from './oracle';

export class MarketDataService {
  constructor(
    private database: DatabaseService,
    private oracle: OracleService
  ) {}

  async getPrice(symbol: string): Promise<number> {
    const price = await this.oracle.getPrice(symbol);
    await this.database.storePriceUpdate(symbol, price);
    return price;
  }
}
```

## 🌐 Public Services

The following services are available for community use:

### API Services (`src/services/api/`)
- **MarketDataService** - Market data operations
- **UserService** - User management
- **PortfolioService** - Portfolio operations
- **OrderService** - Order management

### Middleware (`src/middleware/`)
- **AuthMiddleware** - JWT authentication
- **RateLimiter** - Request rate limiting
- **ErrorHandler** - Centralized error handling
- **ValidationMiddleware** - Request validation

### Utilities (`src/utils/`)
- **DatabaseService** - Supabase abstraction
- **OracleService** - Pyth Network integration
- **Logger** - Structured logging
- **Validator** - Data validation

## 🔒 Security

- **JWT Authentication** - Secure token-based auth
- **Rate Limiting** - Prevent abuse
- **Input Validation** - Sanitize all inputs
- **SQL Injection Protection** - Parameterized queries
- **CORS Configuration** - Controlled cross-origin requests

## 📚 Examples

See `examples/backend-api-services.ts` for comprehensive service examples.

## 🧪 Testing

```bash
pnpm run test        # Run unit tests
pnpm run test:watch  # Run tests in watch mode
pnpm run coverage    # Generate coverage report
```

## 🚀 Deployment

The backend is deployed to Vercel with automatic deployments from the main branch.

### Environment Configuration
- **Development** - Local development with devnet
- **Staging** - Staging environment with testnet
- **Production** - Production environment with mainnet

## 📖 API Documentation

### Authentication
```typescript
// JWT token required for protected routes
Authorization: Bearer <jwt_token>
```

### Market Data Endpoints
```typescript
GET /api/market-data/:symbol
GET /api/market-data/:symbol/history
GET /api/market-summary
```

### Order Endpoints
```typescript
POST /api/orders          # Create order
GET /api/orders           # Get user orders
DELETE /api/orders/:id    # Cancel order
```

### Portfolio Endpoints
```typescript
GET /api/portfolio        # Get portfolio
GET /api/balance/:symbol  # Get balance
```

## 🔧 Configuration

### Environment Variables
- `DATABASE_URL` - Supabase database URL
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_ANON_KEY` - Supabase anonymous key
- `JWT_SECRET` - JWT signing secret
- `PYTH_NETWORK_URL` - Pyth oracle endpoint
- `PORT` - Server port (default: 3002)

### Database Schema
- **users** - User accounts and authentication
- **markets** - Trading markets configuration
- **positions** - User trading positions
- **orders** - Order management
- **trades** - Trade execution history

## 🐛 Troubleshooting

### Common Issues
1. **Database Connection** - Check Supabase credentials
2. **Oracle Errors** - Verify Pyth Network access
3. **Authentication** - Check JWT secret configuration
4. **Rate Limiting** - Adjust rate limit settings

### Debug Mode
```bash
DEBUG=quantdesk:* pnpm run dev
```

### Health Check
```bash
curl http://localhost:3002/api/health
```

## 📊 Monitoring

### Metrics
- **Request Rate** - API request frequency
- **Response Time** - API response latency
- **Error Rate** - API error frequency
- **Database Performance** - Query performance

### Logging
- **Structured Logs** - JSON formatted logs
- **Log Levels** - DEBUG, INFO, WARN, ERROR
- **Request Tracking** - Request ID correlation

## 📄 License

This backend code is part of QuantDesk and is licensed under Apache License 2.0.
