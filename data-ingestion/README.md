# Data Ingestion Service

The QuantDesk data ingestion service is a real-time market data processing pipeline that collects, normalizes, and distributes market data from multiple sources.

## 🚀 Quick Start

```bash
cd data-ingestion
pnpm install
pnpm run dev
```

## 🏗️ Architecture

### Tech Stack
- **Node.js 20+** - Runtime environment
- **WebSocket** - Real-time data streams
- **Redis** - Data caching and pub/sub
- **TypeScript** - Type safety
- **Express.js** - API server
- **EventEmitter** - Event-driven architecture

### Key Features
- **Real-time Data** - Live market data collection
- **Multi-source Aggregation** - Multiple exchange data sources
- **Data Normalization** - Standardized data formats
- **Price Validation** - Data quality assurance
- **Event Streaming** - Real-time data distribution
- **Pipeline Management** - Data flow orchestration

## 📁 Structure

```
data-ingestion/
├── src/
│   ├── processors/    # Data processing components
│   │   ├── common/   # Public processors
│   │   └── trading/  # Trading processors (proprietary)
│   ├── services/     # Core services
│   ├── middleware/   # Express middleware
│   ├── routes/       # API routes
│   ├── utils/        # Helper functions
│   └── types/        # TypeScript types
├── tests/            # Test files
└── dist/            # Build output
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

### Processor Development
```typescript
// Example processor structure
import { EventEmitter } from 'events';
import WebSocket from 'ws';

export class MarketDataProcessor extends EventEmitter {
  private ws: WebSocket | null = null;

  constructor(private config: any) {
    super();
  }

  async start(): Promise<void> {
    this.ws = new WebSocket(this.config.wsUrl);
    this.ws.on('message', (data) => this.processData(data));
  }

  private processData(data: Buffer): void {
    const message = JSON.parse(data.toString());
    this.emit('marketData', this.normalizeData(message));
  }
}
```

## 🌐 Public Processors

The following processors are available for community use:

### Common Processors (`src/processors/common/`)
- **MarketDataProcessor** - WebSocket data processing
- **PriceAggregator** - Multi-source price aggregation
- **DataValidator** - Data quality validation
- **EventPublisher** - Data event distribution

### Services (`src/services/`)
- **DataSourceService** - Data source management
- **AggregationService** - Price aggregation logic
- **ValidationService** - Data validation rules
- **StorageService** - Data persistence

### Utilities (`src/utils/`)
- **DataNormalizer** - Data format standardization
- **EventBus** - Event distribution system
- **RateLimiter** - API rate limiting
- **Logger** - Structured logging

## 🔒 Security

- **API Key Management** - Secure data source credentials
- **Data Validation** - Input sanitization and validation
- **Rate Limiting** - Prevent API abuse
- **Access Control** - Data source access management
- **Audit Logging** - Data processing audit trails

## 📚 Examples

See `examples/data-ingestion-processors.ts` for comprehensive processor examples.

## 🧪 Testing

```bash
pnpm run test        # Run unit tests
pnpm run test:watch  # Run tests in watch mode
pnpm run coverage    # Generate coverage report
```

## 🚀 Deployment

The data ingestion service is deployed to Vercel with automatic deployments from the main branch.

### Environment Configuration
- **Development** - Local development with test data
- **Staging** - Staging environment with limited sources
- **Production** - Production environment with all sources

## 📖 API Documentation

### Data Endpoints
```typescript
GET /api/data/price/:symbol     # Get current price
GET /api/data/history/:symbol   # Get price history
GET /api/data/sources           # Get available sources
GET /api/data/status            # Get pipeline status
```

### WebSocket Events
```typescript
// Real-time price updates
ws.on('price_update', (data) => {
  console.log('Price update:', data);
});
```

## 🔧 Configuration

### Environment Variables
- `REDIS_URL` - Redis connection URL
- `WS_URL` - WebSocket data source URL
- `API_KEY` - Data source API key
- `PORT` - Server port (default: 3003)

### Data Sources
- **Primary Sources** - Major exchange WebSocket feeds
- **Secondary Sources** - Backup data sources
- **Aggregation** - Multi-source price aggregation
- **Validation** - Data quality checks

## 🐛 Troubleshooting

### Common Issues
1. **WebSocket Errors** - Check data source connectivity
2. **Redis Connection** - Verify Redis configuration
3. **Data Validation** - Check data format compliance
4. **Rate Limiting** - Adjust API rate limits

### Debug Mode
```bash
DEBUG=data-ingestion:* pnpm run dev
```

### Health Check
```bash
curl http://localhost:3003/api/health
```

## 📊 Monitoring

### Metrics
- **Data Throughput** - Messages processed per second
- **Data Quality** - Validation success rate
- **Source Health** - Data source availability
- **Processing Latency** - Data processing time

### Logging
- **Data Processing** - Detailed processing logs
- **Source Monitoring** - Data source health logs
- **Error Tracking** - Processing error logs
- **Performance Metrics** - Processing time logs

## 🔄 Data Pipeline

### Processing Flow
1. **Data Collection** - WebSocket data ingestion
2. **Normalization** - Standardize data formats
3. **Validation** - Quality assurance checks
4. **Aggregation** - Multi-source price aggregation
5. **Distribution** - Real-time data distribution
6. **Storage** - Historical data persistence

### Data Formats
```typescript
// Standardized market data format
interface MarketData {
  symbol: string;
  price: number;
  volume: number;
  timestamp: Date;
  source: string;
}
```

## 📄 License

This data ingestion code is part of QuantDesk and is licensed under Apache License 2.0.