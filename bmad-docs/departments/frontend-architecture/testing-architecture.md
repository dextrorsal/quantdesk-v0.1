# Testing Architecture

## Vitest Configuration
```typescript
// vitest.config.ts - Testing framework setup
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
    exclude: ['node_modules', 'dist', '.idea', '.git', '.cache'],
  },
})
```

## Test Setup and Mocks
```typescript
// test/setup.ts - Test environment configuration
import '@testing-library/jest-dom'

// Mock environment variables
Object.defineProperty(import.meta, 'env', {
  value: {
    VITE_SOLANA_RPC_URL: 'https://api.devnet.solana.com',
    VITE_SUPABASE_URL: 'https://test.supabase.co',
    VITE_SUPABASE_ANON_KEY: 'test-key',
  },
})

// Mock WebSocket
global.WebSocket = class WebSocket {
  constructor() {}
  close() {}
  send() {}
  addEventListener() {}
  removeEventListener() {}
} as any

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))
```

## Testing Patterns
```typescript
// Component testing example
describe('PortfolioDashboard', () => {
  it('renders portfolio data correctly', () => {
    render(<PortfolioDashboard />)
    expect(screen.getByText('Portfolio')).toBeInTheDocument()
  })
  
  it('handles loading state', () => {
    render(<PortfolioDashboard loading={true} />)
    expect(screen.getByText('Loading...')).toBeInTheDocument()
  })
})
```
