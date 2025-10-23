# API Integration Architecture

## HTTP Client Configuration
```typescript
// apiClient.ts - Axios configuration
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:3002',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for authentication
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})
```

## WebSocket Service Architecture
```typescript
// websocketService.ts - Real-time data streaming
export interface WebSocketMessage {
  type: 'market_data' | 'order_book' | 'trade' | 'position_update' | 'order_update'
  channel: string
  data: any
  timestamp: number
}

class WebSocketService {
  private ws: WebSocket | null = null
  private subscribers = new Map<string, Set<Function>>()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  
  // Connection management
  connect(url: string): Promise<void>
  disconnect(): void
  subscribe(channel: string, callback: Function): () => void
  send(message: WebSocketMessage): void
}
```
