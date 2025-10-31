<story-context id="bmad/bmm/workflows/4-implementation/story-context/template" v="1.0">
  <metadata>
    <epicId>0</epicId>
    <storyId>2</storyId>
    <title>Market Data WebSocket Reliability in Pro Terminal</title>
    <status>ready-for-dev</status>
    <generatedAt>2025-10-30</generatedAt>
    <generator>BMAD Story Context Workflow</generator>
    <sourceStoryPath>bmad/stories/0-2-market-data-websocket-reliability-pro-terminal.md</sourceStoryPath>
  </metadata>

  <story>
    <asA>user</asA>
    <iWant>price updates via WebSocket with retry and safe fallback</iWant>
    <soThat>prices remain live</soThat>
    <tasks>
- [ ] Implement WebSocket connection with retry logic [AC1]
  - [ ] Connect via env URL or localhost:3002/ws
  - [ ] Exponential backoff retry (max 15s between retries)
  - [ ] Error throttling to prevent console spam
  - [ ] Connection state management (connecting, connected, disconnected, error)
- [ ] Implement polling fallback [AC2]
  - [ ] Detect WebSocket failure after max retries
  - [ ] Start polling `/api/oracle/prices` every 2s
  - [ ] Suppress console errors/spam during polling
  - [ ] Seamlessly switch back to WebSocket when available
- [ ] Add visual "Live" indicator [AC3]
  - [ ] Show "Live" when WebSocket connected
  - [ ] Show "Polling" when in fallback mode
  - [ ] Show "Disconnected" when neither working
  - [ ] Use appropriate color coding (green/amber/red)
    </tasks>
  </story>

  <acceptanceCriteria>
1. WebSocket connects via env URL or localhost:3002/ws; exponential backoff max 15s; errors throttled
2. On WS failure, polling fallback engages every 2s without console spam
3. A visual "Live" indicator reflects WS/polling status
  </acceptanceCriteria>

  <artifacts>
    <docs>
      <doc path="bmad/docs/tech-spec-epic-0.md" title="Epic 0 Technical Specification" section="WebSocket Reliability">
        Defines requirements for robust WebSocket price updates with retry and polling fallback. Emphasizes seamless failover without console spam and visual status indicators.
      </doc>
      <doc path="bmad/docs/architecture.md" title="QuantDesk Architecture Documentation" section="Frontend Services">
        Architecture outlines WebSocket service pattern for real-time data streaming, with fallback to REST polling when WebSocket unavailable.
      </doc>
    </docs>
    <code>
      <artifact path="frontend/src/services/websocketService.ts" kind="service" symbol="WebSocketService" lines="64-337" reason="Core WebSocket service implementation with connection management, retry logic, and subscription handling. Currently uses exponential backoff but max delay is configurable.">
        Current implementation:
        - Constructor accepts URL (default: 'ws://localhost:3002/') and optional flag
        - `connect()` method with Promise-based connection
        - Exponential backoff retry in `handleReconnect()` (line 184-209): `delay = reconnectInterval * Math.pow(2, reconnectAttempts - 1)`
        - Current `reconnectInterval = 1000` and `maxReconnectAttempts = 5`, so max delay is ~16s
        - Optional mode: resolves instead of rejecting on connection failure
        - Connection state tracking via `isConnected` and `connectionFailed` flags
        - Heartbeat mechanism (ping/pong) every 30 seconds
        - Subscription management with resubscribe on reconnect
      </artifact>
      <artifact path="frontend/src/contexts/PriceContext.tsx" kind="context" symbol="PriceProvider" lines="28-113" reason="Price context provider that uses WebSocketService and PriceStore. Handles WebSocket connection initialization and fallback to polling.">
        Current implementation:
        - Uses `WebSocketService` with optional flag (line 39)
        - Attempts WebSocket connection first (line 57)
        - Falls back to `priceStore.startFallbackPolling()` if WebSocket fails (line 95)
        - Current polling interval: 5000ms (5 seconds), should be 2000ms per AC2
        - Connection status state: 'connecting', 'connected', 'disconnected'
        - Subscribes to 'market_data_update', 'order_book_update', 'trade_update' events
      </artifact>
      <artifact path="frontend/src/stores/PriceStore.ts" kind="store" symbol="PriceStore" lines="31-125" reason="Price store that manages price data and fallback polling. Methods: `startFallbackPolling()`, `stopFallbackPolling()`, `updatePrice()`, `setConnectionStatus()`.">
        Price store singleton that:
        - Manages price data map
        - Handles subscription callbacks
        - Provides fallback polling functionality
        - Tracks connection status
        - Stores last update timestamp
      </artifact>
      <artifact path="frontend/src/pro/index.tsx" kind="component" symbol="QMWindowContent" lines="2389-2498" reason="Quote Monitor component that displays live prices. Currently polls /api/oracle/prices every 5s. Could benefit from WebSocket integration.">
        QM component:
        - Polls `/api/oracle/prices` every 5 seconds (line 2422)
        - Uses `usePrice()` hook for WebSocket prices
        - Uses `backendPrices` state from REST polling
        - Could display connection status indicator
      </artifact>
      <artifact path="backend/src/services/websocket.ts" kind="service" symbol="WebSocketService" lines="77-606" reason="Backend WebSocket service using Socket.IO. Broadcasts market data updates to connected clients.">
        Backend WebSocket service:
        - Uses Socket.IO for WebSocket communication
        - Broadcasts market data via `broadcastMarketData()` (line 347)
        - Emits 'market_data_update' events to connected clients
        - Handles subscriptions and channel management
      </artifact>
      <artifact path="backend/src/server.ts" kind="server" symbol="wss" lines="472-544" reason="Backend WebSocket server using native WebSocket. Handles connections on /ws endpoint with authentication.">
        Backend WebSocket server:
        - Native WebSocket server (ws library)
        - Listens on /ws endpoint
        - Requires authentication via JWT cookie
        - Supports channel-based subscriptions
        - Uses Redis Pub/Sub for message broadcasting
      </artifact>
    </code>
    <dependencies>
      <dependency>
        <ecosystem>Node.js</ecosystem>
        <packages>
          <package>socket.io: ^4.x</package>
          <package>ws: ^8.x</package>
        </packages>
      </dependency>
      <dependency>
        <ecosystem>React</ecosystem>
        <packages>
          <package>react: ^18.x</package>
          <package>socket.io-client: ^4.x</package>
        </packages>
      </dependency>
    </dependencies>
  </artifacts>

  <constraints>
    - WebSocket URL should come from env variable `VITE_WEBSOCKET_URL` or default to `localhost:3002/ws`
    - Exponential backoff max delay: 15 seconds (AC1)
    - Polling fallback interval: 2 seconds (AC2)
    - Error throttling: Prevent console spam during connection failures
    - Visual indicator must be visible in Pro Terminal UI
    - Connection state must be managed: connecting, connected, disconnected, error
    - Seamless failover: Switch back to WebSocket when available without disrupting user experience
    - No console spam during fallback polling mode
  </constraints>
  <interfaces>
    <interface name="WebSocketService.connect()" kind="function signature" signature="connect(): Promise&lt;void&gt;" path="frontend/src/services/websocketService.ts">
      Connects to WebSocket server. Returns Promise that resolves even if connection fails (optional mode).
    </interface>
    <interface name="WebSocketService.getConnectionStatus()" kind="function signature" signature="getConnectionStatus(): boolean" path="frontend/src/services/websocketService.ts">
      Returns true if WebSocket is currently connected.
    </interface>
    <interface name="WebSocketService.getConnectionInfo()" kind="function signature" signature="getConnectionInfo(): { connected: boolean; available: boolean; failed: boolean }" path="frontend/src/services/websocketService.ts">
      Returns detailed connection information including connection state and failure status.
    </interface>
    <interface name="WebSocketService.subscribe()" kind="function signature" signature="subscribe(channel: string, callback: (data: any) =>&gt; void): () =&gt; void" path="frontend/src/services/websocketService.ts">
      Subscribes to a WebSocket channel. Returns unsubscribe function.
    </interface>
    <interface name="PriceStore.startFallbackPolling()" kind="function signature" signature="startFallbackPolling(apiUrl: string, interval: number): void" path="frontend/src/stores/PriceStore.ts">
      Starts polling fallback to fetch prices via REST API. Interval in milliseconds.
    </interface>
    <interface name="PriceStore.stopFallbackPolling()" kind="function signature" signature="stopFallbackPolling(): void" path="frontend/src/stores/PriceStore.ts">
      Stops fallback polling.
    </interface>
    <interface name="GET /api/oracle/prices" kind="REST endpoint" signature="GET /api/oracle/prices" path="backend/src/routes/oracle.ts">
      Polling fallback endpoint. Returns { success: true, data: Record&lt;string, number&gt;, timestamp, source }.
    </interface>
    <interface name="ws://localhost:3002/ws" kind="WebSocket endpoint" signature="WebSocket connection endpoint" path="backend/src/server.ts">
      Backend WebSocket endpoint. Requires authentication via JWT cookie. Supports market data updates.
    </interface>
  </interfaces>
  <tests>
    <standards>
      Frontend tests use Vitest and React Testing Library. Test files located in `frontend/src/tests/` with subdirectories: `unit/`, `integration/`, `e2e/`. Mock WebSocket connections for unit tests. Use `@testing-library/react` for component testing.
    </standards>
    <locations>
      - `frontend/src/tests/unit/` - Unit tests for services and hooks
      - `frontend/src/tests/integration/` - Integration tests for components with WebSocket
      - `frontend/src/tests/e2e/` - End-to-end tests for Pro Terminal flows
    </locations>
    <ideas>
      <test ac="AC1" idea="Unit test: WebSocket connection with exponential backoff retry logic">
        Mock WebSocket to fail initially. Verify exponential backoff: 1s, 2s, 4s, 8s, 15s (max). Verify max delay capped at 15s. Test error throttling prevents console spam.
      </test>
      <test ac="AC1" idea="Unit test: WebSocket connects via env URL or localhost:3002/ws">
        Test WebSocketService constructor with env variable. Test default to localhost:3002/ws when env not set. Verify connection URL is correct.
      </test>
      <test ac="AC1" idea="Unit test: Connection state management">
        Test state transitions: disconnected → connecting → connected. Test error state handling. Verify state is correctly tracked.
      </test>
      <test ac="AC2" idea="Integration test: Polling fallback engages on WebSocket failure">
        Mock WebSocket to fail after max retries. Verify polling starts automatically. Verify polling interval is 2 seconds. Test no console spam during polling.
      </test>
      <test ac="AC2" idea="Integration test: Seamless switch back to WebSocket">
        Start with polling fallback. Simulate WebSocket becomes available. Verify seamless switch from polling to WebSocket. Verify no data loss during transition.
      </test>
      <test ac="AC3" idea="Component test: Visual 'Live' indicator shows correct status">
        Render Pro Terminal with WebSocket connected. Verify "Live" indicator displays with green color. Test with WebSocket disconnected: show "Polling" (amber) or "Disconnected" (red). Verify color coding matches status.
      </test>
      <test ac="AC3" idea="E2E test: User sees live indicator in Pro Terminal">
        Open Pro Terminal. Verify indicator shows correct status. Test indicator updates when WebSocket connection status changes. Verify indicator is visible and accessible.
      </test>
    </ideas>
  </tests>
</story-context>

