# REST API Spec

```yaml
openapi: 3.0.0
info:
  title: QuantDesk Trading API
  version: 1.0.0
  description: Enterprise-grade perpetual trading platform API with AI-powered assistance
servers:
  - url: https://api.quantdesk.com
    description: Production server
  - url: https://dev-api.quantdesk.com
    description: Development server

paths:
  /api/auth/login:
    post:
      summary: User authentication
      tags: [Authentication]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                wallet_address:
                  type: string
                  description: Solana wallet address
                signature:
                  type: string
                  description: Wallet signature for verification
      responses:
        '200':
          description: Authentication successful
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuthResponse'
        '401':
          description: Authentication failed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /api/trading/orders:
    post:
      summary: Place trading order
      tags: [Trading]
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TradingOrder'
      responses:
        '201':
          description: Order placed successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/OrderResponse'
        '400':
          description: Invalid order parameters
        '401':
          description: Unauthorized
        '429':
          description: Rate limit exceeded

    get:
      summary: Get user orders
      tags: [Trading]
      security:
        - bearerAuth: []
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [pending, filled, cancelled, rejected]
        - name: limit
          in: query
          schema:
            type: integer
            default: 50
      responses:
        '200':
          description: List of orders
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/TradingOrder'

  /api/trading/positions:
    get:
      summary: Get user positions
      tags: [Trading]
      security:
        - bearerAuth: []
      responses:
        '200':
          description: List of active positions
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/TradingPosition'

  /api/portfolio/balance:
    get:
      summary: Get portfolio balance
      tags: [Portfolio]
      security:
        - bearerAuth: []
      responses:
        '200':
          description: Portfolio balance information
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PortfolioBalance'

  /api/market/prices:
    get:
      summary: Get current market prices
      tags: [Market Data]
      parameters:
        - name: symbols
          in: query
          schema:
            type: array
            items:
              type: string
          description: Trading pair symbols
      responses:
        '200':
          description: Current market prices
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/MarketPrice'

  /api/ai/analyze:
    post:
      summary: Request AI trading analysis
      tags: [AI Assistant]
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                symbol:
                  type: string
                  description: Trading pair to analyze
                analysis_type:
                  type: string
                  enum: [market_analysis, trading_signal, risk_assessment]
      responses:
        '200':
          description: AI analysis result
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AIAnalysis'

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  schemas:
    AuthResponse:
      type: object
      properties:
        token:
          type: string
          description: JWT authentication token
        user:
          $ref: '#/components/schemas/User'
        expires_at:
          type: string
          format: date-time

    User:
      type: object
      properties:
        id:
          type: string
          format: uuid
        wallet_address:
          type: string
        email:
          type: string
        created_at:
          type: string
          format: date-time
        kyc_status:
          type: string
          enum: [pending, verified, rejected]

    TradingOrder:
      type: object
      required: [symbol, side, quantity, order_type]
      properties:
        symbol:
          type: string
          description: Trading pair (e.g., SOL-PERP)
        side:
          type: string
          enum: [buy, sell]
        quantity:
          type: number
          format: decimal
          description: Order quantity
        order_type:
          type: string
          enum: [market, limit, stop, stop_limit]
        price:
          type: number
          format: decimal
          description: Limit price (for limit orders)
        stop_price:
          type: number
          format: decimal
          description: Stop price (for stop orders)

    OrderResponse:
      type: object
      properties:
        order_id:
          type: string
          format: uuid
        status:
          type: string
          enum: [pending, filled, cancelled, rejected]
        filled_quantity:
          type: number
          format: decimal
        average_fill_price:
          type: number
          format: decimal
        created_at:
          type: string
          format: date-time

    TradingPosition:
      type: object
      properties:
        id:
          type: string
          format: uuid
        symbol:
          type: string
        side:
          type: string
          enum: [long, short]
        size:
          type: number
          format: decimal
        entry_price:
          type: number
          format: decimal
        current_price:
          type: number
          format: decimal
        unrealized_pnl:
          type: number
          format: decimal
        margin_used:
          type: number
          format: decimal
        leverage:
          type: number
          format: decimal

    PortfolioBalance:
      type: object
      properties:
        total_balance:
          type: number
          format: decimal
        available_balance:
          type: number
          format: decimal
        margin_used:
          type: number
          format: decimal
        unrealized_pnl:
          type: number
          format: decimal
        realized_pnl:
          type: number
          format: decimal

    MarketPrice:
      type: object
      properties:
        symbol:
          type: string
        price:
          type: number
          format: decimal
        timestamp:
          type: string
          format: date-time
        confidence:
          type: number
          format: decimal
        source:
          type: string

    AIAnalysis:
      type: object
      properties:
        analysis_id:
          type: string
          format: uuid
        symbol:
          type: string
        analysis_type:
          type: string
        confidence_score:
          type: number
          format: decimal
        recommendation:
          type: string
          enum: [buy, sell, hold, close]
        reasoning:
          type: string
        created_at:
          type: string
          format: date-time
        expires_at:
          type: string
          format: date-time

    ErrorResponse:
      type: object
      properties:
        error:
          type: string
        message:
          type: string
        code:
          type: string
        timestamp:
          type: string
          format: date-time
```
