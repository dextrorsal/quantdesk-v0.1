import { Request, Response } from 'express';
import { Logger } from '../utils/logger';

const logger = new Logger();

// OpenAPI 3.0 specification generator
export function generateOpenAPISpec(): any {
  return {
    openapi: '3.0.3',
    info: {
      title: 'QuantDesk API',
      description: 'The Bloomberg Terminal for Crypto - Professional-grade decentralized perpetual trading platform',
      version: '1.0.0',
      contact: {
        name: 'QuantDesk Support',
        email: 'support@quantdesk.com',
        url: 'https://quantdesk.com'
      },
      license: {
        name: 'MIT',
        url: 'https://opensource.org/licenses/MIT'
      }
    },
    servers: [
      {
        url: 'https://api.quantdesk.com',
        description: 'Production server'
      },
      {
        url: 'https://api-dev.quantdesk.com',
        description: 'Development server'
      },
      {
        url: 'http://localhost:3002',
        description: 'Local development server'
      }
    ],
    security: [
      {
        bearerAuth: []
      },
      {
        apiKey: []
      }
    ],
    paths: {
      '/api/auth/login': {
        post: {
          tags: ['Authentication'],
          summary: 'User login',
          description: 'Authenticate user and return JWT token',
          requestBody: {
            required: true,
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  required: ['wallet_address', 'signature'],
                  properties: {
                    wallet_address: {
                      type: 'string',
                      description: 'Solana wallet address',
                      example: '9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM'
                    },
                    signature: {
                      type: 'string',
                      description: 'Wallet signature',
                      example: '5J7X8K9L2M3N4O5P6Q7R8S9T0U1V2W3X4Y5Z6A7B8C9D0E1F2G3H4I5J6K7L8M9N0O1P2Q3R4S5T6U7V8W9X0Y1Z2A3B4C5D6E7F8G9H0I1J2K3L4M5N6O7P8Q9R0S1T2U3V4W5X6Y7Z8A9B0C1D2E3F4G5H6I7J8K9L0M1N2O3P4Q5R6S7T8U9V0W1X2Y3Z4A5B6C7D8E9F0G1H2I3J4K5L6M7N8O9P0Q1R2S3T4U5V6W7X8Y9Z0'
                    }
                  }
                }
              }
            }
          },
          responses: {
            '200': {
              description: 'Login successful',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean', example: true },
                      data: {
                        type: 'object',
                        properties: {
                          token: { type: 'string' },
                          user: {
                            type: 'object',
                            properties: {
                              id: { type: 'string' },
                              wallet_address: { type: 'string' },
                              tier: { type: 'string', enum: ['free', 'premium', 'professional', 'enterprise'] }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            },
            '401': {
              description: 'Authentication failed',
              content: {
                'application/json': {
                  schema: { $ref: '#/components/schemas/ErrorResponse' }
                }
              }
            }
          }
        }
      },
      '/api/markets': {
        get: {
          tags: ['Markets'],
          summary: 'Get all markets',
          description: 'Retrieve list of all available perpetual markets',
          responses: {
            '200': {
              description: 'Markets retrieved successfully',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean', example: true },
                      data: {
                        type: 'array',
                        items: { $ref: '#/components/schemas/Market' }
                      },
                      count: { type: 'number', example: 11 }
                    }
                  }
                }
              }
            }
          }
        }
      },
      '/api/orders': {
        post: {
          tags: ['Orders'],
          summary: 'Place new order',
          description: 'Place a new trading order',
          security: [{ bearerAuth: [] }],
          requestBody: {
            required: true,
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/OrderRequest' }
              }
            }
          },
          responses: {
            '201': {
              description: 'Order placed successfully',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean', example: true },
                      data: { $ref: '#/components/schemas/Order' }
                    }
                  }
                }
              }
            },
            '400': {
              description: 'Invalid order request',
              content: {
                'application/json': {
                  schema: { $ref: '#/components/schemas/ErrorResponse' }
                }
              }
            },
            '401': {
              description: 'Authentication required',
              content: {
                'application/json': {
                  schema: { $ref: '#/components/schemas/ErrorResponse' }
                }
              }
            }
          }
        }
      },
      '/api/advanced-orders': {
        post: {
          tags: ['Advanced Orders'],
          summary: 'Place advanced order',
          description: 'Place advanced order types (stop-loss, take-profit, trailing stops, etc.)',
          security: [{ bearerAuth: [] }],
          requestBody: {
            required: true,
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/AdvancedOrderRequest' }
              }
            }
          },
          responses: {
            '201': {
              description: 'Advanced order placed successfully',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean', example: true },
                      data: { $ref: '#/components/schemas/AdvancedOrder' }
                    }
                  }
                }
              }
            }
          }
        }
      },
      '/api/cross-collateral': {
        post: {
          tags: ['Cross Collateral'],
          summary: 'Initialize collateral account',
          description: 'Initialize a new collateral account for cross-collateralization',
          security: [{ bearerAuth: [] }],
          requestBody: {
            required: true,
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  required: ['user_id', 'asset_type', 'amount'],
                  properties: {
                    user_id: { type: 'string' },
                    asset_type: { 
                      type: 'string', 
                      enum: ['SOL', 'USDC', 'BTC', 'ETH', 'USDT', 'AVAX', 'MATIC', 'ARB', 'OP', 'DOGE', 'ADA', 'DOT', 'LINK'] 
                    },
                    amount: { type: 'number', minimum: 0 }
                  }
                }
              }
            }
          },
          responses: {
            '201': {
              description: 'Collateral account initialized successfully',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean', example: true },
                      data: { $ref: '#/components/schemas/CollateralAccount' }
                    }
                  }
                }
              }
            }
          }
        }
      },
      '/api/webhooks/subscriptions': {
        post: {
          tags: ['Webhooks'],
          summary: 'Create webhook subscription',
          description: 'Create a new webhook subscription for real-time notifications',
          security: [{ bearerAuth: [] }],
          requestBody: {
            required: true,
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  required: ['url', 'events'],
                  properties: {
                    url: { type: 'string', format: 'uri' },
                    events: {
                      type: 'array',
                      items: {
                        type: 'string',
                        enum: [
                          'order.placed', 'order.filled', 'order.cancelled',
                          'position.opened', 'position.closed', 'position.liquidated',
                          'collateral.added', 'collateral.removed',
                          'account.created', 'account.updated',
                          'market.updated', 'price.alert', 'system.maintenance'
                        ]
                      }
                    },
                    secret: { type: 'string', description: 'Optional webhook secret for signature verification' }
                  }
                }
              }
            }
          },
          responses: {
            '201': {
              description: 'Webhook subscription created successfully',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean', example: true },
                      data: { $ref: '#/components/schemas/WebhookSubscription' }
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    components: {
      securitySchemes: {
        bearerAuth: {
          type: 'http',
          scheme: 'bearer',
          bearerFormat: 'JWT'
        },
        apiKey: {
          type: 'apiKey',
          in: 'header',
          name: 'X-API-Key'
        }
      },
      schemas: {
        Market: {
          type: 'object',
          properties: {
            id: { type: 'string' },
            symbol: { type: 'string', example: 'BTC-PERP' },
            base_asset: { type: 'string', example: 'BTC' },
            quote_asset: { type: 'string', example: 'USDT' },
            program_id: { type: 'string' },
            is_active: { type: 'boolean' },
            max_leverage: { type: 'number', example: 100 },
            tick_size: { type: 'number', example: 0.01 },
            step_size: { type: 'number', example: 0.001 },
            min_order_size: { type: 'number', example: 0.001 },
            max_order_size: { type: 'number', example: 1000000 }
          }
        },
        OrderRequest: {
          type: 'object',
          required: ['market_id', 'side', 'size', 'leverage'],
          properties: {
            market_id: { type: 'string' },
            side: { type: 'string', enum: ['long', 'short'] },
            size: { type: 'number', minimum: 0 },
            price: { type: 'number', minimum: 0 },
            leverage: { type: 'number', minimum: 1, maximum: 100 }
          }
        },
        AdvancedOrderRequest: {
          type: 'object',
          required: ['user_id', 'market_id', 'order_type', 'side', 'size', 'leverage'],
          properties: {
            user_id: { type: 'string' },
            market_id: { type: 'string' },
            order_type: { 
              type: 'string', 
              enum: ['stop_loss', 'take_profit', 'trailing_stop', 'iceberg', 'twap', 'bracket', 'stop_limit'] 
            },
            side: { type: 'string', enum: ['long', 'short'] },
            size: { type: 'number', minimum: 0 },
            price: { type: 'number', minimum: 0 },
            stop_price: { type: 'number', minimum: 0 },
            trailing_distance: { type: 'number', minimum: 0 },
            leverage: { type: 'number', minimum: 1, maximum: 100 },
            time_in_force: { type: 'string', enum: ['GTC', 'IOC', 'FOK', 'GTD'] }
          }
        },
        Order: {
          type: 'object',
          properties: {
            id: { type: 'string' },
            user_id: { type: 'string' },
            market_id: { type: 'string' },
            order_type: { type: 'string' },
            side: { type: 'string' },
            size: { type: 'number' },
            price: { type: 'number' },
            leverage: { type: 'number' },
            status: { type: 'string' },
            created_at: { type: 'number' },
            filled_size: { type: 'number' }
          }
        },
        AdvancedOrder: {
          allOf: [
            { $ref: '#/components/schemas/Order' },
            {
              type: 'object',
              properties: {
                stop_price: { type: 'number' },
                trailing_distance: { type: 'number' },
                hidden_size: { type: 'number' },
                display_size: { type: 'number' },
                time_in_force: { type: 'string' },
                target_price: { type: 'number' },
                parent_order: { type: 'string' },
                twap_duration: { type: 'number' },
                twap_interval: { type: 'number' }
              }
            }
          ]
        },
        CollateralAccount: {
          type: 'object',
          properties: {
            id: { type: 'string' },
            user_id: { type: 'string' },
            asset_type: { type: 'string' },
            amount: { type: 'number' },
            value_usd: { type: 'number' },
            last_updated: { type: 'number' },
            is_active: { type: 'boolean' }
          }
        },
        WebhookSubscription: {
          type: 'object',
          properties: {
            id: { type: 'string' },
            user_id: { type: 'string' },
            url: { type: 'string' },
            events: { type: 'array', items: { type: 'string' } },
            secret: { type: 'string' },
            is_active: { type: 'boolean' },
            created_at: { type: 'number' },
            last_delivery_at: { type: 'number' },
            failure_count: { type: 'number' },
            max_failures: { type: 'number' }
          }
        },
        ErrorResponse: {
          type: 'object',
          properties: {
            success: { type: 'boolean', example: false },
            error: { type: 'string' },
            code: { type: 'string' },
            message: { type: 'string' },
            details: { type: 'object' },
            timestamp: { type: 'number' },
            request_id: { type: 'string' },
            path: { type: 'string' },
            method: { type: 'string' }
          }
        }
      }
    },
    tags: [
      {
        name: 'Authentication',
        description: 'User authentication and authorization'
      },
      {
        name: 'Markets',
        description: 'Market data and information'
      },
      {
        name: 'Orders',
        description: 'Basic order management'
      },
      {
        name: 'Advanced Orders',
        description: 'Advanced order types (stop-loss, take-profit, etc.)'
      },
      {
        name: 'Cross Collateral',
        description: 'Cross-collateralization features'
      },
      {
        name: 'Webhooks',
        description: 'Real-time webhook notifications'
      },
      {
        name: 'Analytics',
        description: 'Trading analytics and metrics'
      },
      {
        name: 'Admin',
        description: 'Administrative functions'
      }
    ]
  };
}

// API documentation routes
export function createAPIDocRoutes() {
  const express = require('express');
  const router = express.Router();
  
  // Serve OpenAPI specification
  router.get('/openapi.json', (req: Request, res: Response) => {
    try {
      const spec = generateOpenAPISpec();
      res.json(spec);
    } catch (error) {
      logger.error('Error generating OpenAPI spec:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to generate API documentation'
      });
    }
  });
  
  // Serve Swagger UI
  router.get('/docs', (req: Request, res: Response) => {
    const swaggerHtml = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QuantDesk API Documentation</title>
  <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css" />
  <style>
    html {
      box-sizing: border-box;
      overflow: -moz-scrollbars-vertical;
      overflow-y: scroll;
    }
    *, *:before, *:after {
      box-sizing: inherit;
    }
    body {
      margin:0;
      background: #fafafa;
    }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
  <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-standalone-preset.js"></script>
  <script>
    window.onload = function() {
      const ui = SwaggerUIBundle({
        url: '/api/docs/openapi.json',
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIStandalonePreset
        ],
        plugins: [
          SwaggerUIBundle.plugins.DownloadUrl
        ],
        layout: "StandaloneLayout",
        tryItOutEnabled: true,
        requestInterceptor: function(request) {
          // Add authentication header if available
          const token = localStorage.getItem('quantdesk_token');
          if (token) {
            request.headers.Authorization = 'Bearer ' + token;
          }
          return request;
        }
      });
      
      window.ui = ui;
    };
  </script>
</body>
</html>`;
    
    res.send(swaggerHtml);
  });
  
  // API status endpoint
  router.get('/status', (req: Request, res: Response) => {
    res.json({
      success: true,
      data: {
        name: 'QuantDesk API',
        version: '1.0.0',
        status: 'operational',
        timestamp: Date.now(),
        uptime: process.uptime(),
        environment: process.env.NODE_ENV || 'development',
        features: {
          advanced_orders: true,
          cross_collateral: true,
          webhooks: true,
          rate_limiting: true,
          api_documentation: true
        }
      }
    });
  });
  
  return router;
}

export default {
  generateOpenAPISpec,
  createAPIDocRoutes
};
