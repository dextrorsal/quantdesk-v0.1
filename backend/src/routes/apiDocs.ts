import express, { Request, Response } from 'express';
import { Logger } from '../utils/logger';

const router = express.Router();
const logger = new Logger();

/**
 * GET /api/docs/swagger
 * Generate OpenAPI/Swagger documentation for AI agents
 */
router.get('/swagger', (req: Request, res: Response) => {
  const swaggerSpec = {
    openapi: '3.0.0',
    info: {
      title: 'QuantDesk Perpetual DEX API',
      description: 'API for QuantDesk Perpetual DEX - Optimized for Development AI Assistance (Cursor AI, GitHub Copilot, Claude, Grok). Provides structured endpoints for AI assistants to understand the system architecture, market data, and user interactions.',
      version: '1.0.0',
      contact: {
        name: 'QuantDesk Team',
        email: 'support@quantdesk.com'
      }
    },
    servers: [
      {
        url: 'http://localhost:3002/api',
        description: 'Development server'
      }
    ],
    tags: [
      { name: 'Development AI', description: 'Development AI assistance endpoints' },
      { name: 'Markets', description: 'Market data and trading' },
      { name: 'Authentication', description: 'User authentication' },
      { name: 'Oracle', description: 'Price feeds and oracle data' },
      { name: 'Portfolio', description: 'User portfolio management' }
    ],
    paths: {
      '/dev/market-summary': {
        get: {
          tags: ['Development AI'],
          summary: 'Get aggregated market data for development AI assistance',
          description: 'Returns comprehensive market data to help AI assistants understand market structure. For AI Assistants: Use this endpoint to understand how markets are structured, what data is available, and how prices are formatted.',
          responses: {
            '200': {
              description: 'Market summary data',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean' },
                      data: {
                        type: 'object',
                        properties: {
                          markets: {
                            type: 'array',
                            items: {
                              type: 'object',
                              properties: {
                                id: { type: 'string' },
                                symbol: { type: 'string' },
                                baseAsset: { type: 'string' },
                                quoteAsset: { type: 'string' },
                                currentPrice: { type: 'number' },
                                priceChange24h: { type: 'number' },
                                volume24h: { type: 'number' },
                                openInterest: { type: 'number' },
                                maxLeverage: { type: 'number' },
                                isActive: { type: 'boolean' },
                                lastUpdated: { type: 'string', format: 'date-time' }
                              }
                            }
                          },
                          totalMarkets: { type: 'number' },
                          timestamp: { type: 'string', format: 'date-time' }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      },
      '/ai/user-portfolio/{wallet}': {
        get: {
          tags: ['AI Agent'],
          summary: 'Get complete portfolio view for AI agents',
          description: 'Returns comprehensive portfolio data for a specific wallet',
          parameters: [
            {
              name: 'wallet',
              in: 'path',
              required: true,
              schema: { type: 'string' },
              description: 'Wallet address'
            }
          ],
          responses: {
            '200': {
              description: 'Portfolio data',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean' },
                      data: {
                        type: 'object',
                        properties: {
                          user: {
                            type: 'object',
                            properties: {
                              id: { type: 'string' },
                              wallet_address: { type: 'string' },
                              created_at: { type: 'string', format: 'date-time' },
                              total_volume: { type: 'number' },
                              total_trades: { type: 'number' }
                            }
                          },
                          portfolio: {
                            type: 'object',
                            properties: {
                              totalPositions: { type: 'number' },
                              openPositions: { type: 'number' },
                              totalOrders: { type: 'number' },
                              totalTrades: { type: 'number' },
                              totalPnL: { type: 'number' },
                              positions: {
                                type: 'array',
                                items: {
                                  type: 'object',
                                  properties: {
                                    id: { type: 'string' },
                                    market: { type: 'string' },
                                    side: { type: 'string' },
                                    size: { type: 'number' },
                                    entryPrice: { type: 'number' },
                                    currentPrice: { type: 'number' },
                                    unrealizedPnl: { type: 'number' },
                                    leverage: { type: 'number' },
                                    status: { type: 'string' }
                                  }
                                }
                              }
                            }
                          },
                          timestamp: { type: 'string', format: 'date-time' }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      },
      '/ai/trading-signals': {
        get: {
          tags: ['AI Agent'],
          summary: 'Get market analysis and trading signals for AI agents',
          description: 'Returns trading signals based on market analysis',
          responses: {
            '200': {
              description: 'Trading signals',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean' },
                      data: {
                        type: 'object',
                        properties: {
                          signals: {
                            type: 'array',
                            items: {
                              type: 'object',
                              properties: {
                                market: { type: 'string' },
                                baseAsset: { type: 'string' },
                                currentPrice: { type: 'number' },
                                priceChange24h: { type: 'number' },
                                priceChangePercent: { type: 'number' },
                                volume24h: { type: 'number' },
                                signal: { type: 'string', enum: ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'] },
                                confidence: { type: 'number', minimum: 0, maximum: 1 },
                                maxLeverage: { type: 'number' },
                                timestamp: { type: 'string', format: 'date-time' }
                              }
                            }
                          },
                          totalSignals: { type: 'number' },
                          timestamp: { type: 'string', format: 'date-time' }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      },
      '/ai/liquidation-risk': {
        get: {
          tags: ['AI Agent'],
          summary: 'Get liquidation risk analysis for AI agents',
          description: 'Returns liquidation risk analysis for all open positions',
          responses: {
            '200': {
              description: 'Liquidation risk analysis',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean' },
                      data: {
                        type: 'object',
                        properties: {
                          risks: {
                            type: 'array',
                            items: {
                              type: 'object',
                              properties: {
                                positionId: { type: 'string' },
                                userId: { type: 'string' },
                                market: { type: 'string' },
                                side: { type: 'string' },
                                size: { type: 'number' },
                                leverage: { type: 'number' },
                                entryPrice: { type: 'number' },
                                currentPrice: { type: 'number' },
                                liquidationPrice: { type: 'number' },
                                distanceToLiquidation: { type: 'number' },
                                riskLevel: { type: 'string', enum: ['HIGH', 'MEDIUM', 'LOW'] },
                                unrealizedPnl: { type: 'number' },
                                timestamp: { type: 'string', format: 'date-time' }
                              }
                            }
                          },
                          totalPositions: { type: 'number' },
                          highRiskPositions: { type: 'number' },
                          mediumRiskPositions: { type: 'number' },
                          lowRiskPositions: { type: 'number' },
                          timestamp: { type: 'string', format: 'date-time' }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      },
      '/ai/funding-rates': {
        get: {
          tags: ['AI Agent'],
          summary: 'Get funding rate analysis for AI agents',
          description: 'Returns current funding rates for all markets',
          responses: {
            '200': {
              description: 'Funding rate analysis',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean' },
                      data: {
                        type: 'object',
                        properties: {
                          fundingRates: {
                            type: 'array',
                            items: {
                              type: 'object',
                              properties: {
                                marketId: { type: 'string' },
                                marketSymbol: { type: 'string' },
                                baseAsset: { type: 'string' },
                                fundingRate: { type: 'number' },
                                fundingRatePercent: { type: 'number' },
                                timestamp: { type: 'string', format: 'date-time' },
                                nextFundingTime: { type: 'string', format: 'date-time' }
                              }
                            }
                          },
                          totalMarkets: { type: 'number' },
                          averageFundingRate: { type: 'number' },
                          timestamp: { type: 'string', format: 'date-time' }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      },
      '/dev/codebase-structure': {
        get: {
          tags: ['Development AI'],
          summary: 'Get codebase structure information for development AI assistance',
          description: 'Returns comprehensive codebase structure to help AI assistants understand the QuantDesk architecture. For AI Assistants: Use this endpoint to understand the database schema, service architecture, and system relationships.',
          responses: {
            '200': {
              description: 'Codebase structure data',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean' },
                      data: {
                        type: 'object',
                        properties: {
                          database: {
                            type: 'object',
                            properties: {
                              tables: {
                                type: 'array',
                                items: {
                                  type: 'object',
                                  properties: {
                                    table_name: { type: 'string' },
                                    table_type: { type: 'string' }
                                  }
                                }
                              },
                              totalTables: { type: 'number' }
                            }
                          },
                          markets: {
                            type: 'object',
                            properties: {
                              total: { type: 'number' },
                              active: { type: 'number' },
                              structure: {
                                type: 'array',
                                items: {
                                  type: 'object',
                                  properties: {
                                    id: { type: 'string' },
                                    symbol: { type: 'string' },
                                    baseAsset: { type: 'string' },
                                    quoteAsset: { type: 'string' }
                                  }
                                }
                              }
                            }
                          },
                          users: {
                            type: 'object',
                            properties: {
                              total: { type: 'number' }
                            }
                          },
                          positions: {
                            type: 'object',
                            properties: {
                              total: { type: 'number' }
                            }
                          },
                          architecture: {
                            type: 'object',
                            properties: {
                              backend: {
                                type: 'object',
                                properties: {
                                  services: { type: 'array', items: { type: 'string' } },
                                  routes: { type: 'array', items: { type: 'string' } },
                                  middleware: { type: 'array', items: { type: 'string' } }
                                }
                              },
                              smartContract: {
                                type: 'object',
                                properties: {
                                  program: { type: 'string' },
                                  accounts: { type: 'array', items: { type: 'string' } },
                                  instructions: { type: 'array', items: { type: 'string' } }
                                }
                              }
                            }
                          },
                          timestamp: { type: 'string', format: 'date-time' }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      },
      '/markets': {
        get: {
          tags: ['Markets'],
          summary: 'Get all available markets',
          description: 'Returns list of all trading markets with live data',
          responses: {
            '200': {
              description: 'Market data',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean' },
                      markets: {
                        type: 'array',
                        items: {
                          type: 'object',
                          properties: {
                            id: { type: 'string' },
                            symbol: { type: 'string' },
                            baseAsset: { type: 'string' },
                            quoteAsset: { type: 'string' },
                            isActive: { type: 'boolean' },
                            maxLeverage: { type: 'number' },
                            price: { type: 'number' },
                            change24h: { type: 'number' },
                            volume24h: { type: 'number' },
                            openInterest: { type: 'number' },
                            fundingRate: { type: 'number' },
                            timestamp: { type: 'string', format: 'date-time' }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      },
      '/oracle/prices': {
        get: {
          tags: ['Oracle'],
          summary: 'Get current oracle prices',
          description: 'Returns current prices from Pyth oracle feeds',
          responses: {
            '200': {
              description: 'Oracle price data',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean' },
                      data: {
                        type: 'object',
                        additionalProperties: { type: 'number' }
                      },
                      timestamp: { type: 'number' },
                      source: { type: 'string' }
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
        BearerAuth: {
          type: 'http',
          scheme: 'bearer',
          bearerFormat: 'JWT'
        }
      },
      schemas: {
        Error: {
          type: 'object',
          properties: {
            success: { type: 'boolean', example: false },
            error: { type: 'string' },
            code: { type: 'string' },
            message: { type: 'string' },
            timestamp: { type: 'string', format: 'date-time' },
            request_id: { type: 'string' },
            path: { type: 'string' },
            method: { type: 'string' }
          }
        },
        Success: {
          type: 'object',
          properties: {
            success: { type: 'boolean', example: true },
            data: { type: 'object' },
            timestamp: { type: 'string', format: 'date-time' }
          }
        }
      }
    }
  };

  res.json(swaggerSpec);
});

/**
 * GET /api/docs
 * Serve interactive Swagger UI
 */
router.get('/', (req: Request, res: Response) => {
  const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>QuantDesk API Documentation</title>
      <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    </head>
    <body>
      <div id="swagger-ui"></div>
      <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
      <script>
        SwaggerUIBundle({
          url: '/api/docs/swagger',
          dom_id: '#swagger-ui',
          presets: [
            SwaggerUIBundle.presets.apis,
            SwaggerUIBundle.presets.standalone
          ],
          layout: "StandaloneLayout",
          deepLinking: true,
          showExtensions: true,
          showCommonExtensions: true,
          docExpansion: "list",
          filter: true,
          tryItOutEnabled: true
        });
      </script>
    </body>
    </html>
  `;
  
  res.send(html);
});

export default router;
