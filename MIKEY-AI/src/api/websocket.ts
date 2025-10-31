import express from 'express';
import { createServer, Server as HttpServer } from 'http';
import { WebSocket, WebSocketServer } from 'ws';
import { SecurityUtils } from '../utils/security';
import { systemLogger, securityLogger } from '../utils/logger';
import { config } from '../config';
import { officialLLMRouter } from '../services/OfficialLLMRouter';
import { WebSocketMessage } from '../types';

/**
 * MIKEY AI WebSocket Server
 * Handles real-time connections for AI responses, market updates, and trading insights
 */
export class MIKEYWebSocketServer {
  private wss: WebSocketServer;
  private clients: Map<string, WebSocket> = new Map();
  private heartbeatInterval: NodeJS.Timeout | null = null;

  constructor(app: express.Application, server?: HttpServer) {
    // Use provided HTTP server if available; otherwise create and listen
    const httpServer: HttpServer = server ?? createServer(app);

    // Create WebSocket server sharing the same HTTP server
    this.wss = new WebSocketServer({
      server: httpServer,
      path: '/ws/mikey'
    });

    // Only start listening if we created the server here
    if (!server) {
      httpServer.listen(config.api.port, () => {
        console.log(`🌐 MIKEY WebSocket server ready on ws://localhost:${config.api.port}/ws/mikey`);
      });
    } else {
      console.log(`🌐 MIKEY WebSocket server attached at ws://localhost:${config.api.port}/ws/mikey`);
    }

    this.setupWebSocket();
  }

  /**
   * Setup WebSocket event handlers
   */
  private setupWebSocket(): void {
    this.wss.on('connection', (ws: WebSocket, req) => {
      const clientId = this.generateClientId();
      this.clients.set(clientId, ws);
      
      console.log(`✅ WebSocket client connected: ${clientId}`);

      // Send welcome message
      ws.send(JSON.stringify({
        type: 'connection',
        data: {
          clientId,
          message: 'Connected to MIKEY AI',
          timestamp: new Date().toISOString()
        }
      }));

      // Handle incoming messages
      ws.on('message', (message: Buffer) => {
        this.handleMessage(clientId, message);
      });

      // Handle disconnection
      ws.on('close', () => {
        this.clients.delete(clientId);
        console.log(`❌ WebSocket client disconnected: ${clientId}`);
      });

      // Handle errors
      ws.on('error', (error) => {
        console.error(`❌ WebSocket error for ${clientId}:`, error);
        securityLogger.suspiciousActivity('websocket_error', {
          clientId,
          error: error.message
        });
        this.clients.delete(clientId);
      });
    });

    // Setup heartbeat
    this.heartbeatInterval = setInterval(() => {
      this.clients.forEach((ws, clientId) => {
        if (ws.readyState === WebSocket.OPEN) {
          try {
            ws.ping();
          } catch (error) {
            console.error(`Heartbeat failed for ${clientId}:`, error);
            this.clients.delete(clientId);
          }
        }
      });
    }, 30000); // 30 seconds

    systemLogger.startup('WebSocket server', 'ready');
  }

  /**
   * Generate unique client ID
   */
  private generateClientId(): string {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Handle incoming WebSocket messages
   */
  private async handleMessage(clientId: string, message: Buffer): Promise<void> {
    try {
      const ws = this.clients.get(clientId);
      if (!ws) return;

      const data = JSON.parse(message.toString());
      
      // Validate message
      if (!data.type || !data.payload) {
        ws.send(JSON.stringify({
          type: 'error',
          data: { message: 'Invalid message format' }
        }));
        return;
      }

      console.log(`📨 Message from ${clientId}:`, data.type);

      // Handle different message types
      switch (data.type) {
        case 'ai_query':
          await this.handleAIQuery(clientId, data.payload);
          break;
        
        case 'subscribe':
          await this.handleSubscribe(clientId, data.payload);
          break;
        
        case 'unsubscribe':
          await this.handleUnsubscribe(clientId, data.payload);
          break;
        
        case 'ping':
          ws.send(JSON.stringify({
            type: 'pong',
            data: { timestamp: new Date().toISOString() }
          }));
          break;
        
        default:
          ws.send(JSON.stringify({
            type: 'error',
            data: { message: 'Unknown message type' }
          }));
      }
    } catch (error) {
      console.error(`❌ Error handling message from ${clientId}:`, error);
      const ws = this.clients.get(clientId);
      if (ws) {
        ws.send(JSON.stringify({
          type: 'error',
          data: { message: 'Failed to process message' }
        }));
      }
    }
  }

  /**
   * Handle AI query requests
   */
  private async handleAIQuery(clientId: string, payload: any): Promise<void> {
    const ws = this.clients.get(clientId);
    if (!ws) return;

    try {
      const { query, context, sessionId } = payload;

      if (!query) {
        ws.send(JSON.stringify({
          type: 'ai_response',
          data: {
            error: 'Query is required',
            timestamp: new Date().toISOString()
          }
        }));
        return;
      }

      console.log(`🤖 Processing AI query from ${clientId}`);

      // Send status update
      ws.send(JSON.stringify({
        type: 'ai_status',
        data: {
          status: 'processing',
          message: 'AI is thinking...',
          timestamp: new Date().toISOString()
        }
      }));

      // Route request through official LLM router
      const startTime = Date.now();
      const result = await officialLLMRouter.routeRequest(query, 'general', sessionId);
      const duration = Date.now() - startTime;

      // Send response
      ws.send(JSON.stringify({
        type: 'ai_response',
        data: {
          response: result.response,
          provider: result.provider,
          duration,
          timestamp: new Date().toISOString(),
          metadata: {
            sessionId,
            clientId
          }
        }
      }));

      console.log(`✅ AI response sent to ${clientId} via ${result.provider}`);
    } catch (error) {
      console.error(`❌ Error processing AI query for ${clientId}:`, error);
      ws.send(JSON.stringify({
        type: 'ai_response',
        data: {
          error: 'Failed to process AI query',
          message: (error as Error).message,
          timestamp: new Date().toISOString()
        }
      }));
    }
  }

  /**
   * Handle subscription requests
   */
  private async handleSubscribe(clientId: string, payload: any): Promise<void> {
    const ws = this.clients.get(clientId);
    if (!ws) return;

    const { channels } = payload;
    
    console.log(`📡 ${clientId} subscribed to channels:`, channels);

    ws.send(JSON.stringify({
      type: 'subscription',
      data: {
        channels,
        status: 'subscribed',
        timestamp: new Date().toISOString()
      }
    }));

    // Here you would add the client to relevant channels
    // For now, just acknowledge the subscription
  }

  /**
   * Handle unsubscription requests
   */
  private async handleUnsubscribe(clientId: string, payload: any): Promise<void> {
    const ws = this.clients.get(clientId);
    if (!ws) return;

    const { channels } = payload;
    
    console.log(`📴 ${clientId} unsubscribed from channels:`, channels);

    ws.send(JSON.stringify({
      type: 'subscription',
      data: {
        channels,
        status: 'unsubscribed',
        timestamp: new Date().toISOString()
      }
    }));
  }

  /**
   * Broadcast message to all connected clients
   */
  public broadcast(message: WebSocketMessage): void {
    const messageStr = JSON.stringify(message);
    
    this.clients.forEach((ws, clientId) => {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(messageStr);
        } catch (error) {
          console.error(`Failed to send to ${clientId}:`, error);
          this.clients.delete(clientId);
        }
      }
    });
  }

  /**
   * Send message to specific client
   */
  public sendToClient(clientId: string, message: WebSocketMessage): void {
    const ws = this.clients.get(clientId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  /**
   * Get connected clients count
   */
  public getClientCount(): number {
    return this.clients.size;
  }

  /**
   * Graceful shutdown
   */
  public shutdown(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    this.clients.forEach((ws) => {
      ws.close();
    });

    this.wss.close();
    console.log('✅ WebSocket server shutdown complete');
  }
}

// Export singleton instance
let wsServer: MIKEYWebSocketServer | null = null;

export function createWebSocketServer(app: express.Application, server?: HttpServer): MIKEYWebSocketServer {
  if (!wsServer) {
    wsServer = new MIKEYWebSocketServer(app, server);
  }
  return wsServer;
}

export function getWebSocketServer(): MIKEYWebSocketServer | null {
  return wsServer;
}

