import { Request, Response } from 'express';
import axios from 'axios';
import crypto from 'crypto';
import { Logger } from '../utils/logger';

const logger = new Logger();

// Webhook event types
export enum WebhookEventType {
  ORDER_PLACED = 'order.placed',
  ORDER_FILLED = 'order.filled',
  ORDER_CANCELLED = 'order.cancelled',
  POSITION_OPENED = 'position.opened',
  POSITION_CLOSED = 'position.closed',
  POSITION_LIQUIDATED = 'position.liquidated',
  COLLATERAL_ADDED = 'collateral.added',
  COLLATERAL_REMOVED = 'collateral.removed',
  ACCOUNT_CREATED = 'account.created',
  ACCOUNT_UPDATED = 'account.updated',
  MARKET_UPDATED = 'market.updated',
  PRICE_ALERT = 'price.alert',
  SYSTEM_MAINTENANCE = 'system.maintenance'
}

// Webhook payload interface
export interface WebhookPayload {
  id: string;
  event: WebhookEventType;
  timestamp: number;
  data: any;
  user_id?: string;
  market_id?: string;
  order_id?: string;
  position_id?: string;
}

// Webhook subscription interface
export interface WebhookSubscription {
  id: string;
  user_id: string;
  url: string;
  events: WebhookEventType[];
  secret: string;
  is_active: boolean;
  created_at: number;
  last_delivery_at?: number;
  failure_count: number;
  max_failures: number;
}

// Webhook delivery result
export interface WebhookDeliveryResult {
  success: boolean;
  status_code?: number;
  response_time?: number;
  error?: string;
  retry_after?: number;
}

class WebhookService {
  private static instance: WebhookService;
  private subscriptions: Map<string, WebhookSubscription> = new Map();
  private deliveryQueue: WebhookPayload[] = [];
  private isProcessing = false;
  
  // Webhook delivery configuration
  private readonly DELIVERY_CONFIG = {
    maxRetries: 3,
    retryDelay: 1000, // 1 second
    timeout: 10000,   // 10 seconds
    batchSize: 10,
    processingInterval: 1000 // 1 second
  };
  
  private constructor() {
    this.startDeliveryProcessor();
  }
  
  public static getInstance(): WebhookService {
    if (!WebhookService.instance) {
      WebhookService.instance = new WebhookService();
    }
    return WebhookService.instance;
  }
  
  // Create a new webhook subscription
  public createSubscription(
    userId: string,
    url: string,
    events: WebhookEventType[],
    secret?: string
  ): WebhookSubscription {
    const id = crypto.randomUUID();
    const subscription: WebhookSubscription = {
      id,
      user_id: userId,
      url,
      events,
      secret: secret || crypto.randomBytes(32).toString('hex'),
      is_active: true,
      created_at: Date.now(),
      failure_count: 0,
      max_failures: 5
    };
    
    this.subscriptions.set(id, subscription);
    logger.info(`Webhook subscription created: ${id} for user ${userId}`);
    
    return subscription;
  }
  
  // Update webhook subscription
  public updateSubscription(
    id: string,
    updates: Partial<WebhookSubscription>
  ): WebhookSubscription | null {
    const subscription = this.subscriptions.get(id);
    if (!subscription) {
      return null;
    }
    
    Object.assign(subscription, updates);
    this.subscriptions.set(id, subscription);
    
    logger.info(`Webhook subscription updated: ${id}`);
    return subscription;
  }
  
  // Delete webhook subscription
  public deleteSubscription(id: string): boolean {
    const deleted = this.subscriptions.delete(id);
    if (deleted) {
      logger.info(`Webhook subscription deleted: ${id}`);
    }
    return deleted;
  }
  
  // Get user's webhook subscriptions
  public getUserSubscriptions(userId: string): WebhookSubscription[] {
    return Array.from(this.subscriptions.values())
      .filter(sub => sub.user_id === userId && sub.is_active);
  }
  
  // Trigger webhook event
  public async triggerEvent(
    event: WebhookEventType,
    data: any,
    userId?: string,
    metadata?: { market_id?: string; order_id?: string; position_id?: string }
  ): Promise<void> {
    const payload: WebhookPayload = {
      id: crypto.randomUUID(),
      event,
      timestamp: Date.now(),
      data,
      user_id: userId,
      ...metadata
    };
    
    // Find relevant subscriptions
    const relevantSubscriptions = Array.from(this.subscriptions.values())
      .filter(sub => 
        sub.is_active && 
        sub.events.includes(event) &&
        (!userId || sub.user_id === userId)
      );
    
    if (relevantSubscriptions.length === 0) {
      logger.debug(`No webhook subscriptions found for event: ${event}`);
      return;
    }
    
    // Queue webhook deliveries
    for (const subscription of relevantSubscriptions) {
      this.deliveryQueue.push({
        ...payload,
        subscription_id: subscription.id
      } as any);
    }
    
    logger.info(`Webhook event queued: ${event} for ${relevantSubscriptions.length} subscriptions`);
  }
  
  // Process webhook delivery queue
  private async startDeliveryProcessor(): Promise<void> {
    if (this.isProcessing) return;
    
    this.isProcessing = true;
    
    const processQueue = async () => {
      if (this.deliveryQueue.length === 0) {
        setTimeout(processQueue, this.DELIVERY_CONFIG.processingInterval);
        return;
      }
      
      const batch = this.deliveryQueue.splice(0, this.DELIVERY_CONFIG.batchSize);
      
      await Promise.allSettled(
        batch.map(delivery => this.deliverWebhook(delivery))
      );
      
      setTimeout(processQueue, this.DELIVERY_CONFIG.processingInterval);
    };
    
    processQueue();
  }
  
  // Deliver individual webhook
  private async deliverWebhook(payload: WebhookPayload & { subscription_id: string }): Promise<WebhookDeliveryResult> {
    const subscription = this.subscriptions.get(payload.subscription_id);
    if (!subscription) {
      return { success: false, error: 'Subscription not found' };
    }
    
    const startTime = Date.now();
    
    try {
      // Create signature
      const signature = this.createSignature(payload, subscription.secret);
      
      // Send webhook
      const response = await axios.post(subscription.url, payload, {
        headers: {
          'Content-Type': 'application/json',
          'X-Webhook-Signature': signature,
          'X-Webhook-Event': payload.event,
          'X-Webhook-Timestamp': payload.timestamp.toString(),
          'User-Agent': 'QuantDesk-Webhook/1.0'
        },
        timeout: this.DELIVERY_CONFIG.timeout,
        validateStatus: (status) => status < 500 // Don't throw on 4xx errors
      });
      
      const responseTime = Date.now() - startTime;
      
      if (response.status >= 200 && response.status < 300) {
        // Success
        subscription.last_delivery_at = Date.now();
        subscription.failure_count = 0;
        
        logger.info(`Webhook delivered successfully: ${subscription.id} (${response.status})`);
        return { success: true, status_code: response.status, response_time: responseTime };
      } else {
        // Client error (4xx) - don't retry
        logger.warn(`Webhook delivery failed (client error): ${subscription.id} (${response.status})`);
        return { success: false, status_code: response.status, error: 'Client error' };
      }
      
    } catch (error: any) {
      const responseTime = Date.now() - startTime;
      subscription.failure_count++;
      
      logger.error(`Webhook delivery failed: ${subscription.id}`, error);
      
      // Disable subscription if too many failures
      if (subscription.failure_count >= subscription.max_failures) {
        subscription.is_active = false;
        logger.error(`Webhook subscription disabled due to failures: ${subscription.id}`);
      }
      
      return { 
        success: false, 
        error: error.message, 
        response_time: responseTime,
        retry_after: this.calculateRetryDelay(subscription.failure_count)
      };
    }
  }
  
  // Create webhook signature
  private createSignature(payload: WebhookPayload, secret: string): string {
    const payloadString = JSON.stringify(payload);
    return crypto
      .createHmac('sha256', secret)
      .update(payloadString)
      .digest('hex');
  }
  
  // Calculate retry delay with exponential backoff
  private calculateRetryDelay(failureCount: number): number {
    return Math.min(
      this.DELIVERY_CONFIG.retryDelay * Math.pow(2, failureCount - 1),
      30000 // Max 30 seconds
    );
  }
  
  // Verify webhook signature (for incoming webhooks)
  public verifySignature(
    payload: string,
    signature: string,
    secret: string
  ): boolean {
    const expectedSignature = crypto
      .createHmac('sha256', secret)
      .update(payload)
      .digest('hex');
    
    return crypto.timingSafeEqual(
      Buffer.from(signature, 'hex'),
      Buffer.from(expectedSignature, 'hex')
    );
  }
  
  // Get webhook statistics
  public getStats(): any {
    const subscriptions = Array.from(this.subscriptions.values());
    
    return {
      total_subscriptions: subscriptions.length,
      active_subscriptions: subscriptions.filter(s => s.is_active).length,
      queued_deliveries: this.deliveryQueue.length,
      events_by_type: subscriptions.reduce((acc, sub) => {
        sub.events.forEach(event => {
          acc[event] = (acc[event] || 0) + 1;
        });
        return acc;
      }, {} as Record<string, number>)
    };
  }
}

export const webhookService = WebhookService.getInstance();

// Webhook routes
export function createWebhookRoutes() {
  const express = require('express');
  const router = express.Router();
  
  // Create webhook subscription
  router.post('/subscriptions', async (req: Request, res: Response) => {
    try {
      const { url, events, secret } = req.body;
      const userId = (req as any).user?.id;
      
      if (!userId) {
        return res.status(401).json({
          success: false,
          error: 'Authentication required'
        });
      }
      
      if (!url || !events || !Array.isArray(events)) {
        return res.status(400).json({
          success: false,
          error: 'Invalid request: url and events array required'
        });
      }
      
      const subscription = webhookService.createSubscription(userId, url, events, secret);
      
      res.json({
        success: true,
        data: subscription
      });
    } catch (error) {
      logger.error('Error creating webhook subscription:', error);
      res.status(500).json({
        success: false,
        error: 'Internal server error'
      });
    }
  });
  
  // Get user's webhook subscriptions
  router.get('/subscriptions', async (req: Request, res: Response) => {
    try {
      const userId = (req as any).user?.id;
      
      if (!userId) {
        return res.status(401).json({
          success: false,
          error: 'Authentication required'
        });
      }
      
      const subscriptions = webhookService.getUserSubscriptions(userId);
      
      res.json({
        success: true,
        data: subscriptions
      });
    } catch (error) {
      logger.error('Error fetching webhook subscriptions:', error);
      res.status(500).json({
        success: false,
        error: 'Internal server error'
      });
    }
  });
  
  // Update webhook subscription
  router.put('/subscriptions/:id', async (req: Request, res: Response) => {
    try {
      const { id } = req.params;
      const updates = req.body;
      const userId = (req as any).user?.id;
      
      if (!userId) {
        return res.status(401).json({
          success: false,
          error: 'Authentication required'
        });
      }
      
      const subscription = webhookService.updateSubscription(id, updates);
      
      if (!subscription) {
        return res.status(404).json({
          success: false,
          error: 'Webhook subscription not found'
        });
      }
      
      res.json({
        success: true,
        data: subscription
      });
    } catch (error) {
      logger.error('Error updating webhook subscription:', error);
      res.status(500).json({
        success: false,
        error: 'Internal server error'
      });
    }
  });
  
  // Delete webhook subscription
  router.delete('/subscriptions/:id', async (req: Request, res: Response) => {
    try {
      const { id } = req.params;
      const userId = (req as any).user?.id;
      
      if (!userId) {
        return res.status(401).json({
          success: false,
          error: 'Authentication required'
        });
      }
      
      const deleted = webhookService.deleteSubscription(id);
      
      if (!deleted) {
        return res.status(404).json({
          success: false,
          error: 'Webhook subscription not found'
        });
      }
      
      res.json({
        success: true,
        message: 'Webhook subscription deleted'
      });
    } catch (error) {
      logger.error('Error deleting webhook subscription:', error);
      res.status(500).json({
        success: false,
        error: 'Internal server error'
      });
    }
  });
  
  // Get webhook statistics
  router.get('/stats', async (req: Request, res: Response) => {
    try {
      const stats = webhookService.getStats();
      
      res.json({
        success: true,
        data: stats
      });
    } catch (error) {
      logger.error('Error fetching webhook stats:', error);
      res.status(500).json({
        success: false,
        error: 'Internal server error'
      });
    }
  });
  
  return router;
}

export default webhookService;
