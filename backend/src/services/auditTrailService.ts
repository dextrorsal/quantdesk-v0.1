import { Logger } from '../utils/logger';
import { getSupabaseService } from './supabaseService';

const logger = new Logger();

export interface AuditEvent {
  eventType: string;
  userId?: string;
  orderId?: string;
  positionId?: string;
  transactionId?: string;
  action: string;
  details: Record<string, any>;
  timestamp: number;
  ipAddress?: string;
  userAgent?: string;
  sessionId?: string;
}

export interface AuditQuery {
  userId?: string;
  orderId?: string;
  eventType?: string;
  action?: string;
  startTime?: number;
  endTime?: number;
  limit?: number;
  offset?: number;
}

export class AuditTrailService {
  private static instance: AuditTrailService;
  private readonly supabase: ReturnType<typeof getSupabaseService>;

  private constructor() {
    this.supabase = getSupabaseService();
  }

  public static getInstance(): AuditTrailService {
    if (!AuditTrailService.instance) {
      AuditTrailService.instance = new AuditTrailService();
    }
    return AuditTrailService.instance;
  }

  /**
   * Log an audit event
   */
  public async logEvent(event: AuditEvent): Promise<void> {
    try {
      await this.supabase.getClient()
        .from('audit_logs')
        .insert({
          event_type: event.eventType,
          user_id: event.userId,
          order_id: event.orderId,
          position_id: event.positionId,
          transaction_id: event.transactionId,
          action: event.action,
          details: event.details,
          ip_address: event.ipAddress,
          user_agent: event.userAgent,
          session_id: event.sessionId,
          created_at: new Date(event.timestamp).toISOString()
        });

      logger.debug(`📝 Audit event logged: ${event.eventType} - ${event.action}`);

    } catch (error) {
      logger.error('Failed to log audit event:', error);
    }
  }

  /**
   * Log order placement event
   */
  public async logOrderPlacement(
    userId: string,
    orderId: string,
    orderData: Record<string, any>,
    ipAddress?: string,
    userAgent?: string,
    sessionId?: string
  ): Promise<void> {
    await this.logEvent({
      eventType: 'order_placement',
      userId,
      orderId,
      action: 'place_order',
      details: {
        symbol: orderData.symbol,
        side: orderData.side,
        size: orderData.size,
        price: orderData.price,
        orderType: orderData.orderType,
        leverage: orderData.leverage,
        marketId: orderData.marketId
      },
      timestamp: Date.now(),
      ipAddress,
      userAgent,
      sessionId
    });
  }

  /**
   * Log order execution event
   */
  public async logOrderExecution(
    userId: string,
    orderId: string,
    executionData: Record<string, any>,
    ipAddress?: string,
    userAgent?: string,
    sessionId?: string
  ): Promise<void> {
    await this.logEvent({
      eventType: 'order_execution',
      userId,
      orderId,
      transactionId: executionData.transactionSignature,
      action: 'execute_order',
      details: {
        filled: executionData.filled,
        filledSize: executionData.filledSize,
        averageFillPrice: executionData.averageFillPrice,
        smartContractTx: executionData.transactionSignature,
        positionId: executionData.positionId
      },
      timestamp: Date.now(),
      ipAddress,
      userAgent,
      sessionId
    });
  }

  /**
   * Log order cancellation event
   */
  public async logOrderCancellation(
    userId: string,
    orderId: string,
    reason: string,
    ipAddress?: string,
    userAgent?: string,
    sessionId?: string
  ): Promise<void> {
    await this.logEvent({
      eventType: 'order_cancellation',
      userId,
      orderId,
      action: 'cancel_order',
      details: { reason },
      timestamp: Date.now(),
      ipAddress,
      userAgent,
      sessionId
    });
  }

  /**
   * Log position creation event
   */
  public async logPositionCreation(
    userId: string,
    positionId: string,
    orderId: string,
    positionData: Record<string, any>,
    ipAddress?: string,
    userAgent?: string,
    sessionId?: string
  ): Promise<void> {
    await this.logEvent({
      eventType: 'position_creation',
      userId,
      orderId,
      positionId,
      action: 'create_position',
      details: {
        symbol: positionData.symbol,
        side: positionData.side,
        size: positionData.size,
        entryPrice: positionData.entryPrice,
        leverage: positionData.leverage,
        smartContractPositionId: positionData.smartContractPositionId
      },
      timestamp: Date.now(),
      ipAddress,
      userAgent,
      sessionId
    });
  }

  /**
   * Log position modification event
   */
  public async logPositionModification(
    userId: string,
    positionId: string,
    modificationData: Record<string, any>,
    ipAddress?: string,
    userAgent?: string,
    sessionId?: string
  ): Promise<void> {
    await this.logEvent({
      eventType: 'position_modification',
      userId,
      positionId,
      action: 'modify_position',
      details: modificationData,
      timestamp: Date.now(),
      ipAddress,
      userAgent,
      sessionId
    });
  }

  /**
   * Log authorization event
   */
  public async logAuthorization(
    userId: string,
    action: string,
    authorized: boolean,
    reason?: string,
    ipAddress?: string,
    userAgent?: string,
    sessionId?: string
  ): Promise<void> {
    await this.logEvent({
      eventType: 'authorization',
      userId,
      action,
      details: {
        authorized,
        reason,
        timestamp: Date.now()
      },
      timestamp: Date.now(),
      ipAddress,
      userAgent,
      sessionId
    });
  }

  /**
   * Log security event
   */
  public async logSecurityEvent(
    userId: string,
    eventType: string,
    action: string,
    details: Record<string, any>,
    ipAddress?: string,
    userAgent?: string,
    sessionId?: string
  ): Promise<void> {
    await this.logEvent({
      eventType: 'security',
      userId,
      action,
      details: {
        ...details,
        severity: details.severity || 'medium'
      },
      timestamp: Date.now(),
      ipAddress,
      userAgent,
      sessionId
    });
  }

  /**
   * Query audit logs
   */
  public async queryAuditLogs(query: AuditQuery): Promise<{
    logs: any[];
    total: number;
    hasMore: boolean;
  }> {
    try {
      let dbQuery = this.supabase.getClient()
        .from('audit_logs')
        .select('*', { count: 'exact' });

      // Apply filters
      if (query.userId) {
        dbQuery = dbQuery.eq('user_id', query.userId);
      }
      
      if (query.orderId) {
        dbQuery = dbQuery.eq('order_id', query.orderId);
      }
      
      if (query.eventType) {
        dbQuery = dbQuery.eq('event_type', query.eventType);
      }
      
      if (query.action) {
        dbQuery = dbQuery.eq('action', query.action);
      }
      
      if (query.startTime) {
        dbQuery = dbQuery.gte('created_at', new Date(query.startTime).toISOString());
      }
      
      if (query.endTime) {
        dbQuery = dbQuery.lte('created_at', new Date(query.endTime).toISOString());
      }

      // Apply pagination
      const limit = query.limit || 100;
      const offset = query.offset || 0;
      
      dbQuery = dbQuery
        .order('created_at', { ascending: false })
        .range(offset, offset + limit - 1);

      const result = await dbQuery;

      return {
        logs: result.data || [],
        total: result.count || 0,
        hasMore: (offset + limit) < (result.count || 0)
      };

    } catch (error) {
      logger.error('Error querying audit logs:', error);
      return {
        logs: [],
        total: 0,
        hasMore: false
      };
    }
  }

  /**
   * Get audit trail for a specific order
   */
  public async getOrderAuditTrail(orderId: string): Promise<any[]> {
    const result = await this.queryAuditLogs({ orderId, limit: 1000 });
    return result.logs;
  }

  /**
   * Get audit trail for a specific user
   */
  public async getUserAuditTrail(
    userId: string, 
    timeWindow: number = 24 * 60 * 60 * 1000
  ): Promise<any[]> {
    const startTime = Date.now() - timeWindow;
    const result = await this.queryAuditLogs({ 
      userId, 
      startTime, 
      limit: 1000 
    });
    return result.logs;
  }

  /**
   * Get security events for monitoring
   */
  public async getSecurityEvents(
    timeWindow: number = 24 * 60 * 60 * 1000,
    severity?: string
  ): Promise<any[]> {
    const startTime = Date.now() - timeWindow;
    const result = await this.queryAuditLogs({ 
      eventType: 'security',
      startTime,
      limit: 1000 
    });

    if (severity) {
      return result.logs.filter(log => 
        log.details?.severity === severity
      );
    }

    return result.logs;
  }

  /**
   * Generate audit report
   */
  public async generateAuditReport(
    userId?: string,
    timeWindow: number = 24 * 60 * 60 * 1000
  ): Promise<{
    totalEvents: number;
    eventsByType: Record<string, number>;
    eventsByAction: Record<string, number>;
    securityEvents: number;
    orderEvents: number;
    positionEvents: number;
    authorizationEvents: number;
  }> {
    try {
      const startTime = Date.now() - timeWindow;
      const query: AuditQuery = { startTime, limit: 10000 };
      
      if (userId) {
        query.userId = userId;
      }

      const result = await this.queryAuditLogs(query);
      const logs = result.logs;

      const eventsByType: Record<string, number> = {};
      const eventsByAction: Record<string, number> = {};
      let securityEvents = 0;
      let orderEvents = 0;
      let positionEvents = 0;
      let authorizationEvents = 0;

      logs.forEach(log => {
        // Count by event type
        eventsByType[log.event_type] = (eventsByType[log.event_type] || 0) + 1;
        
        // Count by action
        eventsByAction[log.action] = (eventsByAction[log.action] || 0) + 1;

        // Count specific event categories
        if (log.event_type === 'security') securityEvents++;
        if (log.event_type.includes('order')) orderEvents++;
        if (log.event_type.includes('position')) positionEvents++;
        if (log.event_type === 'authorization') authorizationEvents++;
      });

      return {
        totalEvents: logs.length,
        eventsByType,
        eventsByAction,
        securityEvents,
        orderEvents,
        positionEvents,
        authorizationEvents
      };

    } catch (error) {
      logger.error('Error generating audit report:', error);
      return {
        totalEvents: 0,
        eventsByType: {},
        eventsByAction: {},
        securityEvents: 0,
        orderEvents: 0,
        positionEvents: 0,
        authorizationEvents: 0
      };
    }
  }

  /**
   * Clean up old audit logs (retention policy)
   */
  public async cleanupOldLogs(retentionDays: number = 90): Promise<number> {
    try {
      const cutoffDate = new Date(Date.now() - (retentionDays * 24 * 60 * 60 * 1000));
      
      const { data, error } = await this.supabase.getClient()
        .from('audit_logs')
        .delete()
        .lt('created_at', cutoffDate.toISOString())
        .select('id');

      if (error) {
        logger.error('Error cleaning up old audit logs:', error);
        return 0;
      }

      const deletedCount = data?.length || 0;
      logger.info(`Cleaned up ${deletedCount} old audit logs (older than ${retentionDays} days)`);
      
      return deletedCount;

    } catch (error) {
      logger.error('Error cleaning up old audit logs:', error);
      return 0;
    }
  }
}

// Export singleton instance
export const auditTrailService = AuditTrailService.getInstance();
