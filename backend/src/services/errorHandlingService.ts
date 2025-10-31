import { Logger } from '../utils/logger';
import { getSupabaseService } from './supabaseService';

const logger = new Logger();

export interface ErrorContext {
  operation: string;
  userId?: string;
  orderId?: string;
  positionId?: string;
  transactionId?: string;
  metadata?: Record<string, any>;
}

export interface RollbackAction {
  action: string;
  params: Record<string, any>;
  priority: number; // Lower number = higher priority
}

export interface ErrorRecoveryPlan {
  canRecover: boolean;
  rollbackActions: RollbackAction[];
  recoverySteps: string[];
  estimatedRecoveryTime: number; // milliseconds
}

export class ErrorHandlingService {
  private static instance: ErrorHandlingService;
  private readonly supabase: ReturnType<typeof getSupabaseService>;

  private constructor() {
    this.supabase = getSupabaseService();
  }

  public static getInstance(): ErrorHandlingService {
    if (!ErrorHandlingService.instance) {
      ErrorHandlingService.instance = new ErrorHandlingService();
    }
    return ErrorHandlingService.instance;
  }

  /**
   * Handle error with comprehensive recovery planning
   */
  public async handleError(error: Error, context: ErrorContext): Promise<ErrorRecoveryPlan> {
    try {
      logger.error(`🚨 Error in ${context.operation}:`, error);

      // Log error for audit trail
      await this.logError(error, context);

      // Determine recovery plan based on operation and error type
      const recoveryPlan = this.determineRecoveryPlan(error, context);

      // Execute rollback actions if recovery is possible
      if (recoveryPlan.canRecover) {
        await this.executeRollbackActions(recoveryPlan.rollbackActions, context);
      }

      return recoveryPlan;

    } catch (recoveryError) {
      logger.error('❌ Error handling failed:', recoveryError);
      
      // Return minimal recovery plan
      return {
        canRecover: false,
        rollbackActions: [],
        recoverySteps: ['Manual intervention required'],
        estimatedRecoveryTime: 0
      };
    }
  }

  /**
   * Determine recovery plan based on error type and context
   */
  private determineRecoveryPlan(error: Error, context: ErrorContext): ErrorRecoveryPlan {
    const errorMessage = error.message.toLowerCase();

    // Order placement errors
    if (context.operation === 'order_placement') {
      return this.handleOrderPlacementError(error, context);
    }

    // Order execution errors
    if (context.operation === 'order_execution') {
      return this.handleOrderExecutionError(error, context);
    }

    // Position creation errors
    if (context.operation === 'position_creation') {
      return this.handlePositionCreationError(error, context);
    }

    // Smart contract errors
    if (errorMessage.includes('smart contract') || errorMessage.includes('blockchain')) {
      return this.handleSmartContractError(error, context);
    }

    // Database errors
    if (errorMessage.includes('database') || errorMessage.includes('connection')) {
      return this.handleDatabaseError(error, context);
    }

    // Oracle errors
    if (errorMessage.includes('oracle') || errorMessage.includes('price')) {
      return this.handleOracleError(error, context);
    }

    // Generic error handling
    return {
      canRecover: false,
      rollbackActions: [],
      recoverySteps: ['Manual intervention required'],
      estimatedRecoveryTime: 0
    };
  }

  /**
   * Handle order placement errors
   */
  private handleOrderPlacementError(error: Error, context: ErrorContext): ErrorRecoveryPlan {
    const errorMessage = error.message.toLowerCase();

    if (errorMessage.includes('authorization')) {
      return {
        canRecover: false,
        rollbackActions: [],
        recoverySteps: ['User authorization issue - no rollback needed'],
        estimatedRecoveryTime: 0
      };
    }

    if (errorMessage.includes('rate limit')) {
      return {
        canRecover: false,
        rollbackActions: [],
        recoverySteps: ['Rate limit exceeded - user must wait'],
        estimatedRecoveryTime: 60000 // 1 minute
      };
    }

    if (errorMessage.includes('validation')) {
      return {
        canRecover: false,
        rollbackActions: [],
        recoverySteps: ['Input validation failed - no rollback needed'],
        estimatedRecoveryTime: 0
      };
    }

    // Database error during order creation
    return {
      canRecover: true,
      rollbackActions: [
        {
          action: 'cancel_order',
          params: { orderId: context.orderId },
          priority: 1
        }
      ],
      recoverySteps: ['Cancel any partially created order'],
      estimatedRecoveryTime: 1000
    };
  }

  /**
   * Handle order execution errors
   */
  private handleOrderExecutionError(error: Error, context: ErrorContext): ErrorRecoveryPlan {
    const errorMessage = error.message.toLowerCase();

    if (errorMessage.includes('smart contract execution failed')) {
      return {
        canRecover: true,
        rollbackActions: [
          {
            action: 'update_order_status',
            params: { orderId: context.orderId, status: 'failed' },
            priority: 1
          },
          {
            action: 'refund_collateral',
            params: { userId: context.userId, orderId: context.orderId },
            priority: 2
          }
        ],
        recoverySteps: [
          'Mark order as failed',
          'Refund any locked collateral',
          'Notify user of failure'
        ],
        estimatedRecoveryTime: 5000
      };
    }

    if (errorMessage.includes('insufficient funds')) {
      return {
        canRecover: true,
        rollbackActions: [
          {
            action: 'update_order_status',
            params: { orderId: context.orderId, status: 'cancelled' },
            priority: 1
          }
        ],
        recoverySteps: ['Mark order as cancelled due to insufficient funds'],
        estimatedRecoveryTime: 1000
      };
    }

    return {
      canRecover: true,
      rollbackActions: [
        {
          action: 'update_order_status',
          params: { orderId: context.orderId, status: 'failed' },
          priority: 1
        }
      ],
      recoverySteps: ['Mark order as failed'],
      estimatedRecoveryTime: 1000
    };
  }

  /**
   * Handle position creation errors
   */
  private handlePositionCreationError(error: Error, context: ErrorContext): ErrorRecoveryPlan {
    return {
      canRecover: true,
      rollbackActions: [
        {
          action: 'delete_position',
          params: { positionId: context.positionId },
          priority: 1
        },
        {
          action: 'update_order_status',
          params: { orderId: context.orderId, status: 'failed' },
          priority: 2
        }
      ],
      recoverySteps: [
        'Delete any partially created position',
        'Mark order as failed',
        'Refund collateral if needed'
      ],
      estimatedRecoveryTime: 3000
    };
  }

  /**
   * Handle smart contract errors
   */
  private handleSmartContractError(error: Error, context: ErrorContext): ErrorRecoveryPlan {
    return {
      canRecover: true,
      rollbackActions: [
        {
          action: 'revert_transaction',
          params: { transactionId: context.transactionId },
          priority: 1
        },
        {
          action: 'update_order_status',
          params: { orderId: context.orderId, status: 'failed' },
          priority: 2
        }
      ],
      recoverySteps: [
        'Attempt to revert blockchain transaction',
        'Mark order as failed',
        'Refund any locked funds'
      ],
      estimatedRecoveryTime: 10000
    };
  }

  /**
   * Handle database errors
   */
  private handleDatabaseError(error: Error, context: ErrorContext): ErrorRecoveryPlan {
    return {
      canRecover: true,
      rollbackActions: [
        {
          action: 'retry_database_operation',
          params: { operation: context.operation, maxRetries: 3 },
          priority: 1
        }
      ],
      recoverySteps: [
        'Retry database operation',
        'Check database connectivity',
        'Fallback to cached data if available'
      ],
      estimatedRecoveryTime: 5000
    };
  }

  /**
   * Handle Oracle errors
   */
  private handleOracleError(error: Error, context: ErrorContext): ErrorRecoveryPlan {
    return {
      canRecover: true,
      rollbackActions: [
        {
          action: 'use_cached_price',
          params: { symbol: context.metadata?.symbol },
          priority: 1
        },
        {
          action: 'fallback_to_alternative_oracle',
          params: { symbol: context.metadata?.symbol },
          priority: 2
        }
      ],
      recoverySteps: [
        'Use cached price data',
        'Try alternative Oracle source',
        'Mark order as pending if no price available'
      ],
      estimatedRecoveryTime: 2000
    };
  }

  /**
   * Execute rollback actions in priority order
   */
  private async executeRollbackActions(actions: RollbackAction[], context: ErrorContext): Promise<void> {
    // Sort by priority (lower number = higher priority)
    const sortedActions = actions.sort((a, b) => a.priority - b.priority);

    for (const action of sortedActions) {
      try {
        await this.executeRollbackAction(action, context);
        logger.info(`✅ Rollback action executed: ${action.action}`);
      } catch (actionError) {
        logger.error(`❌ Rollback action failed: ${action.action}`, actionError);
        // Continue with other actions even if one fails
      }
    }
  }

  /**
   * Execute individual rollback action
   */
  private async executeRollbackAction(action: RollbackAction, context: ErrorContext): Promise<void> {
    switch (action.action) {
      case 'cancel_order':
        await this.cancelOrder(action.params.orderId);
        break;
      
      case 'update_order_status':
        await this.updateOrderStatus(action.params.orderId, action.params.status);
        break;
      
      case 'delete_position':
        await this.deletePosition(action.params.positionId);
        break;
      
      case 'refund_collateral':
        await this.refundCollateral(action.params.userId, action.params.orderId);
        break;
      
      case 'revert_transaction':
        await this.revertTransaction(action.params.transactionId);
        break;
      
      case 'retry_database_operation':
        await this.retryDatabaseOperation(action.params.operation, action.params.maxRetries);
        break;
      
      case 'use_cached_price':
        await this.useCachedPrice(action.params.symbol);
        break;
      
      case 'fallback_to_alternative_oracle':
        await this.fallbackToAlternativeOracle(action.params.symbol);
        break;
      
      default:
        logger.warn(`Unknown rollback action: ${action.action}`);
    }
  }

  /**
   * Rollback action implementations
   */
  private async cancelOrder(orderId: string): Promise<void> {
    await this.supabase.getClient()
      .from('orders')
      .update({ status: 'cancelled', updated_at: new Date().toISOString() })
      .eq('id', orderId);
  }

  private async updateOrderStatus(orderId: string, status: string): Promise<void> {
    await this.supabase.getClient()
      .from('orders')
      .update({ status, updated_at: new Date().toISOString() })
      .eq('id', orderId);
  }

  private async deletePosition(positionId: string): Promise<void> {
    await this.supabase.getClient()
      .from('positions')
      .delete()
      .eq('id', positionId);
  }

  private async refundCollateral(userId: string, orderId: string): Promise<void> {
    // Implementation would depend on collateral system
    logger.info(`Refunding collateral for user ${userId}, order ${orderId}`);
  }

  private async revertTransaction(transactionId: string): Promise<void> {
    // Implementation would depend on blockchain interaction
    logger.info(`Reverting transaction ${transactionId}`);
  }

  private async retryDatabaseOperation(operation: string, maxRetries: number): Promise<void> {
    // Implementation would retry the failed operation
    logger.info(`Retrying database operation: ${operation}, max retries: ${maxRetries}`);
  }

  private async useCachedPrice(symbol: string): Promise<void> {
    // Implementation would use cached price data
    logger.info(`Using cached price for symbol: ${symbol}`);
  }

  private async fallbackToAlternativeOracle(symbol: string): Promise<void> {
    // Implementation would try alternative Oracle
    logger.info(`Falling back to alternative Oracle for symbol: ${symbol}`);
  }

  /**
   * Log error for audit trail
   */
  private async logError(error: Error, context: ErrorContext): Promise<void> {
    try {
      await this.supabase.getClient()
        .from('error_logs')
        .insert({
          operation: context.operation,
          error_message: error.message,
          error_stack: error.stack,
          user_id: context.userId,
          order_id: context.orderId,
          position_id: context.positionId,
          transaction_id: context.transactionId,
          metadata: context.metadata,
          created_at: new Date().toISOString()
        });
    } catch (logError) {
      logger.error('Failed to log error:', logError);
    }
  }

  /**
   * Get error statistics for monitoring
   */
  public async getErrorStatistics(timeWindow: number = 24 * 60 * 60 * 1000): Promise<{
    totalErrors: number;
    errorsByOperation: Record<string, number>;
    errorsByType: Record<string, number>;
    recoverySuccessRate: number;
  }> {
    try {
      const since = new Date(Date.now() - timeWindow).toISOString();
      
      const { data, error } = await this.supabase.getClient()
        .from('error_logs')
        .select('operation, error_message, created_at')
        .gte('created_at', since);

      if (error) {
        logger.error('Error fetching error logs:', error);
        return {
          totalErrors: 0,
          errorsByOperation: {},
          errorsByType: {},
          recoverySuccessRate: 0
        };
      }

      const totalErrors = data?.length || 0;
      const errorsByOperation: Record<string, number> = {};
      const errorsByType: Record<string, number> = {};

      if (data) {
        data.forEach(error => {
          errorsByOperation[error.operation] = (errorsByOperation[error.operation] || 0) + 1;
          
          const errorType = this.categorizeError(error.error_message);
          errorsByType[errorType] = (errorsByType[errorType] || 0) + 1;
        });
      }

      return {
        totalErrors,
        errorsByOperation,
        errorsByType,
        recoverySuccessRate: 0.95 // Placeholder - would be calculated from recovery logs
      };

    } catch (error) {
      logger.error('Error getting error statistics:', error);
      return {
        totalErrors: 0,
        errorsByOperation: {},
        errorsByType: {},
        recoverySuccessRate: 0
      };
    }
  }

  /**
   * Categorize error message
   */
  private categorizeError(errorMessage: string): string {
    const message = errorMessage.toLowerCase();
    
    if (message.includes('authorization')) return 'authorization';
    if (message.includes('rate limit')) return 'rate_limit';
    if (message.includes('validation')) return 'validation';
    if (message.includes('smart contract')) return 'smart_contract';
    if (message.includes('database')) return 'database';
    if (message.includes('oracle')) return 'oracle';
    if (message.includes('network')) return 'network';
    
    return 'unknown';
  }
}

// Export singleton instance
export const errorHandlingService = ErrorHandlingService.getInstance();
