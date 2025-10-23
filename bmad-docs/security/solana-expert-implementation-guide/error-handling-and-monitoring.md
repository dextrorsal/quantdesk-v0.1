# Error Handling and Monitoring

## Expert Recommendations

### 1. Comprehensive Error Handling
```typescript
// Error handling middleware
export const errorHandler = (error: Error, req: Request, res: Response, next: NextFunction) => {
  logger.error('Database error:', {
    error: error.message,
    stack: error.stack,
    query: req.body,
    user: req.user?.id
  });

  // Don't expose internal errors to clients
  if (error instanceof DatabaseError) {
    return res.status(500).json({
      error: 'Database operation failed',
      code: 'DATABASE_ERROR'
    });
  }

  // Handle specific error types
  if (error.message.includes('RLS')) {
    return res.status(403).json({
      error: 'Access denied',
      code: 'ACCESS_DENIED'
    });
  }

  res.status(500).json({
    error: 'Internal server error',
    code: 'INTERNAL_ERROR'
  });
};
```

### 2. Event Processing Monitoring
```typescript
// Event processing health check
export class EventProcessor {
  private processedEvents = new Map<string, number>();
  private failedEvents = new Map<string, number>();

  async processEvent(event: any, signature: string) {
    try {
      await this.handleEvent(event);
      this.processedEvents.set(signature, Date.now());
    } catch (error) {
      this.failedEvents.set(signature, Date.now());
      logger.error('Event processing failed:', { signature, error });
      throw error;
    }
  }

  getHealthStatus() {
    return {
      processed: this.processedEvents.size,
      failed: this.failedEvents.size,
      success_rate: this.processedEvents.size / (this.processedEvents.size + this.failedEvents.size)
    };
  }
}
```
