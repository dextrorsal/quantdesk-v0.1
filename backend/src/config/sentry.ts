import * as Sentry from '@sentry/node';
import { nodeProfilingIntegration } from '@sentry/profiling-node';

/**
 * Sentry Configuration for QuantDesk Backend
 * Error tracking and performance monitoring
 */
export function initSentry() {
  if (!process.env.SENTRY_DSN) {
    console.warn('⚠️ SENTRY_DSN not configured. Sentry monitoring disabled.');
    return;
  }

  Sentry.init({
    dsn: process.env.SENTRY_DSN,
    environment: process.env.NODE_ENV || 'development',
    
    // Performance Monitoring
    tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
    profilesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
    
    // Integrations
    integrations: [
      nodeProfilingIntegration(),
      new Sentry.Integrations.Http({ tracing: true }),
      new Sentry.Integrations.Express({ app: undefined }),
      new Sentry.Integrations.OnUncaughtException(),
      new Sentry.Integrations.OnUnhandledRejection(),
    ],
    
    // Release tracking
    release: process.env.SENTRY_RELEASE || 'quantdesk-backend@1.0.0',
    
    // Custom tags
    initialScope: {
      tags: {
        component: 'quantdesk-backend',
        version: '1.0.0',
        service: 'backend',
      },
    },
    
    // Filter sensitive data
    beforeSend(event) {
      // Remove sensitive data from events
      if (event.request?.data) {
        delete event.request.data.password;
        delete event.request.data.token;
        delete event.request.data.secret;
        delete event.request.data.privateKey;
      }
      return event;
    },
  });

  console.log('✅ Sentry initialized for QuantDesk Backend');
}

/**
 * Capture custom errors with context
 */
export function captureError(error: Error, context?: Record<string, any>) {
  Sentry.withScope((scope) => {
    if (context) {
      Object.entries(context).forEach(([key, value]) => {
        scope.setContext(key, value);
      });
    }
    Sentry.captureException(error);
  });
}

/**
 * Capture custom messages
 */
export function captureMessage(message: string, level: 'info' | 'warning' | 'error' = 'info') {
  Sentry.captureMessage(message, level);
}

/**
 * Add user context to Sentry
 */
export function setUserContext(user: { id: string; email?: string; username?: string }) {
  Sentry.setUser({
    id: user.id,
    email: user.email,
    username: user.username,
  });
}

/**
 * Add custom tags
 */
export function setTag(key: string, value: string) {
  Sentry.setTag(key, value);
}

/**
 * Add breadcrumb for debugging
 */
export function addBreadcrumb(message: string, category?: string, level?: 'info' | 'warning' | 'error') {
  Sentry.addBreadcrumb({
    message,
    category: category || 'custom',
    level: level || 'info',
    timestamp: Date.now() / 1000,
  });
}
