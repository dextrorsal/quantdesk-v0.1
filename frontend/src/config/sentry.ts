import * as Sentry from '@sentry/react';

/**
 * Sentry Configuration for QuantDesk Frontend
 * Error tracking and performance monitoring
 */
export function initSentry() {
  if (!import.meta.env.VITE_SENTRY_DSN) {
    console.warn('⚠️ VITE_SENTRY_DSN not configured. Sentry monitoring disabled.');
    return;
  }

  Sentry.init({
    dsn: import.meta.env.VITE_SENTRY_DSN,
    // Setting this option to true will send default PII data to Sentry.
    // For example, automatic IP address collection on events
    sendDefaultPii: true,
    environment: import.meta.env.MODE || 'development',
    
    // Performance Monitoring
    tracesSampleRate: import.meta.env.MODE === 'production' ? 0.1 : 1.0,
    
    // Integrations
    integrations: [
      Sentry.browserTracingIntegration(),
      Sentry.replayIntegration(),
      // Send console.log, console.warn, and console.error calls as logs to Sentry
      Sentry.consoleLoggingIntegration({ levels: ["log", "warn", "error"] }),
    ],
    
    // Set 'tracePropagationTargets' to control for which URLs distributed tracing should be enabled
    tracePropagationTargets: [
      'localhost',
      /^https:\/\/api\.quantdesk\.com/,
      /^https:\/\/.*\.vercel\.app/,
      /^https:\/\/.*\.railway\.app\/api/
    ],
    
    // Session Replay
    replaysSessionSampleRate: import.meta.env.MODE === 'production' ? 0.1 : 1.0, // 100% in dev, 10% in production
    replaysOnErrorSampleRate: 1.0, // 100% when sampling sessions where errors occur
    
    // Enable logs to be sent to Sentry
    enableLogs: true,
    
    // Release tracking
    release: import.meta.env.VITE_SENTRY_RELEASE || 'quantdesk-frontend@1.0.0',
    
    // Custom tags
    initialScope: {
      tags: {
        component: 'quantdesk-frontend',
        version: '1.0.0',
        service: 'frontend',
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

  console.log('✅ Sentry initialized for QuantDesk Frontend');
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
