import * as Sentry from '@sentry/node';
import { httpIntegration } from '@sentry/node';

/**
 * Sentry Configuration for QuantDesk MIKEY-AI
 * Error monitoring and performance tracking
 */

export function initSentry() {
  if (!process.env.SENTRY_DSN) {
    console.warn('⚠️ SENTRY_DSN not configured. Sentry monitoring disabled.');
    return;
  }

  Sentry.init({
    dsn: process.env.SENTRY_DSN,
    environment: process.env.NODE_ENV || 'development',
    integrations: [
      httpIntegration(),
    ],
    tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
    profilesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
    release: process.env.SENTRY_RELEASE || 'quantdesk-mikey-ai@1.0.0',
    beforeSend(event) {
      // Filter out development noise
      if (process.env.NODE_ENV === 'development') {
        // Only send errors, not warnings
        if (event.level === 'warning') {
          return null;
        }
      }
      return event;
    },
    beforeSendTransaction(event) {
      // Filter out health check transactions
      if (event.transaction?.includes('/health') || event.transaction?.includes('/ping')) {
        return null;
      }
      return event;
    },
  });

  console.log('✅ Sentry initialized for QuantDesk MIKEY-AI');
}

/**
 * Capture an error with context
 */
export function captureError(error: Error, context?: Record<string, any>) {
  Sentry.withScope((scope) => {
    if (context) {
      Object.keys(context).forEach(key => {
        scope.setContext(key, context[key]);
      });
    }
    Sentry.captureException(error);
  });
}

/**
 * Capture a message
 */
export function captureMessage(message: string, level: Sentry.SeverityLevel = 'info') {
  Sentry.captureMessage(message, level);
}

/**
 * Add user context to Sentry
 */
export function setUserContext(user: {
  id?: string;
  email?: string;
  username?: string;
  wallet?: string;
}) {
  Sentry.setUser({
    id: user.id,
    email: user.email,
    username: user.username,
    wallet: user.wallet,
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
export function addBreadcrumb(message: string, category?: string, data?: Record<string, any>) {
  Sentry.addBreadcrumb({
    message,
    category: category || 'custom',
    data,
    level: 'info',
  });
}