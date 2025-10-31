console.log('🚀 MIKEY-AI instrument.js loaded!');

// Import with `import * as Sentry from "@sentry/node"` if you are using ESM
const Sentry = require("@sentry/node");

console.log('🔧 Initializing Sentry for MIKEY-AI...');
console.log('🔧 SENTRY_DSN_MIKEY_AI:', process.env.SENTRY_DSN_MIKEY_AI ? 'Set' : 'Not set');
console.log('🔧 NODE_ENV:', process.env.NODE_ENV || 'development');

try {
  Sentry.init({
    dsn: process.env.SENTRY_DSN_MIKEY_AI || "https://2b186ea51eca740226881585ff59fb83@o4510110939217920.ingest.us.sentry.io/4510248958885888",
    integrations: [
      // Basic integrations only - profiling has native module issues
    ],

    // Send structured logs to Sentry
    enableLogs: true,
    // Tracing
    tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0, // Capture 100% in dev, 10% in production
    // Setting this option to true will send default PII data to Sentry.
    // For example, automatic IP address collection on events
    sendDefaultPii: true,
    
    // Environment and release info
    environment: process.env.NODE_ENV || 'development',
    release: process.env.SENTRY_RELEASE_MIKEY_AI || 'quantdesk-mikey-ai@1.0.0',
    
    // Custom tags for MIKEY-AI
    initialScope: {
      tags: {
        component: 'quantdesk-mikey-ai',
        version: '1.0.0',
        service: 'ai-agent',
      },
    },
  });

  console.log('✅ Sentry initialized for QuantDesk MIKEY-AI');

  // Test Sentry with a simple span
  Sentry.startSpan({
    name: "MIKEY-AI Initialization",
  }, () => {
    console.log('✅ Sentry tracing started for MIKEY-AI');
  });
} catch (error) {
  console.error('❌ Error initializing Sentry:', error);
}
