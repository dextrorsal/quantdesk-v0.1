// instrument.js
// Initialize Sentry as early as possible in your application's lifecycle
const Sentry = require("@sentry/node");

Sentry.init({
  dsn: "https://80b2a47ae8405d0eb03a01a0f33a135d@o4510110939217920.ingest.us.sentry.io/4510248890269696",
  // Setting this option to true will send default PII data to Sentry.
  // For example, automatic IP address collection on events
  sendDefaultPii: true,
  environment: process.env.NODE_ENV || 'development',
  release: process.env.SENTRY_RELEASE || 'quantdesk-backend@1.0.0',
  tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
});
