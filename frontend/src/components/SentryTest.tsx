import * as Sentry from '@sentry/react';

// Add this button component to your app to test Sentry's error tracking
export default function SentryTestButton() {
  return (
    <button
      className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-md text-sm font-medium"
      onClick={() => {
        // Send a log before throwing the error
        Sentry.logger.info('User triggered test error', {
          action: 'test_error_button_click',
        });
        throw new Error('This is your first frontend error!');
      }}
    >
      🚨 Test Sentry Error
    </button>
  );
}

// Test component for Sentry tracing
export function SentryTestTracing() {
  const handleTestTracing = () => {
    // Create a transaction/span to measure performance
    Sentry.startSpan(
      {
        op: "ui.click",
        name: "Test Button Click",
      },
      (span) => {
        const value = "some config";
        const metric = "some metric";

        // Metrics can be added to the span
        span.setAttribute("config", value);
        span.setAttribute("metric", metric);

        // Simulate some work
        setTimeout(() => {
          console.log('Tracing test completed');
        }, 100);
      },
    );
  };

  return (
    <button
      className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium ml-2"
      onClick={handleTestTracing}
    >
      📊 Test Sentry Tracing
    </button>
  );
}
