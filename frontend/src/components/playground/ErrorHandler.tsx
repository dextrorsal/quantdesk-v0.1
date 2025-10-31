// QuantDesk Developer Playground - Error Handler Component
// Phase 2: Enhanced Features (Exceed Drift's Capabilities)
// Strategy: "More Open Than Drift" - Comprehensive error handling

import React from 'react';

interface ErrorHandlerProps {
  error: string;
  onClose: () => void;
}

export const ErrorHandler: React.FC<ErrorHandlerProps> = ({
  error,
  onClose
}) => {
  const getErrorType = (error: string) => {
    if (error.includes('401') || error.includes('Unauthorized')) {
      return {
        type: 'Authentication',
        icon: '🔐',
        color: 'red',
        suggestions: [
          'Check if your API key is correct',
          'Verify that you are authenticated',
          'Ensure your API key has the required permissions'
        ]
      };
    }
    
    if (error.includes('403') || error.includes('Forbidden')) {
      return {
        type: 'Authorization',
        icon: '🚫',
        color: 'orange',
        suggestions: [
          'Verify your API key permissions',
          'Check if the endpoint requires special access',
          'Contact support if you believe this is an error'
        ]
      };
    }
    
    if (error.includes('404') || error.includes('Not Found')) {
      return {
        type: 'Not Found',
        icon: '🔍',
        color: 'blue',
        suggestions: [
          'Verify the endpoint URL is correct',
          'Check if the resource exists',
          'Ensure you are using the correct API version'
        ]
      };
    }
    
    if (error.includes('429') || error.includes('Rate Limit')) {
      return {
        type: 'Rate Limited',
        icon: '⏱️',
        color: 'yellow',
        suggestions: [
          'Wait before making another request',
          'Check your rate limit status',
          'Consider upgrading your API plan'
        ]
      };
    }
    
    if (error.includes('500') || error.includes('Internal Server Error')) {
      return {
        type: 'Server Error',
        icon: '⚠️',
        color: 'red',
        suggestions: [
          'Try again in a few moments',
          'Check our status page for outages',
          'Contact support if the issue persists'
        ]
      };
    }
    
    return {
      type: 'General Error',
      icon: '❌',
      color: 'gray',
      suggestions: [
        'Check your request parameters',
        'Verify your API key is valid',
        'Contact support for assistance'
      ]
    };
  };

  const errorInfo = getErrorType(error);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        {/* Header */}
        <div className={`p-6 rounded-t-lg ${
          errorInfo.color === 'red' ? 'bg-red-50' :
          errorInfo.color === 'orange' ? 'bg-orange-50' :
          errorInfo.color === 'blue' ? 'bg-blue-50' :
          errorInfo.color === 'yellow' ? 'bg-yellow-50' :
          'bg-gray-50'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <span className="text-3xl mr-3">{errorInfo.icon}</span>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">
                  {errorInfo.type} Error
                </h3>
                <p className="text-sm text-gray-600">
                  Something went wrong with your API request
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <span className="text-2xl">×</span>
            </button>
          </div>
        </div>

        {/* Error Message */}
        <div className="p-6">
          <div className="mb-4">
            <h4 className="font-medium text-gray-900 mb-2">Error Details</h4>
            <div className="bg-gray-100 rounded-lg p-3">
              <code className="text-sm text-gray-800">{error}</code>
            </div>
          </div>

          {/* Suggestions */}
          <div className="mb-6">
            <h4 className="font-medium text-gray-900 mb-3">Suggested Solutions</h4>
            <ul className="space-y-2">
              {errorInfo.suggestions.map((suggestion, index) => (
                <li key={index} className="flex items-start">
                  <span className="text-blue-500 mr-2 mt-1">•</span>
                  <span className="text-sm text-gray-700">{suggestion}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Help Resources */}
          <div className="border-t pt-4">
            <h4 className="font-medium text-gray-900 mb-3">Need More Help?</h4>
            <div className="grid grid-cols-1 gap-2">
              <a
                href="#"
                className="flex items-center p-2 text-sm text-blue-600 hover:bg-blue-50 rounded"
              >
                <span className="mr-2">📚</span>
                View API Documentation
              </a>
              <a
                href="#"
                className="flex items-center p-2 text-sm text-blue-600 hover:bg-blue-50 rounded"
              >
                <span className="mr-2">💬</span>
                Contact Support
              </a>
              <a
                href="#"
                className="flex items-center p-2 text-sm text-blue-600 hover:bg-blue-50 rounded"
              >
                <span className="mr-2">🐛</span>
                Report Bug
              </a>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="bg-gray-50 px-6 py-4 rounded-b-lg">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              QuantDesk Developer Playground
            </div>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-600 text-white text-sm rounded hover:bg-gray-700"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
