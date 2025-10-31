// QuantDesk Developer Playground - API Tester Component
// Phase 1: Core Playground (Match Drift's Data API Playground)
// Strategy: "More Open Than Drift" - Enhanced developer experience

import React, { useState, useEffect } from 'react';

interface ApiEndpoint {
  path: string;
  method: string;
  description: string;
}

interface ApiCategory {
  category: string;
  endpoints: ApiEndpoint[];
}

interface ApiTesterProps {
  apiEndpoints: ApiCategory[];
  selectedEndpoint: string | null;
  onApiCall: (endpoint: string, method: string, params: any) => void;
  lastResponse: any;
  error: string | null;
  isAuthenticated: boolean;
}

export const ApiTester: React.FC<ApiTesterProps> = ({
  apiEndpoints,
  selectedEndpoint,
  onApiCall,
  lastResponse,
  error,
  isAuthenticated
}) => {
  const [selectedMethod, setSelectedMethod] = useState<string>('GET');
  const [requestParams, setRequestParams] = useState<any>({});
  const [isLoading, setIsLoading] = useState(false);
  const [responseTime, setResponseTime] = useState<number | null>(null);

  // Find the selected endpoint details
  const endpointDetails = apiEndpoints
    .flatMap(cat => cat.endpoints)
    .find(ep => ep.path === selectedEndpoint);

  // Reset params when endpoint changes
  useEffect(() => {
    if (selectedEndpoint) {
      const endpoint = apiEndpoints
        .flatMap(cat => cat.endpoints)
        .find(ep => ep.path === selectedEndpoint);
      if (endpoint) {
        setSelectedMethod(endpoint.method);
        setRequestParams({});
      }
    }
  }, [selectedEndpoint, apiEndpoints]);

  const handleApiCall = async () => {
    if (!selectedEndpoint) return;
    
    setIsLoading(true);
    const startTime = Date.now();
    
    try {
      await onApiCall(selectedEndpoint, selectedMethod, requestParams);
      setResponseTime(Date.now() - startTime);
    } finally {
      setIsLoading(false);
    }
  };

  const updateRequestParam = (key: string, value: any) => {
    setRequestParams(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const addRequestParam = () => {
    const key = prompt('Parameter key:');
    if (key) {
      updateRequestParam(key, '');
    }
  };

  const removeRequestParam = (key: string) => {
    setRequestParams(prev => {
      const newParams = { ...prev };
      delete newParams[key];
      return newParams;
    });
  };

  const formatResponse = (response: any) => {
    return JSON.stringify(response, null, 2);
  };

  const getMethodColor = (method: string) => {
    switch (method.toUpperCase()) {
      case 'GET': return 'bg-green-100 text-green-800';
      case 'POST': return 'bg-blue-100 text-blue-800';
      case 'PUT': return 'bg-yellow-100 text-yellow-800';
      case 'DELETE': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          🧪 API Tester
        </h2>
        <p className="text-gray-600">
          Test QuantDesk API endpoints in real-time - More Open Than Drift Protocol
        </p>
      </div>

      {/* Authentication Warning */}
      {!isAuthenticated && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center">
            <span className="text-2xl mr-3">⚠️</span>
            <div>
              <h3 className="font-medium text-yellow-800">Authentication Required</h3>
              <p className="text-sm text-yellow-700">
                Please authenticate with your API key to test endpoints
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Endpoint Selection */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Select Endpoint
        </h3>
        
        {selectedEndpoint ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-4">
                <span className={`px-3 py-1 text-sm font-medium rounded ${getMethodColor(selectedMethod)}`}>
                  {selectedMethod.toUpperCase()}
                </span>
                <span className="font-mono text-sm text-gray-700">
                  {selectedEndpoint}
                </span>
              </div>
              <button
                onClick={() => window.location.reload()}
                className="text-sm text-blue-600 hover:text-blue-800"
              >
                Change Endpoint
              </button>
            </div>
            
            {endpointDetails && (
              <p className="text-sm text-gray-600">
                {endpointDetails.description}
              </p>
            )}
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="text-4xl mb-4">🔗</div>
            <h4 className="text-lg font-medium text-gray-900 mb-2">
              No Endpoint Selected
            </h4>
            <p className="text-gray-600 mb-4">
              Please select an endpoint from the Dashboard to start testing
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Go to Dashboard
            </button>
          </div>
        )}
      </div>

      {/* Request Configuration */}
      {selectedEndpoint && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Request Configuration
          </h3>
          
          <div className="space-y-4">
            {/* Method Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                HTTP Method
              </label>
              <select
                value={selectedMethod}
                onChange={(e) => setSelectedMethod(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="GET">GET</option>
                <option value="POST">POST</option>
                <option value="PUT">PUT</option>
                <option value="DELETE">DELETE</option>
              </select>
            </div>

            {/* Request Parameters */}
            {(selectedMethod === 'POST' || selectedMethod === 'PUT') && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-gray-700">
                    Request Parameters
                  </label>
                  <button
                    onClick={addRequestParam}
                    className="text-sm text-blue-600 hover:text-blue-800"
                  >
                    + Add Parameter
                  </button>
                </div>
                
                <div className="space-y-2">
                  {Object.entries(requestParams).map(([key, value]) => (
                    <div key={key} className="flex items-center space-x-2">
                      <input
                        type="text"
                        value={key}
                        onChange={(e) => {
                          const newParams = { ...requestParams };
                          delete newParams[key];
                          newParams[e.target.value] = value;
                          setRequestParams(newParams);
                        }}
                        className="flex-1 p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        placeholder="Parameter key"
                      />
                      <input
                        type="text"
                        value={value as string}
                        onChange={(e) => updateRequestParam(key, e.target.value)}
                        className="flex-1 p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        placeholder="Parameter value"
                      />
                      <button
                        onClick={() => removeRequestParam(key)}
                        className="p-2 text-red-600 hover:text-red-800"
                      >
                        🗑️
                      </button>
                    </div>
                  ))}
                  
                  {Object.keys(requestParams).length === 0 && (
                    <div className="text-center py-4 text-gray-500">
                      No parameters added yet
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Execute Button */}
            <div className="pt-4">
              <button
                onClick={handleApiCall}
                disabled={!isAuthenticated || isLoading}
                className={`w-full py-3 px-4 rounded-lg font-medium ${
                  !isAuthenticated || isLoading
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
              >
                {isLoading ? (
                  <span className="flex items-center justify-center">
                    <span className="animate-spin mr-2">⏳</span>
                    Testing API...
                  </span>
                ) : (
                  <span className="flex items-center justify-center">
                    <span className="mr-2">🚀</span>
                    Test API Endpoint
                  </span>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Response Display */}
      {(lastResponse || error) && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Response
            {responseTime && (
              <span className="ml-2 text-sm font-normal text-gray-500">
                ({responseTime}ms)
              </span>
            )}
          </h3>
          
          {error ? (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center">
                <span className="text-2xl mr-3">❌</span>
                <div>
                  <h4 className="font-medium text-red-800">Error</h4>
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <span className="text-2xl mr-3">✅</span>
                  <h4 className="font-medium text-green-800">Success</h4>
                </div>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-4">
                <pre className="text-sm text-gray-800 overflow-x-auto">
                  {formatResponse(lastResponse)}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Help Section */}
      <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
        <h3 className="text-lg font-semibold text-blue-900 mb-4">
          💡 Tips for API Testing
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Authentication</h4>
            <p className="text-sm text-blue-700">
              Make sure to authenticate with your API key before testing endpoints
            </p>
          </div>
          
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Parameters</h4>
            <p className="text-sm text-blue-700">
              Add request parameters for POST and PUT methods to send data
            </p>
          </div>
          
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Response Time</h4>
            <p className="text-sm text-blue-700">
              Monitor response times to ensure optimal API performance
            </p>
          </div>
          
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Error Handling</h4>
            <p className="text-sm text-blue-700">
              Check error messages for debugging and troubleshooting
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
