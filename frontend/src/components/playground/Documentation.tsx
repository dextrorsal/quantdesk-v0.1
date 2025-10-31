// QuantDesk Developer Playground - Documentation Component
// Phase 2: Enhanced Features (Exceed Drift's Capabilities)
// Strategy: "More Open Than Drift" - Comprehensive documentation

import React, { useState } from 'react';

interface ApiEndpoint {
  path: string;
  method: string;
  description: string;
}

interface ApiCategory {
  category: string;
  endpoints: ApiEndpoint[];
}

interface DocumentationProps {
  apiEndpoints: ApiCategory[];
}

export const Documentation: React.FC<DocumentationProps> = ({
  apiEndpoints
}) => {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>('');

  const getMethodColor = (method: string) => {
    switch (method.toUpperCase()) {
      case 'GET': return 'bg-green-100 text-green-800';
      case 'POST': return 'bg-blue-100 text-blue-800';
      case 'PUT': return 'bg-yellow-100 text-yellow-800';
      case 'DELETE': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Market Data': return '📈';
      case 'Trading': return '💼';
      case 'AI Integration': return '🤖';
      case 'Community': return '👥';
      case 'Analytics': return '📊';
      case 'System': return '⚙️';
      default: return '🔗';
    }
  };

  const filteredEndpoints = apiEndpoints
    .filter(category => 
      !selectedCategory || category.category === selectedCategory
    )
    .map(category => ({
      ...category,
      endpoints: category.endpoints.filter(endpoint =>
        endpoint.path.toLowerCase().includes(searchTerm.toLowerCase()) ||
        endpoint.description.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }))
    .filter(category => category.endpoints.length > 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          📚 API Documentation
        </h2>
        <p className="text-gray-600">
          Comprehensive API reference - Exceeding Drift's basic documentation
        </p>
      </div>

      {/* Search and Filter */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Search Endpoints
            </label>
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search by endpoint or description..."
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Filter by Category
            </label>
            <select
              value={selectedCategory || ''}
              onChange={(e) => setSelectedCategory(e.target.value || null)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">All Categories</option>
              {apiEndpoints.map(category => (
                <option key={category.category} value={category.category}>
                  {category.category}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* API Overview */}
      <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
        <h3 className="text-lg font-semibold text-blue-900 mb-4">
          🚀 QuantDesk API Overview
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Base URL</h4>
            <code className="text-sm text-blue-700 bg-blue-100 px-2 py-1 rounded">
              https://api.quantdesk.com
            </code>
          </div>
          
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Authentication</h4>
            <code className="text-sm text-blue-700 bg-blue-100 px-2 py-1 rounded">
              X-API-Key: your_api_key
            </code>
          </div>
          
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Content Type</h4>
            <code className="text-sm text-blue-700 bg-blue-100 px-2 py-1 rounded">
              application/json
            </code>
          </div>
        </div>
      </div>

      {/* API Categories */}
      <div className="space-y-6">
        {filteredEndpoints.map((category) => (
          <div key={category.category} className="bg-white rounded-lg shadow-sm border">
            <div className="p-6 border-b">
              <div className="flex items-center">
                <span className="text-2xl mr-3">
                  {getCategoryIcon(category.category)}
                </span>
                <h3 className="text-lg font-semibold text-gray-900">
                  {category.category}
                </h3>
                <span className="ml-auto text-sm text-gray-500">
                  {category.endpoints.length} endpoints
                </span>
              </div>
            </div>
            
            <div className="p-6">
              <div className="space-y-4">
                {category.endpoints.map((endpoint) => (
                  <div
                    key={`${endpoint.method}-${endpoint.path}`}
                    className="p-4 border rounded-lg hover:bg-gray-50"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className={`px-3 py-1 text-sm font-medium rounded ${getMethodColor(endpoint.method)}`}>
                        {endpoint.method.toUpperCase()}
                      </span>
                      <span className="font-mono text-sm text-gray-700">
                        {endpoint.path}
                      </span>
                    </div>
                    
                    <p className="text-sm text-gray-700 mb-3">
                      {endpoint.description}
                    </p>
                    
                    {/* Example Request */}
                    <div className="bg-gray-50 rounded p-3">
                      <h5 className="text-xs font-medium text-gray-600 mb-2">Example Request</h5>
                      <pre className="text-xs text-gray-800">
                        <code>
                          {endpoint.method.toUpperCase()} {endpoint.path}
                          {endpoint.method !== 'GET' && '\nContent-Type: application/json\nX-API-Key: your_api_key'}
                        </code>
                      </pre>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Rate Limiting */}
      <div className="bg-yellow-50 rounded-lg p-6 border border-yellow-200">
        <h3 className="text-lg font-semibold text-yellow-900 mb-4">
          ⏱️ Rate Limiting
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-yellow-900 mb-2">Rate Limits</h4>
            <ul className="text-sm text-yellow-700 space-y-1">
              <li>• General API: 1000 requests/hour</li>
              <li>• Trading API: 100 requests/hour</li>
              <li>• AI API: 50 requests/hour</li>
              <li>• Analytics API: 200 requests/hour</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-yellow-900 mb-2">Headers</h4>
            <ul className="text-sm text-yellow-700 space-y-1">
              <li>• X-RateLimit-Limit: 1000</li>
              <li>• X-RateLimit-Remaining: 999</li>
              <li>• X-RateLimit-Reset: 1642248600</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Error Codes */}
      <div className="bg-red-50 rounded-lg p-6 border border-red-200">
        <h3 className="text-lg font-semibold text-red-900 mb-4">
          ❌ Error Codes
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-red-900 mb-2">HTTP Status Codes</h4>
            <ul className="text-sm text-red-700 space-y-1">
              <li>• 200: Success</li>
              <li>• 400: Bad Request</li>
              <li>• 401: Unauthorized</li>
              <li>• 403: Forbidden</li>
              <li>• 404: Not Found</li>
              <li>• 429: Rate Limited</li>
              <li>• 500: Internal Server Error</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-red-900 mb-2">Custom Error Codes</h4>
            <ul className="text-sm text-red-700 space-y-1">
              <li>• INSUFFICIENT_BALANCE</li>
              <li>• INVALID_MARKET</li>
              <li>• POSITION_SIZE_TOO_LARGE</li>
              <li>• MARKET_CLOSED</li>
              <li>• RATE_LIMIT_EXCEEDED</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Competitive Advantages */}
      <div className="bg-green-50 rounded-lg p-6 border border-green-200">
        <h3 className="text-lg font-semibold text-green-900 mb-4">
          🎯 Documentation Advantages Over Drift
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-green-900 mb-2">Drift Protocol</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Basic endpoint documentation</li>
              <li>• Limited examples</li>
              <li>• Simple error handling</li>
              <li>• Basic rate limiting info</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-green-900 mb-2">QuantDesk</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Comprehensive API reference</li>
              <li>• Interactive examples</li>
              <li>• Detailed error handling</li>
              <li>• Advanced rate limiting</li>
              <li>• Search and filtering</li>
              <li>• Category organization</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Getting Started */}
      <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
        <h3 className="text-lg font-semibold text-blue-900 mb-4">
          🚀 Getting Started
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-start">
            <span className="text-2xl mr-3">1️⃣</span>
            <div>
              <h4 className="font-medium text-blue-900">Get API Key</h4>
              <p className="text-sm text-blue-700">
                Sign up and generate your API key
              </p>
            </div>
          </div>
          
          <div className="flex items-start">
            <span className="text-2xl mr-3">2️⃣</span>
            <div>
              <h4 className="font-medium text-blue-900">Test Endpoints</h4>
              <p className="text-sm text-blue-700">
                Use the API Tester to make requests
              </p>
            </div>
          </div>
          
          <div className="flex items-start">
            <span className="text-2xl mr-3">3️⃣</span>
            <div>
              <h4 className="font-medium text-blue-900">Generate Code</h4>
              <p className="text-sm text-blue-700">
                Get code examples in multiple languages
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
