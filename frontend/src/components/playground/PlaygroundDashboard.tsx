// QuantDesk Developer Playground - Dashboard Component
// Phase 1: Core Playground (Match Drift's Data API Playground)
// Strategy: "More Open Than Drift" - Enhanced developer experience

import React from 'react';

interface ApiEndpoint {
  path: string;
  method: string;
  description: string;
}

interface ApiCategory {
  category: string;
  endpoints: ApiEndpoint[];
}

interface PlaygroundDashboardProps {
  apiEndpoints: ApiCategory[];
  onSelectEndpoint: (endpoint: string) => void;
  isAuthenticated: boolean;
}

export const PlaygroundDashboard: React.FC<PlaygroundDashboardProps> = ({
  apiEndpoints,
  onSelectEndpoint,
  isAuthenticated
}) => {
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

  return (
    <div className="space-y-8">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-8 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold mb-2">
              Welcome to QuantDesk Developer Playground
            </h2>
            <p className="text-blue-100 text-lg mb-4">
              Interactive API testing and documentation - More Open Than Drift Protocol
            </p>
            <div className="flex items-center space-x-6">
              <div className="flex items-center">
                <span className="text-2xl mr-2">🚀</span>
                <span className="font-medium">30+ API Endpoints</span>
              </div>
              <div className="flex items-center">
                <span className="text-2xl mr-2">⚡</span>
                <span className="font-medium">5+ Programming Languages</span>
              </div>
              <div className="flex items-center">
                <span className="text-2xl mr-2">🔐</span>
                <span className="font-medium">Enterprise Security</span>
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-4xl mb-2">🎯</div>
            <div className="text-sm text-blue-100">
              Competitive Advantage
            </div>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="text-2xl mr-3">📊</div>
            <div>
              <div className="text-2xl font-bold text-gray-900">30+</div>
              <div className="text-sm text-gray-600">API Endpoints</div>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="text-2xl mr-3">🤖</div>
            <div>
              <div className="text-2xl font-bold text-gray-900">AI</div>
              <div className="text-sm text-gray-600">MIKEY Integration</div>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="text-2xl mr-3">👥</div>
            <div>
              <div className="text-2xl font-bold text-gray-900">Community</div>
              <div className="text-sm text-gray-600">Points & Referrals</div>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="text-2xl mr-3">⚡</div>
            <div>
              <div className="text-2xl font-bold text-gray-900">5+</div>
              <div className="text-sm text-gray-600">Languages</div>
            </div>
          </div>
        </div>
      </div>

      {/* Authentication Status */}
      <div className={`p-4 rounded-lg border ${
        isAuthenticated 
          ? 'bg-green-50 border-green-200' 
          : 'bg-yellow-50 border-yellow-200'
      }`}>
        <div className="flex items-center">
          <span className="text-2xl mr-3">
            {isAuthenticated ? '✅' : '⚠️'}
          </span>
          <div>
            <h3 className="font-medium text-gray-900">
              {isAuthenticated ? 'Authenticated' : 'Authentication Required'}
            </h3>
            <p className="text-sm text-gray-600">
              {isAuthenticated 
                ? 'You can test all API endpoints with your API key'
                : 'Please authenticate to test API endpoints'
              }
            </p>
          </div>
        </div>
      </div>

      {/* API Categories */}
      <div className="space-y-6">
        <h3 className="text-xl font-semibold text-gray-900">
          Available API Endpoints
        </h3>
        
        {apiEndpoints.map((category) => (
          <div key={category.category} className="bg-white rounded-lg shadow-sm border">
            <div className="p-6 border-b">
              <div className="flex items-center">
                <span className="text-2xl mr-3">
                  {getCategoryIcon(category.category)}
                </span>
                <h4 className="text-lg font-semibold text-gray-900">
                  {category.category}
                </h4>
                <span className="ml-auto text-sm text-gray-500">
                  {category.endpoints.length} endpoints
                </span>
              </div>
            </div>
            
            <div className="p-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {category.endpoints.map((endpoint) => (
                  <div
                    key={`${endpoint.method}-${endpoint.path}`}
                    className="p-4 border rounded-lg hover:bg-gray-50 cursor-pointer transition-colors"
                    onClick={() => onSelectEndpoint(endpoint.path)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className={`px-2 py-1 text-xs font-medium rounded ${getMethodColor(endpoint.method)}`}>
                        {endpoint.method.toUpperCase()}
                      </span>
                      <span className="text-sm text-gray-500 font-mono">
                        {endpoint.path}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700">
                      {endpoint.description}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
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
              <h4 className="font-medium text-blue-900">Authenticate</h4>
              <p className="text-sm text-blue-700">
                Get your API key from the Authentication tab
              </p>
            </div>
          </div>
          
          <div className="flex items-start">
            <span className="text-2xl mr-3">2️⃣</span>
            <div>
              <h4 className="font-medium text-blue-900">Test APIs</h4>
              <p className="text-sm text-blue-700">
                Use the API Tester to make real requests
              </p>
            </div>
          </div>
          
          <div className="flex items-start">
            <span className="text-2xl mr-3">3️⃣</span>
            <div>
              <h4 className="font-medium text-blue-900">Generate Code</h4>
              <p className="text-sm text-blue-700">
                Get code examples in 5+ programming languages
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Competitive Advantages */}
      <div className="bg-green-50 rounded-lg p-6 border border-green-200">
        <h3 className="text-lg font-semibold text-green-900 mb-4">
          🎯 Why Choose QuantDesk Over Drift?
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-green-900 mb-2">Complete Ecosystem</h4>
            <p className="text-sm text-green-700">
              Multi-service architecture with AI integration, community features, and analytics
            </p>
          </div>
          
          <div>
            <h4 className="font-medium text-green-900 mb-2">Multi-Language Support</h4>
            <p className="text-sm text-green-700">
              Code generation for JavaScript, Python, TypeScript, Rust, and Go
            </p>
          </div>
          
          <div>
            <h4 className="font-medium text-green-900 mb-2">AI Integration</h4>
            <p className="text-sm text-green-700">
              MIKEY AI provides intelligent trading insights and analysis
            </p>
          </div>
          
          <div>
            <h4 className="font-medium text-green-900 mb-2">Community Features</h4>
            <p className="text-sm text-green-700">
              Points system, referrals, and gamification for user engagement
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
