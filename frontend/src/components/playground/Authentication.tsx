// QuantDesk Developer Playground - Authentication Component
// Phase 1: Core Playground (Match Drift's Data API Playground)
// Strategy: "More Open Than Drift" - Enhanced security

import React, { useState } from 'react';

interface AuthenticationProps {
  apiKey: string | null;
  onApiKeyChange: (apiKey: string | null) => void;
}

export const Authentication: React.FC<AuthenticationProps> = ({
  apiKey,
  onApiKeyChange
}) => {
  const [inputApiKey, setInputApiKey] = useState<string>(apiKey || '');
  const [showApiKey, setShowApiKey] = useState<boolean>(false);
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [validationResult, setValidationResult] = useState<{
    success: boolean;
    message: string;
  } | null>(null);

  const handleSaveApiKey = () => {
    if (inputApiKey.trim()) {
      onApiKeyChange(inputApiKey.trim());
      setValidationResult({
        success: true,
        message: 'API key saved successfully'
      });
    } else {
      onApiKeyChange(null);
      setValidationResult({
        success: false,
        message: 'API key removed'
      });
    }
  };

  const handleClearApiKey = () => {
    setInputApiKey('');
    onApiKeyChange(null);
    setValidationResult({
      success: false,
      message: 'API key cleared'
    });
  };

  const validateApiKey = async () => {
    if (!inputApiKey.trim()) return;
    
    setIsValidating(true);
    setValidationResult(null);
    
    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:3002'}/api/health`, {
        headers: {
          'X-API-Key': inputApiKey.trim()
        }
      });
      
      if (response.ok) {
        setValidationResult({
          success: true,
          message: 'API key is valid and working'
        });
      } else {
        setValidationResult({
          success: false,
          message: 'API key validation failed'
        });
      }
    } catch (error) {
      setValidationResult({
        success: false,
        message: 'Failed to validate API key'
      });
    } finally {
      setIsValidating(false);
    }
  };

  const generateTestApiKey = () => {
    // Generate a test API key for demonstration
    const testKey = 'qd_test_' + Math.random().toString(36).substr(2, 9);
    setInputApiKey(testKey);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          🔐 Authentication
        </h2>
        <p className="text-gray-600">
          Secure API key management - Enterprise-grade security exceeding Drift Protocol
        </p>
      </div>

      {/* Current Status */}
      <div className={`p-6 rounded-lg border ${
        apiKey 
          ? 'bg-green-50 border-green-200' 
          : 'bg-yellow-50 border-yellow-200'
      }`}>
        <div className="flex items-center">
          <span className="text-3xl mr-4">
            {apiKey ? '✅' : '⚠️'}
          </span>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              {apiKey ? 'Authenticated' : 'Not Authenticated'}
            </h3>
            <p className="text-gray-600">
              {apiKey 
                ? 'You can now test all API endpoints with your API key'
                : 'Please enter your API key to test endpoints'
              }
            </p>
          </div>
        </div>
      </div>

      {/* API Key Input */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          API Key Configuration
        </h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              API Key
            </label>
            <div className="flex items-center space-x-2">
              <input
                type={showApiKey ? 'text' : 'password'}
                value={inputApiKey}
                onChange={(e) => setInputApiKey(e.target.value)}
                placeholder="Enter your API key"
                className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              <button
                onClick={() => setShowApiKey(!showApiKey)}
                className="p-3 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                {showApiKey ? '🙈' : '👁️'}
              </button>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <button
              onClick={handleSaveApiKey}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              💾 Save API Key
            </button>
            
            <button
              onClick={validateApiKey}
              disabled={!inputApiKey.trim() || isValidating}
              className={`px-4 py-2 rounded-lg ${
                !inputApiKey.trim() || isValidating
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-green-600 text-white hover:bg-green-700'
              }`}
            >
              {isValidating ? (
                <span className="flex items-center">
                  <span className="animate-spin mr-2">⏳</span>
                  Validating...
                </span>
              ) : (
                <span className="flex items-center">
                  <span className="mr-2">🔍</span>
                  Validate
                </span>
              )}
            </button>
            
            <button
              onClick={handleClearApiKey}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              🗑️ Clear
            </button>
          </div>

          {/* Validation Result */}
          {validationResult && (
            <div className={`p-4 rounded-lg ${
              validationResult.success 
                ? 'bg-green-50 border border-green-200' 
                : 'bg-red-50 border border-red-200'
            }`}>
              <div className="flex items-center">
                <span className="text-2xl mr-3">
                  {validationResult.success ? '✅' : '❌'}
                </span>
                <span className={`font-medium ${
                  validationResult.success ? 'text-green-800' : 'text-red-800'
                }`}>
                  {validationResult.message}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Test API Key */}
      <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
        <h3 className="text-lg font-semibold text-blue-900 mb-4">
          🧪 Test API Key
        </h3>
        <p className="text-blue-700 mb-4">
          Generate a test API key for demonstration purposes
        </p>
        <button
          onClick={generateTestApiKey}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          🎲 Generate Test Key
        </button>
      </div>

      {/* Security Features */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          🔒 Security Features
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Authentication Methods</h4>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• JWT Token Authentication</li>
              <li>• API Key Authentication</li>
              <li>• Rate Limiting Protection</li>
              <li>• CORS Policy Enforcement</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Security Measures</h4>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• HTTPS Only Communication</li>
              <li>• Input Validation & Sanitization</li>
              <li>• Request Rate Limiting</li>
              <li>• Fraud Detection System</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Competitive Advantages */}
      <div className="bg-green-50 rounded-lg p-6 border border-green-200">
        <h3 className="text-lg font-semibold text-green-900 mb-4">
          🎯 Security Advantages Over Drift
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-green-900 mb-2">Drift Protocol</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Basic API key authentication</li>
              <li>• Simple rate limiting</li>
              <li>• Limited security features</li>
              <li>• Basic error handling</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-green-900 mb-2">QuantDesk</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Multiple authentication methods</li>
              <li>• Advanced rate limiting</li>
              <li>• Comprehensive security features</li>
              <li>• Enterprise-grade protection</li>
            </ul>
          </div>
        </div>
      </div>

      {/* API Key Management */}
      <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          📋 API Key Management
        </h3>
        
        <div className="space-y-4">
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Getting Your API Key</h4>
            <ol className="text-sm text-gray-600 space-y-1 list-decimal list-inside">
              <li>Sign up for a QuantDesk account</li>
              <li>Navigate to your account settings</li>
              <li>Generate a new API key</li>
              <li>Copy and paste it here</li>
            </ol>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Best Practices</h4>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Keep your API key secure and private</li>
              <li>• Don't share your API key with others</li>
              <li>• Regenerate your key if compromised</li>
              <li>• Use environment variables in production</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
