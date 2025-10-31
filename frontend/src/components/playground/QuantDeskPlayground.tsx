// QuantDesk Developer Playground - Main Dashboard
// Phase 1: Core Playground (Match Drift's Data API Playground)
// Strategy: "More Open Than Drift" - Enhanced developer experience

import React, { useState, useEffect } from 'react';
import { PlaygroundDashboard } from './PlaygroundDashboard';
import { ApiTester } from './ApiTester';
import { CodeGenerator } from './CodeGenerator';
import { Authentication } from './Authentication';
import { ErrorHandler } from './ErrorHandler';
import { Documentation } from './Documentation';
import { SdkIntegrationExamples } from './SdkIntegrationExamples';
import { AdvancedFeatures } from './AdvancedFeatures';
import { RealTimeCollaboration } from './RealTimeCollaboration';
import { ApiAnalytics } from './ApiAnalytics';
import { MikeyIntegrationShowcase } from './MikeyIntegrationShowcase';
import { AIToolsIntegration } from './AIToolsIntegration';

interface PlaygroundState {
  currentTab: 'dashboard' | 'tester' | 'generator' | 'auth' | 'docs' | 'sdk' | 'advanced' | 'collaboration' | 'analytics' | 'mikey' | 'ai-tools';
  isAuthenticated: boolean;
  apiKey: string | null;
  selectedEndpoint: string | null;
  lastResponse: any;
  error: string | null;
}

export const QuantDeskPlayground: React.FC = () => {
  const [state, setState] = useState<PlaygroundState>({
    currentTab: 'dashboard',
    isAuthenticated: false,
    apiKey: null,
    selectedEndpoint: null,
    lastResponse: null,
    error: null
  });

  // Available API endpoints (matching our actual backend routes)
  const apiEndpoints = [
    // Market Data APIs
    { category: 'Market Data', endpoints: [
      { path: '/api/markets', method: 'GET', description: 'Get available markets' },
      { path: '/api/markets/{symbol}', method: 'GET', description: 'Get market data' },
      { path: '/api/markets/{symbol}/history', method: 'GET', description: 'Get historical data' },
      { path: '/api/oracle', method: 'GET', description: 'Get oracle price feeds' }
    ]},
    
    // Trading APIs
    { category: 'Trading', endpoints: [
      { path: '/api/positions', method: 'GET', description: 'Get positions' },
      { path: '/api/positions', method: 'POST', description: 'Open position' },
      { path: '/api/positions/{id}', method: 'DELETE', description: 'Close position' },
      { path: '/api/orders', method: 'GET', description: 'Get orders' },
      { path: '/api/orders', method: 'POST', description: 'Place order' },
      { path: '/api/orders/{id}', method: 'DELETE', description: 'Cancel order' },
      { path: '/api/portfolio', method: 'GET', description: 'Get portfolio' }
    ]},
    
    // AI Integration APIs
    { category: 'AI Integration', endpoints: [
      { path: '/api/ai/analysis/{market}', method: 'GET', description: 'Get AI analysis' },
      { path: '/api/ai/signals', method: 'GET', description: 'Get trading signals' },
      { path: '/api/ai/risk-assessment', method: 'GET', description: 'Get risk assessment' },
      { path: '/api/ai/chat', method: 'POST', description: 'Chat with MIKEY AI' },
      { path: '/api/mikeyAI', method: 'GET', description: 'MIKEY AI integration' }
    ]},
    
    // Community Features
    { category: 'Community', endpoints: [
      { path: '/api/communityPoints', method: 'GET', description: 'Get user points' },
      { path: '/api/communityPoints/award', method: 'POST', description: 'Award points' },
      { path: '/api/referrals', method: 'GET', description: 'Get referrals' },
      { path: '/api/referrals/generate', method: 'POST', description: 'Generate referral code' },
      { path: '/api/referralSystem/stats', method: 'GET', description: 'Get referral stats' }
    ]},
    
    // Analytics APIs
    { category: 'Analytics', endpoints: [
      { path: '/api/analytics/trading-history', method: 'GET', description: 'Get trading history' },
      { path: '/api/analytics/performance', method: 'GET', description: 'Get performance metrics' },
      { path: '/api/performance', method: 'GET', description: 'Get system performance' },
      { path: '/api/metrics', method: 'GET', description: 'Get system metrics' }
    ]},
    
    // System APIs
    { category: 'System', endpoints: [
      { path: '/api/health', method: 'GET', description: 'Health check' },
      { path: '/api/status', method: 'GET', description: 'System status' },
      { path: '/api/users/profile', method: 'GET', description: 'Get user profile' },
      { path: '/api/users/settings', method: 'PUT', description: 'Update user settings' }
    ]}
  ];

  const handleApiCall = async (endpoint: string, method: string, params: any) => {
    try {
      setState(prev => ({ ...prev, error: null }));
      
      const url = `${process.env.REACT_APP_API_URL || 'http://localhost:3002'}${endpoint}`;
      const headers: HeadersInit = {
        'Content-Type': 'application/json'
      };
      
      if (state.apiKey) {
        headers['X-API-Key'] = state.apiKey;
      }
      
      const response = await fetch(url, {
        method,
        headers,
        body: method !== 'GET' ? JSON.stringify(params) : undefined
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error?.message || `HTTP ${response.status}`);
      }
      
      setState(prev => ({ ...prev, lastResponse: data }));
      
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      }));
    }
  };

  const generateCodeExample = (endpoint: string, method: string, params: any, language: string) => {
    const baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:3002';
    
    switch (language) {
      case 'javascript':
        return `// JavaScript Example
const response = await fetch('${baseUrl}${endpoint}', {
  method: '${method.toUpperCase()}',
  headers: {
    'Content-Type': 'application/json'${state.apiKey ? `,\n    'X-API-Key': '${state.apiKey}'` : ''}
  }${method !== 'GET' ? `,\n  body: JSON.stringify(${JSON.stringify(params, null, 2)})` : ''}
});

const data = await response.json();
console.log(data);`;

      case 'python':
        return `# Python Example
import requests
import json

url = '${baseUrl}${endpoint}'
headers = {
    'Content-Type': 'application/json'${state.apiKey ? `,\n    'X-API-Key': '${state.apiKey}'` : ''}
}${method !== 'GET' ? `\ndata = ${JSON.stringify(params, null, 2)}` : ''}

response = requests.${method.toLowerCase()}(
    url${method !== 'GET' ? ',\n    json=data' : ''},
    headers=headers
)

result = response.json()
print(result)`;

      case 'typescript':
        return `// TypeScript Example
interface ApiResponse {
  success: boolean;
  data: any;
  error?: {
    code: string;
    message: string;
  };
}

const response = await fetch('${baseUrl}${endpoint}', {
  method: '${method.toUpperCase()}',
  headers: {
    'Content-Type': 'application/json'${state.apiKey ? `,\n    'X-API-Key': '${state.apiKey}'` : ''}
  }${method !== 'GET' ? `,\n  body: JSON.stringify(${JSON.stringify(params, null, 2)})` : ''}
});

const data: ApiResponse = await response.json();
console.log(data);`;

      case 'rust':
        return `// Rust Example
use reqwest;
use serde_json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let url = "${baseUrl}${endpoint}";
    
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("Content-Type", "application/json".parse()?);${state.apiKey ? `\n    headers.insert("X-API-Key", "${state.apiKey}".parse()?);` : ''}
    
    let response = client
        .${method.toLowerCase()}("${baseUrl}${endpoint}")${method !== 'GET' ? `\n        .json(&${JSON.stringify(params)})` : ''}
        .headers(headers)
        .send()
        .await?;
    
    let data: serde_json::Value = response.json().await?;
    println!("{:?}", data);
    
    Ok(())
}`;

      case 'go':
        return `// Go Example
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
)

func main() {
    url := "${baseUrl}${endpoint}"
    ${method !== 'GET' ? `data := map[string]interface{}{${Object.entries(params).map(([key, value]) => `\n        "${key}": ${JSON.stringify(value)}`).join(',')}\n    }\n    jsonData, _ := json.Marshal(data)` : ''}
    
    req, _ := http.NewRequest("${method.toUpperCase()}", url${method !== 'GET' ? ', bytes.NewBuffer(jsonData)' : ', nil'})
    req.Header.Set("Content-Type", "application/json")${state.apiKey ? `\n    req.Header.Set("X-API-Key", "${state.apiKey}")` : ''}
    
    client := &http.Client{}
    resp, _ := client.Do(req)
    defer resp.Body.Close()
    
    body, _ := io.ReadAll(resp.Body)
    fmt.Println(string(body))
}`;

      default:
        return `// ${language} Example\n// Code generation for ${language} not yet implemented`;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">
                QuantDesk Developer Playground
              </h1>
              <span className="ml-3 px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full">
                More Open Than Drift
              </span>
            </div>
            <div className="flex items-center space-x-4">
              {state.isAuthenticated ? (
                <span className="text-sm text-gray-600">
                  ✅ Authenticated
                </span>
              ) : (
                <span className="text-sm text-gray-600">
                  ⚠️ Not Authenticated
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8">
            {[
              { id: 'dashboard', label: 'Dashboard', icon: '📊' },
              { id: 'tester', label: 'API Tester', icon: '🧪' },
              { id: 'generator', label: 'Code Generator', icon: '⚡' },
              { id: 'auth', label: 'Authentication', icon: '🔐' },
              { id: 'docs', label: 'Documentation', icon: '📚' },
              { id: 'sdk', label: 'SDK Examples', icon: '🛠️' },
              { id: 'advanced', label: 'Advanced Features', icon: '🌟' },
              { id: 'collaboration', label: 'Collaboration', icon: '🤝' },
              { id: 'analytics', label: 'Analytics', icon: '📊' },
              { id: 'mikey', label: 'MIKEY AI', icon: '🤖' },
              { id: 'ai-tools', label: 'AI Tools', icon: '🛠️' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setState(prev => ({ ...prev, currentTab: tab.id as any }))}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  state.currentTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {state.currentTab === 'dashboard' && (
          <PlaygroundDashboard 
            apiEndpoints={apiEndpoints}
            onSelectEndpoint={(endpoint) => setState(prev => ({ ...prev, selectedEndpoint: endpoint }))}
            isAuthenticated={state.isAuthenticated}
          />
        )}
        
        {state.currentTab === 'tester' && (
          <ApiTester
            apiEndpoints={apiEndpoints}
            selectedEndpoint={state.selectedEndpoint}
            onApiCall={handleApiCall}
            lastResponse={state.lastResponse}
            error={state.error}
            isAuthenticated={state.isAuthenticated}
          />
        )}
        
        {state.currentTab === 'generator' && (
          <CodeGenerator
            apiEndpoints={apiEndpoints}
            selectedEndpoint={state.selectedEndpoint}
            generateCodeExample={generateCodeExample}
            isAuthenticated={state.isAuthenticated}
          />
        )}
        
        {state.currentTab === 'auth' && (
          <Authentication
            apiKey={state.apiKey}
            onApiKeyChange={(apiKey) => setState(prev => ({ 
              ...prev, 
              apiKey, 
              isAuthenticated: !!apiKey 
            }))}
          />
        )}
        
        {state.currentTab === 'docs' && (
          <Documentation apiEndpoints={apiEndpoints} />
        )}
        
        {state.currentTab === 'sdk' && (
          <SdkIntegrationExamples />
        )}
        
        {state.currentTab === 'advanced' && (
          <AdvancedFeatures />
        )}
        
        {state.currentTab === 'collaboration' && (
          <RealTimeCollaboration />
        )}
        
        {state.currentTab === 'analytics' && (
          <ApiAnalytics />
        )}
        
        {state.currentTab === 'mikey' && (
          <MikeyIntegrationShowcase />
        )}
        
        {state.currentTab === 'ai-tools' && (
          <AIToolsIntegration />
        )}
      </div>

      {/* Error Handler */}
      {state.error && (
        <ErrorHandler 
          error={state.error}
          onClose={() => setState(prev => ({ ...prev, error: null }))}
        />
      )}
    </div>
  );
};

export default QuantDeskPlayground;
