// QuantDesk Developer Playground - Code Generator Component
// Phase 2: Enhanced Features (Exceed Drift's Capabilities)
// Strategy: "More Open Than Drift" - Multi-language support

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

interface CodeGeneratorProps {
  apiEndpoints: ApiCategory[];
  selectedEndpoint: string | null;
  generateCodeExample: (endpoint: string, method: string, params: any, language: string) => string;
  isAuthenticated: boolean;
}

export const CodeGenerator: React.FC<CodeGeneratorProps> = ({
  apiEndpoints,
  selectedEndpoint,
  generateCodeExample,
  isAuthenticated
}) => {
  const [selectedLanguage, setSelectedLanguage] = useState<string>('javascript');
  const [requestParams, setRequestParams] = useState<any>({});
  const [generatedCode, setGeneratedCode] = useState<string>('');
  const [copied, setCopied] = useState<boolean>(false);

  const languages = [
    { id: 'javascript', name: 'JavaScript', icon: '🟨', description: 'Web development' },
    { id: 'python', name: 'Python', icon: '🐍', description: 'Data science & automation' },
    { id: 'typescript', name: 'TypeScript', icon: '🔷', description: 'Type-safe JavaScript' },
    { id: 'rust', name: 'Rust', icon: '🦀', description: 'Systems programming' },
    { id: 'go', name: 'Go', icon: '🐹', description: 'Backend services' }
  ];

  // Find the selected endpoint details
  const endpointDetails = apiEndpoints
    .flatMap(cat => cat.endpoints)
    .find(ep => ep.path === selectedEndpoint);

  // Generate code when dependencies change
  useEffect(() => {
    if (selectedEndpoint && endpointDetails) {
      const code = generateCodeExample(
        selectedEndpoint,
        endpointDetails.method,
        requestParams,
        selectedLanguage
      );
      setGeneratedCode(code);
    }
  }, [selectedEndpoint, selectedLanguage, requestParams, generateCodeExample, endpointDetails]);

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

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(generatedCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  const downloadCode = () => {
    const extension = selectedLanguage === 'typescript' ? 'ts' : selectedLanguage;
    const filename = `quantdesk-api-example.${extension}`;
    
    const blob = new Blob([generatedCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          ⚡ Code Generator
        </h2>
        <p className="text-gray-600">
          Generate code examples in 5+ programming languages - Exceeding Drift's single language support
        </p>
      </div>

      {/* Language Selection */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Select Programming Language
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {languages.map((lang) => (
            <button
              key={lang.id}
              onClick={() => setSelectedLanguage(lang.id)}
              className={`p-4 rounded-lg border-2 transition-all ${
                selectedLanguage === lang.id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="text-center">
                <div className="text-3xl mb-2">{lang.icon}</div>
                <div className="font-medium text-gray-900">{lang.name}</div>
                <div className="text-sm text-gray-600">{lang.description}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Endpoint Selection */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          API Endpoint
        </h3>
        
        {selectedEndpoint ? (
          <div className="space-y-4">
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <span className="px-3 py-1 bg-blue-100 text-blue-800 text-sm font-medium rounded">
                    {endpointDetails?.method.toUpperCase()}
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
                <p className="text-sm text-gray-600 mt-2">
                  {endpointDetails.description}
                </p>
              )}
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="text-4xl mb-4">🔗</div>
            <h4 className="text-lg font-medium text-gray-900 mb-2">
              No Endpoint Selected
            </h4>
            <p className="text-gray-600 mb-4">
              Please select an endpoint from the Dashboard to generate code
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

      {/* Request Parameters */}
      {selectedEndpoint && endpointDetails && (endpointDetails.method === 'POST' || endpointDetails.method === 'PUT') && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Request Parameters
          </h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-sm text-gray-600">
                Add parameters to include in the generated code
              </p>
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
        </div>
      )}

      {/* Generated Code */}
      {generatedCode && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">
              Generated Code ({languages.find(l => l.id === selectedLanguage)?.name})
            </h3>
            <div className="flex items-center space-x-2">
              <button
                onClick={copyToClipboard}
                className={`px-3 py-1 text-sm rounded ${
                  copied 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {copied ? '✅ Copied!' : '📋 Copy'}
              </button>
              <button
                onClick={downloadCode}
                className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded hover:bg-blue-200"
              >
                💾 Download
              </button>
            </div>
          </div>
          
          <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <pre className="text-green-400 text-sm">
              <code>{generatedCode}</code>
            </pre>
          </div>
        </div>
      )}

      {/* Language Comparison */}
      <div className="bg-green-50 rounded-lg p-6 border border-green-200">
        <h3 className="text-lg font-semibold text-green-900 mb-4">
          🎯 Multi-Language Advantage Over Drift
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-green-900 mb-2">Drift Protocol</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• JavaScript only</li>
              <li>• Basic code examples</li>
              <li>• Limited language support</li>
              <li>• Manual code writing</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-green-900 mb-2">QuantDesk</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• 5+ programming languages</li>
              <li>• Auto-generated code</li>
              <li>• Comprehensive language support</li>
              <li>• Copy & download functionality</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Usage Examples */}
      <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
        <h3 className="text-lg font-semibold text-blue-900 mb-4">
          💡 Usage Examples
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Web Development</h4>
            <p className="text-sm text-blue-700">
              Use JavaScript/TypeScript examples for frontend integration
            </p>
          </div>
          
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Data Science</h4>
            <p className="text-sm text-blue-700">
              Use Python examples for data analysis and automation
            </p>
          </div>
          
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Backend Services</h4>
            <p className="text-sm text-blue-700">
              Use Go/Rust examples for high-performance backend services
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
