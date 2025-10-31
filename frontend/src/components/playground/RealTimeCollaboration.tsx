// QuantDesk Developer Playground - Real-time Collaboration Component
// Phase 3: Advanced Capabilities (Beyond Drift)
// Strategy: "More Open Than Drift" - Enterprise collaboration features

import React, { useState, useEffect } from 'react';

interface CollaborationSession {
  id: string;
  name: string;
  participants: string[];
  status: 'active' | 'inactive';
  createdAt: Date;
  lastActivity: Date;
}

interface CollaborationMessage {
  id: string;
  userId: string;
  userName: string;
  type: 'api_call' | 'message' | 'code_share' | 'error';
  content: any;
  timestamp: Date;
}

export const RealTimeCollaboration: React.FC = () => {
  const [sessions, setSessions] = useState<CollaborationSession[]>([]);
  const [currentSession, setCurrentSession] = useState<CollaborationSession | null>(null);
  const [messages, setMessages] = useState<CollaborationMessage[]>([]);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [newMessage, setNewMessage] = useState<string>('');

  // Mock WebSocket connection for demonstration
  useEffect(() => {
    // Simulate WebSocket connection
    const connectWebSocket = () => {
      setIsConnected(true);
      console.log('WebSocket connected for real-time collaboration');
    };

    connectWebSocket();

    // Mock session data
    setSessions([
      {
        id: 'session-1',
        name: 'Team Trading Session',
        participants: ['user1', 'user2', 'user3'],
        status: 'active',
        createdAt: new Date(Date.now() - 3600000),
        lastActivity: new Date(Date.now() - 300000)
      },
      {
        id: 'session-2',
        name: 'API Testing Workshop',
        participants: ['user4', 'user5'],
        status: 'active',
        createdAt: new Date(Date.now() - 7200000),
        lastActivity: new Date(Date.now() - 600000)
      }
    ]);

    // Mock messages
    setMessages([
      {
        id: 'msg-1',
        userId: 'user1',
        userName: 'Alice',
        type: 'api_call',
        content: {
          endpoint: '/api/positions',
          method: 'GET',
          response: { success: true, data: [] }
        },
        timestamp: new Date(Date.now() - 300000)
      },
      {
        id: 'msg-2',
        userId: 'user2',
        userName: 'Bob',
        type: 'message',
        content: 'Great! The API is working perfectly.',
        timestamp: new Date(Date.now() - 240000)
      },
      {
        id: 'msg-3',
        userId: 'user1',
        userName: 'Alice',
        type: 'code_share',
        content: {
          language: 'typescript',
          code: 'const position = await client.openPosition({...});'
        },
        timestamp: new Date(Date.now() - 180000)
      }
    ]);

    return () => {
      setIsConnected(false);
    };
  }, []);

  const createSession = () => {
    const newSession: CollaborationSession = {
      id: `session-${Date.now()}`,
      name: 'New Collaboration Session',
      participants: ['current-user'],
      status: 'active',
      createdAt: new Date(),
      lastActivity: new Date()
    };
    setSessions(prev => [...prev, newSession]);
    setCurrentSession(newSession);
  };

  const joinSession = (session: CollaborationSession) => {
    setCurrentSession(session);
  };

  const sendMessage = () => {
    if (!newMessage.trim() || !currentSession) return;

    const message: CollaborationMessage = {
      id: `msg-${Date.now()}`,
      userId: 'current-user',
      userName: 'You',
      type: 'message',
      content: newMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, message]);
    setNewMessage('');
  };

  const shareApiCall = (endpoint: string, method: string, response: any) => {
    if (!currentSession) return;

    const message: CollaborationMessage = {
      id: `msg-${Date.now()}`,
      userId: 'current-user',
      userName: 'You',
      type: 'api_call',
      content: {
        endpoint,
        method,
        response
      },
      timestamp: new Date()
    };

    setMessages(prev => [...prev, message]);
  };

  const shareCode = (language: string, code: string) => {
    if (!currentSession) return;

    const message: CollaborationMessage = {
      id: `msg-${Date.now()}`,
      userId: 'current-user',
      userName: 'You',
      type: 'code_share',
      content: {
        language,
        code
      },
      timestamp: new Date()
    };

    setMessages(prev => [...prev, message]);
  };

  const getMessageIcon = (type: string) => {
    switch (type) {
      case 'api_call': return '🔗';
      case 'message': return '💬';
      case 'code_share': return '📝';
      case 'error': return '❌';
      default: return '📄';
    }
  };

  const formatMessageContent = (message: CollaborationMessage) => {
    switch (message.type) {
      case 'api_call':
        return (
          <div className="bg-blue-50 p-3 rounded">
            <div className="font-medium text-blue-900">
              API Call: {message.content.method} {message.content.endpoint}
            </div>
            <pre className="text-xs text-blue-700 mt-1">
              {JSON.stringify(message.content.response, null, 2)}
            </pre>
          </div>
        );
      case 'code_share':
        return (
          <div className="bg-gray-50 p-3 rounded">
            <div className="font-medium text-gray-900">
              Code ({message.content.language})
            </div>
            <pre className="text-xs text-gray-700 mt-1">
              {message.content.code}
            </pre>
          </div>
        );
      default:
        return <div className="text-gray-800">{message.content}</div>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              🤝 Real-time Collaboration
            </h2>
            <p className="text-gray-600">
              Shared testing sessions and collaborative development - Beyond Drift's capabilities
            </p>
          </div>
          <div className="flex items-center">
            <div className={`w-3 h-3 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm text-gray-600">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sessions List */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Active Sessions</h3>
            <button
              onClick={createSession}
              className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
            >
              + New Session
            </button>
          </div>
          
          <div className="space-y-3">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                  currentSession?.id === session.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => joinSession(session)}
              >
                <div className="flex items-center justify-between">
                  <h4 className="font-medium text-gray-900">{session.name}</h4>
                  <span className={`w-2 h-2 rounded-full ${
                    session.status === 'active' ? 'bg-green-500' : 'bg-gray-400'
                  }`}></span>
                </div>
                <div className="text-sm text-gray-600 mt-1">
                  {session.participants.length} participants
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Last activity: {session.lastActivity.toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Chat Interface */}
        <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow-sm border">
          {currentSession ? (
            <div className="h-96 flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  {currentSession.name}
                </h3>
                <div className="text-sm text-gray-600">
                  {currentSession.participants.length} participants
                </div>
              </div>
              
              {/* Messages */}
              <div className="flex-1 overflow-y-auto space-y-3 mb-4">
                {messages.map((message) => (
                  <div key={message.id} className="flex items-start space-x-3">
                    <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
                      <span className="text-sm">{getMessageIcon(message.type)}</span>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium text-gray-900">{message.userName}</span>
                        <span className="text-xs text-gray-500">
                          {message.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="mt-1">
                        {formatMessageContent(message)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              {/* Message Input */}
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  placeholder="Type a message..."
                  className="flex-1 p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                />
                <button
                  onClick={sendMessage}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  Send
                </button>
              </div>
            </div>
          ) : (
            <div className="h-96 flex items-center justify-center text-gray-500">
              <div className="text-center">
                <div className="text-4xl mb-4">🤝</div>
                <h3 className="text-lg font-medium mb-2">No Session Selected</h3>
                <p>Select a session from the list to start collaborating</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Collaboration Features */}
      <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
        <h3 className="text-lg font-semibold text-blue-900 mb-4">
          🚀 Collaboration Features
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Real-time Features</h4>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• Multi-user testing sessions</li>
              <li>• Shared API key management</li>
              <li>• Real-time code sharing</li>
              <li>• Collaborative debugging</li>
              <li>• Session recording and playback</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Team Management</h4>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• Team workspace management</li>
              <li>• Role-based permissions</li>
              <li>• Activity tracking</li>
              <li>• Session history</li>
              <li>• Export collaboration logs</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Competitive Advantages */}
      <div className="bg-green-50 rounded-lg p-6 border border-green-200">
        <h3 className="text-lg font-semibold text-green-900 mb-4">
          🎯 Collaboration Advantages Over Drift
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-green-900 mb-2">Drift Protocol</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Individual testing only</li>
              <li>• No collaboration features</li>
              <li>• No team management</li>
              <li>• No session sharing</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-green-900 mb-2">QuantDesk</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Multi-user collaboration</li>
              <li>• Real-time session sharing</li>
              <li>• Team workspace management</li>
              <li>• Collaborative debugging</li>
              <li>• Session recording</li>
              <li>• Activity tracking</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
