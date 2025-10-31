// Enhanced MIKEY AI Chat with Real-time Updates and Conversation Memory
// Story 2.2: AC 2.2.4 - Frontend Integration

import React, { useState, useEffect, useRef } from 'react';
import { 
  enhancedMikeyAI, 
  EnhancedMessage, 
  ConversationSession, 
  UserPreferences 
} from '../services/mikeyAIEnhanced';

interface EnhancedMikeyAIChatProps {
  onClose?: () => void;
}

export const EnhancedMikeyAIChat: React.FC<EnhancedMikeyAIChatProps> = ({ onClose }) => {
  const [messages, setMessages] = useState<EnhancedMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [sessionInfo, setSessionInfo] = useState<ConversationSession | null>(null);
  const [userPreferences, setUserPreferences] = useState<UserPreferences>(
    enhancedMikeyAI.getPreferences()
  );
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize service
  useEffect(() => {
    // Load saved conversation
    const savedSession = enhancedMikeyAI.loadSession();
    if (savedSession && savedSession.messages.length > 0) {
      setSessionInfo(savedSession);
      setMessages(savedSession.messages);
    } else {
      enhancedMikeyAI.startNewSession();
      setSessionInfo(enhancedMikeyAI.getCurrentSession());
    }

    // Register for real-time updates
    enhancedMikeyAI.onMessage((message: EnhancedMessage) => {
      setMessages(prev => [...prev, message]);
    });

    enhancedMikeyAI.onStatus((status) => {
      setIsConnected(status === 'connected');
    });

    // Cleanup on unmount
    return () => {
      enhancedMikeyAI.disconnect();
    };
  }, []);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: EnhancedMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const aiMessage = await enhancedMikeyAI.queryAI(inputMessage.trim());
      
      // Message will be added via WebSocket or return value
      if (!isConnected) {
        setMessages(prev => [...prev, aiMessage]);
      }
      
      setSessionInfo(enhancedMikeyAI.getCurrentSession());
    } catch (error) {
      const errorMessage: EnhancedMessage = {
        id: `error-${Date.now()}`,
        type: 'system',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
        confidence: 0
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearConversation = () => {
    enhancedMikeyAI.startNewSession();
    setMessages([]);
    setSessionInfo(enhancedMikeyAI.getCurrentSession());
  };

  const updatePreferences = (prefs: Partial<UserPreferences>) => {
    const newPrefs = { ...userPreferences, ...prefs };
    setUserPreferences(newPrefs);
    enhancedMikeyAI.updatePreferences(prefs);
  };

  const quickActions = [
    { label: 'Market Analysis', query: 'Analyze the current market for SOL and ETH' },
    { label: 'Portfolio Review', query: 'Review my portfolio and suggest optimizations' },
    { label: 'Risk Assessment', query: 'Assess current market risks for DeFi trading' },
    { label: 'Trading Strategy', query: 'Suggest a trading strategy for current conditions' }
  ];

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header with Status */}
      <div className="px-4 py-3 border-b border-border flex justify-between items-center bg-card">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🤖</span>
          <div>
            <div className="font-semibold">MIKEY AI Assistant</div>
            <div className="text-xs text-muted-foreground">
              {isConnected ? 'Connected' : 'Disconnected'}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          {onClose && (
            <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
              ×
            </button>
          )}
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex flex-col ${message.type === 'user' ? 'items-end' : 'items-start'}`}
          >
            <div
              className={`max-w-[80%] px-4 py-3 rounded-lg ${
                message.type === 'user'
                  ? 'bg-primary text-white'
                  : 'bg-card text-foreground'
              }`}
            >
              {message.content}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {message.timestamp.toLocaleTimeString()}
              {message.provider && ` • ${message.provider}`}
              {message.cost && ` • $${message.cost.toFixed(4)}`}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            MIKEY is thinking...
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Quick Actions */}
      {messages.length <= 1 && (
        <div className="px-4 py-3 border-t border-border bg-card">
          <div className="text-xs text-muted-foreground mb-2">Quick Actions:</div>
          <div className="flex gap-2 flex-wrap">
            {quickActions.map((action, index) => (
              <button
                key={index}
                onClick={() => setInputMessage(action.query)}
                className="px-3 py-1 text-xs bg-background border border-border rounded-md hover:bg-primary hover:text-white transition-colors"
              >
                {action.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="p-4 border-t border-border bg-card">
        <div className="flex gap-2">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask MIKEY anything about trading..."
            disabled={isLoading}
            className="flex-1 px-3 py-2 border border-border rounded-lg bg-background text-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary"
            rows={1}
            style={{ minHeight: '40px', maxHeight: '120px' }}
          />
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className="px-4 py-2 bg-primary text-white rounded-lg disabled:bg-muted disabled:text-muted-foreground hover:bg-primary/90 transition-colors"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default EnhancedMikeyAIChat;

