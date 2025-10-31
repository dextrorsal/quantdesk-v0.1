// MIKEY AI Trading Assistant Chat Component
// Dedicated AI chat interface for the Pro Terminal

import React, { useState, useEffect, useRef } from 'react';
import { mikeyAI, MikeyAIResponse, LLMStatus } from '../services/mikeyAI';

interface Message {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  confidence?: number;
  sources?: string[];
}

interface MikeyAIChatProps {
  onClose?: () => void;
}

export const MikeyAIChat: React.FC<MikeyAIChatProps> = ({ onClose }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [serviceStatus, setServiceStatus] = useState(false);
  const [llmStatus, setLlmStatus] = useState<LLMStatus[]>([]);
  const [error, setError] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Check service health on mount
  useEffect(() => {
    const checkHealth = async () => {
      const isHealthy = await mikeyAI.healthCheck();
      setServiceStatus(isHealthy);
      
      if (isHealthy) {
        const status = await mikeyAI.getLLMStatus();
        setLlmStatus(status);
        
        // Add welcome message
        setMessages([{
          id: 'welcome',
          type: 'ai',
          content: `🤖 **MIKEY AI Trading Assistant Online**

I'm your AI trading companion, powered by multiple LLM providers:
${status.map(llm => `• ${llm.name} (${llm.status})`).join('\n')}

**What I can help you with:**
• Market analysis and sentiment
• Trading strategy recommendations  
• Risk assessment and portfolio optimization
• Real-time market insights
• Technical analysis and patterns

Ask me anything about crypto trading!`,
          timestamp: new Date(),
          confidence: 1.0,
          sources: ['MIKEY-AI Service']
        }]);
      } else {
        setError('MIKEY-AI service is not available. Please ensure the AI service is running.');
        setMessages([{
          id: 'error',
          type: 'ai',
          content: '❌ **Service Unavailable**\n\nMIKEY-AI service is not running. Please ensure the AI service is started.',
          timestamp: new Date(),
          confidence: 0,
          sources: ['Error']
        }]);
      }
    };
    
    checkHealth();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading || !serviceStatus) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setError('');

    try {
      const response: MikeyAIResponse = await mikeyAI.queryAI(inputMessage.trim());
      
      const aiMessage: Message = {
        id: `ai-${Date.now()}`,
        type: 'ai',
        content: response.success ? response.data.response : 'Sorry, I encountered an error processing your request.',
        timestamp: new Date(),
        confidence: response.success ? response.data.confidence : 0,
        sources: response.success ? response.data.sources : ['Error']
      };

      setMessages(prev => [...prev, aiMessage]);

      // Parse mock trade intents from AI response and persist demo position
      try {
        const content = aiMessage.content || '';
        const MOCK_KEY = 'quantdesk_mock_positions';
        const load = (): any[] => {
          try { const raw = localStorage.getItem(MOCK_KEY); return raw ? JSON.parse(raw) : []; } catch { return []; }
        };
        const save = (list: any[]) => localStorage.setItem(MOCK_KEY, JSON.stringify(list));
        
        // Format 1: Structured token e.g. QDX:OPEN_MOCK_POSITION {"pair":"SOL/USDT","side":"LONG","size":1}
        const tokenMatch = content.match(/QDX:OPEN_MOCK_POSITION\s*({[\s\S]*?})/i);
        // Format 2: Natural language e.g. "open a long on SOL size 1" / "open a short position on BTC 0.5"
        const nlMatch = content.match(/open\s+a\s+(long|short)\s+(?:position\s+)?on\s+\$?([a-z0-9]+)\b(?:[^0-9]+(\d+(?:\.\d+)?))?/i);
        let mock: any | null = null;
        if (tokenMatch) {
          try { mock = JSON.parse(tokenMatch[1]); } catch {}
        } else if (nlMatch) {
          const side = nlMatch[1].toUpperCase();
          const sym = nlMatch[2].toUpperCase();
          const size = nlMatch[3] ? parseFloat(nlMatch[3]) : 1;
          mock = { pair: sym.includes('/') ? sym : `${sym}/USDT`, side, size };
        }
        if (mock && mock.pair && mock.side) {
          const now = Date.now();
          const leverage = 5;
          const entry = 0; // filled by Positions via live price; store 0 as placeholder
          const record = { pair: mock.pair, side: mock.side, size: Number(mock.size || 1), entry, leverage, openedAt: now };
          const list = load();
          list.push(record);
          save(list);
        }
      } catch (e) {
        // Swallow parsing errors silently
      }
    } catch (error) {
      console.error('MIKEY-AI query error:', error);
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        type: 'ai',
        content: `❌ **Network Error**\n\nI encountered a network error: ${error instanceof Error ? error.message : 'Unknown error'}\n\nPlease check your connection and try again.`,
        timestamp: new Date(),
        confidence: 0,
        sources: ['Network Error']
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

  const quickActions = [
    { label: 'Market Analysis', query: 'What is the current market sentiment for SOL?' },
    { label: 'Portfolio Review', query: 'Analyze my portfolio and suggest optimizations' },
    { label: 'Risk Assessment', query: 'What are the current market risks I should be aware of?' },
    { label: 'Trading Strategy', query: 'Suggest a trading strategy for the current market conditions' }
  ];

  return (
    <div style={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: 'var(--bg-primary)',
      color: 'var(--text-primary)'
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--border-base)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        backgroundColor: 'var(--bg-secondary)',
        fontFamily: "'JetBrains Mono', monospace"
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '20px' }}>🤖</span>
          <div>
            <div style={{ fontSize: '14px', fontWeight: '600', fontFamily: "'JetBrains Mono', monospace" }}>MIKEY AI Assistant</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: "'JetBrains Mono', monospace" }}>
              {serviceStatus ? (
                `${llmStatus.length} LLM providers available`
              ) : (
                'Service offline'
              )}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: serviceStatus ? '#52c41a' : '#ff4d4f',
            animation: serviceStatus ? 'pulse 2s infinite' : 'none'
          }} />
          {onClose && (
            <button
              onClick={onClose}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                fontSize: '18px',
                padding: '4px',
                fontFamily: "'JetBrains Mono', monospace",
                transition: 'color 0.2s ease'
              }}
              onMouseOver={(e) => e.currentTarget.style.color = 'var(--primary-blue)'}
              onMouseOut={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
            >
              ×
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px'
      }}>
        {messages.map((message) => (
          <div
            key={message.id}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: message.type === 'user' ? 'flex-end' : 'flex-start',
              gap: '4px'
            }}
          >
            <div style={{
              maxWidth: '80%',
              padding: '12px 16px',
              border: '1px solid var(--border-base)',
              backgroundColor: message.type === 'user' 
                ? 'var(--primary-blue)' 
                : 'var(--bg-secondary)',
              color: message.type === 'user' 
                ? '#fff' 
                : 'var(--text-primary)',
              fontSize: '13px',
              lineHeight: '1.4',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontFamily: "'JetBrains Mono', monospace",
              transition: 'border-color 0.2s ease, background-color 0.2s ease'
            }}>
              {message.content}
            </div>
            <div style={{
              fontSize: '10px',
              color: 'var(--text-muted)',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontFamily: "'JetBrains Mono', monospace"
            }}>
              <span>{message.timestamp.toLocaleTimeString()}</span>
              {message.confidence !== undefined && (
                <span>Conf: {(message.confidence * 100).toFixed(0)}%</span>
              )}
              {message.sources && message.sources.length > 0 && (
                <span>Src: {message.sources.join(', ')}</span>
              )}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '12px 16px',
            backgroundColor: 'var(--bg-secondary)',
            borderRadius: '12px',
            fontSize: '14px',
            color: 'var(--text-muted)'
          }}>
            <div style={{
              width: '16px',
              height: '16px',
              border: '2px solid var(--primary-500)',
              borderTop: '2px solid transparent',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }} />
            MIKEY is thinking...
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Quick Actions */}
      {serviceStatus && messages.length <= 1 && (
        <div style={{
          padding: '12px 16px',
          borderTop: '1px solid var(--bg-tertiary)',
          backgroundColor: 'var(--bg-secondary)'
        }}>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>
            Quick Actions:
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {quickActions.map((action, index) => (
              <button
                key={index}
                onClick={() => setInputMessage(action.query)}
                style={{
                  padding: '6px 12px',
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--bg-tertiary)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '12px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.backgroundColor = 'var(--primary-500)';
                  e.currentTarget.style.color = '#fff';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.backgroundColor = 'var(--bg-primary)';
                  e.currentTarget.style.color = 'var(--text-primary)';
                }}
              >
                {action.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div style={{
        padding: '16px',
        borderTop: '1px solid var(--border-base)',
        backgroundColor: 'var(--bg-secondary)'
      }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={serviceStatus ? "Ask MIKEY anything about trading..." : "MIKEY service is offline"}
            disabled={!serviceStatus || isLoading}
            style={{
              flex: 1,
              padding: '12px',
              border: '1px solid var(--border-base)',
              borderRadius: '4px',
              backgroundColor: 'var(--bg-primary)',
              color: 'var(--text-primary)',
              fontSize: '13px',
              resize: 'none',
              minHeight: '40px',
              maxHeight: '120px',
              fontFamily: "'JetBrains Mono', monospace",
              transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
              outline: 'none'
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = 'var(--border-accent)';
              e.currentTarget.style.boxShadow = '0 0 0 2px rgba(59, 130, 246, 0.1)';
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = 'var(--border-base)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          />
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isLoading || !serviceStatus}
            style={{
              padding: '12px 16px',
              backgroundColor: serviceStatus && inputMessage.trim() && !isLoading 
                ? 'var(--primary-blue)' 
                : 'var(--bg-tertiary)',
              border: '1px solid var(--border-base)',
              borderRadius: '4px',
              color: serviceStatus && inputMessage.trim() && !isLoading 
                ? '#fff' 
                : 'var(--text-muted)',
              fontSize: '13px',
              fontFamily: "'JetBrains Mono', monospace",
              cursor: serviceStatus && inputMessage.trim() && !isLoading 
                ? 'pointer' 
                : 'not-allowed',
              transition: 'all 0.2s ease'
            }}
          >
            Send
          </button>
        </div>
        {error && (
          <div style={{
            marginTop: '8px',
            padding: '8px 12px',
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: '6px',
            color: '#dc2626',
            fontSize: '12px'
          }}>
            {error}
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default MikeyAIChat;
