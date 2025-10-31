import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useWalletAuth } from '../hooks/useWalletAuth';
import { chatService } from '../services/chatService';
import { TickerText, ClickableTicker } from './ClickableTicker';

interface Message {
  id: string;
  author_pubkey: string;
  message: string;
  mentions: string[];
  created_at: string;
}

interface ChatChannel {
  id: string;
  name: string;
  description: string;
}

interface ChatWindowProps {
  channelId?: string;
}

const ChatWindow: React.FC<ChatWindowProps> = ({ channelId }) => {
  const { user, isAuthenticated, isLoading } = useWalletAuth();
  const [channels, setChannels] = useState<ChatChannel[]>([]);
  const [currentChannelId, setCurrentChannelId] = useState<string>(channelId || 'global');
  const [messages, setMessages] = useState<Message[]>([]);
  const [onlineUsers, setOnlineUsers] = useState<string[]>([]);
  const [showSidebar, setShowSidebar] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const ws = useRef<WebSocket | null>(null);
  
  // Debug logging
  React.useEffect(() => {
    console.log('ChatWindow Debug:', { 
      user, 
      isAuthenticated, 
      isLoading,
      hasUser: !!user,
      userWallet: user?.wallet_pubkey 
    });
  }, [user, isAuthenticated, isLoading]);

  const fetchChannels = useCallback(async () => {
    // Only fetch channels if authenticated
    if (!isAuthenticated || !user) {
      return;
    }
    
    try {
      const channelsData = await chatService.getChannels();
      setChannels(channelsData);

      // If no specific channelId provided, use the first channel
      if (!channelId && channelsData.length > 0) {
        setCurrentChannelId(channelsData[0].id);
      }
    } catch (error) {
      console.error('Error fetching channels:', error);
    }
  }, [channelId, isAuthenticated, user]);

  const fetchChatHistory = useCallback(async () => {
    if (!user?.wallet_pubkey || !currentChannelId) return;
    try {
      const history = await chatService.getChatHistory(currentChannelId, user.wallet_pubkey);
      setMessages(history);
    } catch (error) {
      console.error('Error fetching chat history:', error);
    }
  }, [currentChannelId, user]);

  useEffect(() => {
    fetchChannels();
  }, [fetchChannels]);

  useEffect(() => {
    fetchChatHistory();
  }, [fetchChatHistory]);

  // WebSocket connection and real-time updates
  useEffect(() => {
    if (!user?.wallet_pubkey || !currentChannelId) return;

    const connectWebSocket = async () => {
      try {
        // Get a chat token for WebSocket authentication
        const { token } = await chatService.getChatToken(user.wallet_pubkey);
        if (!token) {
          console.error('Failed to get chat token');
          return;
        }

        const wsUrl = (import.meta.env.VITE_WS_URL || 'ws://localhost:3002') + `?token=${token}&channelId=${currentChannelId}`;
        ws.current = new WebSocket(wsUrl);

        ws.current.onopen = () => {
          console.log(`WebSocket connected to channel ${currentChannelId}`);
        };

        ws.current.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === 'chat_message') {
            setMessages((prevMessages) => [...prevMessages, data]);
          } else if (data.type === 'presence_update') {
            setOnlineUsers(data.payload.onlineUsers);
          }
        };

        ws.current.onclose = () => {
          console.log(`WebSocket disconnected from channel ${currentChannelId}. Attempting to reconnect in 5 seconds...`);
          setTimeout(connectWebSocket, 5000);
        };

        ws.current.onerror = (error) => {
          console.error('WebSocket error:', error);
          ws.current?.close();
        };
      } catch (error) {
        console.error('Failed to establish WebSocket connection:', error);
        setTimeout(connectWebSocket, 5000);
      }
    };

    connectWebSocket();

    return () => {
      ws.current?.close();
    };
  }, [currentChannelId, user]);

  // Helper function to parse mentions and ticker symbols (Godel Terminal style)
  const parseMessageContent = (message: string) => {
    // Split by mentions (@), $ tickers, or regular ticker symbols
    // First handle @mentions
    const parts = message.split(/(@[A-Za-z0-9]{32,44}|\$[A-Z]+)/g);
    
    return parts.map((part, index) => {
      const uniqueKey = `${message}-${index}-${part.slice(0, 10)}`;
      
      // Check if it's a mention
      if (part.startsWith('@') && part.length >= 32) {
        return (
          <span key={uniqueKey} style={{ color: 'var(--primary-500)', fontWeight: 'bold' }}>
            {part}
          </span>
        );
      }
      
      // Check if it's a $ ticker (Godel Terminal style)
      if (part.startsWith('$') && part.length > 1) {
        const ticker = part.substring(1);
        return (
          <ClickableTicker key={uniqueKey} symbol={ticker}>
            {part}
          </ClickableTicker>
        );
      }
      
      // Regular text - parse for other tickers
      return (
        <TickerText key={uniqueKey} text={part} />
      );
    });
  };

  // Scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Add some default channels if we don't have any yet
  const displayChannels = channels.length > 0 ? channels : [
    { id: 'global', name: 'global', description: 'General chat' },
    { id: 'trading', name: 'trading', description: 'Trading discussions' },
    { id: 'general', name: 'general', description: 'General discussions' }
  ];

  // Show loading or not authenticated states
  if (isLoading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', backgroundColor: 'var(--bg-primary)' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ display: 'inline-block', width: '32px', height: '32px', border: '2px solid var(--bg-tertiary)', borderBottomColor: 'var(--primary-500)', borderRadius: '50%', animation: 'spin 1s linear infinite', marginBottom: '16px' }}></div>
          <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>Connecting...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated || !user) {
    return (
      <div style={{ display: 'flex', height: '100%', backgroundColor: 'var(--bg-primary)' }}>
        {/* Channels Sidebar */}
        {showSidebar && (
          <div style={{ width: '192px', backgroundColor: 'var(--bg-secondary)', borderRight: '1px solid var(--bg-tertiary)', display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '12px', borderBottom: '1px solid var(--bg-tertiary)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ color: 'var(--text-primary)', fontSize: '14px', fontWeight: '600' }}>CHANNELS</span>
                <button 
                  onClick={() => setShowSidebar(false)}
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: '12px', cursor: 'pointer' }}
                >
                  ✕
                </button>
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
                👥 0 online
              </div>
            </div>

            <div style={{ flex: 1, overflowY: 'auto' }}>
              {displayChannels.map((channel) => (
                <button
                  key={channel.id}
                  onClick={() => setCurrentChannelId(channel.id)}
                  style={{
                    width: '100%',
                    textAlign: 'left',
                    padding: '8px 12px',
                    fontSize: '14px',
                    backgroundColor: currentChannelId === channel.id ? 'var(--primary-500)' : 'transparent',
                    color: currentChannelId === channel.id ? 'var(--text-primary)' : 'var(--text-secondary)',
                    fontWeight: currentChannelId === channel.id ? '600' : '400',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                  onMouseEnter={(e) => {
                    if (currentChannelId !== channel.id) {
                      e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)';
                      e.currentTarget.style.color = 'var(--text-primary)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (currentChannelId !== channel.id) {
                      e.currentTarget.style.backgroundColor = 'transparent';
                      e.currentTarget.style.color = 'var(--text-secondary)';
                    }
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span style={{ marginRight: '4px' }}>#</span>
                    <span>{channel.name}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Chat Area */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', backgroundColor: 'var(--bg-primary)' }}>
          <div style={{ backgroundColor: 'var(--bg-secondary)', borderBottom: '1px solid var(--bg-tertiary)', padding: '8px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              {!showSidebar && (
                <button 
                  onClick={() => setShowSidebar(true)}
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '16px' }}
                  title="Show channels"
                >
                  ☰
                </button>
              )}
              <h3 style={{ color: 'var(--text-primary)', fontWeight: '600', fontSize: '14px' }}>
                #{currentChannelId}
              </h3>
            </div>
          </div>

          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ textAlign: 'center', backgroundColor: 'var(--bg-secondary)', padding: '24px', borderRadius: '8px', border: '1px solid var(--bg-tertiary)', maxWidth: '400px' }}>
              <p style={{ color: 'var(--text-primary)', fontSize: '14px', fontWeight: '600', marginBottom: '8px' }}>🔐 Authentication Required</p>
              <p style={{ color: 'var(--text-muted)', fontSize: '12px', marginBottom: '16px' }}>
                Click the wallet button in the top right to authenticate and join the chat.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', height: '100%', backgroundColor: 'var(--bg-primary)' }}>
      {/* Channels Sidebar - Godel Terminal Style */}
      {showSidebar && (
        <div style={{ width: '192px', backgroundColor: 'var(--bg-secondary)', borderRight: '1px solid var(--bg-tertiary)', display: 'flex', flexDirection: 'column' }}>
          {/* Sidebar Header */}
          <div style={{ padding: '12px', borderBottom: '1px solid var(--bg-tertiary)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ color: 'var(--text-primary)', fontSize: '14px', fontWeight: '600' }}>CHANNELS</span>
              <button 
                onClick={() => setShowSidebar(false)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: '12px', cursor: 'pointer' }}
              >
                ✕
              </button>
            </div>
            <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
              👥 {onlineUsers.length} online
            </div>
          </div>

          {/* Channel List */}
          <div style={{ flex: 1, overflowY: 'auto' }}>
            {displayChannels.map((channel) => (
              <button
                key={channel.id}
                onClick={() => setCurrentChannelId(channel.id)}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 12px',
                  fontSize: '14px',
                  backgroundColor: currentChannelId === channel.id ? 'var(--primary-500)' : 'transparent',
                  color: currentChannelId === channel.id ? 'var(--text-primary)' : 'var(--text-secondary)',
                  fontWeight: currentChannelId === channel.id ? '600' : '400',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onMouseEnter={(e) => {
                  if (currentChannelId !== channel.id) {
                    e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)';
                    e.currentTarget.style.color = 'var(--text-primary)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (currentChannelId !== channel.id) {
                    e.currentTarget.style.backgroundColor = 'transparent';
                    e.currentTarget.style.color = 'var(--text-secondary)';
                  }
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={{ marginRight: '4px' }}>#</span>
                  <span>{channel.name}</span>
                  {channel.name === 'general' && (
                    <span style={{ marginLeft: 'auto', fontSize: '10px', backgroundColor: 'var(--success-500)', color: 'var(--text-primary)', padding: '2px 4px', borderRadius: '4px' }}>LIVE</span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Chat Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', backgroundColor: 'var(--bg-primary)' }}>
        {/* Header */}
        <div style={{ backgroundColor: 'var(--bg-secondary)', borderBottom: '1px solid var(--bg-tertiary)', padding: '8px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {!showSidebar && (
              <button 
                onClick={() => setShowSidebar(true)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '16px' }}
                title="Show channels"
              >
                ☰
              </button>
            )}
            <h3 style={{ color: 'var(--text-primary)', fontWeight: '600', fontSize: '14px' }}>
              #{currentChannelId}
            </h3>
            <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
              {messages.length} messages
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '12px', color: 'var(--success-500)' }}>●</span>
            <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
              {onlineUsers.length} online
            </span>
          </div>
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
          {messages.length === 0 ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
              <div style={{ textAlign: 'center' }}>
                <p style={{ fontSize: '14px' }}>No messages yet</p>
                <p style={{ fontSize: '12px', marginTop: '4px' }}>Start the conversation!</p>
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} style={{ marginBottom: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'start', gap: '8px' }}>
                  <span style={{ fontWeight: 'bold', color: 'var(--primary-500)', fontSize: '12px' }}>
                    {msg.author_pubkey.slice(0, 8)}...{msg.author_pubkey.slice(-4)}
                  </span>
                  <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
                    {new Date(msg.created_at).toLocaleTimeString()}
                  </span>
                </div>
                <div style={{ color: 'var(--text-primary)', fontSize: '14px', marginLeft: '8px' }}>
                  {parseMessageContent(msg.message)}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>
    </div>
  );
};

export default ChatWindow;
