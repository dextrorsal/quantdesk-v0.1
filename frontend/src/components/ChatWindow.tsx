import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useWalletAuth } from '../hooks/useWalletAuth';
import { chatService } from '../services/chatService';
import { TickerText } from './ClickableTicker';

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
  const { user } = useWalletAuth();
  const [channels, setChannels] = useState<ChatChannel[]>([]);
  const [currentChannelId, setCurrentChannelId] = useState<string>(channelId || 'global');
  const [messages, setMessages] = useState<Message[]>([]);
  const [onlineUsers, setOnlineUsers] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const ws = useRef<WebSocket | null>(null);

  const fetchChannels = useCallback(async () => {
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
  }, [channelId]);

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

  // Helper function to parse mentions and ticker symbols
  const parseMessageContent = (message: string) => {
    const mentionRegex = /@([A-Za-z0-9]{32,44})/g;
    const parts = message.split(mentionRegex);

    return parts.map((part, index) => {
      const uniqueKey = `${message}-${index}-${part.slice(0, 10)}`;
      if (part.match(mentionRegex)) {
        return (
          <span key={uniqueKey} className="text-blue-400 font-bold">
            @{part}
          </span>
        );
      }
      return (
        <TickerText key={uniqueKey} text={part} />
      );
    });
  };

  // Scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (!user) {
    return <div className="text-white">Please connect your wallet to join the chat.</div>;
  }

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg p-4">
      {/* Channel Switcher */}
      <div className="mb-4">
        <select
          value={currentChannelId}
          onChange={(e) => setCurrentChannelId(e.target.value)}
          className="bg-gray-700 text-white px-3 py-2 rounded border border-gray-600"
          aria-label="Select chat channel"
        >
          {channels.map((channel) => (
            <option key={channel.id} value={channel.id}>
              #{channel.name}
            </option>
          ))}
        </select>
      </div>

      {/* Online Users */}
      <div className="text-gray-400 text-sm mb-4">
        Online: {onlineUsers.length} users
      </div>

      {/* Message List */}
      <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 mb-4">
        {messages.map((msg) => (
          <div key={msg.id} className="mb-2">
            <span className="font-bold text-blue-400">{msg.author_pubkey.slice(0, 6)}...{msg.author_pubkey.slice(-4)}:</span>
            <span className="text-white ml-2">
              {parseMessageContent(msg.message)}
            </span>
            <span className="text-gray-500 text-xs ml-2">{new Date(msg.created_at).toLocaleTimeString()}</span>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};

export default ChatWindow;
