import { useState, useEffect, useRef, useCallback } from 'react';

export interface WebSocketMessage {
  type: string;
  channel?: string;
  data?: any;
  [key: string]: any;
}

export interface WebSocketHook {
  subscribe: (channel: string, callback: (message: WebSocketMessage) => void) => Promise<() => void>;
  unsubscribe: (channel: string) => Promise<void>;
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  sendMessage: (message: WebSocketMessage) => void;
}

export const useWebSocket = (url?: string): WebSocketHook => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const subscribersRef = useRef<Map<string, (message: WebSocketMessage) => void>>(new Map());
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 1000; // Start with 1 second

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      setConnectionStatus('connecting');
      const wsUrl = url || import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:3002';
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setConnectionStatus('connected');
        reconnectAttemptsRef.current = 0;
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          // Handle different message types
          if (message.type === 'order_book' && message.channel) {
            const callback = subscribersRef.current.get(message.channel);
            if (callback) {
              callback(message);
            }
          } else if (message.type === 'price_update') {
            // Broadcast price updates to all subscribers
            subscribersRef.current.forEach((callback) => {
              callback(message);
            });
          } else if (message.type === 'error') {
            console.error('WebSocket error message:', message);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      wsRef.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        setConnectionStatus('disconnected');
        
        // Attempt to reconnect if not a manual close
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = reconnectDelay * Math.pow(2, reconnectAttemptsRef.current);
          console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectAttemptsRef.current + 1})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            connect();
          }, delay);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          setConnectionStatus('error');
          console.error('Max reconnection attempts reached');
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      };

    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      setConnectionStatus('error');
    }
  }, [url]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setIsConnected(false);
    setConnectionStatus('disconnected');
    reconnectAttemptsRef.current = 0;
  }, []);

  const subscribe = useCallback(async (channel: string, callback: (message: WebSocketMessage) => void): Promise<() => void> => {
    // Ensure connection is established
    if (!isConnected) {
      connect();
    }

    // Add subscriber
    subscribersRef.current.set(channel, callback);

    // Send subscription message if connected
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const subscribeMessage: WebSocketMessage = {
        type: 'subscribe',
        channel: channel
      };
      wsRef.current.send(JSON.stringify(subscribeMessage));
    }

    // Return unsubscribe function
    return () => {
      subscribersRef.current.delete(channel);
      
      // Send unsubscribe message if connected
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const unsubscribeMessage: WebSocketMessage = {
          type: 'unsubscribe',
          channel: channel
        };
        wsRef.current.send(JSON.stringify(unsubscribeMessage));
      }
    };
  }, [isConnected, connect]);

  const unsubscribe = useCallback(async (channel: string): Promise<void> => {
    subscribersRef.current.delete(channel);
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const unsubscribeMessage: WebSocketMessage = {
        type: 'unsubscribe',
        channel: channel
      };
      wsRef.current.send(JSON.stringify(unsubscribeMessage));
    }
  }, []);

  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send message:', message);
    }
  }, []);

  // Connect on mount
  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return {
    subscribe,
    unsubscribe,
    isConnected,
    connectionStatus,
    sendMessage
  };
};
