// Enhanced MIKEY-AI Service with Real-time WebSocket Support
// Story 2.2: AC 2.2.4 - Frontend Integration

export interface EnhancedMessage {
  id: string;
  type: 'user' | 'ai' | 'system';
  content: string;
  timestamp: Date;
  confidence?: number;
  sources?: string[];
  provider?: string;
  cost?: number;
}

export interface ConversationSession {
  sessionId: string;
  userId?: string;
  messages: EnhancedMessage[];
  startTime: Date;
  metadata: {
    totalTokens: number;
    totalCost: number;
    provider: string;
  };
}

export interface UserPreferences {
  aiProvider?: 'auto' | 'openai' | 'gemini' | 'cohere' | 'mistral' | 'xai';
  conversationStyle: 'professional' | 'casual' | 'analytical';
  autoConnect: boolean;
  notifications: boolean;
  theme: 'light' | 'dark' | 'auto';
}

export class EnhancedMikeyAIService {
  private baseUrl: string = 'http://localhost:3000';
  private websocket: WebSocket | null = null;
  private conversationSession: ConversationSession | null = null;
  private userPreferences: UserPreferences;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  
  // Event listeners
  private messageCallbacks: Array<(message: EnhancedMessage) => void> = [];
  private statusCallbacks: Array<(status: 'connected' | 'disconnected' | 'error') => void> = [];

  constructor() {
    // Load user preferences from localStorage
    this.userPreferences = this.loadPreferences();
    
    // Initialize WebSocket connection
    this.connectWebSocket();
  }

  /**
   * Load user preferences from localStorage
   */
  private loadPreferences(): UserPreferences {
    try {
      const stored = localStorage.getItem('mikey_preferences');
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (error) {
      console.error('Failed to load preferences:', error);
    }
    
    return {
      conversationStyle: 'professional',
      autoConnect: true,
      notifications: true,
      theme: 'auto'
    };
  }

  /**
   * Save user preferences to localStorage
   */
  public updatePreferences(preferences: Partial<UserPreferences>): void {
    this.userPreferences = { ...this.userPreferences, ...preferences };
    localStorage.setItem('mikey_preferences', JSON.stringify(this.userPreferences));
  }

  /**
   * Connect to WebSocket for real-time updates
   */
  private connectWebSocket(): void {
    if (this.websocket?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    try {
      const wsUrl = this.baseUrl.replace('http://', 'ws://').replace('https://', 'wss://');
      this.websocket = new WebSocket(`${wsUrl}/ws/mikey`);

      this.websocket.onopen = () => {
        console.log('MIKEY WebSocket connected');
        this.reconnectAttempts = 0;
        this.notifyStatus('connected');
      };

      this.websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Handle different message types
          if (data.type === 'message') {
            this.handleIncomingMessage(data);
          } else if (data.type === 'status') {
            console.log('MIKEY status update:', data);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.websocket.onerror = (error) => {
        console.error('MIKEY WebSocket error:', error);
        this.notifyStatus('error');
      };

      this.websocket.onclose = () => {
        console.log('MIKEY WebSocket disconnected');
        this.notifyStatus('disconnected');
        
        // Attempt reconnection
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          setTimeout(() => this.connectWebSocket(), 1000 * this.reconnectAttempts);
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.notifyStatus('error');
    }
  }

  /**
   * Handle incoming real-time messages
   */
  private handleIncomingMessage(data: any): void {
    const message: EnhancedMessage = {
      id: data.id || `msg-${Date.now()}`,
      type: data.role === 'user' ? 'user' : 'ai',
      content: data.content,
      timestamp: new Date(data.timestamp || Date.now()),
      confidence: data.confidence,
      sources: data.sources,
      provider: data.provider,
      cost: data.cost
    };

    this.messageCallbacks.forEach(callback => callback(message));
  }

  /**
   * Send AI query with session management
   */
  async queryAI(query: string, sessionId?: string): Promise<EnhancedMessage> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/ai/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query,
          sessionId: sessionId || this.conversationSession?.sessionId,
          preferences: this.userPreferences
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Create conversation session if it doesn't exist
      if (!this.conversationSession) {
        this.conversationSession = {
          sessionId: sessionId || `session-${Date.now()}`,
          messages: [],
          startTime: new Date(),
          metadata: {
            totalTokens: 0,
            totalCost: 0,
            provider: data.data.provider || 'unknown'
          }
        };
      }

      // Add AI response to session
      const aiMessage: EnhancedMessage = {
        id: `ai-${Date.now()}`,
        type: 'ai',
        content: data.success ? data.data.response : 'Error occurred',
        timestamp: new Date(),
        confidence: data.success ? data.data.confidence : 0,
        sources: data.success ? data.data.sources : [],
        provider: data.data.provider,
        cost: data.metadata?.cost
      };

      if (this.conversationSession) {
        this.conversationSession.messages.push(aiMessage);
        this.saveSession();
      }

      return aiMessage;
    } catch (error) {
      console.error('Enhanced MIKEY query failed:', error);
      throw error;
    }
  }

  /**
   * Start new conversation session
   */
  startNewSession(sessionId?: string): void {
    this.conversationSession = {
      sessionId: sessionId || `session-${Date.now()}`,
      messages: [],
      startTime: new Date(),
      metadata: {
        totalTokens: 0,
        totalCost: 0,
        provider: 'unknown'
      }
    };
    this.saveSession();
  }

  /**
   * Save conversation session to localStorage
   */
  private saveSession(): void {
    if (this.conversationSession) {
      try {
        localStorage.setItem('mikey_session', JSON.stringify(this.conversationSession));
      } catch (error) {
        console.error('Failed to save session:', error);
      }
    }
  }

  /**
   * Load conversation session from localStorage
   */
  loadSession(): ConversationSession | null {
    try {
      const stored = localStorage.getItem('mikey_session');
      if (stored) {
        const session = JSON.parse(stored);
        session.startTime = new Date(session.startTime);
        session.messages = session.messages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        this.conversationSession = session;
        return session;
      }
    } catch (error) {
      console.error('Failed to load session:', error);
    }
    return null;
  }

  /**
   * Register message callback for real-time updates
   */
  onMessage(callback: (message: EnhancedMessage) => void): void {
    this.messageCallbacks.push(callback);
  }

  /**
   * Register status callback
   */
  onStatus(callback: (status: 'connected' | 'disconnected' | 'error') => void): void {
    this.statusCallbacks.push(callback);
  }

  /**
   * Notify all status callbacks
   */
  private notifyStatus(status: 'connected' | 'disconnected' | 'error'): void {
    this.statusCallbacks.forEach(callback => callback(status));
  }

  /**
   * Get current session
   */
  getCurrentSession(): ConversationSession | null {
    return this.conversationSession;
  }

  /**
   * Get user preferences
   */
  getPreferences(): UserPreferences {
    return this.userPreferences;
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const data = await response.json();
      return data.success && data.data.status === 'healthy';
    } catch (error) {
      console.error('MIKEY-AI health check failed:', error);
      return false;
    }
  }

  /**
   * Disconnect WebSocket
   */
  disconnect(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }

  /**
   * Reconnect WebSocket
   */
  reconnect(): void {
    this.disconnect();
    this.reconnectAttempts = 0;
    this.connectWebSocket();
  }
}

// Export singleton instance
export const enhancedMikeyAI = new EnhancedMikeyAIService();
export default enhancedMikeyAI;

