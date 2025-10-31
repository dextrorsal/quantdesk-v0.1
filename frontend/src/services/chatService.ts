import { Connection } from '@solana/web3.js';
import { useWallet } from '@solana/wallet-adapter-react';

import { WalletAuthService } from './walletAuth';
import { smartContractService } from './smartContractService';
import { balanceService } from './balanceService';

export const apiClient = {
  // Implement common API client methods here if needed, or use fetch directly
};

class ChatService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:3002';
  }

  async getChannels(): Promise<any[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/chat/channels`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          // The qd_session cookie will be sent automatically by the browser
        },
      });
      const data = await response.json();
      if (response.ok && data.channels) {
        return data.channels;
      } else {
        throw new Error(data.error || 'Failed to fetch channels');
      }
    } catch (error) {
      console.error('Error fetching channels:', error);
      throw error;
    }
  }

  async getChatHistory(channelId: string, walletPubkey: string): Promise<any[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/chat/history?channelId=${channelId}&wallet=${walletPubkey}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          // The qd_session cookie will be sent automatically by the browser
        },
      });
      const data = await response.json();
      if (response.ok && data.messages) {
        return data.messages;
      } else {
        throw new Error(data.error || 'Failed to fetch chat history');
      }
    } catch (error) {
      console.error('Error fetching chat history:', error);
      throw error;
    }
  }

  async getChatToken(walletPubkey: string): Promise<{ token: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/chat/token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ wallet_pubkey: walletPubkey }),
      });
      const data = await response.json();
      if (response.ok && data.token) {
        return { token: data.token };
      } else {
        throw new Error(data.error || 'Failed to get chat token');
      }
    } catch (error) {
      console.error('Error getting chat token:', error);
      throw error;
    }
  }

  async sendMessage(channelId: string, walletPubkey: string, message: string): Promise<void> {
    try {
      // Get a fresh chat token for sending the message
      const { token } = await this.getChatToken(walletPubkey);

      const response = await fetch(`${this.baseUrl}/api/chat/send`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ token, channelId, message }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to send message');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  }

  // This function can be used to send presence heartbeats via REST if WS is not yet established
  // Or, more efficiently, via WebSocket as part of the WS protocol
  async sendPresenceHeartbeat(channel: string, walletPubkey: string): Promise<void> {
    // For now, this is a no-op as presence is handled via WS handshake and `setPresence` in backend
    // and the GET /api/chat/history endpoint also sets presence.
    console.log(`Sending presence heartbeat for ${walletPubkey} on ${channel}`);
  }
}

export const chatService = new ChatService();
