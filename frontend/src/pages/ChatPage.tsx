import React, { useState, useEffect, useCallback } from 'react';
import ChatWindow from '../components/ChatWindow';
import MessageInput from '../components/MessageInput';
import { useWalletAuth } from '../hooks/useWalletAuth';
import { chatService } from '../services/chatService';

const ChatPage: React.FC = () => {
  const { user, isLoading: isAuthLoading } = useWalletAuth();
  const [isSendingMessage, setIsSendingMessage] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const globalChannel = 'global'; // Default global chat channel

  const handleSendMessage = useCallback(async (message: string) => {
    if (!user?.wallet_pubkey) {
      setChatError('Please connect your wallet to send messages.');
      return;
    }
    setIsSendingMessage(true);
    setChatError(null);
    try {
      await chatService.sendMessage(globalChannel, user.wallet_pubkey, message);
    } catch (error: any) {
      console.error('Error sending message:', error);
      setChatError(error.message || 'Failed to send message.');
    } finally {
      setIsSendingMessage(false);
    }
  }, [user]);

  if (isAuthLoading) {
    return <div className="flex items-center justify-center h-screen text-white">Loading authentication...</div>;
  }

  return (
    <div className="flex flex-col h-screen bg-gray-800 p-6">
      {chatError && <div className="text-red-500 mb-4">{chatError}</div>}
      <div className="flex-1 mb-4">
        <ChatWindow channelId={globalChannel} />
      </div>
      <div className="flex-none">
        <MessageInput onSendMessage={handleSendMessage} isLoading={isSendingMessage} />
      </div>
    </div>
  );
};

export default ChatPage;
