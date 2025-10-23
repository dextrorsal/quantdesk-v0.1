import React, { useState, useRef, useEffect } from 'react';
import { PaperAirplaneIcon } from '@heroicons/react/outline';

interface MentionSuggestion {
  pubkey: string;
  username?: string;
}

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  mentions?: string[]; // Available users for mentions
}

const MessageInput: React.FC<MessageInputProps> = ({ onSendMessage, isLoading, mentions = [] }) => {
  const [message, setMessage] = useState('');
  const [showMentions, setShowMentions] = useState(false);
  const [mentionFilter, setMentionFilter] = useState('');
  const [mentionCursor, setMentionCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const filteredMentions = mentions.filter(pubkey =>
    pubkey.toLowerCase().includes(mentionFilter.toLowerCase())
  ).slice(0, 5);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    const cursorPos = e.target.selectionStart || 0;

    // Check for @ symbol
    const beforeCursor = value.substring(0, cursorPos);
    const mentionMatch = beforeCursor.match(/@([A-Za-z0-9]*)$/);

    if (mentionMatch) {
      setShowMentions(true);
      setMentionFilter(mentionMatch[1]);
      setMentionCursor(cursorPos);
    } else {
      setShowMentions(false);
      setMentionFilter('');
    }

    setMessage(value);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showMentions) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setMentionCursor(prev => Math.min(prev + 1, filteredMentions.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setMentionCursor(prev => Math.max(prev - 1, 0));
      } else if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        if (filteredMentions[mentionCursor]) {
          insertMention(filteredMentions[mentionCursor]);
        }
      } else if (e.key === 'Escape') {
        setShowMentions(false);
      }
    }
  };

  const insertMention = (pubkey: string) => {
    const beforeMention = message.substring(0, mentionCursor - mentionFilter.length - 1); // -1 for @
    const afterMention = message.substring(mentionCursor);
    const newMessage = `${beforeMention}@${pubkey} ${afterMention}`;
    setMessage(newMessage);
    setShowMentions(false);
    setMentionFilter('');
    inputRef.current?.focus();
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSendMessage(message);
      setMessage('');
      setShowMentions(false);
    }
  };

  // Close mentions dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = () => setShowMentions(false);
    if (showMentions) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [showMentions]);

  return (
    <div className="relative">
      <form onSubmit={handleSubmit} className="flex items-center space-x-2">
        <div className="flex-1 relative">
          <input
            ref={inputRef}
            type="text"
            value={message}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Type your message... (@mention users)"
            className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />

          {/* Mentions dropdown */}
          {showMentions && filteredMentions.length > 0 && (
            <div className="absolute bottom-full left-0 right-0 bg-gray-800 border border-gray-600 rounded-lg shadow-lg z-10 max-h-40 overflow-y-auto">
              {filteredMentions.map((pubkey, index) => (
                <div
                  key={pubkey}
                  className={`p-2 cursor-pointer hover:bg-gray-700 ${
                    index === mentionCursor ? 'bg-gray-600' : ''
                  }`}
                  onClick={() => insertMention(pubkey)}
                >
                  <span className="text-white font-mono text-sm">
                    @{pubkey.slice(0, 8)}...{pubkey.slice(-4)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <button
          type="submit"
          className="bg-blue-600 hover:bg-blue-700 text-white font-bold p-3 rounded-lg transition-colors flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={isLoading || !message.trim()}
        >
          {isLoading ? (
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
          ) : (
            <PaperAirplaneIcon className="w-5 h-5" />
          )}
        </button>
      </form>
    </div>
  );
};

export default MessageInput;
