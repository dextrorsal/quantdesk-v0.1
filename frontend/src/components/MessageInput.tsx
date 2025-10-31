import React, { useState, useRef, useEffect } from 'react';

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
  const [showTickerSearch, setShowTickerSearch] = useState(false);
  const [tickerFilter, setTickerFilter] = useState('');
  const [tickerCursor, setTickerCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Popular tickers for autocomplete
  const popularTickers = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'MATIC', 'AVAX', 'LINK', 'UNI', 'ATOM', 'DOGE', 'SHIB'];

  const filteredMentions = mentions.filter(pubkey =>
    pubkey.toLowerCase().includes(mentionFilter.toLowerCase())
  ).slice(0, 5);
  
  const filteredTickers = popularTickers.filter(ticker =>
    ticker.toLowerCase().includes(tickerFilter.toLowerCase())
  ).slice(0, 5);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    const cursorPos = e.target.selectionStart || 0;

    // Check for @ symbol for mentions
    const beforeCursor = value.substring(0, cursorPos);
    const mentionMatch = beforeCursor.match(/@([A-Za-z0-9]*)$/);
    const tickerMatch = beforeCursor.match(/\$([A-Za-z]*)$/);

    if (mentionMatch) {
      setShowMentions(true);
      setShowTickerSearch(false);
      setMentionFilter(mentionMatch[1]);
      setMentionCursor(cursorPos);
    } else if (tickerMatch) {
      setShowTickerSearch(true);
      setShowMentions(false);
      setTickerFilter(tickerMatch[1]);
      setTickerCursor(cursorPos);
    } else {
      setShowMentions(false);
      setShowTickerSearch(false);
      setMentionFilter('');
      setTickerFilter('');
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
    } else if (showTickerSearch) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setTickerCursor(prev => Math.min(prev + 1, filteredTickers.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setTickerCursor(prev => Math.max(prev - 1, 0));
      } else if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        if (filteredTickers[tickerCursor]) {
          insertTicker(filteredTickers[tickerCursor]);
        }
      } else if (e.key === 'Escape') {
        setShowTickerSearch(false);
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
  
  const insertTicker = (ticker: string) => {
    const beforeTicker = message.substring(0, tickerCursor - tickerFilter.length - 1); // -1 for $
    const afterTicker = message.substring(tickerCursor);
    const newMessage = `${beforeTicker}$${ticker} ${afterTicker}`;
    setMessage(newMessage);
    setShowTickerSearch(false);
    setTickerFilter('');
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

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      setShowMentions(false);
      setShowTickerSearch(false);
    };
    if (showMentions || showTickerSearch) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [showMentions, showTickerSearch]);

  return (
    <div style={{ position: 'relative' }}>
      <form onSubmit={handleSubmit} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <input
            ref={inputRef}
            type="text"
            value={message}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Type your message... (Press $ for tickers, @ for mentions)"
            style={{
              width: '100%',
              padding: '12px',
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--bg-tertiary)',
              borderRadius: '4px',
              color: 'var(--text-primary)',
              fontSize: '14px',
              outline: 'none'
            }}
            disabled={isLoading}
            onFocus={(e) => e.currentTarget.style.borderColor = 'var(--primary-500)'}
            onBlur={(e) => e.currentTarget.style.borderColor = 'var(--bg-tertiary)'}
          />

          {/* Ticker search dropdown - Godel Terminal style */}
          {showTickerSearch && filteredTickers.length > 0 && (
            <div style={{
              position: 'absolute',
              bottom: '100%',
              left: 0,
              right: 0,
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--primary-500)',
              borderRadius: '4px',
              zIndex: 10,
              maxHeight: '160px',
              overflowY: 'auto',
              marginBottom: '4px'
            }}>
              {filteredTickers.map((ticker, index) => (
                <div
                  key={ticker}
                  style={{
                    padding: '8px',
                    cursor: 'pointer',
                    backgroundColor: index === tickerCursor ? 'var(--bg-tertiary)' : 'transparent',
                    transition: 'background-color 0.2s'
                  }}
                  onClick={() => insertTicker(ticker)}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
                  onMouseLeave={(e) => {
                    if (index !== tickerCursor) {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }
                  }}
                >
                  <span style={{ color: 'var(--primary-500)', fontFamily: 'monospace', fontSize: '14px' }}>
                    ${ticker}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Mentions dropdown */}
          {showMentions && filteredMentions.length > 0 && (
            <div style={{
              position: 'absolute',
              bottom: '100%',
              left: 0,
              right: 0,
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--primary-500)',
              borderRadius: '4px',
              zIndex: 10,
              maxHeight: '160px',
              overflowY: 'auto',
              marginBottom: '4px'
            }}>
              {filteredMentions.map((pubkey, index) => (
                <div
                  key={pubkey}
                  style={{
                    padding: '8px',
                    cursor: 'pointer',
                    backgroundColor: index === mentionCursor ? 'var(--bg-tertiary)' : 'transparent',
                    transition: 'background-color 0.2s'
                  }}
                  onClick={() => insertMention(pubkey)}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
                  onMouseLeave={(e) => {
                    if (index !== mentionCursor) {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }
                  }}
                >
                  <span style={{ color: 'var(--text-primary)', fontFamily: 'monospace', fontSize: '14px' }}>
                    @{pubkey.slice(0, 8)}...{pubkey.slice(-4)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <button
          type="submit"
          style={{
            backgroundColor: isLoading || !message.trim() ? 'var(--bg-tertiary)' : 'var(--primary-500)',
            color: 'var(--text-primary)',
            fontWeight: '600',
            padding: '12px 24px',
            borderRadius: '4px',
            border: 'none',
            cursor: isLoading || !message.trim() ? 'not-allowed' : 'pointer',
            opacity: isLoading || !message.trim() ? 0.5 : 1,
            transition: 'all 0.2s',
            fontSize: '14px'
          }}
          disabled={isLoading || !message.trim()}
          onMouseEnter={(e) => {
            if (!isLoading && message.trim()) {
              e.currentTarget.style.opacity = '0.9';
            }
          }}
          onMouseLeave={(e) => {
            if (!isLoading && message.trim()) {
              e.currentTarget.style.opacity = '1';
            }
          }}
        >
          {isLoading ? (
            <span style={{ fontSize: '14px' }}>...</span>
          ) : (
            <span style={{ fontSize: '14px' }}>Send</span>
          )}
        </button>
      </form>
    </div>
  );
};

export default MessageInput;
