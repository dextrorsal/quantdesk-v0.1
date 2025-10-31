import React, { useEffect, useState } from 'react';

interface TwitterFeedProps {
  account?: string;
  keywords?: string;
  usernames?: string;
  hashtags?: string;
  height?: number;
}

interface Tweet {
  id: string;
  text: string;
  author: string;
  timestamp: string;
  likes: number;
  retweets: number;
  url: string;
}

const TwitterFeed: React.FC<TwitterFeedProps> = ({ 
  keywords = 'bitcoin ethereum crypto',
  usernames,
  hashtags,
  height = 600
}) => {
  const [tweets, setTweets] = useState<Tweet[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTweets = async () => {
      try {
        setLoading(true);
        
        // Build query params
        const params = new URLSearchParams();
        if (keywords) params.append('keywords', keywords);
        if (usernames) params.append('usernames', usernames);
        if (hashtags) params.append('hashtags', hashtags);
        
        const response = await fetch(`/api/twitter/feed?${params.toString()}`);
        const data = await response.json();
        
        if (data.success && data.tweets) {
          setTweets(data.tweets);
        }
      } catch (error) {
        console.error('Error fetching tweets:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTweets();
    
    // Refresh every 60 seconds
    const interval = setInterval(fetchTweets, 60000);
    return () => clearInterval(interval);
  }, [keywords, usernames, hashtags]);

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  if (loading) {
    return (
      <div style={{ 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        color: 'var(--text-muted)'
      }}>
        Loading tweets...
      </div>
    );
  }

  return (
    <div style={{ 
      width: '100%', 
      height: '100%', 
      overflow: 'auto',
      backgroundColor: '#000',
      padding: '10px'
    }}>
      <div style={{ marginBottom: '10px', fontSize: '12px', color: 'var(--text-muted)' }}>
        📊 Live Crypto Tweets • {tweets.length} results
      </div>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {tweets.map((tweet) => (
          <div 
            key={tweet.id}
            style={{
              padding: '12px',
              backgroundColor: 'var(--bg-secondary)',
              borderRadius: '8px',
              border: '1px solid var(--border-base)',
              cursor: 'pointer',
              transition: 'border-color 0.2s'
            }}
            onMouseEnter={(e) => e.currentTarget.style.borderColor = 'var(--primary-blue)'}
            onMouseLeave={(e) => e.currentTarget.style.borderColor = 'var(--border-base)'}
            onClick={() => window.open(tweet.url, '_blank')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ color: 'var(--primary-blue)', fontWeight: 'bold', fontSize: '13px' }}>
                {tweet.author}
              </span>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                {formatTime(tweet.timestamp)}
              </span>
            </div>
            
            <div style={{ fontSize: '13px', color: 'var(--text-primary)', marginBottom: '8px', lineHeight: '1.5' }}>
              {tweet.text}
            </div>
            
            <div style={{ display: 'flex', gap: '20px', fontSize: '11px', color: 'var(--text-muted)' }}>
              <span>❤️ {tweet.likes.toLocaleString()}</span>
              <span>🔄 {tweet.retweets.toLocaleString()}</span>
              <span>💬 {Math.floor(tweet.retweets * 0.3).toLocaleString()}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TwitterFeed;

