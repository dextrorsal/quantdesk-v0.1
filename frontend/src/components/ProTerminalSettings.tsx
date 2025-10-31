import React, { useState, useEffect } from 'react';
import WalletButton from './WalletButton';

interface CommandLayout {
  id: string;
  name: string;
  description: string;
  commands: string[];
  isDefault: boolean;
  createdAt: Date;
  lastUsed: Date;
}

interface TerminalProfile {
  id: string;
  name: string;
  description: string;
  layout: CommandLayout;
  theme: 'dark' | 'light' | 'terminal';
  fontSize: number;
  fontFamily: string;
  windowDefaults: {
    defaultWidth: number;
    defaultHeight: number;
    maxWindows: number;
  };
  shortcuts: Record<string, string>;
  isActive: boolean;
}

interface SettingsManagerProps {
  isOpen: boolean;
  onClose: () => void;
  onProfileChange: (profile: TerminalProfile) => void;
  currentProfile?: TerminalProfile;
  savedLayouts?: any[];
  onSaveLayout?: (name: string) => any;
  onLoadLayout?: (layout: any) => void;
}

const ProTerminalSettings: React.FC<SettingsManagerProps> = ({ 
  isOpen, 
  onClose, 
  onProfileChange, 
  currentProfile,
  savedLayouts = [],
  onSaveLayout,
  onLoadLayout
}) => {
  const [profiles, setProfiles] = useState<TerminalProfile[]>([]);
  const [layouts, setLayouts] = useState<CommandLayout[]>([]);
  const [activeTab, setActiveTab] = useState<'profiles' | 'layouts' | 'appearance' | 'shortcuts' | 'wallet'>('profiles');
  const [newLayoutName, setNewLayoutName] = useState('');
  const [showCreateProfile, setShowCreateProfile] = useState(false);
  const [showCreateLayout, setShowCreateLayout] = useState(false);
  const [editingProfile, setEditingProfile] = useState<TerminalProfile | null>(null);
  const [editingLayout, setEditingLayout] = useState<CommandLayout | null>(null);
  const [showEditProfileModal, setShowEditProfileModal] = useState(false);

  // Available commands from our pro terminal (full catalog)
  const availableCommands = [
    { command: 'QM', description: 'Quote Monitor', category: 'Market Data' },
    { command: 'N', description: 'News', category: 'News' },
    { command: 'CHART', description: 'Charting', category: 'Analysis' },
    { command: 'CHAT', description: 'Chat', category: 'Social' },
    { command: 'CF', description: 'SEC Filings', category: 'Research' },
    { command: 'EQS', description: 'Equity Screener', category: 'Screening' },
    { command: 'CALC', description: 'Calculator', category: 'Tools' },
    { command: 'PAT', description: 'Pattern Search', category: 'Analysis' },
    { command: 'MOST', description: 'Most Active', category: 'Market Data' },
    { command: 'IPO', description: 'IPOs', category: 'Market Data' },
    { command: 'AL', description: 'Alerts', category: 'Alerts' },
    { command: 'NOTE', description: 'Notes', category: 'Tools' },
    { command: 'BROK', description: 'Brokerage', category: 'Trading' },
    { command: 'TOP', description: 'Top News', category: 'News' },
    { command: 'GR', description: 'Graph Relations', category: 'Analysis' },
    { command: 'HPS', description: 'Historical Compare', category: 'Analysis' },
    { command: 'OPT', description: 'Options Chain', category: 'Options' },
    { command: 'WATCH', description: 'Watchlist', category: 'Market Data' },
    { command: 'SCAN', description: 'Scanner', category: 'Screening' },
    { command: 'ORDER', description: 'Place Orders', category: 'Trading' },
    { command: 'POSITIONS', description: 'View Positions', category: 'Trading' },
    { command: 'PF', description: 'Portfolio', category: 'Portfolio' },
    { command: 'VOLUME', description: 'Volume Analysis', category: 'Analysis' },
    { command: 'FEAR', description: 'Fear & Greed', category: 'Sentiment' },
    { command: 'CORR', description: 'Correlation', category: 'Analysis' },
    { command: 'FLOW', description: 'On-chain Flow', category: 'On-chain' },
    { command: 'DEFI', description: 'DeFi Protocols', category: 'DeFi' },
    { command: 'NFT', description: 'NFT Analysis', category: 'NFT' },
    { command: 'RES', description: 'Research Reports', category: 'Research' },
    { command: 'WHITEPAPER', description: 'Whitepapers', category: 'Research' },
    { command: 'ACCT', description: 'Account Management', category: 'Account' },
    { command: 'WALLET', description: 'Wallet', category: 'Wallet' },
    { command: 'API', description: 'API Management', category: 'API' },
    { command: 'GAS', description: 'Gas Tracker', category: 'Tools' },
    { command: 'STAKING', description: 'Staking Calculator', category: 'Staking' },
    { command: 'TWITTER', description: 'Twitter Feed', category: 'Social' },
    { command: 'REDDIT', description: 'Crypto Reddit discussions', category: 'Social' },
    { command: 'ORDERBOOK', description: 'Real-time order book with bids/asks', category: 'Market Data' },
    { command: 'HEATMAP', description: 'Market heatmap showing price changes', category: 'Analysis' },
    { command: 'STRATEGIES', description: 'Trading strategies management', category: 'Trading' },
    { command: 'NOTIFICATIONS', description: 'Comprehensive notification center', category: 'Trading' },
    { command: 'ANALYSIS', description: 'Technical analysis tools and indicators', category: 'Analysis' },
    { command: 'OVERVIEW', description: 'Trading dashboard with key metrics', category: 'System' },
    { command: 'MIKEY', description: 'AI Trading Assistant', category: 'AI' },
    { command: 'MIKEY_MONITOR', description: 'MIKEY AI Monitoring Dashboard', category: 'AI' },
    { command: 'HELP', description: 'Crypto Terminal Documentation', category: 'System' },
    { command: 'S', description: 'Keyboard Shortcuts', category: 'System' },
    { command: 'CLEAR', description: 'Clear all windows', category: 'System' },
    { command: 'LAYOUT', description: 'Save/Load window layouts', category: 'System' },
    { command: 'ERR', description: 'Report bugs and get support', category: 'System' },
    { command: 'BT', description: 'Backtesting', category: 'Backtesting' }
  ];

  // Only show commands that are implemented in the current crypto build
  // Show everything that appears in the backtick menu
  const visibleCommands = availableCommands;

  // Load profiles from localStorage and initialize
  useEffect(() => {
    // Try to load saved profiles from localStorage first
    try {
      const savedProfiles = localStorage.getItem('quantdesk_profiles');
      if (savedProfiles) {
        const parsed = JSON.parse(savedProfiles);
        if (Array.isArray(parsed) && parsed.length > 0) {
          // Load saved profiles but preserve their structure and ensure all required fields exist
          setProfiles(parsed.map((p: any) => {
            const normalizedLayout = p.layout || {
              id: p.id || '0',
              name: p.name,
              commands: (p.layout?.commands || p.commands || []),
              isDefault: true
            };

            // If Master profile (P0) has empty commands (older saves), default to all visible commands
            const isMaster = p.id === '0' || (p.name || '').toLowerCase().includes('master');
            const commands = (normalizedLayout.commands && normalizedLayout.commands.length > 0)
              ? normalizedLayout.commands
              : (isMaster ? visibleCommands.map(c => c.command) : (normalizedLayout.commands || []));

            return {
              ...p,
              layout: { ...normalizedLayout, commands },
              windowDefaults: p.windowDefaults || {
                defaultWidth: 400,
                defaultHeight: 300,
                maxWindows: 8
              },
              shortcuts: p.shortcuts || {},
              theme: p.theme || 'dark',
              fontSize: p.fontSize || 14,
              fontFamily: p.fontFamily || 'JetBrains Mono'
            };
          }));
          
          // If currentProfile is provided, sync it with saved data
          if (currentProfile) {
            const savedCurrent = parsed.find((p: any) => p.id === currentProfile.id);
            if (savedCurrent) {
              onProfileChange(savedCurrent);
            }
          }
          
          // Don't override with mock data if we have saved profiles
          return;
        }
      }
    } catch (e) {
      console.error('Failed to load saved profiles:', e);
    }

    // If no saved profiles, initialize with mock data
    // Get allowed commands for the master profile (implemented only)
    const allCommands = visibleCommands.map(cmd => cmd.command);

    const mockLayouts: CommandLayout[] = [
      {
        id: '0',
        name: 'Master - ALL Commands',
        description: 'Complete access to all terminal commands',
        commands: allCommands,
        isDefault: true,
        createdAt: new Date(Date.now() - 86400000),
        lastUsed: new Date(Date.now() - 3600000)
      },
      {
        id: '1',
        name: 'Trading Focus',
        description: 'Essential commands for active trading',
        commands: ['QM', 'CHART', 'ORDER', 'POSITIONS', 'AL', 'FOCUS'],
        isDefault: false,
        createdAt: new Date(Date.now() - 86400000),
        lastUsed: new Date(Date.now() - 3600000)
      },
      {
        id: '2',
        name: 'Research Mode',
        description: 'Comprehensive research and analysis tools',
        commands: ['CF', 'EQS', 'PAT', 'GR', 'HPS', 'N', 'TOP', 'NOTE', 'RES'],
        isDefault: false,
        createdAt: new Date(Date.now() - 172800000),
        lastUsed: new Date(Date.now() - 7200000)
      },
      {
        id: '3',
        name: 'Minimal',
        description: 'Clean interface with core functions only',
        commands: ['QM', 'CHART', 'WATCH'],
        isDefault: false,
        createdAt: new Date(Date.now() - 259200000),
        lastUsed: new Date(Date.now() - 86400000)
      },
      {
        id: '4',
        name: 'Crypto Pro',
        description: 'Specialized for crypto trading and analysis',
        commands: ['QM', 'CHART', 'ORDER', 'POSITIONS', 'PF', 'VOLUME', 'FEAR', 'CORR', 'FLOW', 'DEFI'],
        isDefault: false,
        createdAt: new Date(Date.now() - 345600000),
        lastUsed: new Date(Date.now() - 172800000)
      }
    ];

    // Generate keyboard shortcuts for all visible commands
    const allShortcuts: Record<string, string> = {};
    visibleCommands.forEach((cmd, index) => {
      if (index < 26) {
        // Use Ctrl+1-9, then Ctrl+Shift+A-Z
        if (index < 9) {
          allShortcuts[`Ctrl+${index + 1}`] = cmd.command;
        } else {
          const letter = String.fromCharCode(65 + index - 9); // A-Z
          allShortcuts[`Ctrl+Shift+${letter}`] = cmd.command;
        }
      }
    });

    const mockProfiles: TerminalProfile[] = [
      {
        id: '0',
        name: 'P0 - Master (ALL COMMANDS)',
        description: 'Complete access to all features - Default active profile',
        layout: mockLayouts[0],
        theme: 'dark',
        fontSize: 14, // Default font size
        fontFamily: 'JetBrains Mono',
        windowDefaults: {
          defaultWidth: 400,
          defaultHeight: 300,
          maxWindows: 20 // Max windows enabled
        },
        shortcuts: allShortcuts, // All shortcuts configured
        isActive: true // This is the active default
      },
      {
        id: '1',
        name: 'P1 - Day Trader',
        description: 'High-frequency trading setup',
        layout: mockLayouts[1],
        theme: 'dark',
        fontSize: 14,
        fontFamily: 'JetBrains Mono',
        windowDefaults: {
          defaultWidth: 400,
          defaultHeight: 300,
          maxWindows: 8
        },
        shortcuts: {
          'Ctrl+1': 'QM',
          'Ctrl+2': 'CHART',
          'Ctrl+3': 'ORDER',
          'Ctrl+4': 'AL'
        },
        isActive: false
      },
      {
        id: '2',
        name: 'P2 - Research',
        description: 'Deep analysis and research',
        layout: mockLayouts[1],
        theme: 'dark',
        fontSize: 16,
        fontFamily: 'Fira Code',
        windowDefaults: {
          defaultWidth: 500,
          defaultHeight: 400,
          maxWindows: 6
        },
        shortcuts: {
          'Ctrl+1': 'CF',
          'Ctrl+2': 'EQS',
          'Ctrl+3': 'PAT',
          'Ctrl+4': 'GR'
        },
        isActive: false
      },
      {
        id: '3',
        name: 'P3 - Crypto Pro',
        description: 'Crypto trading specialist',
        layout: mockLayouts[3],
        theme: 'terminal',
        fontSize: 13,
        fontFamily: 'Consolas',
        windowDefaults: {
          defaultWidth: 450,
          defaultHeight: 350,
          maxWindows: 10
        },
        shortcuts: {
          'Ctrl+1': 'QM',
          'Ctrl+2': 'CHART',
          'Ctrl+3': 'ORDER',
          'Ctrl+4': 'PF'
        },
        isActive: false
      }
    ];

    setLayouts(mockLayouts);
    setProfiles(mockProfiles);
    
    // Save initial profiles to localStorage
    localStorage.setItem('quantdesk_profiles', JSON.stringify(mockProfiles));
  }, []);

  const switchProfile = (profile: TerminalProfile) => {
    const updatedProfiles = profiles.map(p => ({ ...p, isActive: p.id === profile.id }));
    setProfiles(updatedProfiles);
    
    // Save to localStorage immediately
    localStorage.setItem('quantdesk_profiles', JSON.stringify(updatedProfiles));
    
    // Dispatch custom event to notify parent component
    window.dispatchEvent(new Event('profileSaved'));
    
    onProfileChange(profile);
  };

  const createProfile = (profileData: Partial<TerminalProfile>) => {
    const newProfile: TerminalProfile = {
      id: Date.now().toString(),
      name: profileData.name || 'New Profile',
      description: profileData.description || '',
      layout: profileData.layout || layouts[0],
      theme: profileData.theme || 'dark',
      fontSize: profileData.fontSize || 14,
      fontFamily: profileData.fontFamily || 'JetBrains Mono',
      windowDefaults: profileData.windowDefaults || {
        defaultWidth: 400,
        defaultHeight: 300,
        maxWindows: 8
      },
      shortcuts: profileData.shortcuts || {},
      isActive: false
    };
    const updatedProfiles = [...profiles, newProfile];
    setProfiles(updatedProfiles);
    
    // Save to localStorage immediately
    localStorage.setItem('quantdesk_profiles', JSON.stringify(updatedProfiles));
    
    // Dispatch custom event to notify parent component
    window.dispatchEvent(new Event('profileSaved'));
    
    setShowCreateProfile(false);
  };

  const createLayout = (layoutData: Partial<CommandLayout>) => {
    const newLayout: CommandLayout = {
      id: Date.now().toString(),
      name: layoutData.name || 'New Layout',
      description: layoutData.description || '',
      commands: layoutData.commands || [],
      isDefault: false,
      createdAt: new Date(),
      lastUsed: new Date()
    };
    setLayouts(prev => [...prev, newLayout]);
    setShowCreateLayout(false);
  };

  const updateProfile = (profileId: string, updates: Partial<TerminalProfile>) => {
    const updatedProfiles = profiles.map(p => 
      p.id === profileId ? { ...p, ...updates } : p
    );
    setProfiles(updatedProfiles);
    
    if (editingProfile?.id === profileId) {
      setEditingProfile({ ...editingProfile, ...updates });
    }
    
      // Save to localStorage immediately
      localStorage.setItem('quantdesk_profiles', JSON.stringify(updatedProfiles));
      
      // Dispatch custom event to notify parent component
      window.dispatchEvent(new Event('profileSaved'));
      
      // Notify parent component if profile is active
      const updatedProfile = updatedProfiles.find(p => p.id === profileId);
      if (updatedProfile && (updatedProfile.isActive || updatedProfile.id === currentProfile?.id)) {
        onProfileChange(updatedProfile);
    }
  };

  const handleEditProfile = (profile: TerminalProfile) => {
    // Ensure all required fields are initialized
    setEditingProfile({
      ...profile,
      windowDefaults: profile.windowDefaults || {
        defaultWidth: 400,
        defaultHeight: 300,
        maxWindows: 8
      },
      shortcuts: profile.shortcuts || {},
      theme: profile.theme || 'dark',
      fontSize: profile.fontSize || 14,
      fontFamily: profile.fontFamily || 'JetBrains Mono'
    });
    setShowEditProfileModal(true);
  };

  const saveProfileEdits = () => {
    if (editingProfile) {
      const updatedProfiles = profiles.map(p => 
        p.id === editingProfile.id ? editingProfile : p
      );
      setProfiles(updatedProfiles);
      
      // Save to localStorage immediately
      localStorage.setItem('quantdesk_profiles', JSON.stringify(updatedProfiles));
      
      // Dispatch custom event to notify parent component
      window.dispatchEvent(new Event('profileSaved'));
      
      // Notify parent component if this profile is active
      if (editingProfile.isActive || editingProfile.id === currentProfile?.id) {
        onProfileChange(editingProfile);
      }
      
      setShowEditProfileModal(false);
      setEditingProfile(null);
    }
  };

  const deleteProfile = (profileId: string) => {
    const updatedProfiles = profiles.filter(p => p.id !== profileId);
    setProfiles(updatedProfiles);
    
    // Save to localStorage immediately
    localStorage.setItem('quantdesk_profiles', JSON.stringify(updatedProfiles));
    
    // Dispatch custom event to notify parent component
    window.dispatchEvent(new Event('profileSaved'));
  };

  const duplicateProfile = (profile: TerminalProfile) => {
    const duplicated: TerminalProfile = {
      ...profile,
      id: Date.now().toString(),
      name: `${profile.name} (Copy)`,
      isActive: false
    };
    const updatedProfiles = [...profiles, duplicated];
    setProfiles(updatedProfiles);
    
    // Save to localStorage immediately
    localStorage.setItem('quantdesk_profiles', JSON.stringify(updatedProfiles));
    
    // Dispatch custom event to notify parent component
    window.dispatchEvent(new Event('profileSaved'));
  };

  if (!isOpen) return null;

  return (
    <>
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 10000
    }}>
      <div style={{
        backgroundColor: 'var(--bg-primary)',
        borderRadius: '12px',
        width: '90vw',
        maxWidth: '1200px',
        height: '80vh',
        border: '1px solid var(--bg-tertiary)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden'
      }}>
        {/* Header */}
        <div style={{
          padding: '20px 24px',
          borderBottom: '1px solid var(--bg-tertiary)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <h2 style={{ margin: '0 0 4px 0', fontSize: '20px', color: 'var(--text-primary)' }}>
              ⚙️ Terminal Settings
            </h2>
            <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '14px' }}>
              Manage profiles, layouts, and personalization
            </p>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              fontSize: '24px',
              cursor: 'pointer',
              padding: '4px'
            }}
          >
            ×
          </button>
        </div>

        {/* Navigation Tabs */}
        <div style={{
          display: 'flex',
          borderBottom: '1px solid var(--bg-tertiary)',
          padding: '0 24px'
        }}>
          {[
            { id: 'profiles', label: '👤 Profiles' },
            { id: 'layouts', label: '🪟 Window Layouts' },
            { id: 'wallet', label: '🔗 Wallet' },
            { id: 'appearance', label: '🎨 Appearance' },
            { id: 'shortcuts', label: '⌨️ Shortcuts' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              style={{
                background: 'none',
                border: 'none',
                color: activeTab === tab.id ? 'var(--primary-500)' : 'var(--text-muted)',
                fontSize: '14px',
                fontWeight: activeTab === tab.id ? '600' : '400',
                cursor: 'pointer',
                padding: '16px 20px',
                borderBottom: activeTab === tab.id ? '2px solid var(--primary-500)' : '2px solid transparent',
                transition: 'all 0.2s ease'
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div style={{ flex: 1, padding: '24px', overflow: 'auto' }}>
          {activeTab === 'profiles' && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h3 style={{ margin: 0, fontSize: '18px', color: 'var(--text-primary)' }}>
                  👤 Terminal Profiles ({profiles.length})
                </h3>
                <button
                  onClick={() => setShowCreateProfile(true)}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: 'var(--primary-500)',
                    border: 'none',
                    borderRadius: '6px',
                    color: '#fff',
                    fontSize: '14px',
                    cursor: 'pointer'
                  }}
                >
                  + New Profile
                </button>
              </div>

              <div style={{ display: 'grid', gap: '16px' }}>
                {profiles.map((profile) => (
                  <div key={profile.id} style={{
                    backgroundColor: profile.isActive ? 'var(--bg-secondary)' : 'var(--bg-primary)',
                    borderRadius: '12px',
                    padding: '20px',
                    border: `1px solid ${profile.isActive ? 'var(--primary-500)' : 'var(--bg-tertiary)'}`,
                    position: 'relative'
                  }}>
                    {profile.isActive && (
                      <div style={{
                        position: 'absolute',
                        top: '12px',
                        right: '12px',
                        padding: '4px 8px',
                        backgroundColor: 'var(--primary-500)',
                        borderRadius: '12px',
                        fontSize: '10px',
                        fontWeight: '600',
                        color: '#fff'
                      }}>
                        ACTIVE
                      </div>
                    )}

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                      <div>
                        <div style={{ fontSize: '18px', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '4px' }}>
                          {profile.name}
                        </div>
                        <div style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                          {profile.description}
                        </div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                          Layout: {profile.layout.name} • Theme: {profile.theme} • Font: {profile.fontFamily}
                        </div>
                      </div>
                      <div style={{ display: 'flex', gap: '8px' }}>
                        {!profile.isActive && (
                          <button
                            onClick={() => switchProfile(profile)}
                            style={{
                              padding: '6px 12px',
                              backgroundColor: 'var(--primary-500)',
                              border: 'none',
                              borderRadius: '4px',
                              color: '#fff',
                              fontSize: '12px',
                              cursor: 'pointer'
                            }}
                          >
                            Activate
                          </button>
                        )}
                        <button
                          onClick={() => handleEditProfile(profile)}
                          style={{
                            padding: '6px 12px',
                            backgroundColor: 'transparent',
                            border: '1px solid var(--warning-500)',
                            borderRadius: '4px',
                            color: 'var(--warning-500)',
                            fontSize: '12px',
                            cursor: 'pointer'
                          }}
                        >
                          Edit
                        </button>
                        <button
                          onClick={() => duplicateProfile(profile)}
                          style={{
                            padding: '6px 12px',
                            backgroundColor: 'transparent',
                            border: '1px solid var(--primary-500)',
                            borderRadius: '4px',
                            color: 'var(--primary-500)',
                            fontSize: '12px',
                            cursor: 'pointer'
                          }}
                        >
                          Duplicate
                        </button>
                        <button
                          onClick={() => deleteProfile(profile.id)}
                          style={{
                            padding: '6px 12px',
                            backgroundColor: 'transparent',
                            border: '1px solid var(--danger-500)',
                            borderRadius: '4px',
                            color: 'var(--danger-500)',
                            fontSize: '12px',
                            cursor: 'pointer'
                          }}
                        >
                          Delete
                        </button>
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px' }}>
                      <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Commands</div>
                        <div style={{ fontSize: '14px', color: 'var(--text-primary)' }}>
                          {profile.layout.commands.length} selected
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Font Size</div>
                        <div style={{ fontSize: '14px', color: 'var(--text-primary)' }}>
                          {profile.fontSize}px
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Max Windows</div>
                        <div style={{ fontSize: '14px', color: 'var(--text-primary)' }}>
                          {profile.windowDefaults?.maxWindows || 8}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Shortcuts</div>
                        <div style={{ fontSize: '14px', color: 'var(--text-primary)' }}>
                          {Object.keys(profile.shortcuts).length} configured
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'layouts' && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h3 style={{ margin: 0, fontSize: '18px', color: 'var(--text-primary)' }}>
                  🪟 Window Layouts ({savedLayouts.length})
                </h3>
                {onSaveLayout && (
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <input
                      type="text"
                      value={newLayoutName}
                      onChange={(e) => setNewLayoutName(e.target.value)}
                      placeholder="Layout name..."
                      style={{
                        padding: '6px 12px',
                        backgroundColor: 'var(--bg-secondary)',
                        border: '1px solid var(--border-base)',
                        borderRadius: '4px',
                        color: 'var(--text-primary)',
                        fontSize: '14px',
                        fontFamily: "'JetBrains Mono', monospace"
                      }}
                    />
                    <button
                      onClick={() => {
                        if (newLayoutName.trim() && onSaveLayout) {
                          onSaveLayout(newLayoutName.trim());
                          setNewLayoutName('');
                          alert(`✅ Layout "${newLayoutName.trim()}" saved!`);
                        }
                      }}
                      disabled={!newLayoutName.trim()}
                      style={{
                        padding: '8px 16px',
                        backgroundColor: newLayoutName.trim() ? 'var(--primary-500)' : 'var(--bg-tertiary)',
                        border: 'none',
                        borderRadius: '6px',
                        color: '#fff',
                        fontSize: '14px',
                        cursor: newLayoutName.trim() ? 'pointer' : 'not-allowed'
                      }}
                    >
                      💾 Save Current
                    </button>
                  </div>
                )}
              </div>

              <div style={{ display: 'grid', gap: '16px' }}>
                {savedLayouts.length === 0 ? (
                  <div style={{
                    backgroundColor: 'var(--bg-primary)',
                    borderRadius: '12px',
                    padding: '60px 20px',
                    textAlign: 'center',
                    border: '1px solid var(--bg-tertiary)'
                  }}>
                    <div style={{ fontSize: '48px', marginBottom: '20px' }}>🪟</div>
                    <div style={{ fontSize: '24px', marginBottom: '12px', color: 'var(--text-primary)' }}>
                      No Saved Window Layouts
                    </div>
                    <div style={{ fontSize: '16px', color: 'var(--text-muted)' }}>
                      Arrange your windows and save your layout with "💾 Save Current"
                    </div>
                  </div>
                ) : (
                  savedLayouts.map((layout) => (
                    <div key={layout.id} style={{
                      backgroundColor: 'var(--bg-primary)',
                      borderRadius: '12px',
                      padding: '20px',
                      border: '1px solid var(--bg-tertiary)'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                        <div>
                          <div style={{ fontSize: '18px', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '4px' }}>
                            {layout.name}
                          </div>
                          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                            Saved: {new Date(layout.timestamp).toLocaleString()} • {layout.windows?.length || 0} windows
                          </div>
                        </div>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          {onLoadLayout && (
                            <button
                              onClick={() => { onLoadLayout(layout); onClose(); }}
                              style={{
                                padding: '6px 12px',
                                backgroundColor: 'var(--primary-500)',
                                border: 'none',
                                borderRadius: '4px',
                                color: '#fff',
                                fontSize: '12px',
                                cursor: 'pointer'
                              }}
                            >
                              📂 Load
                            </button>
                          )}
                          <button
                            onClick={() => {
                              const updated = savedLayouts.filter(l => l.id !== layout.id);
                              localStorage.setItem('quantdesk_layouts', JSON.stringify(updated));
                              window.location.reload();
                            }}
                            style={{
                              padding: '6px 12px',
                              backgroundColor: 'transparent',
                              border: '1px solid var(--danger-500)',
                              borderRadius: '4px',
                              color: 'var(--danger-500)',
                              fontSize: '12px',
                              cursor: 'pointer'
                            }}
                          >
                            🗑️ Delete
                          </button>
                        </div>
                      </div>

                      <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>
                          Windows ({layout.windows?.length || 0})
                        </div>
                        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                          {layout.windows?.map((window: any, idx: number) => (
                            <span key={idx} style={{
                              padding: '4px 8px',
                              backgroundColor: 'var(--bg-tertiary)',
                              borderRadius: '4px',
                              fontSize: '12px',
                              color: 'var(--text-primary)'
                            }}>
                              {window.type || window.title || 'Window'}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {activeTab === 'wallet' && (
            <div>
              <h3 style={{ margin: '0 0 20px 0', fontSize: '18px', color: 'var(--text-primary)' }}>
                🔗 Wallet Connection
              </h3>
              
              {/* Wallet Status */}
              <div style={{
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: '12px',
                padding: '20px',
                border: '1px solid var(--bg-tertiary)',
                marginBottom: '20px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <div>
                    <div style={{ fontSize: '18px', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '4px' }}>
                      Wallet Status
                    </div>
                    <div style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                      Connect your wallet to access trading features
                    </div>
                  </div>
                </div>
                
                {/* Use the same WalletButton as lite version */}
                <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
                  <WalletButton />
                </div>
              </div>

              {/* Wallet Settings */}
              <div style={{ marginTop: '24px' }}>
                <h4 style={{ margin: '0 0 12px 0', fontSize: '16px', color: 'var(--text-primary)' }}>
                  Wallet Settings
                </h4>
                <div style={{
                  backgroundColor: 'var(--bg-primary)',
                  borderRadius: '12px',
                  padding: '16px',
                  border: '1px solid var(--bg-tertiary)'
                }}>
                  <div style={{ display: 'grid', gap: '12px' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                      <input type="checkbox" style={{ transform: 'scale(1.2)' }} />
                      <span style={{ color: 'var(--text-primary)', fontSize: '14px' }}>Auto-connect on startup</span>
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                      <input type="checkbox" style={{ transform: 'scale(1.2)' }} />
                      <span style={{ color: 'var(--text-primary)', fontSize: '14px' }}>Remember wallet connection</span>
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                      <input type="checkbox" style={{ transform: 'scale(1.2)' }} />
                      <span style={{ color: 'var(--text-primary)', fontSize: '14px' }}>Show balance in taskbar</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'appearance' && (
            <div>
              <h3 style={{ margin: '0 0 20px 0', fontSize: '18px', color: 'var(--text-primary)' }}>
                🎨 Appearance Settings
              </h3>
              <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
                <div style={{ fontSize: '48px', marginBottom: '20px' }}>🎨</div>
                <div style={{ fontSize: '24px', marginBottom: '12px', color: 'var(--text-primary)' }}>
                  Appearance Customization
                </div>
                <div style={{ fontSize: '16px' }}>
                  Theme customization, font settings, and visual preferences coming soon...
                </div>
              </div>
            </div>
          )}

          {activeTab === 'shortcuts' && (
            <div>
              <h3 style={{ margin: '0 0 20px 0', fontSize: '18px', color: 'var(--text-primary)' }}>
                ⌨️ Keyboard Shortcuts
              </h3>
              <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
                <div style={{ fontSize: '48px', marginBottom: '20px' }}>⌨️</div>
                <div style={{ fontSize: '24px', marginBottom: '12px', color: 'var(--text-primary)' }}>
                  Shortcut Management
                </div>
                <div style={{ fontSize: '16px' }}>
                  Custom keyboard shortcuts and hotkeys coming soon...
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>

    {/* Edit Profile Modal */}
    {showEditProfileModal && editingProfile !== null && editingProfile && (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 20000
      }}>
        <div style={{
          backgroundColor: 'var(--bg-primary)',
          borderRadius: '12px',
          width: '90vw',
          maxWidth: '800px',
          maxHeight: '90vh',
          overflow: 'auto',
          border: '1px solid var(--bg-tertiary)',
          padding: '24px'
        }}>
          <h2 style={{ margin: '0 0 20px 0', fontSize: '20px', color: 'var(--text-primary)' }}>
            Edit Profile: {editingProfile.name}
          </h2>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Profile Name */}
            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-primary)', fontSize: '14px' }}>
                Profile Name
              </label>
              <input
                type="text"
                value={editingProfile.name}
                onChange={(e) => setEditingProfile({ ...editingProfile, name: e.target.value })}
                style={{
                  width: '100%',
                  padding: '10px',
                  backgroundColor: 'var(--bg-secondary)',
                  border: '1px solid var(--border-base)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '14px',
                  fontFamily: 'JetBrains Mono'
                }}
              />
            </div>

            {/* Description */}
            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-primary)', fontSize: '14px' }}>
                Description
              </label>
              <textarea
                value={editingProfile.description}
                onChange={(e) => setEditingProfile({ ...editingProfile, description: e.target.value })}
                style={{
                  width: '100%',
                  padding: '10px',
                  backgroundColor: 'var(--bg-secondary)',
                  border: '1px solid var(--border-base)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '14px',
                  fontFamily: 'JetBrains Mono',
                  minHeight: '60px'
                }}
              />
            </div>

            {/* Font Settings */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-primary)', fontSize: '14px' }}>
                  Font Size
                </label>
                <input
                  type="number"
                  value={editingProfile.fontSize}
                  onChange={(e) => setEditingProfile({ ...editingProfile, fontSize: parseInt(e.target.value) || 14 })}
                  min="8"
                  max="24"
                  style={{
                    width: '100%',
                    padding: '10px',
                    backgroundColor: 'var(--bg-secondary)',
                    border: '1px solid var(--border-base)',
                    borderRadius: '6px',
                    color: 'var(--text-primary)',
                    fontSize: '14px',
                    fontFamily: 'JetBrains Mono'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-primary)', fontSize: '14px' }}>
                  Font Family
                </label>
                <select
                  value={editingProfile.fontFamily}
                  onChange={(e) => setEditingProfile({ ...editingProfile, fontFamily: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px',
                    backgroundColor: 'var(--bg-secondary)',
                    border: '1px solid var(--border-base)',
                    borderRadius: '6px',
                    color: 'var(--text-primary)',
                    fontSize: '14px',
                    fontFamily: 'JetBrains Mono'
                  }}
                >
                  <option value="JetBrains Mono">JetBrains Mono</option>
                  <option value="Fira Code">Fira Code</option>
                  <option value="Consolas">Consolas</option>
                  <option value="Monaco">Monaco</option>
                  <option value="Courier New">Courier New</option>
                </select>
              </div>
            </div>

            {/* Theme */}
            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-primary)', fontSize: '14px' }}>
                Theme
              </label>
              <select
                value={editingProfile.theme}
                onChange={(e) => setEditingProfile({ ...editingProfile, theme: e.target.value as 'dark' | 'light' | 'terminal' })}
                style={{
                  width: '100%',
                  padding: '10px',
                  backgroundColor: 'var(--bg-secondary)',
                  border: '1px solid var(--border-base)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '14px',
                  fontFamily: 'JetBrains Mono'
                }}
              >
                <option value="dark">Dark</option>
                <option value="light">Light</option>
                <option value="terminal">Terminal</option>
              </select>
            </div>

            {/* Window Settings */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-primary)', fontSize: '14px' }}>
                  Default Width
                </label>
                <input
                  type="number"
                  value={editingProfile?.windowDefaults?.defaultWidth || 400}
                  onChange={(e) => setEditingProfile({
                    ...editingProfile,
                    windowDefaults: { 
                      ...(editingProfile?.windowDefaults || { defaultWidth: 400, defaultHeight: 300, maxWindows: 8 }),
                      defaultWidth: parseInt(e.target.value) || 400
                    }
                  })}
                  style={{
                    width: '100%',
                    padding: '10px',
                    backgroundColor: 'var(--bg-secondary)',
                    border: '1px solid var(--border-base)',
                    borderRadius: '6px',
                    color: 'var(--text-primary)',
                    fontSize: '14px',
                    fontFamily: 'JetBrains Mono'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-primary)', fontSize: '14px' }}>
                  Default Height
                </label>
                <input
                  type="number"
                  value={editingProfile?.windowDefaults?.defaultHeight || 300}
                  onChange={(e) => setEditingProfile({
                    ...editingProfile,
                    windowDefaults: { 
                      ...(editingProfile?.windowDefaults || { defaultWidth: 400, defaultHeight: 300, maxWindows: 8 }),
                      defaultHeight: parseInt(e.target.value) || 300
                    }
                  })}
                  style={{
                    width: '100%',
                    padding: '10px',
                    backgroundColor: 'var(--bg-secondary)',
                    border: '1px solid var(--border-base)',
                    borderRadius: '6px',
                    color: 'var(--text-primary)',
                    fontSize: '14px',
                    fontFamily: 'JetBrains Mono'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-primary)', fontSize: '14px' }}>
                  Max Windows
                </label>
                <input
                  type="number"
                  value={editingProfile?.windowDefaults?.maxWindows || 8}
                  onChange={(e) => setEditingProfile({
                    ...editingProfile,
                    windowDefaults: { 
                      ...(editingProfile?.windowDefaults || { defaultWidth: 400, defaultHeight: 300, maxWindows: 8 }),
                      maxWindows: parseInt(e.target.value) || 8
                    }
                  })}
                  min="1"
                  max="50"
                  style={{
                    width: '100%',
                    padding: '10px',
                    backgroundColor: 'var(--bg-secondary)',
                    border: '1px solid var(--border-base)',
                    borderRadius: '6px',
                    color: 'var(--text-primary)',
                    fontSize: '14px',
                    fontFamily: 'JetBrains Mono'
                  }}
                />
              </div>
            </div>

            {/* Commands Selection */}
            <div>
              <label style={{ display: 'block', marginBottom: '12px', color: 'var(--text-primary)', fontSize: '14px' }}>
                Available Commands ({editingProfile.layout.commands.length} selected)
              </label>
              <div style={{
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: '6px',
                padding: '12px',
                maxHeight: '200px',
                overflow: 'auto',
                display: 'flex',
                flexWrap: 'wrap',
                gap: '8px'
              }}>
                {visibleCommands.map(cmd => (
                  <label key={cmd.command} style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={editingProfile.layout.commands.includes(cmd.command)}
                      onChange={(e) => {
                        const currentCommands = editingProfile.layout.commands;
                        const newCommands = e.target.checked
                          ? [...currentCommands, cmd.command]
                          : currentCommands.filter(c => c !== cmd.command);
                        setEditingProfile({
                          ...editingProfile,
                          layout: { ...editingProfile.layout, commands: newCommands }
                        });
                      }}
                      style={{ marginRight: '6px' }}
                    />
                    <span style={{
                      padding: '4px 8px',
                      backgroundColor: 'var(--bg-tertiary)',
                      borderRadius: '4px',
                      fontSize: '12px',
                      color: 'var(--text-primary)'
                    }}>
                      {cmd.command} - {cmd.description}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', marginTop: '24px' }}>
            <button
              onClick={() => {
                setShowEditProfileModal(false);
                setEditingProfile(null);
              }}
              style={{
                padding: '10px 20px',
                backgroundColor: 'transparent',
                border: '1px solid var(--danger-500)',
                borderRadius: '6px',
                color: 'var(--danger-500)',
                fontSize: '14px',
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
            <button
              onClick={saveProfileEdits}
              style={{
                padding: '10px 20px',
                backgroundColor: 'var(--primary-500)',
                border: 'none',
                borderRadius: '6px',
                color: '#fff',
                fontSize: '14px',
                cursor: 'pointer'
              }}
            >
              Save Changes
            </button>
          </div>
        </div>
      </div>
    )}
    </>
  );
};

export default ProTerminalSettings;
