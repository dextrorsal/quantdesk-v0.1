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
}

const ProTerminalSettings: React.FC<SettingsManagerProps> = ({ 
  isOpen, 
  onClose, 
  onProfileChange, 
  currentProfile 
}) => {
  const [profiles, setProfiles] = useState<TerminalProfile[]>([]);
  const [layouts, setLayouts] = useState<CommandLayout[]>([]);
  const [activeTab, setActiveTab] = useState<'profiles' | 'layouts' | 'appearance' | 'shortcuts' | 'wallet'>('profiles');
  const [showCreateProfile, setShowCreateProfile] = useState(false);
  const [showCreateLayout, setShowCreateLayout] = useState(false);
  const [editingProfile, setEditingProfile] = useState<TerminalProfile | null>(null);
  const [editingLayout, setEditingLayout] = useState<CommandLayout | null>(null);

  // Available commands from our pro terminal
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
    { command: 'FOCUS', description: 'Focus Mode', category: 'Tools' },
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
    { command: 'BT', description: 'Backtesting', category: 'Backtesting' }
  ];

  // Mock data initialization
  useEffect(() => {
    const mockLayouts: CommandLayout[] = [
      {
        id: '1',
        name: 'Trading Focus',
        description: 'Essential commands for active trading',
        commands: ['QM', 'CHART', 'ORDER', 'POSITIONS', 'AL', 'FOCUS'],
        isDefault: true,
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

    const mockProfiles: TerminalProfile[] = [
      {
        id: '1',
        name: 'P1 - Day Trader',
        description: 'High-frequency trading setup',
        layout: mockLayouts[0],
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
        isActive: true
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
  }, []);

  const switchProfile = (profile: TerminalProfile) => {
    setProfiles(prev => prev.map(p => ({ ...p, isActive: p.id === profile.id })));
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
    setProfiles(prev => [...prev, newProfile]);
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
    setProfiles(prev => prev.map(p => 
      p.id === profileId ? { ...p, ...updates } : p
    ));
  };

  const deleteProfile = (profileId: string) => {
    setProfiles(prev => prev.filter(p => p.id !== profileId));
  };

  const duplicateProfile = (profile: TerminalProfile) => {
    const duplicated: TerminalProfile = {
      ...profile,
      id: Date.now().toString(),
      name: `${profile.name} (Copy)`,
      isActive: false
    };
    setProfiles(prev => [...prev, duplicated]);
  };

  if (!isOpen) return null;

  return (
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
            { id: 'layouts', label: '📋 Layouts' },
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
                          onClick={() => setEditingProfile(profile)}
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
                          {profile.windowDefaults.maxWindows}
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
                  📋 Command Layouts ({layouts.length})
                </h3>
                <button
                  onClick={() => setShowCreateLayout(true)}
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
                  + New Layout
                </button>
              </div>

              <div style={{ display: 'grid', gap: '16px' }}>
                {layouts.map((layout) => (
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
                          {layout.isDefault && (
                            <span style={{ 
                              marginLeft: '8px',
                              padding: '2px 6px',
                              backgroundColor: 'var(--primary-500)',
                              borderRadius: '4px',
                              fontSize: '10px',
                              color: '#fff'
                            }}>
                              DEFAULT
                            </span>
                          )}
                        </div>
                        <div style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                          {layout.description}
                        </div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                          Created: {layout.createdAt.toLocaleDateString()} • 
                          Last used: {layout.lastUsed.toLocaleDateString()}
                        </div>
                      </div>
                      <div style={{ display: 'flex', gap: '8px' }}>
                        <button
                          onClick={() => setEditingLayout(layout)}
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
                          onClick={() => {
                            const duplicated: CommandLayout = {
                              ...layout,
                              id: Date.now().toString(),
                              name: `${layout.name} (Copy)`,
                              isDefault: false,
                              createdAt: new Date()
                            };
                            setLayouts(prev => [...prev, duplicated]);
                          }}
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
                      </div>
                    </div>

                    <div>
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>
                        Commands ({layout.commands.length})
                      </div>
                      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                        {layout.commands.map((cmd, idx) => {
                          const commandInfo = availableCommands.find(c => c.command === cmd);
                          return (
                            <span key={idx} style={{
                              padding: '4px 8px',
                              backgroundColor: 'var(--bg-tertiary)',
                              borderRadius: '4px',
                              fontSize: '12px',
                              color: 'var(--text-primary)',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '4px'
                            }}>
                              <span style={{ fontWeight: '600' }}>{cmd}</span>
                              <span style={{ color: 'var(--text-muted)' }}>
                                {commandInfo?.description || 'Unknown'}
                              </span>
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                ))}
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
  );
};

export default ProTerminalSettings;
