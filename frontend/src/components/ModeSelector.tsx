import React from 'react'

interface ModeSelectorProps {
  currentMode: 'lite' | 'pro'
  onModeChange: (mode: 'lite' | 'pro') => void
  style?: 'floating' | 'taskbar'
}

const ModeSelector: React.FC<ModeSelectorProps> = ({ currentMode, onModeChange, style = 'floating' }) => {
  if (style === 'taskbar') {
    return (
      <div style={{
        display: 'flex',
        gap: '4px',
        alignItems: 'center'
      }}>
        <button
          onClick={() => onModeChange('lite')}
          style={{
            padding: '2px 8px',
            backgroundColor: currentMode === 'lite' ? '#52c41a' : 'transparent',
            color: currentMode === 'lite' ? '#000' : '#666',
            border: '1px solid #333',
            borderRadius: '2px',
            cursor: 'pointer',
            fontSize: '10px',
            fontWeight: 'bold'
          }}
          onMouseEnter={(e) => { if (currentMode !== 'lite') (e.currentTarget as HTMLButtonElement).style.color = '#fff' }}
          onMouseLeave={(e) => { if (currentMode !== 'lite') (e.currentTarget as HTMLButtonElement).style.color = '#666' }}
        >
          LITE
        </button>
        <button
          onClick={() => onModeChange('pro')}
          style={{
            padding: '2px 8px',
            backgroundColor: currentMode === 'pro' ? '#1890ff' : 'transparent',
            color: currentMode === 'pro' ? '#fff' : '#666',
            border: '1px solid #333',
            borderRadius: '2px',
            cursor: 'pointer',
            fontSize: '10px',
            fontWeight: 'bold'
          }}
          onMouseEnter={(e) => { if (currentMode !== 'pro') (e.currentTarget as HTMLButtonElement).style.color = '#fff' }}
          onMouseLeave={(e) => { if (currentMode !== 'pro') (e.currentTarget as HTMLButtonElement).style.color = '#666' }}
        >
          PRO
        </button>
      </div>
    )
  }

  return (
    <div style={{
      position: 'fixed',
      top: '20px',
      right: '20px',
      zIndex: 10000,
      display: 'flex',
      gap: '8px',
      backgroundColor: '#1a1a1a',
      padding: '8px',
      borderRadius: '4px',
      border: '1px solid #333'
    }}>
      <button
        onClick={() => onModeChange('lite')}
        style={{
          padding: '8px 16px',
          backgroundColor: currentMode === 'lite' ? '#52c41a' : '#333',
          color: currentMode === 'lite' ? '#000' : '#fff',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '12px',
          fontWeight: 'bold'
        }}
      >
        LITE MODE
      </button>
      <button
        onClick={() => onModeChange('pro')}
        style={{
          padding: '8px 16px',
          backgroundColor: currentMode === 'pro' ? '#1890ff' : '#333',
          color: currentMode === 'pro' ? '#fff' : '#fff',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '12px',
          fontWeight: 'bold'
        }}
      >
        PRO MODE
      </button>
    </div>
  )
}

export default ModeSelector


