import React from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import ModeSelector from './ModeSelector'

const BottomTaskbar: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()

  const [mode, setMode] = React.useState<'lite' | 'pro'>(() => (localStorage.getItem('quantdesk_ui_mode') as 'lite' | 'pro') || 'lite')
  const [now, setNow] = React.useState(new Date())

  React.useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  const onModeChange = (next: 'lite' | 'pro') => {
    setMode(next)
    localStorage.setItem('quantdesk_ui_mode', next)
    navigate(next === 'pro' ? '/pro' : '/lite')
  }

  const currentPage = React.useMemo(() => {
    if (location.pathname.startsWith('/lite')) {
      const hash = (location.hash || '').replace('#', '') || 'overview'
      return hash.replace('-', ' ')
    }
    if (location.pathname.startsWith('/pro')) return 'pro terminal'
    return 'dashboard'
  }, [location.pathname, location.hash])

  return (
    <div style={{
      position: 'fixed',
      bottom: 0,
      left: 0,
      right: 0,
      height: '28px',
      backgroundColor: '#0a0a0a',
      borderTop: '1px solid #333',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 8px',
      fontSize: '11px',
      zIndex: 999,
      fontFamily: 'Monaco, Consolas, "Courier New", monospace'
    }}>
      {/* Left Side - Mode & Tools (verbatim style) */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <ModeSelector currentMode={mode} onModeChange={onModeChange} style="taskbar" />

        <button
          onClick={() => navigate('/lite#overview')}
          style={{
            background: 'none',
            border: '1px solid #333',
            color: '#999',
            padding: '2px 8px',
            fontSize: '10px',
            cursor: 'pointer',
            borderRadius: '2px'
          }}
          onMouseEnter={(e) => (e.currentTarget as HTMLButtonElement).style.color = '#fff'}
          onMouseLeave={(e) => (e.currentTarget as HTMLButtonElement).style.color = '#999'}
        >
          Dashboard
        </button>

        <button
          onClick={() => alert(`Debug Info:\n- Mode: ${mode}\n- Page: ${currentPage}`)}
          style={{
            background: 'none',
            border: '1px solid #333',
            color: '#999',
            padding: '2px 8px',
            fontSize: '10px',
            cursor: 'pointer',
            borderRadius: '2px'
          }}
          onMouseEnter={(e) => (e.currentTarget as HTMLButtonElement).style.color = '#fff'}
          onMouseLeave={(e) => (e.currentTarget as HTMLButtonElement).style.color = '#999'}
        >
          Debug
        </button>
      </div>

      {/* Center - Current Page */}
      <div style={{ color: '#666', fontSize: '10px', textTransform: 'capitalize' }}>
        {currentPage}
      </div>

      {/* Right Side - Version & Status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: '#666', fontSize: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#52c41a' }}></div>
          <span>Ready</span>
        </div>
        <div>v1.0.1</div>
        <div style={{ padding: '2px 6px', backgroundColor: '#1a1a1a', borderRadius: '2px', fontSize: '9px' }}>
          {now.toLocaleTimeString()}
        </div>
      </div>
    </div>
  )
}

export default BottomTaskbar

