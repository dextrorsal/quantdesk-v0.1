import React, { useState } from 'react'
import { useTheme } from '../contexts/ThemeContext'

const ThemeToggle: React.FC = () => {
  const { theme } = useTheme()
  const [open, setOpen] = useState(false)

  const palettes = {
    lite: [
      { name: 'Ocean Blue', primary: '#3b82f6' },
      { name: 'Forest Green', primary: '#10b981' },
      { name: 'Royal Purple', primary: '#8b5cf6' },
      { name: 'Crimson Red', primary: '#ef4444' },
      { name: 'Sunset Orange', primary: '#f97316' },
      { name: 'Sky Cyan', primary: '#06b6d4' }
    ],
    pro: [
      { name: 'Fire Orange', primary: '#f97316' },
      { name: 'Electric Blue', primary: '#3b82f6' },
      { name: 'Neon Green', primary: '#22c55e' },
      { name: 'Hot Pink', primary: '#ec4899' },
      { name: 'Gold Yellow', primary: '#eab308' },
      { name: 'Deep Purple', primary: '#7c3aed' }
    ]
  } as const

  const handleSelect = (color: string) => {
    localStorage.setItem(`quantdesk_${theme}_color`, color)
    // apply immediately (ThemeProvider will also sync on next render)
    document.documentElement.style.setProperty('--primary-500', color)
    setOpen(false)
  }

  const currentColor = localStorage.getItem(`quantdesk_${theme}_color`) || palettes[theme][0].primary

  return (
    <div className="position-relative">
      <button
        className="btn btn-sm btn-outline-light d-flex align-items-center"
        style={{
          background: 'transparent',
          borderColor: 'var(--bg-tertiary, #1a1a1a)',
          fontFamily: 'Monaco, Consolas, "Courier New", monospace'
        }}
        onClick={() => setOpen(!open)}
        title="Customize theme colors"
      >
        <span
          style={{
            display: 'inline-block', width: 12, height: 12, borderRadius: 9999,
            background: currentColor, marginRight: 8
          }}
        />
        <span style={{ fontSize: 12, color: 'var(--text-muted, #a3a3a3)' }}>{theme.toUpperCase()} COLORS</span>
      </button>

      {open && (
        <div
          className="position-absolute p-2"
          style={{ right: 0, top: '100%', background: '#0a0a0a', border: '1px solid #1a1a1a', zIndex: 1000, minWidth: 200 }}
        >
          <div className="row g-2" style={{ width: 220 }}>
            {palettes[theme].map((p) => (
              <div key={p.name} className="col-4">
                <button
                  onClick={() => handleSelect(p.primary)}
                  className="w-100"
                  style={{
                    height: 36,
                    background: p.primary,
                    border: '1px solid rgba(255,255,255,0.2)'
                  }}
                  title={p.name}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default ThemeToggle


