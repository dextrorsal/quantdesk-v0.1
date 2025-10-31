import React, { useState } from 'react'
import { useTheme } from '../contexts/ThemeContext'

const ThemeToggle: React.FC = () => {
  const { theme, setTheme } = useTheme()
  const [showColorPicker, setShowColorPicker] = useState(false)

  // Color palettes for each mode
  const colorPalettes = {
    lite: [
      { name: 'Ocean Blue', primary: '#3b82f6', secondary: '#1e40af' },
      { name: 'Forest Green', primary: '#10b981', secondary: '#047857' },
      { name: 'Royal Purple', primary: '#8b5cf6', secondary: '#6d28d9' },
      { name: 'Crimson Red', primary: '#ef4444', secondary: '#dc2626' },
      { name: 'Sunset Orange', primary: '#f97316', secondary: '#ea580c' },
      { name: 'Sky Cyan', primary: '#06b6d4', secondary: '#0891b2' }
    ],
    pro: [
      { name: 'Fire Orange', primary: '#f97316', secondary: '#ea580c' },
      { name: 'Electric Blue', primary: '#3b82f6', secondary: '#2563eb' },
      { name: 'Neon Green', primary: '#22c55e', secondary: '#16a34a' },
      { name: 'Hot Pink', primary: '#ec4899', secondary: '#db2777' },
      { name: 'Gold Yellow', primary: '#eab308', secondary: '#ca8a04' },
      { name: 'Deep Purple', primary: '#7c3aed', secondary: '#6d28d9' }
    ]
  }

  const currentPalettes = colorPalettes[theme]
  const currentColor = localStorage.getItem(`quantdesk_${theme}_color`) || currentPalettes[0].primary

  const handleColorSelect = (color: string) => {
    localStorage.setItem(`quantdesk_${theme}_color`, color)
    
    // Calculate darker shade for secondary color
    const hexToRgb = (hex: string) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
      return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
      } : null
    }
    
    const rgbToHex = (r: number, g: number, b: number) => {
      return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)
    }
    
    const rgb = hexToRgb(color)
    if (rgb) {
      const darkerRgb = {
        r: Math.max(0, rgb.r - 30),
        g: Math.max(0, rgb.g - 30),
        b: Math.max(0, rgb.b - 30)
      }
      const darkerColor = rgbToHex(darkerRgb.r, darkerRgb.g, darkerRgb.b)
      
      // Update CSS custom properties
      document.documentElement.style.setProperty('--primary-500', color)
      document.documentElement.style.setProperty('--primary-600', darkerColor)
      
      // Generate success colors based on primary color
      const isGreenish = rgb.g > Math.max(rgb.r, rgb.b) + 50
      if (isGreenish) {
        // If it's already greenish, use it as success
        document.documentElement.style.setProperty('--success-500', color)
        document.documentElement.style.setProperty('--success-600', darkerColor)
      } else {
        // For non-green themes, create a green success color that complements the primary
        const successRgb = {
          r: Math.max(0, rgb.r - 50),
          g: Math.min(255, rgb.g + 80),
          b: Math.max(0, rgb.b - 30)
        }
        const successColor = rgbToHex(successRgb.r, successRgb.g, successRgb.b)
        const successDarkerRgb = {
          r: Math.max(0, successRgb.r - 30),
          g: Math.max(0, successRgb.g - 30),
          b: Math.max(0, successRgb.b - 30)
        }
        const successDarkerColor = rgbToHex(successDarkerRgb.r, successDarkerRgb.g, successDarkerRgb.b)
        document.documentElement.style.setProperty('--success-500', successColor)
        document.documentElement.style.setProperty('--success-600', successDarkerColor)
      }
    }
    
    setShowColorPicker(false)
  }

  // Check if we're in pro mode (terminal-style interface)
  const isProMode = window.location.pathname.includes('/pro') || document.documentElement.getAttribute('data-theme') === 'pro'

  return (
    <div className="relative">
      <button
        onClick={() => setShowColorPicker(!showColorPicker)}
        className={isProMode ? 
          "flex items-center gap-1 px-2 py-1 text-xs transition-all duration-200 hover:bg-bg-secondary border border-bg-tertiary rounded" :
          "btn-ghost flex items-center gap-2 px-3 py-2 rounded-lg transition-all duration-200 hover:bg-bg-secondary"
        }
        title="Customize theme colors"
        style={isProMode ? {
          backgroundColor: 'transparent',
          color: 'var(--text-muted)',
          fontSize: '10px',
          fontFamily: 'Monaco, Consolas, "Courier New", monospace'
        } : {}}
        onMouseEnter={isProMode ? (e) => e.currentTarget.style.color = 'var(--text-primary)' : undefined}
        onMouseLeave={isProMode ? (e) => e.currentTarget.style.color = 'var(--text-muted)' : undefined}
      >
        <div className="flex items-center gap-1">
          {/* Color Circle */}
          <div 
            className={`rounded-full border-2 border-white/20 ${isProMode ? 'w-3 h-3' : 'w-4 h-4'}`}
            style={{ backgroundColor: currentColor }}
          />
          
          {/* Theme Label - Only show in lite mode */}
          {!isProMode && (
            <span className="text-sm font-medium capitalize">
              {theme} Colors
            </span>
          )}
          
          {/* Dropdown Arrow - Only show in lite mode */}
          {!isProMode && (
            <svg 
              className={`w-3 h-3 transition-transform duration-200 ${showColorPicker ? 'rotate-180' : ''}`}
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M19 9l-7 7-7-7" 
              />
            </svg>
          )}
        </div>
      </button>

      {/* Color Picker Dropdown */}
      {showColorPicker && (
        <div className={`absolute right-0 bg-bg-secondary border border-bg-tertiary rounded-lg shadow-lg z-50 ${isProMode ? 'p-2 min-w-[150px] bottom-full mb-2' : 'p-4 min-w-[200px] top-full mt-2'}`}>
          <div className={`text-xs text-text-muted mb-3 ${isProMode ? 'text-center' : ''}`}>
            {isProMode ? `${theme.toUpperCase()}` : `Customize ${theme.toUpperCase()} colors`}
          </div>
          <div className={`grid gap-2 ${isProMode ? 'grid-cols-2' : 'grid-cols-3'}`}>
            {currentPalettes.map((palette, index) => (
              <button
                key={index}
                onClick={() => handleColorSelect(palette.primary)}
                className={`flex flex-col items-center rounded-lg hover:bg-bg-tertiary transition-colors ${isProMode ? 'p-1' : 'p-2'}`}
                title={palette.name}
              >
                <div 
                  className={`rounded-full border-2 border-white/20 mb-1 ${isProMode ? 'w-6 h-6' : 'w-8 h-8'}`}
                  style={{ backgroundColor: palette.primary }}
                />
                {!isProMode && (
                  <span className="text-xs text-text-secondary">{palette.name}</span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default ThemeToggle
