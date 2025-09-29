import React, { createContext, useContext, useEffect, useState } from 'react'

type Theme = 'lite' | 'pro'

interface ThemeContextType {
  theme: Theme
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export const useTheme = () => {
  const context = useContext(ThemeContext)
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

interface ThemeProviderProps {
  children: React.ReactNode
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setThemeState] = useState<Theme>('lite')

  // Load theme from localStorage on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('quantdesk_theme') as Theme
    if (savedTheme && (savedTheme === 'lite' || savedTheme === 'pro')) {
      setThemeState(savedTheme)
    }
  }, [])

  // Apply theme to document and load saved colors
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('quantdesk_theme', theme)
    
    // Load saved color for this theme
    const savedColor = localStorage.getItem(`quantdesk_${theme}_color`)
    if (savedColor) {
      document.documentElement.style.setProperty('--primary-500', savedColor)
      
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
      
      const rgb = hexToRgb(savedColor)
      if (rgb) {
        const darkerRgb = {
          r: Math.max(0, rgb.r - 30),
          g: Math.max(0, rgb.g - 30),
          b: Math.max(0, rgb.b - 30)
        }
        const darkerColor = rgbToHex(darkerRgb.r, darkerRgb.g, darkerRgb.b)
        document.documentElement.style.setProperty('--primary-600', darkerColor)
        
        // Generate success colors based on primary color
        // For green themes, use the primary color as success
        // For other themes, create a complementary success color
        const isGreenish = rgb.g > Math.max(rgb.r, rgb.b) + 50
        if (isGreenish) {
          // If it's already greenish, use it as success
          document.documentElement.style.setProperty('--success-500', savedColor)
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
    } else {
      // Reset to default colors
      const defaultColors = {
        lite: '#3b82f6',
        pro: '#f97316'
      }
      const defaultSecondaryColors = {
        lite: '#2563eb',
        pro: '#ea580c'
      }
      const defaultSuccessColors = {
        lite: '#22c55e',
        pro: '#22c55e'
      }
      const defaultSuccessSecondaryColors = {
        lite: '#16a34a',
        pro: '#16a34a'
      }
      document.documentElement.style.setProperty('--primary-500', defaultColors[theme])
      document.documentElement.style.setProperty('--primary-600', defaultSecondaryColors[theme])
      document.documentElement.style.setProperty('--success-500', defaultSuccessColors[theme])
      document.documentElement.style.setProperty('--success-600', defaultSuccessSecondaryColors[theme])
    }
  }, [theme])

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme)
  }

  const toggleTheme = () => {
    setThemeState(prev => prev === 'lite' ? 'pro' : 'lite')
  }

  const value = {
    theme,
    setTheme,
    toggleTheme,
  }

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  )
}
