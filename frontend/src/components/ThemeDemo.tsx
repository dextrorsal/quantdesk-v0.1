import React from 'react'
import { useTheme } from '../contexts/ThemeContext'

const ThemeDemo: React.FC = () => {
  const { theme } = useTheme()

  return (
    <div className="p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold gradient-text mb-2">QuantDesk Design System</h1>
        <p className="text-text-secondary">Current theme: <span className="font-semibold capitalize">{theme}</span></p>
      </div>

      {/* Color Palette Demo */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Color Palette</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="space-y-2">
            <div className="w-full h-16 rounded-lg" style={{ backgroundColor: 'var(--primary-500)' }}></div>
            <p className="text-sm text-text-secondary">Primary</p>
          </div>
          <div className="space-y-2">
            <div className="w-full h-16 rounded-lg" style={{ backgroundColor: 'var(--success-500)' }}></div>
            <p className="text-sm text-text-secondary">Success</p>
          </div>
          <div className="space-y-2">
            <div className="w-full h-16 rounded-lg" style={{ backgroundColor: 'var(--danger-500)' }}></div>
            <p className="text-sm text-text-secondary">Danger</p>
          </div>
          <div className="space-y-2">
            <div className="w-full h-16 rounded-lg" style={{ backgroundColor: 'var(--warning-500)' }}></div>
            <p className="text-sm text-text-secondary">Warning</p>
          </div>
        </div>
      </div>

      {/* Button Demo */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Buttons</h2>
        <div className="flex flex-wrap gap-4">
          <button className="btn-primary">Primary Button</button>
          <button className="btn-secondary">Secondary Button</button>
          <button className="btn-success">Success Button</button>
          <button className="btn-danger">Danger Button</button>
          <button className="btn-ghost">Ghost Button</button>
        </div>
      </div>

      {/* Cards Demo */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card">
          <h3 className="font-semibold mb-2">Standard Card</h3>
          <p className="text-text-secondary text-sm">This is a standard card with our theme colors.</p>
        </div>
        <div className="trading-card">
          <h3 className="font-semibold mb-2">Trading Card</h3>
          <p className="text-text-secondary text-sm">Elevated card for trading data.</p>
        </div>
        <div className="glass-card">
          <h3 className="font-semibold mb-2">Glass Card</h3>
          <p className="text-text-secondary text-sm">Glass morphism effect.</p>
        </div>
      </div>

      {/* Financial Data Demo */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Financial Data</h2>
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span>BTC/USDT</span>
            <span className="price-positive">+2.45%</span>
          </div>
          <div className="flex justify-between items-center">
            <span>ETH/USDT</span>
            <span className="price-negative">-1.23%</span>
          </div>
          <div className="flex justify-between items-center">
            <span>SOL/USDT</span>
            <span className="price-neutral">0.00%</span>
          </div>
        </div>
      </div>

      {/* Status Indicators */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Status Indicators</h2>
        <div className="space-y-2">
          <div className="flex items-center">
            <span className="status-indicator status-active"></span>
            <span>Active Connection</span>
          </div>
          <div className="flex items-center">
            <span className="status-indicator status-inactive"></span>
            <span>Inactive</span>
          </div>
          <div className="flex items-center">
            <span className="status-indicator status-warning"></span>
            <span>Warning</span>
          </div>
        </div>
      </div>

      {/* Input Demo */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Form Elements</h2>
        <div className="space-y-4">
          <input 
            type="text" 
            className="input w-full" 
            placeholder="Enter amount..."
          />
          <input 
            type="text" 
            className="input w-full" 
            placeholder="Enter symbol..."
          />
        </div>
      </div>
    </div>
  )
}

export default ThemeDemo
