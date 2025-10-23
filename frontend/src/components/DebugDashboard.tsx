import React, { useState, useEffect } from 'react'

interface DebugInfo {
  component: string
  renderCount: number
  lastRender: number
  averageRenderTime: number
}

export const DebugDashboard: React.FC = () => {
  const [debugInfo, setDebugInfo] = useState<DebugInfo[]>([])
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    // Listen for debug messages from console
    const originalLog = console.log
    console.log = (...args) => {
      if (args[0]?.includes('rendered') && args[0]?.includes('times')) {
        const match = args[0].match(/(\w+) rendered (\d+) times/)
        if (match) {
          const [, component, count] = match
          setDebugInfo(prev => {
            const existing = prev.find(item => item.component === component)
            if (existing) {
              existing.renderCount = parseInt(count)
              existing.lastRender = Date.now()
              return [...prev]
            } else {
              return [...prev, {
                component,
                renderCount: parseInt(count),
                lastRender: Date.now(),
                averageRenderTime: 0
              }]
            }
          })
        }
      }
      originalLog(...args)
    }

    return () => {
      console.log = originalLog
    }
  }, [])

  // Only show in development
  if (!import.meta.env.DEV) {
    return null
  }

  return (
    <>
      {/* Toggle Button */}
      <button
        onClick={() => setIsVisible(!isVisible)}
        style={{
          position: 'fixed',
          top: '10px',
          right: '10px',
          background: '#1a1a1a',
          color: '#52c41a',
          border: '1px solid #52c41a',
          padding: '8px 12px',
          borderRadius: '4px',
          fontSize: '12px',
          cursor: 'pointer',
          zIndex: 10000,
          fontFamily: 'monospace'
        }}
      >
        DEBUG {isVisible ? '▼' : '▲'}
      </button>

      {/* Debug Panel */}
      {isVisible && (
        <div
          style={{
            position: 'fixed',
            top: '50px',
            right: '10px',
            width: '300px',
            maxHeight: '400px',
            background: '#1a1a1a',
            border: '1px solid #333',
            borderRadius: '8px',
            padding: '12px',
            fontSize: '11px',
            fontFamily: 'monospace',
            color: '#fff',
            zIndex: 10000,
            overflow: 'auto'
          }}
        >
          <h3 style={{ margin: '0 0 10px 0', color: '#52c41a' }}>Debug Info</h3>
          
          <div style={{ marginBottom: '10px' }}>
            <strong>Performance Warnings:</strong>
            <div style={{ fontSize: '10px', color: '#ff6b6b' }}>
              {debugInfo.filter(item => item.renderCount > 5).length > 0 && 
                `${debugInfo.filter(item => item.renderCount > 5).length} components with excessive renders`
              }
            </div>
          </div>

          <div>
            <strong>Component Renders:</strong>
            {debugInfo.map((item, index) => (
              <div key={index} style={{ 
                marginTop: '5px',
                padding: '4px',
                background: item.renderCount > 5 ? 'rgba(255, 107, 107, 0.1)' : 'rgba(82, 196, 26, 0.1)',
                borderRadius: '4px',
                borderLeft: `3px solid ${item.renderCount > 5 ? '#ff6b6b' : '#52c41a'}`
              }}>
                <div style={{ fontWeight: 'bold' }}>{item.component}</div>
                <div style={{ fontSize: '10px', color: '#999' }}>
                  Renders: {item.renderCount} | Last: {new Date(item.lastRender).toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: '10px', fontSize: '10px', color: '#666' }}>
            <div>🟢 Normal (&lt;5 renders)</div>
            <div>🔴 Excessive (&gt;5 renders)</div>
          </div>
        </div>
      )}
    </>
  )
}

export default DebugDashboard
