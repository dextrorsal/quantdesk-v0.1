import React from 'react'

interface DebugInfoProps {
  componentName: string
}

const DebugInfoComponent: React.FC<DebugInfoProps> = ({ componentName }) => {
  const renderCount = React.useRef(0)
  renderCount.current += 1

  React.useEffect(() => {
    console.log(`🔄 ${componentName} rendered ${renderCount.current} times`)
    console.trace(`${componentName} render stack trace`)
  })

  // Only show in development
  if (!import.meta.env.DEV) {
    return null
  }

  return (
    <div style={{
      position: 'fixed',
      top: '10px',
      right: '10px',
      background: 'rgba(0,0,0,0.8)',
      color: 'white',
      padding: '4px 8px',
      fontSize: '10px',
      borderRadius: '4px',
      zIndex: 9999,
      fontFamily: 'monospace'
    }}>
      {componentName}: {renderCount.current}
    </div>
  )
}

export const DebugInfo = React.memo(DebugInfoComponent)

export default DebugInfo
