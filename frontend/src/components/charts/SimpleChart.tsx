import React from 'react'

interface Props {
  symbol?: string
  timeframe?: string
  height?: number
}

const SimpleChart: React.FC<Props> = ({ symbol = 'BTC', timeframe = '1h', height = 300 }) => {
  return (
    <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#9ca3af' }}>
      SimpleChart placeholder • {symbol} • {timeframe}
    </div>
  )
}

export default SimpleChart


