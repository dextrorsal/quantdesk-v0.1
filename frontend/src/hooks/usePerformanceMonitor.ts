import React, { useEffect, useRef } from 'react'

interface PerformanceMetrics {
  renderCount: number
  lastRenderTime: number
  averageRenderTime: number
  renderTimes: number[]
}

export const usePerformanceMonitor = (componentName: string) => {
  const metrics = useRef<PerformanceMetrics>({
    renderCount: 0,
    lastRenderTime: 0,
    averageRenderTime: 0,
    renderTimes: []
  })

  const startTime = useRef<number>(0)

  // Throttle performance logging to prevent console spam
  const logPerformance = React.useCallback(() => {
    const endTime = performance.now()
    const renderTime = endTime - startTime.current
    
    metrics.current.renderCount += 1
    metrics.current.lastRenderTime = renderTime
    metrics.current.renderTimes.push(renderTime)
    
    // Keep only last 10 render times for average
    if (metrics.current.renderTimes.length > 10) {
      metrics.current.renderTimes.shift()
    }
    
    metrics.current.averageRenderTime = 
      metrics.current.renderTimes.reduce((a, b) => a + b, 0) / metrics.current.renderTimes.length

    // Log performance warnings (throttled)
    if (renderTime > 16 && metrics.current.renderCount % 5 === 0) { // Log every 5th slow render
      console.warn(`🐌 ${componentName} slow render: ${renderTime.toFixed(2)}ms`)
    }
    
    if (metrics.current.renderCount > 10 && metrics.current.renderCount % 5 === 0) {
      console.warn(`🔄 ${componentName} excessive renders: ${metrics.current.renderCount} times`)
    }
  }, [componentName])

  useEffect(() => {
    logPerformance()
  })

  useEffect(() => {
    startTime.current = performance.now()
  })

  return metrics.current
}

export default usePerformanceMonitor
