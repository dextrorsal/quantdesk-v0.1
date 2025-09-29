import React, { createContext, useContext, ReactNode } from 'react'
// import { useTradingStore } from '../stores/tradingStore' // For future use

interface TradingContextType {
  // This will be populated with trading-related context
}

const TradingContext = createContext<TradingContextType | undefined>(undefined)

interface TradingProviderProps {
  children: ReactNode
}

export const TradingProvider: React.FC<TradingProviderProps> = ({ children }) => {
  // const tradingStore = useTradingStore() // For future use

  // Memoize the context value to prevent unnecessary re-renders
  const value = React.useMemo(() => ({
    // Add trading context values here
  }), [])

  return (
    <TradingContext.Provider value={value}>
      {children}
    </TradingContext.Provider>
  )
}

export const useTrading = () => {
  const context = useContext(TradingContext)
  if (context === undefined) {
    throw new Error('useTrading must be used within a TradingProvider')
  }
  return context
}
