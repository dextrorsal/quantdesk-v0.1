import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react'

interface TabContextType {
  activeTab: string
  setActiveTab: (tab: string) => void
  goToTab: (tab: string) => void
}

const TabContext = createContext<TabContextType | undefined>(undefined)

interface TabProviderProps {
  children: ReactNode
  defaultTab?: string
}

export const TabProvider: React.FC<TabProviderProps> = ({ children, defaultTab = 'overview' }) => {
  const [activeTab, setActiveTabState] = useState<string>(() => {
    // Prioritize hash over localStorage for initial load
    const hash = typeof window !== 'undefined' ? window.location.hash.replace('#', '') : ''
    const saved = typeof window !== 'undefined' ? (localStorage.getItem('quantdesk_lite_tab') || defaultTab) : defaultTab
    const finalTab = hash || saved
    console.log(`🚀 TabProvider initializing - saved: ${saved}, hash: ${hash}, final: ${finalTab}`)
    return finalTab
  })

  const setActiveTab = useCallback((tab: string) => {
    console.log(`🔄 TabProvider setting activeTab to: ${tab}`)
    setActiveTabState(tab)
    localStorage.setItem('quantdesk_lite_tab', tab)
  }, [])

  const goToTab = useCallback((tab: string) => {
    console.log(`🖱️ TabProvider navigating to tab: ${tab}`)
    setActiveTab(tab)
    // Update URL hash
    if (window.location.hash !== `#${tab}`) {
      window.history.replaceState(null, '', `#${tab}`)
    }
  }, [setActiveTab])

  useEffect(() => {
    const onHashChange = () => {
      const next = window.location.hash.replace('#', '') || defaultTab
      console.log(`🔄 TabProvider hash change detected: ${next}`)
      setActiveTabState(next)
      localStorage.setItem('quantdesk_lite_tab', next)
    }
    
    const onStorageChange = (e: StorageEvent) => {
      if (e.key === 'quantdesk_lite_tab' && e.newValue) {
        const next = e.newValue
        console.log(`🔄 TabProvider storage change detected: ${next}`)
        setActiveTabState(next)
        // Update URL hash to match
        if (window.location.hash !== `#${next}`) {
          window.history.replaceState(null, '', `#${next}`)
        }
      }
    }
    
    window.addEventListener('hashchange', onHashChange)
    window.addEventListener('storage', onStorageChange)
    
    return () => {
      window.removeEventListener('hashchange', onHashChange)
      window.removeEventListener('storage', onStorageChange)
    }
  }, [defaultTab])

  const value: TabContextType = {
    activeTab,
    setActiveTab,
    goToTab
  }

  return (
    <TabContext.Provider value={value}>
      {children}
    </TabContext.Provider>
  )
}

export const useTabContext = (): TabContextType => {
  const context = useContext(TabContext)
  if (context === undefined) {
    throw new Error('useTabContext must be used within a TabProvider')
  }
  return context
}
