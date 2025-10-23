import React from 'react'
import Header from './Header'
import BottomTaskbar from './BottomTaskbar'
import { TabProvider } from '../contexts/TabContext'
import { ThemeProvider } from '../contexts/ThemeContext'

interface LayoutProps {
  children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <ThemeProvider>
      <TabProvider defaultTab="trading">
        <div className="h-screen flex flex-col overflow-hidden" style={{ backgroundColor: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
          <Header />
          <main className="flex-1 overflow-auto pb-16">
            {children}
          </main>
          <BottomTaskbar />
        </div>
      </TabProvider>
    </ThemeProvider>
  )
}

export default Layout
