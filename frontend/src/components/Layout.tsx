import React from 'react'
import Header from './Header'
import BottomTaskbar from './BottomTaskbar'
import { TabProvider } from '../contexts/TabContext'

interface LayoutProps {
  children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <TabProvider defaultTab="trading">
      <div className="h-screen bg-black text-white flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-auto pb-16">
          {children}
        </main>
        <BottomTaskbar />
      </div>
    </TabProvider>
  )
}

export default Layout
