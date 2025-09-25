import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { 
  TrendingUp, 
  BarChart3, 
  Wallet, 
  History, 
  Settings,
  HelpCircle 
} from 'lucide-react'

const Sidebar: React.FC = () => {
  const location = useLocation()

  const navItems = [
    { path: '/trading', icon: TrendingUp, label: 'Trading' },
    { path: '/portfolio', icon: BarChart3, label: 'Portfolio' },
    { path: '/markets', icon: Wallet, label: 'Markets' },
    { path: '/history', icon: History, label: 'History' },
    { path: '/settings', icon: Settings, label: 'Settings' },
    { path: '/help', icon: HelpCircle, label: 'Help' },
  ]

  return (
    <aside className="fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 bg-gray-900 border-r border-gray-800 overflow-y-auto hidden lg:block">
      <nav className="p-4 space-y-2">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = location.pathname === item.path
          
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              <Icon className="h-5 w-5" />
              <span className="font-medium text-sm">{item.label}</span>
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}

export default Sidebar
