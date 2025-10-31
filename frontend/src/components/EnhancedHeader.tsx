import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useWallet } from '@solana/wallet-adapter-react';
import { NavigationManager, NavigationUtils, MAIN_NAVIGATION_ITEMS, useNavigation } from '../utils/navigation';
import { EnhancedWalletButton } from './wallet/EnhancedWalletButton';

export const EnhancedHeader: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { connected } = useWallet();
  const { navigateTo, getBreadcrumbs } = useNavigation();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  // Initialize navigation manager
  useEffect(() => {
    NavigationManager.getInstance().initialize(navigate);
  }, [navigate]);

  // Update breadcrumbs when location changes
  useEffect(() => {
    const breadcrumbs = NavigationUtils.generateBreadcrumbs(location.pathname);
    NavigationManager.getInstance().setBreadcrumbs(breadcrumbs);
  }, [location.pathname]);

  const handleNavigation = (item: typeof MAIN_NAVIGATION_ITEMS[0]) => {
    // Check if user is authenticated for protected routes
    if (item.requiresAuth && !connected) {
      // Show wallet connection prompt
      setIsMenuOpen(false);
      return;
    }

    navigateTo(item.path);
    setIsMenuOpen(false);
  };

  const getCurrentNavigationItem = () => {
    return NavigationUtils.getNavigationItemByPath(location.pathname);
  };

  const breadcrumbs = getBreadcrumbs();

  return (
    <header className="bg-gray-900 border-b border-gray-700 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <div className="flex items-center space-x-4">
            <button
              onClick={() => navigateTo('/')}
              className="flex items-center space-x-2 text-white hover:text-gray-300 transition-colors"
            >
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">Q</span>
              </div>
              <span className="font-bold text-lg">QuantDesk</span>
            </button>

            {/* Breadcrumbs */}
            {breadcrumbs.length > 1 && (
              <nav className="hidden md:flex items-center space-x-2 text-sm">
                {breadcrumbs.map((crumb, index) => (
                  <React.Fragment key={crumb.id}>
                    {index > 0 && <span className="text-gray-500">/</span>}
                    <button
                      onClick={() => navigateTo(crumb.path)}
                      className={`text-gray-300 hover:text-white transition-colors ${
                        index === breadcrumbs.length - 1 ? 'font-medium' : ''
                      }`}
                    >
                      {crumb.icon && <span className="mr-1">{crumb.icon}</span>}
                      {crumb.label}
                    </button>
                  </React.Fragment>
                ))}
              </nav>
            )}
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            {MAIN_NAVIGATION_ITEMS.map((item) => {
              const isActive = location.pathname === item.path;
              const isDisabled = item.requiresAuth && !connected;
              
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavigation(item)}
                  disabled={isDisabled}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-blue-600 text-white'
                      : isDisabled
                      ? 'text-gray-500 cursor-not-allowed'
                      : 'text-gray-300 hover:text-white hover:bg-gray-800'
                  }`}
                  title={isDisabled ? 'Connect wallet to access this feature' : item.label}
                >
                  {item.icon && <span className="mr-1">{item.icon}</span>}
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Right side - Wallet and Mobile Menu */}
          <div className="flex items-center space-x-4">
            {/* Wallet Button */}
            <EnhancedWalletButton />

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="md:hidden p-2 text-gray-300 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        {isMenuOpen && (
          <div className="md:hidden border-t border-gray-700 py-4">
            <nav className="space-y-2">
              {MAIN_NAVIGATION_ITEMS.map((item) => {
                const isActive = location.pathname === item.path;
                const isDisabled = item.requiresAuth && !connected;
                
                return (
                  <button
                    key={item.id}
                    onClick={() => handleNavigation(item)}
                    disabled={isDisabled}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-blue-600 text-white'
                        : isDisabled
                        ? 'text-gray-500 cursor-not-allowed'
                        : 'text-gray-300 hover:text-white hover:bg-gray-800'
                    }`}
                  >
                    {item.icon && <span className="mr-2">{item.icon}</span>}
                    {item.label}
                    {isDisabled && <span className="ml-2 text-xs">(Connect Wallet)</span>}
                  </button>
                );
              })}
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};

export default EnhancedHeader;
