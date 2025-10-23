import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { ConnectionProvider, WalletProvider, useConnection, useAnchorWallet } from '@solana/wallet-adapter-react'
import { WalletAdapterNetwork } from '@solana/wallet-adapter-base'
import { PhantomWalletAdapter, SolflareWalletAdapter } from '@solana/wallet-adapter-wallets'
import { WalletModalProvider } from '@solana/wallet-adapter-react-ui'
import { clusterApiUrl } from '@solana/web3.js'
import { TradingProvider } from './providers/TradingProvider'
import { ProgramProvider } from './contexts/ProgramContext'
import { PriceProvider } from './contexts/PriceContext'
import { AccountProvider } from './contexts/AccountContext'
import { MarketProvider } from './contexts/MarketContext'
import { TabProvider } from './contexts/TabContext'
import { ThemeProvider } from './contexts/ThemeContext'
import Layout from './components/Layout'
import React from 'react'
import PortfolioPage from './pages/PortfolioPage'
import MarketsPage from './pages/MarketsPage'
import LandingPage from './pages/LandingPage'
import ThemeDemo from './components/ThemeDemo'
import TickerClickTest from './components/TickerClickTest'
// import TradingInterface from './components/TradingInterface'
import DexTradingInterface from './components/DexTradingInterface'
import ChatIntegrationTest from './components/ChatIntegrationTest'
import LiteRouter from './lite/LiteRouter'
import AdminRedirect from './components/AdminRedirect'
// Dev preview imports for static routing (avoid lazy issues with dot filenames)
// Archived variants removed from routes; keep files for reference

// Import wallet adapter CSS
import '@solana/wallet-adapter-react-ui/styles.css'

function AppRoutes() {

  // Wallet configuration
  const network = WalletAdapterNetwork.Devnet
  const endpoint = clusterApiUrl(network)
  const wallets = [
    new PhantomWalletAdapter(),
    new SolflareWalletAdapter(),
  ]

  // Memoize the redirect logic to prevent unnecessary re-runs
  // Commented out auto-redirect to allow landing page to be visible
  // React.useEffect(() => {
  //   if (location.pathname === '/') {
  //     const stored = (localStorage.getItem('quantdesk_ui_mode') || 'lite').toLowerCase()
  //     let route = '/lite'
  //     if (stored === 'pro') route = '/pro'
  //     else if (stored === 'trading') route = '/trading'
  //     // Do not redirect if already at explicit route, only from root
  //     navigate(route, { replace: true })
  //   }
  // }, [location.pathname, navigate])

  const location = useLocation();
  const liteKey = `lite-${location.hash || ''}`;

  // Dev wrapper to bridge wallet/connection into ProgramProvider
  const DevProgramWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { connection } = useConnection();
    const anchorWallet = useAnchorWallet();
    if (!anchorWallet) {
      return <div className="p-6 text-white">Connect your wallet to load the program…</div>;
    }
    return (
      <ProgramProvider connection={connection as any} wallet={anchorWallet as any}>
        {children}
      </ProgramProvider>
    );
  };

  return (
      <ConnectionProvider endpoint={endpoint}>
        <WalletProvider wallets={wallets} autoConnect>
          <WalletModalProvider>
            <MarketProvider>
              <PriceProvider websocketUrl="ws://localhost:3002/" fallbackApiUrl="/api/prices">
                <TradingProvider>
                  <AccountProvider>
                    <Routes>
                    <Route path="/" element={<LandingPage />} />
                  {/* Lite (new) → QuantDesk chrome with QuantDesk Lite content */}
                  <Route path="/lite" element={
                    <Layout>
                      {/* Force remount on hash/tab change to ensure tab content refreshes */}
                      <div key={liteKey}>
                        <LiteRouter />
                      </div>
                    </Layout>
                  } />
                  {/* Pro mode will host the terminal + taskbar shell */}
                  <Route path="/pro" element={
                    <React.Suspense fallback={<div className="p-6 text-white">Loading Pro…</div>}>
                      {React.createElement(React.lazy(() => import('./pro/index')))}
                    </React.Suspense>
                  } />
                  {/* Trading routes removed - use /lite instead */}
                  <Route path="/portfolio" element={
                    <Layout>
                      <PortfolioPage />
                    </Layout>
                  } />
                  {/* dev/trading-interface route removed per request */}
                  <Route path="/markets" element={
                    <Layout>
                      <MarketsPage />
                    </Layout>
                  } />
                  <Route path="/theme-demo" element={
                    <Layout>
                      <ThemeDemo />
                    </Layout>
                  } />
                  <Route path="/ticker-test" element={
                    <Layout>
                      <TickerClickTest />
                    </Layout>
                  } />
                  {/* removed legacy TradingInterface route */}
                  <Route path="/chat-integration" element={
                    <Layout>
                      <ChatIntegrationTest />
                    </Layout>
                  } />
                  <Route path="/dex-trading" element={
                    <Layout>
                      <DexTradingInterface />
                    </Layout>
                  } />
                              {/* Admin Dashboard - Redirect to standalone app */}
                              <Route path="/admin/*" element={<AdminRedirect />} />
                  {/* Trading routes removed - use /lite instead */}
                  
                  {/* Catch-all route for unknown paths */}
                  <Route path="*" element={<LandingPage />} />
                  </Routes>
                  </AccountProvider>
                </TradingProvider>
              </PriceProvider>
            </MarketProvider>
          </WalletModalProvider>
        </WalletProvider>
      </ConnectionProvider>
  )
}

function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  )
}

export default App
