import { Routes, Route } from 'react-router-dom'
import { WalletProvider } from './providers/WalletProvider'
import { TradingProvider } from './providers/TradingProvider'
import Layout from './components/Layout'
import React from 'react'
import PortfolioPage from './pages/PortfolioPage'
import MarketsPage from './pages/MarketsPage'
import LandingPage from './pages/LandingPage'

function App() {
  // const navigate = useNavigate()
  // const location = useLocation()

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

  return (
    <WalletProvider>
        <TradingProvider>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            {/* Lite (new) → QuantDesk chrome with Quantify Lite content */}
            <Route path="/lite" element={
              <Layout>
                <React.Suspense fallback={<div className="p-6 text-white">Loading Lite…</div>}>
                  {React.createElement(React.lazy(() => import('./lite/LiteRouter')))}
                </React.Suspense>
              </Layout>
            } />
            {/* Pro mode will host the terminal + taskbar shell */}
            <Route path="/pro" element={
              <React.Suspense fallback={<div className="p-6 text-white">Loading Pro…</div>}>
                {React.createElement(React.lazy(() => import('./pro/index')))}
              </React.Suspense>
            } />
            {/* Trading mode - standalone trading interface */}
            <Route path="/trading" element={
              <React.Suspense fallback={<div className="p-6 text-white">Loading Trading…</div>}>
                {React.createElement(React.lazy(() => import('./lite/TradingTab')))}
              </React.Suspense>
            } />
            <Route path="/portfolio" element={
              <Layout>
                <PortfolioPage />
              </Layout>
            } />
            <Route path="/markets" element={
              <Layout>
                <MarketsPage />
              </Layout>
            } />
          </Routes>
        </TradingProvider>
      </WalletProvider>
  )
}

export default App
