import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/api\.quantdesk\.app\/.*/i,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'api-cache',
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 // 24 hours
              }
            }
          }
        ]
      },
      manifest: {
        name: 'QuantDesk Trading Platform',
        short_name: 'QuantDesk',
        description: 'Professional crypto trading platform',
        theme_color: '#3b82f6',
        background_color: '#000000',
        display: 'standalone',
        orientation: 'portrait',
        icons: [
          {
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ]
      }
    })
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  define: {
    global: 'globalThis',
    'process.env': 'import.meta.env',
    // Production environment variables
    'import.meta.env.VITE_API_URL': JSON.stringify(process.env.VITE_API_URL || 'https://api.quantdesk.app'),
    'import.meta.env.VITE_WS_URL': JSON.stringify(process.env.VITE_WS_URL || 'wss://api.quantdesk.app'),
    'import.meta.env.VITE_SOLANA_RPC_URL': JSON.stringify(process.env.VITE_SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com'),
    'import.meta.env.VITE_SOLANA_NETWORK': JSON.stringify(process.env.VITE_SOLANA_NETWORK || 'mainnet-beta'),
    'import.meta.env.VITE_DEBUG': JSON.stringify(process.env.VITE_DEBUG || 'false'),
  },
  envPrefix: 'VITE_',
  server: {
    port: 3001,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:3002',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false, // Disable sourcemaps in production for security
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          solana: ['@solana/web3.js', '@solana/wallet-adapter-base'],
          charts: ['lightweight-charts', 'recharts'],
        },
      },
    },
  },
  optimizeDeps: {
    include: ['buffer', 'process'],
  },
})
