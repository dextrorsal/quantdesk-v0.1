# Environment Configuration

## Vite Configuration
```typescript
// vite.config.ts - Build and development configuration
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
        runtimeCaching: [{
          urlPattern: /^https:\/\/api\.quantdesk\.app\/.*/i,
          handler: 'NetworkFirst',
          options: {
            cacheName: 'api-cache',
            expiration: { maxEntries: 10, maxAgeSeconds: 60 * 60 * 24 }
          }
        }]
      },
      manifest: {
        name: 'QuantDesk Trading Platform',
        short_name: 'QuantDesk',
        description: 'Professional crypto trading platform',
        theme_color: '#3b82f6',
        background_color: '#000000',
        display: 'standalone',
      }
    })
  ],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  define: {
    global: 'globalThis',
    'process.env': 'import.meta.env',
    // Environment variables
    'import.meta.env.VITE_API_URL': JSON.stringify(process.env.VITE_API_URL || 'http://localhost:3002'),
    'import.meta.env.VITE_WS_URL': JSON.stringify(process.env.VITE_WS_URL || 'ws://localhost:3002'),
    'import.meta.env.VITE_SOLANA_RPC_URL': JSON.stringify(process.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com'),
  },
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
})
```

## TypeScript Configuration
```json
// tsconfig.json - TypeScript compiler options
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["src", "../archive/index-stock-backup.tsx"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```
