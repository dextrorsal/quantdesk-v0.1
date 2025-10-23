# Styling System Architecture

## Tailwind CSS Configuration
```typescript
// tailwind.config.js - Comprehensive theme system
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      screens: {
        'xs': '475px',
        'sm': '640px',
        'md': '768px',
        'lg': '1024px',
        'xl': '1280px',
        '2xl': '1536px',
        '3xl': '1920px',
        '4xl': '2560px',
        // Custom breakpoints for trading interfaces
        'mobile': {'max': '768px'},
        'tablet': {'min': '769px', 'max': '1024px'},
        'desktop': {'min': '1025px', 'max': '1920px'},
        'large-desktop': {'min': '1921px', 'max': '2560px'},
        'ultra-wide': {'min': '2561px'},
      },
      colors: {
        // Lite Theme Colors (Blue)
        'lite-primary': {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',  // Main brand blue
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
        // Pro Theme Colors (Orange)
        'pro-primary': {
          50: '#fff7ed',
          100: '#ffedd5',
          200: '#fed7aa',
          300: '#fdba74',
          400: '#fb923c',
          500: '#f97316',  // Main brand orange
          600: '#ea580c',
          700: '#c2410c',
          800: '#9a3412',
          900: '#7c2d12',
        },
        // Universal CSS Variables
        primary: {
          50: 'var(--primary-50)',
          100: 'var(--primary-100)',
          // ... dynamic theme colors
        },
      },
    },
  },
}
```

## Theme System Implementation
```typescript
// ThemeContext.tsx - Dynamic theme management
interface ThemeContextType {
  theme: 'lite' | 'pro'
  setTheme: (theme: 'lite' | 'pro') => void
  colors: ThemeColors
}

// CSS Variables for dynamic theming
:root {
  --primary-50: #eff6ff;
  --primary-500: #3b82f6;
  --bg-primary: #000000;
  --text-primary: #ffffff;
}
```
