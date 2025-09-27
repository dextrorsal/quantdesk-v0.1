# QuantDesk UI/UX Design System

## 🎯 Vision & Goals

**Mission**: Create a professional, cohesive trading platform that instills confidence and clarity in financial decision-making.

**Core Principles**:
- **Clarity**: Information should be immediately understandable
- **Confidence**: Visual design should convey stability and professionalism  
- **Efficiency**: Users should accomplish tasks with minimal friction
- **Consistency**: Every interaction should feel familiar and predictable

---

## 🎨 Color Palette & Themes

### Lite Theme: Professional Blue
**Target Audience**: Casual traders, beginners, mobile users
**Personality**: Clean, approachable, trustworthy

```css
/* Lite Theme Colors */
:root {
  /* Primary Blues */
  --lite-primary-50: #eff6ff;
  --lite-primary-100: #dbeafe;
  --lite-primary-200: #bfdbfe;
  --lite-primary-300: #93c5fd;
  --lite-primary-400: #60a5fa;
  --lite-primary-500: #3b82f6;  /* Main brand blue */
  --lite-primary-600: #2563eb;
  --lite-primary-700: #1d4ed8;
  --lite-primary-800: #1e40af;
  --lite-primary-900: #1e3a8a;
  
  /* Base Colors */
  --lite-bg-primary: #0f172a;    /* Deep slate */
  --lite-bg-secondary: #1e293b;   /* Slate 800 */
  --lite-bg-tertiary: #334155;    /* Slate 700 */
  --lite-text-primary: #f8fafc;   /* Slate 50 */
  --lite-text-secondary: #cbd5e1; /* Slate 300 */
  --lite-text-muted: #94a3b8;     /* Slate 400 */
  
  /* Accent Colors */
  --lite-success: #10b981;        /* Emerald 500 */
  --lite-danger: #ef4444;         /* Red 500 */
  --lite-warning: #f59e0b;        /* Amber 500 */
}
```

### Pro Theme: Premium Orange
**Target Audience**: Professional traders, power users, desktop users
**Personality**: Bold, sophisticated, high-performance

```css
/* Pro Theme Colors */
:root {
  /* Primary Oranges */
  --pro-primary-50: #fff7ed;
  --pro-primary-100: #ffedd5;
  --pro-primary-200: #fed7aa;
  --pro-primary-300: #fdba74;
  --pro-primary-400: #fb923c;
  --pro-primary-500: #f97316;  /* Main brand orange */
  --pro-primary-600: #ea580c;
  --pro-primary-700: #c2410c;
  --pro-primary-800: #9a3412;
  --pro-primary-900: #7c2d12;
  
  /* Base Colors */
  --pro-bg-primary: #000000;       /* Pure black */
  --pro-bg-secondary: #0a0a0a;    /* Near black */
  --pro-bg-tertiary: #1a1a1a;     /* Dark gray */
  --pro-text-primary: #ffffff;    /* Pure white */
  --pro-text-secondary: #e5e5e5;  /* Light gray */
  --pro-text-muted: #a3a3a3;      /* Medium gray */
  
  /* Accent Colors */
  --pro-success: #10b981;         /* Emerald 500 */
  --pro-danger: #ef4444;          /* Red 500 */
  --pro-warning: #f59e0b;         /* Amber 500 */
}
```

### Universal Accent Colors
**Green**: Used for positive financial indicators (profits, gains, success)
```css
--success-50: #f0fdf4;
--success-500: #22c55e;  /* Main success green */
--success-600: #16a34a;
--success-700: #15803d;
```

---

## 🔤 Typography

### Font Family
**Primary**: Inter (Google Fonts)
- Clean, modern, excellent readability
- Professional appearance
- Great for both UI and data display

**Monospace**: JetBrains Mono
- For code, trading data, numbers
- Clear character distinction
- Professional developer aesthetic

### Typography Scale
```css
/* Text Sizes */
--text-xs: 0.75rem;     /* 12px - Labels, captions */
--text-sm: 0.875rem;    /* 14px - Body text, buttons */
--text-base: 1rem;      /* 16px - Default body */
--text-lg: 1.125rem;    /* 18px - Subheadings */
--text-xl: 1.25rem;     /* 20px - Headings */
--text-2xl: 1.5rem;     /* 24px - Large headings */
--text-3xl: 1.875rem;   /* 30px - Page titles */
--text-4xl: 2.25rem;    /* 36px - Hero text */

/* Font Weights */
--font-light: 300;
--font-normal: 400;
--font-medium: 500;
--font-semibold: 600;
--font-bold: 700;
```

---

## 🧩 Component Design System

### Buttons

#### Primary Button
```css
.btn-primary {
  background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
  color: white;
  padding: 0.75rem 1.5rem;
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
  transition: all 0.2s ease;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.btn-primary:hover {
  background: linear-gradient(135deg, var(--primary-700), var(--primary-800));
  transform: translateY(-1px);
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}
```

#### Secondary Button
```css
.btn-secondary {
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--bg-tertiary);
  padding: 0.75rem 1.5rem;
  border-radius: 0.5rem;
  font-weight: 500;
  transition: all 0.2s ease;
}

.btn-secondary:hover {
  background: var(--bg-tertiary);
  border-color: var(--primary-500);
}
```

#### Success Button (Green Accent)
```css
.btn-success {
  background: linear-gradient(135deg, var(--success-600), var(--success-700));
  color: white;
  /* Same styling as primary but with success colors */
}
```

### Cards

#### Standard Card
```css
.card {
  background: var(--bg-secondary);
  border: 1px solid var(--bg-tertiary);
  border-radius: 0.75rem;
  padding: 1.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(8px);
}
```

#### Trading Card (Elevated)
```css
.trading-card {
  background: var(--bg-secondary);
  border: 1px solid var(--bg-tertiary);
  border-radius: 1rem;
  padding: 1.25rem;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(12px);
}
```

### Input Fields
```css
.input {
  background: var(--bg-tertiary);
  border: 1px solid var(--bg-tertiary);
  color: var(--text-primary);
  padding: 0.75rem 1rem;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  transition: all 0.2s ease;
}

.input:focus {
  outline: none;
  border-color: var(--primary-500);
  box-shadow: 0 0 0 3px rgba(var(--primary-500), 0.1);
}
```

---

## 📊 Financial Data Visualization

### Price Indicators
```css
.price-positive {
  color: var(--success-500);
  font-weight: 600;
}

.price-negative {
  color: var(--danger-500);
  font-weight: 600;
}

.price-neutral {
  color: var(--text-muted);
}
```

### Status Indicators
```css
.status-indicator {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 0.5rem;
}

.status-active {
  background-color: var(--success-500);
  animation: pulse 2s infinite;
}

.status-inactive {
  background-color: var(--text-muted);
}

.status-warning {
  background-color: var(--warning-500);
}
```

---

## 🎭 Theme Implementation Strategy

### CSS Custom Properties Approach
```css
/* Theme Variables */
[data-theme="lite"] {
  --primary-500: var(--lite-primary-500);
  --primary-600: var(--lite-primary-600);
  --bg-primary: var(--lite-bg-primary);
  --bg-secondary: var(--lite-bg-secondary);
  --text-primary: var(--lite-text-primary);
  /* ... other lite variables */
}

[data-theme="pro"] {
  --primary-500: var(--pro-primary-500);
  --primary-600: var(--pro-primary-600);
  --bg-primary: var(--pro-bg-primary);
  --bg-secondary: var(--pro-bg-secondary);
  --text-primary: var(--pro-text-primary);
  /* ... other pro variables */
}
```

### Tailwind Configuration
```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {
          50: 'var(--primary-50)',
          500: 'var(--primary-500)',
          600: 'var(--primary-600)',
          // ... etc
        },
        success: {
          500: 'var(--success-500)',
          600: 'var(--success-600)',
        }
      }
    }
  }
}
```

---

## 📱 Responsive Design Principles

### Breakpoints
- **Mobile**: < 640px
- **Tablet**: 640px - 1024px  
- **Desktop**: 1024px - 1440px
- **Large Desktop**: > 1440px

### Layout Guidelines
- **Lite**: Mobile-first, simplified layouts
- **Pro**: Desktop-optimized, information-dense layouts
- **Consistent spacing**: 4px base unit (0.25rem)
- **Touch targets**: Minimum 44px for mobile

---

## 🎨 Visual Hierarchy

### Z-Index Scale
```css
--z-dropdown: 1000;
--z-sticky: 1020;
--z-fixed: 1030;
--z-modal-backdrop: 1040;
--z-modal: 1050;
--z-popover: 1060;
--z-tooltip: 1070;
```

### Shadow System
```css
--shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
--shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
--shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
--shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
```

---

## 🚀 Implementation Roadmap

### Phase 1: Foundation
1. ✅ Create design system documentation
2. 🔄 Implement CSS custom properties
3. 🔄 Update Tailwind configuration
4. 🔄 Standardize typography

### Phase 2: Components
1. 🔄 Update button components
2. 🔄 Standardize card components
3. 🔄 Implement input field styling
4. 🔄 Update navigation components

### Phase 3: Themes
1. 🔄 Implement Lite theme (blue)
2. 🔄 Implement Pro theme (orange)
3. 🔄 Add theme switching logic
4. 🔄 Test across all components

### Phase 4: Polish
1. 🔄 Add animations and transitions
2. 🔄 Implement responsive optimizations
3. 🔄 Add accessibility improvements
4. 🔄 Performance optimization

---

## 📋 Component Checklist

### Before Implementation
- [ ] Does it follow the color palette?
- [ ] Is typography consistent?
- [ ] Does it work in both themes?
- [ ] Is it accessible?
- [ ] Is it responsive?

### After Implementation
- [ ] Tested in Lite theme
- [ ] Tested in Pro theme
- [ ] Tested on mobile
- [ ] Tested on desktop
- [ ] Accessibility verified
- [ ] Performance checked

---

## 🎯 Success Metrics

### Visual Consistency
- All components use the same color palette
- Typography is consistent across the platform
- Spacing follows the 4px grid system
- Shadows and borders are standardized

### User Experience
- Theme switching is seamless
- Components feel cohesive
- Information hierarchy is clear
- Interactions are predictable

### Technical
- CSS is maintainable and scalable
- Theme switching is performant
- Components are reusable
- Code is well-documented

---

*This design system is a living document. Update it as the platform evolves and new patterns emerge.*
