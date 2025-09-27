# 🎨 QuantDesk Lite Mode - Complete Color Scheme Documentation

## 📍 **Color Definition Locations**

### 1. **Main CSS Variables** (`frontend/src/index.css`)
```css
:root {
  /* Lite Theme Variables (Default) - Minimal */
  --primary-50: #f0f9ff;
  --primary-100: #e0f2fe;
  --primary-200: #bae6fd;
  --primary-300: #7dd3fc;
  --primary-400: #38bdf8;
  --primary-500: #0ea5e9;  /* Subtle blue accent */
  --primary-600: #0284c7;
  --primary-700: #0369a1;
  --primary-800: #075985;
  --primary-900: #0c4a6e;
  
  --bg-primary: #000000;      /* Pure black */
  --bg-secondary: #0a0a0a;   /* Near black */
  --bg-tertiary: #1a1a1a;    /* Dark gray */
  --text-primary: #ffffff;   /* Pure white */
  --text-secondary: #e5e5e5; /* Light gray */
  --text-muted: #a3a3a3;     /* Medium gray */
  
  --success-500: #22c55e;
  --danger-500: #ef4444;
  --warning-500: #f59e0b;
}
```

### 2. **Tailwind Config** (`frontend/tailwind.config.js`)
```javascript
colors: {
  'lite-primary': {
    50: '#eff6ff', 100: '#dbeafe', 200: '#bfdbfe', 300: '#93c5fd',
    400: '#60a5fa', 500: '#3b82f6', 600: '#2563eb', 700: '#1d4ed8',
    800: '#1e40af', 900: '#1e3a8a'
  },
  'pro-primary': { /* Orange theme for Pro mode */ },
  primary: {
    50: 'var(--primary-50)', 100: 'var(--primary-100)', /* ... */
  },
  'bg-primary': 'var(--bg-primary)',
  'bg-secondary': 'var(--bg-secondary)',
  'bg-tertiary': 'var(--bg-tertiary)',
  'text-primary': 'var(--text-primary)',
  'text-secondary': 'var(--text-secondary)',
  'text-muted': 'var(--text-muted)',
}
```

## 🎯 **Component-Specific Color Usage**

### **Header Component** (`frontend/src/components/Header.tsx`)
- **Background**: `var(--bg-secondary)` (#0a0a0a)
- **Border**: `var(--bg-tertiary)` (#1a1a1a)
- **Active Tab**: `var(--primary-500)` (#0ea5e9)
- **Inactive Tab**: `var(--bg-secondary)` (#0a0a0a)
- **Logo Text**: `gradient-text` (white to blue gradient)

### **LiteRouter Component** (`frontend/src/lite/LiteRouter.tsx`)
**Current Usage Count:**
- `bg-gray-*`: 255 instances
- `text-gray-*`: 255 instances  
- `border-gray-*`: 255 instances
- `bg-primary-*`: 63 instances
- `text-primary-*`: 63 instances
- `border-primary-*`: 63 instances

**Key Color Patterns:**
- **Market Data Cards**: `bg-gray-800` with `border-gray-700`
- **Active States**: `text-primary-500` (blue)
- **Price Changes**: `text-primary-500` (positive), `text-red-400` (negative)
- **Loading States**: `border-blue-500`
- **Hover States**: `hover:bg-gray-700`

### **Theme Toggle** (`frontend/src/components/ThemeToggle.tsx`)
- **Active Button**: `var(--primary-500)` background
- **Inactive Button**: `transparent` background
- **Border**: `var(--bg-tertiary)` for inactive, `var(--primary-500)` for active

## 🔧 **Easy Color Customization Guide**

### **To Change Primary Accent Color:**
1. **Update CSS Variables** in `frontend/src/index.css`:
   ```css
   --primary-500: #0ea5e9;  /* Change this to your desired blue */
   ```

2. **Update Tailwind Config** in `frontend/tailwind.config.js`:
   ```javascript
   'lite-primary': {
     500: '#0ea5e9',  // Change this to match
   }
   ```

### **To Change Background Colors:**
```css
--bg-primary: #000000;      /* Main background */
--bg-secondary: #0a0a0a;   /* Card backgrounds */
--bg-tertiary: #1a1a1a;    /* Borders, dividers */
```

### **To Change Text Colors:**
```css
--text-primary: #ffffff;   /* Main text */
--text-secondary: #e5e5e5; /* Secondary text */
--text-muted: #a3a3a3;     /* Muted text */
```

## 📊 **Color Usage Statistics**

### **Most Used Colors in Lite Mode:**
1. **Gray Scale**: 255+ instances
   - `bg-gray-800`: Card backgrounds
   - `text-gray-400`: Secondary text
   - `border-gray-700`: Card borders

2. **Primary Blue**: 63 instances
   - `text-primary-500`: Active states, positive changes
   - `bg-primary-500`: Active buttons
   - `border-primary-500`: Active borders

3. **Status Colors**: 
   - `text-red-400`: Negative changes, sell orders
   - `text-green-400`: Success states (now changed to primary-500)

## 🎨 **Design System Hierarchy**

### **Background Hierarchy:**
- **Level 1**: `--bg-primary` (#000000) - Main page background
- **Level 2**: `--bg-secondary` (#0a0a0a) - Card backgrounds  
- **Level 3**: `--bg-tertiary` (#1a1a1a) - Borders, dividers

### **Text Hierarchy:**
- **Primary**: `--text-primary` (#ffffff) - Headings, important text
- **Secondary**: `--text-secondary` (#e5e5e5) - Body text
- **Muted**: `--text-muted` (#a3a3a3) - Labels, captions

### **Accent Colors:**
- **Primary**: `--primary-500` (#0ea5e9) - Active states, CTAs
- **Success**: `--success-500` (#22c55e) - Positive indicators
- **Danger**: `--danger-500` (#ef4444) - Negative indicators
- **Warning**: `--warning-500` (#f59e0b) - Caution states

## 🚀 **Quick Redesign Commands**

### **Change to Dark Purple Theme:**
```css
--primary-500: #8b5cf6;  /* Purple accent */
--bg-secondary: #0f0a1a; /* Slightly purple tint */
```

### **Change to Dark Green Theme:**
```css
--primary-500: #10b981;  /* Green accent */
--bg-secondary: #0a1a0f; /* Slightly green tint */
```

### **Change to Minimal Gray Theme:**
```css
--primary-500: #6b7280;  /* Gray accent */
--bg-secondary: #111111; /* Pure dark gray */
```

## 📝 **Notes for Future Updates**

1. **Always update both CSS variables AND Tailwind config** for consistency
2. **Test both light and dark text contrast** with new backgrounds
3. **Maintain the 3-level background hierarchy** for visual depth
4. **Keep accent colors minimal** - only for active states and CTAs
5. **Use CSS variables** instead of hardcoded colors for easy theming

---

**Last Updated**: $(date)
**Total Color Instances**: 500+ across Lite mode components
**Main Files**: `index.css`, `tailwind.config.js`, `LiteRouter.tsx`, `Header.tsx`
