# Bloomberg Terminal & Godel Terminal Inspired Polish

**Inspiration:** Bloomberg Terminal + Godel Terminal aesthetic  
**Date:** 2025-01-27  
**Designer:** Sally (UX Designer)  

---

## 🎯 **Bloomberg Terminal Key Aesthetics**

### **1. Data Density**
- 📊 **Tight spacing** - Maximum information, minimal whitespace
- 📏 **Fixed-width fonts** - JetBrains Mono for all numbers
- 🔢 **Right-aligned numbers** - Professional data presentation
- 🎨 **Subtle color coding** - Green/red for up/down

### **2. Keyboard Shortcuts**
- ⌨️ **Everything has shortcuts** - Efficiency over mouse
- 🔤 **Visible hints** - Shortcuts shown in UI
- ⚡ **Fast navigation** - Bloomberg command system

### **3. Real-Time Data Visualization**
- ⚡ **Price flash animations** - Color highlight on change
- 📈 **Depth visualization** - Order book with visual depth
- 🎯 **Status indicators** - Pulse animations on active data
- 📊 **Ticker tape style** - Scrolling data feeds

### **4. Professional Color Scheme**
- 🖤 **Black backgrounds** - Terminal aesthetic
- 🔵 **Blue accents** - Active/highlighted elements
- 🟢 **Green** - Positive/profitable
- 🔴 **Red** - Negative/losses
- ⚪ **Subtle grays** - Inactive/muted

### **5. Visual Hierarchy**
- 📏 **Bold headers** - Uppercase, small font
- 🔤 **Monospace everywhere** - Numbers in terminal font
- 📊 **Clean borders** - 1px solid, no rounded corners
- 🎨 **Accent on hover** - Blue glow

---

## ✨ **Polish Items Added to Theme**

### **1. Bloomberg Table Styling**
```css
.bloomberg-table {
  width: 100%;
  border-collapse: collapse;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; /* Compact like Bloomberg */
}
```

**Features:**
- ✅ Monospace font for numbers
- ✅ Hover effect with blue background tint
- ✅ Uppercase, small headers
- ✅ Tight spacing

---

### **2. Order Book Depth Visualization**
```css
.orderbook-bid {
  background: linear-gradient(90deg, rgba(34, 197, 94, 0.1) 0%, transparent 100%);
}

.orderbook-ask {
  background: linear-gradient(90deg, transparent 0%, rgba(239, 68, 68, 0.1) 100%);
}
```

**Bloomberg Inspiration:** Visual depth bars in order book

---

### **3. Real-Time Price Flash Animation**
```css
@keyframes priceFlash {
  0% { background-color: transparent; }
  50% { background-color: var(--primary-blue); }
  100% { background-color: transparent; }
}
```

**Bloomberg Inspiration:** Prices flash when they update

---

### **4. Status Bar Style**
```css
.bloomberg-status-bar {
  background: var(--bg-secondary);
  border-top: 1px solid var(--border-base);
  padding: 4px 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
}
```

**Bloomberg Inspiration:** Bottom status bar with ticker

---

### **5. Keyboard Shortcut Hints**
```css
.shortcut-hint {
  font-size: 9px;
  color: var(--text-muted);
  background: var(--bg-tertiary);
  padding: 2px 4px;
  border: 1px solid var(--border-base);
  font-family: 'JetBrains Mono', monospace;
}
```

**GodelTerminal Inspiration:** Keyboard shortcuts visible in UI

---

### **6. Bloomberg Tab System**
```css
.bloomberg-tab {
  padding: 8px 16px;
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--text-muted);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
}

.bloomberg-tab.active {
  color: var(--primary-blue);
  border-bottom-color: var(--primary-blue);
}
```

**Bloomberg Inspiration:** Clean tab system with blue underline for active

---

### **7. Button Groups (Bloomberg Style)**
```css
.btn-group {
  display: inline-flex;
  border: 1px solid var(--border-base);
}

.btn-group button.active {
  background: var(--primary-blue);
  color: white;
}
```

**Bloomberg Inspiration:** Grouped buttons for related actions

---

### **8. Form Input Polish**
```css
input[type="text"],
input[type="number"],
textarea,
select {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-base);
  color: var(--text-primary);
  font-family: 'JetBrains Mono', monospace;
  transition: all 0.2s ease;
}
```

**Bloomberg Inspiration:** Terminal-style inputs with blue focus

---

## 🎬 **Demo Recommendations**

### **High-Impact Polish for Demo:**

1. **Add Bloomberg Table Classes to Positions Table**
   - Use `.bloomberg-table` on positions
   - Add JetBrains Mono to all numbers
   - Right-align numeric columns

2. **Enable Price Flash Animation**
   - Add `.price-positive-flash` / `.price-negative-flash`
   - Flash when price updates

3. **Order Book Depth Visualization**
   - Use `.orderbook-bid` / `.orderbook-ask`
   - Visual depth bars

4. **Bloomberg Tab System**
   - Switch to `.bloomberg-tabs` in dashboard
   - Clean, professional tabs

5. **Keyboard Shortcut Hints**
   - Show shortcuts with `.shortcut-hint`
   - Bloomberg efficiency

---

## ✅ **What's Now Available**

All these Bloomberg-inspired classes are now in `theme.css` and ready to use:

- ✅ `.bloomberg-table` - Professional data tables
- ✅ `.bloomberg-status-bar` - Bottom status bar
- ✅ `.shortcut-hint` - Keyboard shortcuts
- ✅ `.bloomberg-tabs` - Clean tab system
- ✅ `.btn-group` - Button groups
- ✅ `.price-flash` animations - Real-time updates
- ✅ `.orderbook-bid` / `.orderbook-ask` - Depth visualization
- ✅ Terminal-style form inputs

---

**Next:** Apply these classes to existing components for Bloomberg Terminal quality!

