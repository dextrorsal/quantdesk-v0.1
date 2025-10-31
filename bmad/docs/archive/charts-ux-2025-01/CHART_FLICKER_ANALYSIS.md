# Chart Flickering Analysis & Solutions (January 2025)

## 🔍 Problem
Chart flickers every 5 seconds when polling for new candle data.

## 📊 Observations from Playwright Testing

### Current Behavior:
- ✅ Chart initializes successfully
- ✅ Initial 500 candles load properly
- ❌ **Chart flickers every 5 seconds** when `setData()` is called during polling
- ⚠️ Multiple `useEffect` calls detected in logs

### Key Log Pattern:
```
[LOG] 🎨 Initializing chart...
[LOG] 🔄 Fetching candles... (happens 4x simultaneously)
[LOG] ✅ Candles set successfully!
[LOG] 🔄 Fetching candles... (every 5 seconds - CAUSES FLICKER)
```

## 🎯 Root Causes

### 1. **Polling calls `setData()`** (Entire chart replacement)
We're calling `seriesRef.current.setData(candles)` every 5 seconds, which:
- **Completely replaces** all chart data
- Causes **full re-render** = FLICKER
- This is **WRONG** for real-time updates

### 2. **Multiple useEffect Calls**
The data fetching `useEffect` is running multiple times simultaneously, causing redundant fetches.

### 3. **Wrong Update Method**
- ❌ Using `setData()` for real-time updates (for initial load only)
- ✅ Should use `update()` for incremental real-time data

## 🏆 Professional Solutions (2024-2025)

### **Solution A: Incremental Updates (Proper Way)**

From `crypto-position-calculator` and TradingView docs:

**Pattern:**
```typescript
// INITIAL DATA: Use setData() once
useEffect(() => {
  if (!seriesRef.current) return;
  
  fetchInitialData().then((candles) => {
    seriesRef.current?.setData(candles); // ✅ Set initial data
  });
}, [symbol, interval]);

// REAL-TIME UPDATES: Use update() for new candles
useEffect(() => {
  const ws = new WebSocket(/*...*/);
  
  ws.onmessage = (event) => {
    const newCandle = parseWebSocketData(event);
    
    // ✅ Update incrementally - no flicker!
    seriesRef.current?.update(newCandle);
  };
  
  return () => ws.close();
}, [symbol, interval]);
```

**Key Difference:**
- `setData()` = Full replacement (use once for initial load)
- `update()` = Incremental update (use for real-time)

### **Solution B: Add Data Merging (If Using Polling)**

If you must use polling instead of WebSockets:

```typescript
useEffect(() => {
  const pollInterval = setInterval(async () => {
    const newCandles = await fetchRecentCandles();
    
    // Only add NEW candles, don't replace entire dataset
    const lastTimestamp = currentCandles[currentCandles.length - 1]?.time;
    const newOnly = newCandles.filter(c => c.time > lastTimestamp);
    
    // Update only new candles
    newOnly.forEach(candle => {
      seriesRef.current?.update(candle);
    });
  }, 5000);
  
  return () => clearInterval(pollInterval);
}, [symbol, interval]);
```

### **Solution C: Use TradingView's WebSocket Pattern**

From official TradingView docs (2024):

```typescript
// Separate useEffect for WebSocket
useEffect(() => {
  if (!seriesRef.current) return;
  
  const ws = new WebSocket(`wss://stream.binance.com:9443/ws/${symbol.toLowerCase()}@kline_${interval}`);
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const k = data.k;
    
    // ✅ Update single candle - no re-render
    seriesRef.current?.update({
      time: k.t / 1000,
      open: parseFloat(k.o),
      high: parseFloat(k.h),
      low: parseFloat(k.l),
      close: parseFloat(k.c),
    });
  };
  
  return () => ws.close();
}, [symbol, interval]);
```

## 📚 Best Practices from Research (2024-2025)

### **Pattern 1: Two Separate useEffects**
```typescript
// Effect 1: Initialize chart (runs once)
useEffect(() => {
  chartRef.current = createChart(containerRef.current, options);
  seriesRef.current = chartRef.current.addCandlestickSeries();
}, []); // ✅ Empty deps

// Effect 2: Fetch initial data
useEffect(() => {
  if (!seriesRef.current) return;
  
  fetchData().then(data => {
    seriesRef.current?.setData(data); // Initial load
  });
}, [symbol, interval]); // ✅ Runs when symbol/interval change

// Effect 3: Real-time updates
useEffect(() => {
  if (!seriesRef.current) return;
  
  const ws = new WebSocket(/*...*/);
  ws.onmessage = (event) => {
    seriesRef.current?.update(parseCandle(event)); // Updates only
  };
  
  return () => ws.close();
}, [symbol, interval]); // ✅ Separate WebSocket effect
```

### **Pattern 2: Official TradingView React Example**

From TradingView docs:
- ✅ Initialize chart in first `useEffect`
- ✅ Use `setData()` for initial historical data
- ✅ Use `update()` for real-time WebSocket updates
- ✅ Cleanup WebSocket properly in return function

### **Pattern 3: useTransition for Smooth Updates**

From `crypto-position-calculator` repo:
```typescript
import { useTransition } from 'react';

const [isPending, startTransition] = useTransition();

useEffect(() => {
  const conn = new WebSocket(/*...*/);
  
  conn.onmessage = (event) => {
    startTransition(() => {
      // Smooth, non-blocking update
      seriesRef.current?.update(parseCandle(event));
    });
  };
  
  return () => conn.close();
}, [symbol, interval]);
```

## 🎯 Recommended Fix for Our Code

### **Current Issue:**
```typescript
// ❌ WRONG: Calling setData() every 5 seconds
const pollInterval = setInterval(() => {
  fetchCandles().then(candles => {
    seriesRef.current?.setData(candles); // FLICKERS!
  });
}, 5000);
```

### **Recommended Fix:**
```typescript
// ✅ OPTION 1: Use WebSocket with update()
useEffect(() => {
  if (!seriesRef.current) return;
  
  const ws = new WebSocket(`wss://stream.binance.com:9443/ws/${symbol.toLowerCase()}@kline_${interval}`);
  
  ws.onmessage = (event) => {
    const k = JSON.parse(event.data).k;
    seriesRef.current?.update({
      time: k.t / 1000,
      open: parseFloat(k.o),
      high: parseFloat(k.h),
      low: parseFloat(k.l),
      close: parseFloat(k.c),
    });
  };
  
  return () => ws.close();
}, [symbol, interval]);

// ✅ OPTION 2: If polling is required, only fetch NEW candles
useEffect(() => {
  if (!seriesRef.current) return;
  
  const pollInterval = setInterval(async () => {
    // Only fetch latest candle (not all 500)
    const latestCandle = await fetchLatestCandle(symbol, interval);
    
    if (latestCandle) {
      seriesRef.current?.update(latestCandle); // Update, don't replace!
    }
  }, 5000);
  
  return () => clearInterval(pollInterval);
}, [symbol, interval]);
```

## 📝 Summary

**The flicker is caused by:**
1. ❌ Using `setData()` for real-time updates
2. ❌ Replacing entire dataset every 5 seconds
3. ❌ Not using WebSocket pattern

**The proper solution:**
1. ✅ Use `setData()` only for initial load
2. ✅ Use `update()` for real-time increments
3. ✅ Use WebSockets for true real-time (or fetch only latest candle if polling)
4. ✅ Separate useEffect hooks for chart init vs data updates

**Next Steps:**
- Implement WebSocket pattern OR
- Fetch only latest candle (not all 500) when polling
- Use `update()` instead of `setData()` for updates

---

**References:**
- TradingView Lightweight Charts Docs: https://tradingview.github.io/lightweight-charts/
- crypto-position-calculator (GitHub): https://github.com/Hamz-06/crypto-position-calculator
- React + Lightweight Charts: https://tradingview.github.io/lightweight-charts/tutorials/react/

