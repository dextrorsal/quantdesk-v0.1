# Dev Agent Review: Chart Flickering Issue

**Date:** January 2025  
**Issue:** Chart flickers every 5 seconds during polling  
**Status:** Needs Fix - Professional Implementation Pattern

---

## 🔍 Dev Agent Analysis

### Current Implementation (Our Code)
```typescript
// ❌ PROBLEM: Calls setData() every 5 seconds
const pollInterval = setInterval(() => {
  fetchCandles().then(candles => {
    seriesRef.current?.setData(candles); // Full replace - FLICKERS!
  });
}, 5000);
```

### Reference Implementation (crypto-position-calculator)
```javascript
// ✅ CORRECT: Uses update() for real-time, setData() only once
useEffect(() => {
  const conn = new WebSocket(GetLiveCandle(timeFrame, symbol));
  
  // Real-time updates use update()
  conn.onmessage = (event) => {
    const liveData = JSON.parse(event.data);
    newSeries.current.update(liveData); // ✅ Incremental
  };
  
  // Initial load uses setData() ONCE
  GetCandles(timeFrame, symbol).then(Resp => {
    const candles = Resp.data.map(/*...*/);
    newSeries.current.setData(candles); // ✅ Initial load only
  });
  
  return () => conn.close();
}, [timeFrame, symbol]);
```

---

## ✅ Dev Agent Recommendation

### Pattern 1: WebSocket (Best Practice - Production Ready)

```typescript
useEffect(() => {
  if (!seriesRef.current) return;
  
  // Fetch initial historical data ONCE
  fetch('/api/oracle/binance/BTCUSDT?interval=1h&limit=500')
    .then(res => res.json())
    .then(data => {
      const candles = transformToCandles(data);
      seriesRef.current?.setData(candles); // ✅ Initial load
    });
  
  // Connect WebSocket for real-time updates
  const ws = new WebSocket(`wss://stream.binance.com:9443/ws/btcusdt@kline_1h`);
  
  ws.onmessage = (event) => {
    const k = JSON.parse(event.data).k;
    
    // ✅ Use update() for incremental updates
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

**Why This Works:**
- `setData()` called **once** for initial load
- `update()` used for each new candle
- **No flicker** - only adds new data points
- Professional trading platform standard

### Pattern 2: If WebSocket Fails, Use Smart Polling

```typescript
useEffect(() => {
  if (!seriesRef.current) return;
  
  let lastTimestamp = 0;
  
  const fetchInitial = async () => {
    const response = await fetch(`/api/oracle/binance/${symbol}?interval=${interval}&limit=500`);
    const data = await response.json();
    const candles = transformToCandles(data);
    
    seriesRef.current?.setData(candles); // Initial load
    
    // Track last candle
    lastTimestamp = candles[candles.length - 1].time;
  };
  
  fetchInitial();
  
  // Poll for NEW candles only
  const pollInterval = setInterval(async () => {
    try {
      // Fetch only recent candles
      const response = await fetch(`/api/oracle/binance/${symbol}?interval=${interval}&limit=10`);
      const data = await response.json();
      const newCandles = transformToCandles(data);
      
      // Filter to only new candles
      const latest = newCandles.filter(c => c.time > lastTimestamp);
      
      // Update only new ones
      latest.forEach(candle => {
        seriesRef.current?.update(candle); // ✅ Update, don't replace
        lastTimestamp = Math.max(lastTimestamp, candle.time);
      });
    } catch (err) {
      console.error('Polling error:', err);
    }
  }, 5000);
  
  return () => clearInterval(pollInterval);
}, [symbol, interval]);
```

**Why This Works:**
- Still fetches initial data with `setData()`
- Updates only new candles with `update()`
- **No flicker** - doesn't replace entire dataset
- Fallback if WebSocket unavailable

---

## 🎯 Dev Agent Verdict

### Root Cause
Using `setData()` for updates instead of `update()` causes full chart replacement = flicker.

### Solution
- **Initial load:** `setData(allCandles)` ✅
- **Real-time updates:** `update(newCandle)` ✅

### Implementation Priority
1. **HIGH**: Implement WebSocket pattern (most professional)
2. **MEDIUM**: Smart polling (fallback if WebSocket fails)
3. **LOW**: Current approach (causes flicker - not acceptable)

### Acceptance Criteria
- [ ] Chart displays initial 500 candles without flicker
- [ ] Real-time updates add new candles incrementally
- [ ] No full chart replacement during updates
- [ ] WebSocket connection properly managed (connect/disconnect)
- [ ] Loading state shows during initial fetch only
- [ ] Error handling for failed connections

---

## 📋 Dev Agent Recommendation

**Immediate Fix:**
Replace the polling `setInterval` with WebSocket pattern using `update()`.

**Code Change:**
```diff
- seriesRef.current?.setData(candles); // In polling loop
+ seriesRef.current?.update(candle);  // In real-time loop
```

**Expected Outcome:**
- ✅ No more flickering
- ✅ Smooth real-time updates
- ✅ Professional trading platform experience
- ✅ Matches industry standard (TradingView, Bloomberg, etc.)

---

**Status:** READY TO IMPLEMENT  
**Risk:** LOW (pattern verified from working reference implementation)  
**Effort:** ~30 minutes to implement WebSocket pattern

