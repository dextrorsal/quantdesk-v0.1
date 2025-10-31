# Manual Testing Guide for Demo Functionality Validation
**Story:** 0-validate-demo-functionality  
**Status:** In Progress  
**Date:** 2025-01-28

## Overview

This guide provides step-by-step instructions for manually testing all demo functionality to validate Story 0-validate-demo-functionality acceptance criteria.

**Prerequisites:**
- All services running (Backend:3002, Frontend:3001, MIKEY-AI:3000)
- Wallet extension (Phantom) installed in browser
- Test account with some SOL on devnet

---

## Test 1: Pro Trading Terminal Initialization

### Steps:
1. **Start Services** (if not already running):
   ```bash
   cd /home/dex/Desktop/quantdesk
   npm run dev
   ```

2. **Navigate to Frontend:**
   - Open browser: `http://localhost:3001`
   - Open Developer Tools (F12)

3. **Check Console for Errors:**
   - Console should show zero errors
   - Look for any red error messages
   - Note any warnings (yellow)

4. **Navigate to Pro Trading Terminal:**
   - Click "Pro Trading" or equivalent navigation
   - URL should be: `http://localhost:3001/pro` or similar

5. **Verify All Panels Render:**
   - [ ] Positions panel (left or top)
   - [ ] Order book (middle/top-right)
   - [ ] Trades feed (right side)
   - [ ] Charts (main area)
   - [ ] Market selector dropdown
   - [ ] Account balance/health indicator

6. **Check Account Information:**
   - [ ] Account balance displays correctly
   - [ ] Available margin shows
   - [ ] Total equity displayed
   - [ ] Leverage multiplier visible

### Expected Result:
✅ All panels visible, zero console errors, account info displays correctly

### Screenshot Area:
Take screenshot showing all panels loaded successfully.

---

## Test 2: Trading Flow End-to-End

### Steps:
1. **Open Network Tab:**
   - DevTools → Network tab
   - Enable "Preserve log"

2. **Select Market:**
   - Use market selector dropdown
   - Select BTC, ETH, or SOL

3. **Enter Test Order:**
   - **Type:** Market Order (or Limit Order if testing)
   - **Side:** BUY
   - **Size:** 0.1 SOL (or appropriate amount)
   - **Price:** Current market price (for limit orders)

4. **Submit Order:**
   - Click Submit or Place Order button

5. **Monitor Network Tab:**
   - [ ] Look for POST request to `/api/orders`
   - [ ] Status should be 200 (OK) or check response
   - [ ] Request payload should contain order details

6. **Verify Order Appears:**
   - [ ] Order appears in "Active Orders" table
   - [ ] Order shows correct market, side, size
   - [ ] Status is "pending" or "filled" appropriately

7. **Check WebSocket Updates:**
   - Network tab → Filter to "WS" (WebSocket)
   - [ ] WebSocket connection established
   - [ ] Should receive updates about order execution
   - [ ] Position may be created if order executes

8. **Order Book Updates** (for limit orders):
   - [ ] If limit order, verify it appears in order book
   - [ ] Order book shows your order with correct price/size

### Expected Result:
✅ Order submitted successfully, API called, UI updates, WebSocket receives updates

### Screenshot Area:
Take screenshot of Network tab showing successful API call.

---

## Test 3: Positions Display Real Data

### Prerequisites:
- Must have at least one open position (complete Test 2 first)

### Steps:
1. **Open Positions Table:**
   - Navigate to positions panel (if not already visible)
   - Scroll through all positions if multiple

2. **Verify All Columns Display:**
   - [ ] Symbol (e.g., BTC/USD, ETH/USD)
   - [ ] Side (Long or Short)
   - [ ] Size (position size)
   - [ ] Entry Price (your entry price)
   - [ ] Current Price (from oracle)
   - [ ] Unrealized P&L (calculated value)
   - [ ] Liquidation Price (calculated value)
   - [ ] Health Factor (account health percentage)

3. **Verify Data is Real (Not Hardcoded):**
   - Open Backend console
   - Check current market price at `/api/oracle/prices`
   - Compare with position's "Current Price"
   - [ ] Prices should match or be very close (within 0.01%)

4. **Check Account Health Metric:**
   - [ ] Health metric displays correctly
   - [ ] Shows percentage (e.g., "85.2%")
   - [ ] Color coding appropriate (green/yellow/red)

5. **Verify Liquidation Risk Indicator:**
   - [ ] Risk level shown (Low, Medium, High, Critical)
   - [ ] Visual indicator (color/icon)

### Expected Result:
✅ All fields populated with real data, prices match oracle, health indicators accurate

### Screenshot Area:
Take screenshot of positions table with real data.

---

## Test 4: MIKEY AI Chat Backend Integration

### Steps:
1. **Open MIKEY AI Chat:**
   - Find MIKEY AI chat panel or button
   - Click to open chat interface

2. **Check Network Tab:**
   - Keep Network tab open
   - Enable "Preserve log"

3. **Send Test Message:**
   - Type: "What's the current market sentiment for SOL?"
   - Send message (Enter key or Send button)

4. **Monitor Network Tab:**
   - [ ] Look for API call to backend
   - [ ] Check for `/api/ai/*` or `/api/chat/*` or `/api/mikey/*`
   - [ ] Status should be 200 OK
   - [ ] Response should contain AI message

5. **Verify Response Display:**
   - [ ] AI response appears in chat
   - [ ] Response is relevant to the question
   - [ ] Response appears professional
   - [ ] Response mentions trading or market analysis

6. **Test Error Handling:**
   - Stop MIKEY-AI service: `cd MIKEY-AI && pnpm run stop` (if running)
   - Send another message
   - [ ] Error message displayed gracefully
   - [ ] No application crash
   - [ ] Error is user-friendly (not technical stack trace)

### Expected Result:
✅ Chat connects to backend, responses are relevant and professional, errors handled gracefully

### Screenshot Area:
Take screenshot of chat showing AI response.

---

## Test 5: WebSocket Real-Time Updates

### Steps:
1. **Open WebSocket Tab:**
   - DevTools → Network → Filter: "WS"
   - [ ] WebSocket connection should be established
   - Connection URL: `ws://localhost:3002` or similar

2. **Monitor Position Price Updates:**
   - Watch positions table for price changes
   - [ ] Prices update automatically without page refresh
   - Updates should occur within <2 seconds of market price change

3. **Verify P&L Recalculates:**
   - [ ] Unrealized P&L updates automatically
   - [ ] Recalculation happens when prices change
   - [ ] No manual refresh needed

4. **Check Order Book Updates:**
   - Observe order book depth
   - [ ] Order book updates in real-time
   - [ ] New orders appear as they're placed
   - [ ] Orders fill/remove as they execute

5. **Verify Trades Feed:**
   - Check trades feed component
   - [ ] New trades appear as they happen
   - [ ] Trade information is accurate
   - [ ] Timestamps are recent

6. **Check Network Traffic:**
   - WebSocket tab should show messages
   - [ ] Messages received regularly
   - [ ] No connection drops (unless intentional)

### Expected Result:
✅ Real-time updates visible, <2 second latency, no page refresh required

### Screenshot Area:
Take screenshot of WebSocket messages and positions updating.

---

## Test 6: Account Information Display

### Steps:
1. **Verify Account Balance:**
   - Find account balance display
   - [ ] Balance shows correct amount
   - [ ] Currency is SOL or USD equivalent
   - [ ] Format is readable (e.g., "1.25 SOL" not "1250000000")

2. **Check Available Margin:**
   - [ ] Available margin calculated correctly
   - [ ] Calculation: Total Balance - Used Margin = Available
   - [ ] Updates when positions change

3. **Verify Total Equity:**
   - [ ] Total equity shows open position value
   - [ ] Includes unrealized P&L
   - [ ] Updates in real-time

4. **Check Leverage Multiplier:**
   - [ ] Leverage displays correctly (e.g., "2x", "5x")
   - [ ] Matches settings or position leverage
   - [ ] Visual indicator is clear

5. **Test with Different Margin Levels:**
   - Open/close positions to change margin usage
   - [ ] Available margin updates correctly
   - [ ] Warnings appear if margin is low
   - [ ] Account health reflects changes

### Expected Result:
✅ All account metrics display correctly and update in real-time

### Screenshot Area:
Take screenshot of account information panel.

---

## Test 7: Demo Recording

### Prerequisites:
- All previous tests (1-6) must pass
- Screen recording software ready
- Test wallet has some SOL

### Steps:
1. **Start Recording:**
   - Begin screen recording (5 minutes planned)
   - Narrate the demo as you perform actions

2. **Demo Flow (5 minutes):**
   - **0:00-0:30** - Login and Navigate
     - [ ] Connect wallet
     - [ ] Navigate to Pro Trading Terminal
     - [ ] Show all panels loading
   
   - **0:30-2:00** - Trading Demonstration
     - [ ] Select market (BTC, ETH, or SOL)
     - [ ] Enter market order (BUY)
     - [ ] Show order submission
     - [ ] Show order appearing in active orders
     - [ ] Show WebSocket updates
     - [ ] Show position creation
   
   - **2:00-3:00** - Positions Display
     - [ ] Show positions table with real data
     - [ ] Point out P&L, health factors
     - [ ] Show real-time price updates
     - [ ] Explain liquidation risk indicators
   
   - **3:00-4:00** - AI Chat Integration
     - [ ] Open MIKEY AI chat
     - [ ] Ask market analysis question
     - [ ] Show AI response
     - [ ] Demonstrate professional responses
   
   - **4:00-5:00** - Multiple Markets & Wrap-up
     - [ ] Switch to different market
     - [ ] Show account balance/health
     - [ ] Summarize key features
     - [ ] End recording

3. **Verify Demo Quality:**
   - [ ] No console errors in demo
   - [ ] Smooth flow without interruptions
   - [ ] All data appears realistic/real
   - [ ] Demo shows key platform features

### Expected Result:
✅ Demo-ready, 5-minute video shows all flows working smoothly

### Deliverables:
- 5-minute demo video file
- Documentation of any issues found
- Demo script used

---

## Test Results Summary

After completing all tests, fill out this summary:

| Test | Status | Notes |
|------|--------|-------|
| Test 1: Pro Trading Terminal | [ ] PASS / [ ] FAIL | |
| Test 2: Trading Flow End-to-End | [ ] PASS / [ ] FAIL | |
| Test 3: Positions Display | [ ] PASS / [ ] FAIL | |
| Test 4: MIKEY AI Chat | [ ] PASS / [ ] FAIL | |
| Test 5: WebSocket Updates | [ ] PASS / [ ] FAIL | |
| Test 6: Account Information | [ ] PASS / [ ] FAIL | |
| Test 7: Demo Recording | [ ] PASS / [ ] FAIL | |

### Overall Status:
[ ] **READY FOR DEMO** - All tests pass, platform ready for demonstration  
[ ] **ISSUES FOUND** - See notes below for required fixes

### Issues Found:
(List any issues discovered during testing)

1. 
2. 
3. 

### Recommendations:
(List any recommendations for improvements)

1. 
2. 
3. 

---

## Next Steps

- If all tests pass → Mark story as complete, prepare demo
- If issues found → Create follow-up stories to fix issues
- If critical failures → Block demo until fixed

**Generated:** 2025-01-28  
**Last Updated:** 2025-01-28
