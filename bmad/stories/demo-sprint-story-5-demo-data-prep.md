# Story: Prepare Demo Data & Recording Environment

**Story ID:** demo-sprint-story-5  
**Priority:** HIGH  
**Status:** backlog  
**Epic:** Demo Video Sprint  
**Estimated Time:** 1-2 days  

---

## 📋 **User Story**

As the demo video creator, I want realistic demo data and a smooth recording environment, so the video looks professional and showcases QuantDesk flawlessly without bugs or errors.

---

## 🎯 **Acceptance Criteria**

### **Demo Data Setup:**
- [ ] Create demo account with realistic balance
- [ ] Pre-populate sample orders (mix of filled/pending)
- [ ] Set up sample positions with P&L
- [ ] Populate chat history with realistic messages
- [ ] Add sample market data
- [ ] Include referral examples

### **Recording Environment:**
- [ ] All services running smoothly
- [ ] No UI bugs or errors
- [ ] Fast load times
- [ ] Smooth transitions throughout
- [ ] WebSocket connections stable
- [ ] Backend responses are quick

### **Backup Materials:**
- [ ] Screenshots of all key screens
- [ ] Demo script finalized
- [ ] Narration script prepared
- [ ] Timing breakdown confirmed
- [ ] All features tested and working

---

## 🛠️ **Technical Implementation**

### **Demo Data Script:**

```typescript
// Create demo scenarios
const demoData = {
  user: {
    wallet: "DemoWalletAddress",
    balance: "1000 USD",
    accounts: ["master", "sub-account-1"]
  },
  orders: [
    { type: "limit", status: "filled", market: "BTC" },
    { type: "market", status: "pending", market: "ETH" },
    // ... more samples
  ],
  positions: [
    { market: "BTC", size: "1.5", pnl: "+$245.30" },
    { market: "ETH", size: "5.0", pnl: "-$89.50" },
  ]
};
```

### **Recording Checklist:**
- [ ] OBS/ScreenFlow ready
- [ ] Audio input tested
- [ ] Browser cache cleared
- [ ] All tabs/bookmarks hidden
- [ ] Resolution set to 1920x1080
- [ ] Background music prepared (optional)

---

## ✅ **Definition of Done**

- [ ] Demo environment is ready
- [ ] All data looks realistic
- [ ] No errors during demo flow
- [ ] Backup materials prepared
- [ ] Ready to record video
- [ ] Timeline can be completed in 4.5 minutes

---

**Status:** backlog  
**Next:** Final prep before video recording

