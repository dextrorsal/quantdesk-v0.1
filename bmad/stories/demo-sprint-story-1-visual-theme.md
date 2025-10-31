# Story: Polish Existing Terminal Theme for Demo

**Story ID:** demo-sprint-story-1  
**Priority:** CRITICAL  
**Status:** ✅ complete  
**Epic:** Demo Video Sprint  
**Estimated Time:** 2-3 days  

---

## 📋 **User Story**

As a trader watching the demo video, I want to see a polished, professional terminal aesthetic (JetBrains Mono + blue/black hacker aesthetic), so I know QuantDesk is a serious Bloomberg Terminal-level trading platform.

---

## 🎯 **Acceptance Criteria**

### **Visual Polish:**
- [ ] Existing black/blue terminal theme is consistent across all pages
- [ ] JetBrains Mono font used throughout
- [ ] Terminal aesthetic is prominent and professional
- [ ] Smooth animations on state changes
- [ ] Hover effects are polished
- [ ] Loading states look good

### **Component Polish:**
- [ ] ProTradingTerminal layout is refined
- [ ] Trading cards have clean borders (no rounded corners per terminal style)
- [ ] Color scheme is consistent (success green #52c41a, danger red #ff4d4f)
- [ ] Charts fit terminal aesthetic
- [ ] Forms look professional
- [ ] Buttons are polished

### **Animations & Details:**
- [ ] Subtle pulse animations on status indicators
- [ ] Smooth transitions between pages
- [ ] Interactive elements respond well
- [ ] Loading states are polished
- [ ] Terminal aesthetic is minimal but professional

---

## 🛠️ **Technical Implementation**

### **Existing Theme Files:**

**Current Theme (found):**
- `frontend/src/pro/theme.css` ✅
- Colors: Black (#000), dark grays (#1a1a1a, #333)
- Terminal aesthetic: No border-radius, clean borders
- Status colors: Green #52c41a, Red #ff4d4f, Yellow #faad14

**Tailwind Config:**
- JetBrains Mono font family configured
- Blue colors: lite-primary (blue #3b82f6)
- Pro colors: pro-primary (orange #f97316)
- Terminal breakpoints already set

**Components to Polish:**
- Trading cards (refine gradients and borders)
- Metric cards (status indicators)
- Chart containers
- Sidebar styling
- Main content areas
- Status indicators (pulse animations)
- Buttons (polish states)
- Tables (Ant Design customizations)

---

## ✅ **Definition of Done**

- [ ] Terminal theme is polished and consistent
- [ ] JetBrains Mono is used throughout
- [ ] Black/blue colors are consistent
- [ ] Status indicators pulse smoothly
- [ ] No visual bugs or inconsistencies
- [ ] Professional terminal aesthetic achieved
- [ ] Ready for demo video recording

---

## 📸 **Demo Requirements**

This story is CRITICAL because the demo video will showcase the professional UI. The existing terminal aesthetic (already in place) needs polish to look Bloomberg Terminal-level.

---

**Status:** backlog → ready to start  
**Next:** Polish existing terminal theme files

**Current Theme:** Black (#000), Dark gray gradients, Blue accents, JetBrains Mono font

