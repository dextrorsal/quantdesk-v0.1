# Story: Epic 2.0 - Documentation Atomic Refactor

**Epic:** Documentation Quality & Maintenance  
**Story ID:** 2.0-doc-refactor  
**Story Type:** Technical Debt / Infrastructure  
**Priority:** CRITICAL 🚨  
**Estimate:** XL (16+ hours) - This is a MASSIVE undertaking

---

## 📋 Story Description

**As a** development team  
**I need** documentation that is accurate, essential, and well-organized  
**So that** I can efficiently develop features without getting confused by outdated, redundant, or incorrect information

---

## 🎯 Business Value

### Why This Story is CRITICAL

1. **Developer Productivity**: Old/incorrect docs waste hours of developer time
2. **Onboarding**: New team members get misled by conflicting information
3. **Technical Debt**: Documentation debt compounds over time
4. **Code-Doc Mismatch**: Docs don't match the actual codebase
5. **BMad Integration**: BMad workflows require accurate, lean documentation

### Success Metrics

- **Reduce doc count**: 538 → target 100-150 essential docs
- **100% accuracy**: Every remaining doc matches actual codebase
- **0 redundancy**: No duplicate information across files
- **Navigation speed**: Devs can find info in <30 seconds
- **BMad compliance**: All BMad configs work properly

---

## 🔥 Scope

### What's Included

This is an **ATOMIC REFACTOR** affecting:

1. **All documentation files** (538 files to audit)
   - `bmad-docs/` (most files)
   - `docs/` (all files)
   - `docs/archive/` (audit for relevance)

2. **Every file must be:**
   - ✅ Reviewed for accuracy
   - ✅ Checked against actual codebase
   - ✅ Categorized (KEEP / UPDATE / DELETE / ARCHIVE)
   - ✅ Consolidated if redundant

3. **Documentation structure:**
   - Align with BMad's `core-config.yaml` expectations
   - Create proper `docs/departments/` structure
   - Organize by department (frontend, backend, contracts, etc.)
   - Ensure PRD and architecture docs are current

### What's NOT Included (Future Stories)

- Code refactoring (separate story)
- Feature development (pause until this is done)
- Testing (will do after docs are clean)

---

## ✅ Acceptance Criteria

### Phase 1: Audit Complete
- [ ] **Inventory**: Complete list of all docs created
- [ ] **Categorization**: Each doc tagged (KEEP/UPDATE/DELETE/ARCHIVE)
- [ ] **Accuracy Check**: Each doc verified against codebase
- [ ] **Redundancy Map**: All duplicates identified

### Phase 2: Structure Fixed
- [ ] **BMad Compliance**: `docs/` structure matches `core-config.yaml`
- [ ] **Departments Created**: All 9 department docs in `docs/departments/`
- [ ] **PRD Current**: `docs/prd.md` reflects actual state
- [ ] **Architecture Current**: `docs/architecture.md` updated

### Phase 3: Consolidation Complete
- [ ] **Deletions**: All tagged DELETE docs removed
- [ ] **Archives**: Historical docs moved to `docs/archive/`
- [ ] **Updates**: All tagged UPDATE docs fixed
- [ ] **Merges**: Redundant docs consolidated

### Phase 4: Validation
- [ ] **BMad Commands Test**: `@pm`, `@architect`, `@qa` all work
- [ ] **Navigation Test**: Key info findable in <30 seconds
- [ ] **Accuracy Check**: Spot check 10 random docs for accuracy
- [ ] **Size Target**: Doc count reduced to 100-150 files

---

## 🗂️ File Categories

### **KEEP** (Essential, Current, Unique)
- Active PRDs and epics
- Current architecture documentation
- Active department specs
- API documentation (if current)
- Setup/deployment guides (if current)
- Security documentation

### **UPDATE** (Valuable but Outdated)
- Architecture that needs minor updates
- API docs that need endpoint updates
- Guides that reference old code
- Stories that are partially complete

### **DELETE** (Not Needed)
- Superseded by newer docs
- Completely outdated
- Redundant information
- Planned but not implemented features
- Failed experiments/documentation

### **ARCHIVE** (Historical Reference)
- Completed work docs
- Historical decisions
- Old architecture versions
- Previous implementation attempts

---

## 🔍 Audit Process

For **EVERY SINGLE DOCUMENT**:

1. **Read the file**
2. **Check accuracy** against codebase
3. **Identify category** (KEEP/UPDATE/DELETE/ARCHIVE)
4. **Document decision** in audit spreadsheet
5. **Take action** (keep, update, delete, or move)

---

## 📊 Estimated Breakdown

| Phase | Task | Est. Hours | Critical? |
|-------|------|-----------|-----------|
| **Phase 1** | Create audit process & spreadsheet | 2 | ✅ |
| **Phase 1** | Audit 538 files (2 min/file avg) | 18 | ✅ |
| **Phase 2** | Fix directory structure | 4 | ✅ |
| **Phase 3** | Consolidate departments | 6 | ✅ |
| **Phase 3** | Update PRD & Architecture | 4 | ✅ |
| **Phase 3** | Delete unused docs | 2 | ✅ |
| **Phase 4** | Validate & test | 4 | ✅ |
| **Phase 4** | Final accuracy check | 2 | ✅ |
| **Total** | | **42 hours** | |

**This is why it's an XL story!**

---

## 🚨 Risks & Blockers

### Risks
- **Time sink**: This will take weeks if not focused
- **Accidental deletion**: Could delete something important
- **Breaking changes**: BMad configs might break during move
- **Developer confusion**: Docs in flux during refactor

### Mitigation
- ✅ Create backup before starting
- ✅ Use git to track changes
- ✅ Do in phases, test after each
- ✅ Get team buy-in before starting

---

## 📝 Dependencies

### Prerequisites
- Backup of all docs created
- Audit spreadsheet template
- Current codebase reference
- BMad configs understood

### Blocks
- **This story BLOCKS all other work** until complete
- Developers can't trust docs until this is done
- New features will add more docs (problem compounds)

---

## 🎯 Definition of Done

- [ ] All 538 docs audited and categorized
- [ ] Doc count reduced to 100-150 essential files
- [ ] 100% of remaining docs verified for accuracy
- [ ] `docs/` structure matches BMad expectations
- [ ] All BMad commands (`@pm`, `@architect`, etc.) work
- [ ] PRD and architecture docs are current
- [ ] Department docs created and accurate
- [ ] Audit spreadsheet completed
- [ ] Backup created and git committed
- [ ] Team trained on new structure

---

## 💬 Notes

**WHY THIS IS CRITICAL:**

You've built an amazing system, but documentation debt has accumulated. This isn't just cleanup - it's an **atomic refactor of your knowledge base**. Every line of code you've written is documented somewhere, but finding the right docs is becoming impossible.

**The hoarding problem:**
- 538 files is too many to navigate
- Many are outdated (describe code that doesn't exist)
- Many are redundant (same info in multiple places)
- Structure doesn't match what BMad expects
- Devs waste hours searching for correct info

**After this story:**
- Lean, accurate documentation
- BMad workflows work properly
- New team members can onboard fast
- Development speed increases
- Technical debt reduced

**This story is "nuclear" because:**
- Affects EVERY documented feature
- Requires touching hundreds of files
- Can't be done incrementally (needs to be atomic)
- Once done, future docs will stay organized
- Prevents documentation from becoming a problem again

---

**Story Owner:** Documentation Team  
**Reviewers:** All Development Team  
**Status:** Ready for Development (pending approval to start massive refactor)

