# Dual Repository Setup - Execution Plan

## Confirmed Settings ✅
- **Repository Management**: Mirror approach (add "private" remote to same folder)
- **Public Repo**: Keep current filtering (contracts/docs/sdk visible, backend/MIKEY-AI/frontend hidden)
- **Private Repo**: Everything except heavy files (node_modules, target, dist, logs, etc.)
- **Backup Method**: tar.xz compressed archive
- **Timestamp**: 2025-10-28 18:06:52 EDT

## Execution Steps

### Phase 1: Create Backup
1. Create compressed archive: `/home/dex/Desktop/quantdesk-0.1.1-backup-20251028-180652.tar.xz`
2. Verify backup integrity

### Phase 2: Fix Gitignore
1. Rename `gitignore` → `.gitignore`
2. Verify current filtering is appropriate for public repo
3. Stage `.gitignore` for commit

### Phase 3: Update Public Repository
1. Review git status for what should/shouldn't be committed
2. Stage appropriate changes
3. Commit with descriptive message
4. Push to public origin (https://github.com/dextrorsal/quantdesk-v0.1.git)

### Phase 4: Configure Private Repository
1. Add private remote: `git remote add private https://github.com/dextrorsal/quantdesk.git`
2. Push full codebase (including backend, MIKEY-AI, etc.) to private repo
3. Use appropriate gitignore that:
   - Hides: node_modules, target, dist, logs, *.log, test-results
   - Shows: All source code (backend, MIKEY-AI, frontend, BMAD, etc.)

### Phase 5: Verify Both Repos
1. Check public repo has correct filtering
2. Check private repo has full source
3. Restore working gitignore for development

## Safety Notes
- Backup created before any git operations
- Can rollback easily if needed
- Private repo gets full source, public gets filtered view

