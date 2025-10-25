# Rclone Sync Guide for QuantDesk

This guide explains how to efficiently sync your QuantDesk project using rclone while excluding unnecessary files and directories.

## Quick Start

### 1. Test What Will Be Synced (Dry Run)
```bash
cd /home/dex/Desktop/quantdesk-1.0.6
rclone ls . --filter-from rclone-filter.txt
```

### 2. Sync to Your Google Drive
```bash
# For gdrive-dex remote
rclone sync /home/dex/Desktop/quantdesk-1.0.6 gdrive-dex:QuantDesk --filter-from rclone-filter.txt

# For gdrive2 remote  
rclone sync /home/dex/Desktop/quantdesk-1.0.6 gdrive2:QuantDesk --filter-from rclone-filter.txt
```

### 3. Verify Sync Results
```bash
# Check what's on remote
rclone ls gdrive-dex:QuantDesk --filter-from rclone-filter.txt

# Compare sizes
rclone size gdrive-dex:QuantDesk
```

## What Gets Synced ✅

**Source Code:**
- `.ts`, `.tsx`, `.js`, `.jsx` - TypeScript/JavaScript files
- `.rs` - Rust source files  
- `.sol` - Solidity files

**Documentation:**
- All `.md` files in `docs/` and subdirectories
- README files and project documentation

**Configuration Files:**
- `package.json` - Node.js dependencies
- `Cargo.toml` - Rust dependencies
- `tsconfig.json` - TypeScript configuration
- `.yml`, `.yaml`, `.toml` - Configuration files
- `.env.example` - Environment templates

**Lock Files (for reproducibility):**
- `pnpm-lock.yaml` - pnpm lock file
- `package-lock.json` - npm lock file  
- `Cargo.lock` - Rust lock file
- `yarn.lock` - yarn lock file

**Database & Scripts:**
- `.sql` files - Database schemas
- `.sh` files - Shell scripts

**Project Structure:**
- All directory structure is preserved
- Only file contents are filtered

## What Gets Excluded ❌

**Dependencies (~6GB saved):**
- `node_modules/` directories (5,322+ directories)
- Rust `target/` directories (3.8GB)

**Build Artifacts:**
- `dist/`, `build/`, `out/` directories
- `.anchor/` - Anchor build cache

**Test Data (4.9GB saved):**
- `test-ledger/` - Solana test ledger data

**Generated Files:**
- `*.log` files and `logs/` directories
- `*.tsbuildinfo` - TypeScript build info
- `.eslintcache`, `.prettiercache` - Linter caches
- `coverage/`, `test-results/` - Test outputs

**IDE & OS Files:**
- `.vscode/`, `.idea/` - IDE configurations
- `.DS_Store`, `Thumbs.db` - OS metadata

**Version Control:**
- `.git/` directory (no git history)

**Compiled Documentation:**
- `docs-site/html/` - Can be regenerated from markdown

**Caches:**
- `.cache/`, `.parcel-cache/`, `.next/`, `.nuxt/`
- Playwright, Jest, Storybook caches

## Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Total Size** | ~12GB | ~50-100MB | 99% reduction |
| **File Count** | 100,000+ | ~2,000 | 98% reduction |
| **Sync Time** | Hours | Minutes | 95% faster |
| **node_modules** | 5,322 dirs | 0 | 100% excluded |
| **test-ledger** | 4.9GB | 0 | 100% excluded |
| **target/** | 3.8GB | 0 | 100% excluded |

## Advanced Usage

### Sync Specific Directories Only
```bash
# Sync only source code
rclone sync /home/dex/Desktop/quantdesk-1.0.6/frontend/src gdrive-dex:QuantDesk/frontend/src --filter-from rclone-filter.txt

# Sync only documentation
rclone sync /home/dex/Desktop/quantdesk-1.0.6/docs gdrive-dex:QuantDesk/docs --filter-from rclone-filter.txt
```

### Two-Way Sync (Bidirectional)
```bash
# Sync both ways (be careful!)
rclone bisync /home/dex/Desktop/quantdesk-1.0.6 gdrive-dex:QuantDesk --filter-from rclone-filter.txt
```

### Check Differences Without Syncing
```bash
# See what would change
rclone check /home/dex/Desktop/quantdesk-1.0.6 gdrive-dex:QuantDesk --filter-from rclone-filter.txt --one-way
```

### Exclude Additional Files
If you need to exclude more files, add patterns to `rclone-filter.txt`:
```
- additional-pattern/**
- *.additional-extension
```

## Troubleshooting

### Filter Not Working?
1. Check filter file path: `--filter-from rclone-filter.txt`
2. Use absolute path: `--filter-from /home/dex/Desktop/quantdesk-1.0.6/rclone-filter.txt`
3. Test with dry run first: `rclone ls . --filter-from rclone-filter.txt`

### Sync Taking Too Long?
1. Check if large files are still being synced
2. Verify filter patterns are correct
3. Use `--progress` flag to see what's being transferred

### Missing Files After Sync?
1. Check if files match exclusion patterns
2. Use `rclone ls` to verify what's on remote
3. Files might be in excluded directories

## Restoring Project

After syncing, you can restore the full project:

```bash
# Clone from remote
rclone copy gdrive-dex:QuantDesk /path/to/restore/location

# Install dependencies
cd /path/to/restore/location
pnpm install  # Restores node_modules
cd contracts && anchor build  # Restores target/
```

## Best Practices

1. **Always test first**: Use `rclone ls` with filters before actual sync
2. **Backup important data**: Don't rely solely on rclone for backups
3. **Check sync results**: Verify what was actually synced
4. **Use dry-run**: Add `--dry-run` flag for testing
5. **Monitor progress**: Use `--progress` for long syncs

## File Structure After Sync

```
QuantDesk/
├── backend/src/          # TypeScript source
├── frontend/src/         # React source  
├── contracts/programs/   # Rust source
├── docs/                # Markdown documentation
├── MIKEY-AI/src/        # AI agent source
├── database/            # SQL schemas
├── scripts/             # Shell scripts
├── package.json         # Dependencies
├── Cargo.toml          # Rust dependencies
├── pnpm-lock.yaml      # Lock file
└── rclone-filter.txt   # This filter file
```

## Support

- [Rclone Documentation](https://rclone.org/filtering/)
- [Rclone Forum](https://forum.rclone.org/)
- Check `rclone --help` for more options
