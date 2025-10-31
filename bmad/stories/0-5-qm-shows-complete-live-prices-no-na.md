# Story 0-5: QM Shows Complete Live Prices (No N/A)

Status: done

## Story

As a user, I need every instrument in Quick Monitor to show a live, non-zero price.

## Acceptance Criteria

1. ✅ 100% instruments render price and timestamp; snapshot test baselines the list
2. ✅ Missing assets are mapped to appropriate oracle symbols or excluded with reason

## Tasks / Subtasks

- [x] Implement backend price polling for QM [AC1]
  - [x] QM polls `/api/oracle/prices` every 5 seconds
  - [x] Backend prices used as authoritative source
  - [x] Prices displayed correctly for PERP pairs
- [x] Asset symbol mapping [AC2]
  - [x] Map market symbols to oracle symbols
  - [x] Handle missing assets appropriately
- [x] Price display implementation [AC1]
  - [x] All instruments show price and timestamp
  - [x] Format prices correctly (e.g., $3.81K for ETH)
  - [x] Display 24h% change, volume, high/low when available

## Dev Notes

### Status
✅ **COMPLETE** - Implementation verified working as of 2025-10-30
- Quote Monitor (QM) in Pro Terminal showing correct prices via Pyth/Oracle integration
- File: `frontend/src/pro/index.tsx` - `QMWindowContent` component (lines 2388-2498)
- Backend endpoint: `/api/oracle/prices` returns normalized prices via `pythOracleService.getAllPrices()`

### References
- [Source: bmad/docs/tech-spec-epic-0.md#Quote Monitor (QM) Live Prices]
- [Source: frontend/src/pro/index.tsx - QMWindowContent implementation]
- [Source: backend/src/routes/oracle.ts - price endpoint]

