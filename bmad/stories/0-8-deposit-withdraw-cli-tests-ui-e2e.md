# Story 0-8: Deposit/Withdraw CLI Tests and UI E2E

Status: ready-for-dev

## Story

As a maintainer, I need automated checks to ensure deposits/withdraws function through upgrades.

## Acceptance Criteria

1. CLI scripts: `deposit_native_sol`, `withdraw_native_sol`; assert PDA derivations, accounts order, success logs
2. UI e2e: connect wallet → deposit → see USD balance → withdraw → balance reduces; screenshots stored

## Tasks / Subtasks

- [ ] Create CLI test script for deposit [AC1]
  - [ ] Script: `scripts/test-deposit-native-sol.sh`
  - [ ] Assert PDA derivations correct
  - [ ] Verify account order matches IDL
  - [ ] Check success logs
  - [ ] Verify SOL balance increases in protocol_vault
  - [ ] Return transaction signature
- [ ] Create CLI test script for withdraw [AC1]
  - [ ] Script: `scripts/test-withdraw-native-sol.sh`
  - [ ] Assert PDA derivations correct
  - [ ] Verify account order matches IDL
  - [ ] Check success logs
  - [ ] Verify lamports decrease in protocol_vault
  - [ ] Verify lamports increase in user wallet
  - [ ] Return transaction signature
- [ ] Create UI E2E test for deposit flow [AC2]
  - [ ] Connect wallet
  - [ ] Deposit SOL via UI
  - [ ] Verify USD balance updates in UI
  - [ ] Capture screenshots at each step
  - [ ] Store screenshots for regression testing
- [ ] Create UI E2E test for withdraw flow [AC2]
  - [ ] Withdraw SOL via UI
  - [ ] Verify balance reduces correctly
  - [ ] Capture screenshots at each step
  - [ ] Store screenshots for regression testing

## Dev Notes

### Architecture Alignment
- E2E test location: `tests/e2e/deposit-withdraw.spec.ts` (new)
- CLI scripts location: `scripts/` directory
- Use Playwright or Cypress for E2E tests
- Screenshots stored in `tests/e2e/screenshots/`

### References
- [Source: bmad/docs/tech-spec-epic-0.md#Deposit/Withdraw CLI Tests and UI E2E]
- [Source: contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs - deposit/withdraw functions]

## Dev Agent Record

### Context Reference

- bmad/stories/0-8-deposit-withdraw-cli-tests-ui-e2e.context.md

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List

