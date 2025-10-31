# Story 0-9: Drift Parity Review for Deposit/Withdraw

Status: ready-for-dev

## Story

As a protocol engineer, I need a gap analysis vs Drift to minimize bugs.

## Acceptance Criteria

1. Document compares account lists, WSOL vs native SOL, remaining accounts, price checks; recommended approach selected (native/WSOL)
2. Risks and mitigation listed (e.g., rent, signer, account order)

## Tasks / Subtasks

- [ ] Research Drift Protocol deposit/withdraw implementation [AC1]
  - [ ] Document Drift account list for deposit/withdraw
  - [ ] Note WSOL vs native SOL approach
  - [ ] Document additional accounts used
  - [ ] Note price check integration
- [ ] Compare with QuantDesk implementation [AC1]
  - [ ] Document QuantDesk account list
  - [ ] Compare account order differences
  - [ ] Note any missing accounts
  - [ ] Compare price check approaches
- [ ] Make recommendation [AC1]
  - [ ] Evaluate WSOL vs native SOL pros/cons
  - [ ] Select recommended approach
  - [ ] Document rationale
- [ ] Document risks and mitigation [AC2]
  - [ ] List risks (rent, signer, account order, etc.)
  - [ ] Provide mitigation strategies for each risk
  - [ ] Document best practices learned

## Dev Notes

### Architecture Alignment
- Output document: `docs/drift-parity-analysis.md` (new)
- Reference: Drift Protocol documentation and codebase
- Compare with: `contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs`

### References
- [Source: bmad/docs/tech-spec-epic-0.md#Drift Parity Review]
- [Source: contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs - current implementation]

## Dev Agent Record

### Context Reference

- bmad/stories/0-9-drift-parity-review-deposit-withdraw.context.md

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List

