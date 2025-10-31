# Story 0-10: Expert Confirmation of Native SOL Patterns

Status: ready-for-dev

## Story

As a protocol engineer, I need expert-validated patterns to avoid regressions.

## Acceptance Criteria

1. Short note capturing expert guidance: signer first, SystemAccount for vault, when to include rent, invoke vs invoke_signed for PDA vault, Anchor 0.30 accounts/addresses behavior

## Tasks / Subtasks

- [ ] Research expert guidance on native SOL patterns [AC1]
  - [ ] Signer position in account struct (must be first)
  - [ ] SystemAccount usage for vault
  - [ ] When to include rent Sysvar
  - [ ] invoke vs invoke_signed for PDA vault
  - [ ] Anchor 0.30+ accounts/addresses behavior
- [ ] Document expert guidance [AC1]
  - [ ] Create concise document with all patterns
  - [ ] Include examples where helpful
  - [ ] Note any version-specific behaviors
  - [ ] Document best practices

## Dev Notes

### Architecture Alignment
- Output document: `docs/native-sol-patterns-expert-guidance.md` (new)
- Reference Solana/Anchor documentation and expert guidance
- Ensure alignment with current implementation

### References
- [Source: bmad/docs/tech-spec-epic-0.md#Expert Confirmation of Native SOL Patterns]
- [Source: contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs - current implementation]
- [Source: Anchor Framework documentation - account patterns]

## Dev Agent Record

### Context Reference

- bmad/stories/0-10-expert-confirmation-native-sol-patterns.context.md

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List

