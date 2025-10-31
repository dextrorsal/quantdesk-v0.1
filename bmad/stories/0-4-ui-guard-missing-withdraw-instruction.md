# Story 0-4: UI Guard for Missing Withdraw Instruction

Status: ready-for-dev

## Story

As a user, I should not hit runtime errors if withdraw is unavailable.

## Acceptance Criteria

1. Withdraw action is soft-disabled when IDL lacks `withdraw_native_sol`; tooltip explains "Upgrade in progress"
2. No "program.methods.withdrawNativeSol is not a function" appears in console

## Tasks / Subtasks

- [ ] Detect if IDL lacks `withdraw_native_sol` instruction [AC1]
  - [ ] Check `program?.idl?.instructions` for `withdrawNativeSol`
  - [ ] Create utility function to check instruction availability
- [ ] Soft-disable withdraw button [AC1]
  - [ ] Disable button when instruction unavailable
  - [ ] Add tooltip: "Upgrade in progress"
  - [ ] Style appropriately (grayed out, not clickable)
- [ ] Prevent runtime errors [AC2]
  - [ ] Never call `program.methods.withdrawNativeSol` if unavailable
  - [ ] Add guard checks before any withdraw function calls
  - [ ] Test console for any error messages

## Dev Notes

### Architecture Alignment
- Location: `frontend/src/components/WithdrawButton.tsx` or equivalent component
- Check program IDL availability before rendering button
- Use theme tokens for disabled state styling

### References
- [Source: bmad/docs/tech-spec-epic-0.md#UI Guard for Missing Withdraw]
- [Source: frontend/src/contexts/ProgramContext.tsx - program/IDL access]

## Dev Agent Record

### Context Reference

- bmad/stories/0-4-ui-guard-missing-withdraw-instruction.context.md

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List

