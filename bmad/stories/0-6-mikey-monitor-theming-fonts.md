# Story 0-6: MIKEY Monitor Theming and Fonts

Status: ready-for-dev

## Story

As an operator, I want MIKEY Monitor cards to use theme tokens so UI is consistent.

## Acceptance Criteria

1. All cards use theme vars for bg/border/text; no inline grays
2. Light/Dark switch preserves contrast; no color blending artifacts

## Tasks / Subtasks

- [ ] Replace inline gray colors with theme tokens [AC1]
  - [ ] Find all MIKEY Monitor card components
  - [ ] Replace `#1a1a1a`, `#ffffff`, etc. with `var(--bg-tertiary)`, `var(--text-primary)`
  - [ ] Use `var(--bg-secondary)`, `var(--border-primary)` for borders
  - [ ] Verify no inline color values remain
- [ ] Test light/dark theme switch [AC2]
  - [ ] Verify contrast preserved on theme switch
  - [ ] Check for color blending artifacts
  - [ ] Ensure all cards remain readable in both themes
  - [ ] Test theme toggle functionality

## Dev Notes

### Architecture Alignment
- Location: `frontend/src/components/MIKEYMonitor.tsx` or equivalent
- Use existing theme token system
- Test with both light and dark themes

### References
- [Source: bmad/docs/tech-spec-epic-0.md#MIKEY Monitor Theming]
- [Source: frontend/src - theme token definitions]

## Dev Agent Record

### Context Reference

- bmad/stories/0-6-mikey-monitor-theming-fonts.context.md

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List

