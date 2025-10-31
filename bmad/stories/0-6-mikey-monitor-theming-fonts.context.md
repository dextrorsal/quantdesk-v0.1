<story-context id="bmad/bmm/workflows/4-implementation/story-context/template" v="1.0">
  <metadata>
    <epicId>0</epicId>
    <storyId>6</storyId>
    <title>MIKEY Monitor Theming and Fonts</title>
    <status>ready-for-dev</status>
    <generatedAt>2025-10-30</generatedAt>
    <generator>BMAD Story Context Workflow</generator>
    <sourceStoryPath>bmad/stories/0-6-mikey-monitor-theming-fonts.md</sourceStoryPath>
  </metadata>

  <story>
    <asA>operator</asA>
    <iWant>MIKEY Monitor cards to use theme tokens</iWant>
    <soThat>UI is consistent</soThat>
    <tasks>
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
    </tasks>
  </story>

  <acceptanceCriteria>
1. All cards use theme vars for bg/border/text; no inline grays
2. Light/Dark switch preserves contrast; no color blending artifacts
  </acceptanceCriteria>

  <artifacts>
    <docs>
      <doc path="bmad/docs/tech-spec-epic-0.md" title="Epic 0 Technical Specification" section="MIKEY Monitor Theming">
        Defines requirements for consistent theming across MIKEY Monitor cards. Requires all cards use theme tokens instead of inline colors.
      </doc>
      <doc path="frontend/src/index.css" title="Theme Token Definitions" section="CSS Variables">
        Defines theme tokens: `--bg-primary`, `--bg-secondary`, `--bg-tertiary`, `--text-primary`, `--text-secondary`, `--border-primary`, `--primary-400`, `--success-500`, `--warning-400`, `--danger-500`, etc.
      </doc>
    </docs>
    <code>
      <artifact path="frontend/src/components/MikeyAIMonitoringDashboard.tsx" kind="component" symbol="MikeyAIMonitoringDashboard" lines="1-198" reason="MIKEY AI Monitoring Dashboard component. Uses some theme tokens but also has inline Tailwind color classes that need to be replaced.">
        Current implementation:
        - Already uses theme tokens: `var(--bg-secondary)`, `var(--bg-tertiary)`, `var(--text-primary)`, `var(--primary-400)`, `var(--success-500)`, `var(--warning-400)` (lines 115-117, 155-165)
        - **MISSING**: Inline Tailwind color classes:
          - Line 85: `bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900` - should use theme tokens
          - Line 91: `text-gray-400` - should use `var(--text-secondary)`
          - Line 94-95: `bg-green-600`, `bg-gray-700` - should use theme tokens
          - Line 97: `bg-blue-600`, `hover:bg-blue-700` - should use theme tokens
          - Line 103: `bg-red-900/50`, `border-red-500`, `text-red-300` - should use theme tokens
          - Line 124: `text-gray-400` - should use `var(--text-secondary)`
          - Line 129-135: `text-gray-400`, `text-blue-400`, `text-green-400`, `text-gray-500` - should use theme tokens
          - Line 180: `text-gray-400` - should use `var(--text-secondary)`
          - Line 183-184: `text-gray-400` - should use `var(--text-secondary)`
          - Line 193: `text-gray-400` - should use `var(--text-secondary)`
      </artifact>
      <artifact path="frontend/src/components/MikeyAIChat.tsx" kind="component" symbol="MikeyAIChat" lines="1-408" reason="MIKEY AI Chat component. May have inline colors that need theme token replacement.">
        MIKEY AI Chat component:
        - Should check for inline colors in styling
        - May need theme token replacement for consistency
      </artifact>
      <artifact path="frontend/src/components/playground/MikeyIntegrationShowcase.tsx" kind="component" symbol="MikeyIntegrationShowcase" lines="19-554" reason="MIKEY Integration Showcase component. Uses inline colors that need theme token replacement.">
        Current implementation:
        - Line 261-265: `bg-green-50`, `border-green-200`, `bg-red-50`, `border-red-200` - should use theme tokens
        - Line 268-276: `text-green-600`, `text-red-600`, `text-green-800`, `text-red-800` - should use theme tokens
        - Line 297-299: `bg-red-100`, `border-red-300`, `text-red-700` - should use theme tokens
        - Line 304: `bg-white` - should use `var(--bg-primary)`
        - May have more inline colors throughout
      </artifact>
      <artifact path="frontend/src/index.css" kind="stylesheet" symbol="CSS Variables" lines="1-150" reason="Theme token definitions. Defines CSS custom properties for consistent theming across the application.">
        Theme tokens defined:
        - `--bg-primary`, `--bg-secondary`, `--bg-tertiary`
        - `--text-primary`, `--text-secondary`, `--text-muted`
        - `--border-primary`, `--border-secondary`
        - `--primary-400`, `--primary-500`, `--primary-600`
        - `--success-500`, `--warning-400`, `--danger-500`
        - And more theme tokens for consistent theming
      </artifact>
    </code>
    <dependencies>
      <dependency>
        <ecosystem>React</ecosystem>
        <packages>
          <package>react: ^18.x</package>
          <package>tailwindcss: ^3.x</package>
        </packages>
      </dependency>
    </dependencies>
  </artifacts>

  <constraints>
    - All inline color values must be replaced with theme tokens
    - No hardcoded colors like `#1a1a1a`, `#ffffff`, `gray-400`, etc.
    - Use `var(--bg-secondary)`, `var(--bg-tertiary)` for backgrounds
    - Use `var(--text-primary)`, `var(--text-secondary)` for text
    - Use `var(--border-primary)` for borders
    - Use `var(--primary-400)`, `var(--success-500)`, `var(--warning-400)`, `var(--danger-500)` for accent colors
    - Light/dark theme switch must preserve contrast
    - No color blending artifacts on theme switch
    - All cards must remain readable in both light and dark themes
  </constraints>
  <interfaces>
    <interface name="CSS Custom Properties" kind="CSS Variables" signature="var(--token-name)" path="frontend/src/index.css">
      Theme tokens defined as CSS custom properties. Use `var(--token-name)` syntax to reference theme tokens.
    </interface>
    <interface name="Tailwind CSS Classes" kind="CSS Framework" signature="className='bg-gray-400'" path="frontend/src/components/">
      Tailwind CSS classes for styling. Replace inline color classes with style objects using theme tokens.
    </interface>
  </interfaces>
  <tests>
    <standards>
      Frontend tests use Vitest and React Testing Library. Test files located in `frontend/src/tests/` with subdirectories: `unit/`, `integration/`, `e2e/`. Use `@testing-library/react` for component testing. Visual regression tests for theme switching.
    </standards>
    <locations>
      - `frontend/src/tests/unit/` - Unit tests for utilities and components
      - `frontend/src/tests/integration/` - Integration tests for components
      - `frontend/src/tests/e2e/` - End-to-end tests for user flows
    </locations>
    <ideas>
      <test ac="AC1" idea="Component test: Verify all cards use theme tokens">
        Render MikeyAIMonitoringDashboard. Check all cards use `var(--bg-secondary)`, `var(--text-primary)`, etc. in style attributes. Verify no inline color values like `#1a1a1a`, `gray-400` remain.
      </test>
      <test ac="AC2" idea="Integration test: Theme switch preserves contrast">
        Render component in light theme. Switch to dark theme. Verify contrast preserved. Verify all cards remain readable. Check for color blending artifacts.
      </test>
      <test ac="AC2" idea="E2E test: User switches theme and verifies readability">
        Open MIKEY Monitor in light theme. Switch to dark theme. Verify all cards readable. Switch back to light theme. Verify no color artifacts.
      </test>
    </ideas>
  </tests>
</story-context>

