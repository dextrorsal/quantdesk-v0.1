# Development Guidelines

## Component Development Standards
- **Functional Components**: Use functional components with TypeScript
- **Props Interfaces**: Define clear prop interfaces for type safety
- **Error Boundaries**: Implement error boundaries for error handling
- **Loading States**: Always handle loading states for async operations
- **Accessibility**: Follow WCAG 2.1 AA guidelines

## State Management Best Practices
- **Zustand for Global State**: Use Zustand for complex global state
- **React Context for Cross-Component**: Use Context for component tree state
- **Custom Hooks for Reusable Logic**: Extract reusable logic into custom hooks
- **Local State for Component-Specific**: Use useState for component-specific data

## Service Layer Patterns
- **Singleton Pattern**: Use singleton pattern for services
- **Async/Await**: Use async/await for API calls
- **Error Handling**: Implement comprehensive error handling with try-catch
- **TypeScript Interfaces**: Define interfaces for API responses

## Testing Standards
- **Test File Location**: `*.test.tsx` alongside components
- **E2E Tests**: `*.e2e.test.ts` in components directory
- **Service Tests**: `*.test.ts` in services directory
- **Arrange-Act-Assert**: Follow AAA testing pattern
- **Mock External Dependencies**: Mock API calls, WebSocket, Solana
- **Test User Interactions**: Test clicks, form submissions, navigation
- **Test Error States**: Test error handling and edge cases
- **Coverage Requirements**: Aim for 80% code coverage
