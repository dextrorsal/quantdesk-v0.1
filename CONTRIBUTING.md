# Contributing to QuantDesk

Thank you for your interest in contributing to QuantDesk! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Check if the issue already exists
2. Use the appropriate issue template
3. Provide detailed information about the problem

### Suggesting Enhancements

We welcome feature suggestions! Please:
1. Check existing feature requests
2. Provide a clear description of the proposed feature
3. Explain the use case and benefits
4. Consider implementation complexity

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Ensure all tests pass**
6. **Commit your changes**: `git commit -m 'Add amazing feature'`
7. **Push to your branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

## 🏗️ Development Setup

### Prerequisites

- Node.js 18+
- Rust 1.70+
- Solana CLI tools
- Anchor Framework
- Git

### Local Development

1. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/quantdesk.git
   cd quantdesk
   ```

2. **Install dependencies**:
   ```bash
   # Backend
   cd backend && npm install
   
   # Frontend
   cd ../frontend && npm install
   
   # Smart contracts
   cd ../contracts/smart-contracts && npm install
   ```

3. **Set up environment**:
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Start development servers**:
   ```bash
   # Terminal 1: Backend
   cd backend && npm run dev
   
   # Terminal 2: Frontend
   cd frontend && npm run dev
   
   # Terminal 3: Solana validator
   solana-test-validator
   ```

## 📝 Coding Standards

### TypeScript/JavaScript

- Use TypeScript for all new code
- Follow ESLint configuration
- Use meaningful variable and function names
- Add JSDoc comments for public APIs
- Prefer functional programming patterns

### Rust (Smart Contracts)

- Follow Rust naming conventions
- Use `cargo fmt` and `cargo clippy`
- Add comprehensive documentation
- Write unit tests for all functions
- Use Anchor best practices

### React Components

- Use functional components with hooks
- Implement proper TypeScript types
- Follow component composition patterns
- Use Tailwind CSS for styling
- Write component tests

## 🧪 Testing

### Smart Contract Tests

```bash
cd contracts/smart-contracts
anchor test
```

### Backend Tests

```bash
cd backend
npm test
```

### Frontend Tests

```bash
cd frontend
npm test
```

### Test Coverage

- Aim for >80% test coverage
- Write integration tests for critical paths
- Test error conditions and edge cases
- Use mocking for external dependencies

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)

### PR Description

Include:
- **Summary**: Brief description of changes
- **Type**: Bug fix, feature, refactor, docs, etc.
- **Testing**: How you tested the changes
- **Breaking Changes**: Any breaking changes
- **Screenshots**: For UI changes

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in development environment
4. **Approval** from at least one maintainer
5. **Merge** by maintainers

## 🏷️ Commit Message Format

Use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(trading): add stop-loss order support
fix(api): resolve authentication token expiration
docs(readme): update installation instructions
```

## 🐛 Bug Reports

When reporting bugs, include:

1. **Environment**:
   - OS and version
   - Node.js version
   - Browser (for frontend issues)

2. **Steps to reproduce**:
   - Clear, numbered steps
   - Expected vs actual behavior

3. **Additional context**:
   - Screenshots or videos
   - Error messages/logs
   - Related issues

## 💡 Feature Requests

For feature requests, provide:

1. **Problem description**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Alternatives considered**: Other approaches?
4. **Additional context**: Screenshots, mockups, etc.

## 🏆 Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes
- Invited to the contributors' Discord channel

## 📞 Getting Help

- **Discord**: [Join our community](https://discord.gg/quantdesk)
- **GitHub Discussions**: For questions and ideas
- **Email**: [contact@quantdesk.io](mailto:contact@quantdesk.io)

## 📄 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## 🎯 Areas Needing Help

- **Documentation**: API docs, tutorials, guides
- **Testing**: Unit tests, integration tests, E2E tests
- **UI/UX**: Design improvements, accessibility
- **Performance**: Optimization, caching, monitoring
- **Security**: Audits, vulnerability assessments
- **DevOps**: CI/CD, deployment automation

Thank you for contributing to QuantDesk! 🚀
