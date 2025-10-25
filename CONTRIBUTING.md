# Contributing to QuantDesk

Thank you for your interest in contributing to QuantDesk! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions
- **Bug Reports**: Report bugs and issues
- **Feature Requests**: Suggest new features
- **Code Contributions**: Submit code improvements
- **Documentation**: Improve documentation
- **Examples**: Add community examples
- **Testing**: Improve test coverage

### Getting Started
1. **Fork the Repository**: Fork the QuantDesk repository
2. **Clone Your Fork**: Clone your fork locally
3. **Create a Branch**: Create a feature branch for your changes
4. **Make Changes**: Implement your changes
5. **Test Your Changes**: Ensure all tests pass
6. **Submit a Pull Request**: Submit your changes for review

## 📋 Development Setup

### Prerequisites
- **Node.js 20+**: Required for all services
- **pnpm**: Package manager (preferred over npm)
- **Rust**: Required for smart contract development
- **Solana CLI**: Required for blockchain interactions
- **Git**: Version control

### Environment Setup
```bash
# Clone your fork
git clone https://github.com/your-username/quantdesk.git
cd quantdesk

# Install dependencies
pnpm install

# Copy environment files
cp .env.example .env
cp frontend/.env.example frontend/.env
cp backend/.env.example backend/.env
cp MIKEY-AI/.env.example MIKEY-AI/.env
cp data-ingestion/.env.example data-ingestion/.env

# Start development servers
pnpm run dev
```

### Service-Specific Setup
```bash
# Frontend development
cd frontend
pnpm run dev

# Backend development
cd backend
pnpm run dev

# MIKEY-AI development
cd MIKEY-AI
pnpm run dev

# Data ingestion development
cd data-ingestion
pnpm run dev

# Smart contracts
cd contracts
anchor build
anchor test
```

## 🔧 Development Guidelines

### Code Style
- **TypeScript**: Use TypeScript for all new code
- **ESLint**: Follow ESLint configuration
- **Prettier**: Use Prettier for code formatting
- **Conventional Commits**: Use conventional commit messages
- **Documentation**: Document all public APIs

### Testing Requirements
- **Unit Tests**: Write unit tests for new features
- **Integration Tests**: Add integration tests where appropriate
- **Test Coverage**: Maintain test coverage above 80%
- **E2E Tests**: Add end-to-end tests for critical flows

### Code Review Process
1. **Self Review**: Review your own code before submitting
2. **Automated Checks**: Ensure all CI checks pass
3. **Peer Review**: At least one team member must review
4. **Security Review**: Security-sensitive changes require security review
5. **Final Approval**: Maintainer approval required for merge

## 📝 Pull Request Guidelines

### Before Submitting
- [ ] **Tests Pass**: All tests must pass
- [ ] **Code Style**: Code follows project style guidelines
- [ ] **Documentation**: Documentation updated if needed
- [ ] **Breaking Changes**: Breaking changes documented
- [ ] **Security**: No security vulnerabilities introduced

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process
1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: Team members review the code
3. **Testing**: Additional testing if needed
4. **Approval**: Maintainer approval required
5. **Merge**: Changes merged to main branch

## 🐛 Bug Reports

### Before Reporting
1. **Search Issues**: Check if the bug is already reported
2. **Update Software**: Ensure you're using the latest version
3. **Reproduce**: Confirm the bug is reproducible
4. **Check Documentation**: Verify it's not documented behavior

### Bug Report Template
```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g., Windows 10, macOS 12, Ubuntu 20.04]
- Node.js Version: [e.g., 20.0.0]
- Browser: [e.g., Chrome 120, Firefox 119]
- QuantDesk Version: [e.g., 0.1.1]

## Additional Context
Any additional context about the problem
```

## 💡 Feature Requests

### Before Requesting
1. **Search Issues**: Check if the feature is already requested
2. **Consider Alternatives**: Look for existing solutions
3. **Community Discussion**: Discuss in GitHub Discussions first
4. **Implementation**: Consider contributing the feature yourself

### Feature Request Template
```markdown
## Feature Description
Clear description of the feature

## Problem Statement
What problem does this feature solve?

## Proposed Solution
How should this feature work?

## Alternatives Considered
What alternatives have you considered?

## Additional Context
Any additional context or mockups
```

## 🔒 Security Contributions

### Security Guidelines
- **Responsible Disclosure**: Report security issues privately first
- **No Exploitation**: Do not exploit vulnerabilities
- **Documentation**: Document security-related changes thoroughly
- **Testing**: Include security tests for new features

### Security Contact
- **Email**: security@quantdesk.com
- **PGP**: [Download PGP key](https://quantdesk.com/security/pgp-key.asc)
- **GitHub**: Use GitHub Security Advisories for sensitive issues

## 📚 Documentation Contributions

### Documentation Types
- **API Documentation**: Document all public APIs
- **User Guides**: Improve user-facing documentation
- **Developer Guides**: Technical documentation for developers
- **Examples**: Add code examples and tutorials

### Documentation Standards
- **Clear Language**: Use clear, concise language
- **Code Examples**: Include practical code examples
- **Screenshots**: Add screenshots for UI changes
- **Links**: Link to related documentation

## 🧪 Testing Contributions

### Test Types
- **Unit Tests**: Test individual functions and components
- **Integration Tests**: Test service interactions
- **E2E Tests**: Test complete user workflows
- **Performance Tests**: Test performance characteristics

### Test Guidelines
- **Test Coverage**: Aim for high test coverage
- **Test Quality**: Write meaningful tests
- **Test Maintenance**: Keep tests up to date
- **Test Documentation**: Document test requirements

## 🎨 Design Contributions

### Design Guidelines
- **Consistency**: Follow existing design patterns
- **Accessibility**: Ensure accessibility compliance
- **Responsive**: Design for all screen sizes
- **Performance**: Consider performance implications

### Design Process
1. **Design Review**: Submit designs for review
2. **Implementation**: Implement approved designs
3. **Testing**: Test across different devices
4. **Documentation**: Document design decisions

## 🌐 Community Guidelines

### Code of Conduct
- **Be Respectful**: Treat everyone with respect
- **Be Inclusive**: Welcome contributors from all backgrounds
- **Be Constructive**: Provide constructive feedback
- **Be Patient**: Be patient with new contributors

### Communication
- **GitHub Issues**: Use for bug reports and feature requests
- **GitHub Discussions**: Use for general discussions
- **Discord**: Join our Discord community
- **Email**: Use for sensitive communications

## 🏆 Recognition

### Contributor Recognition
- **Contributors List**: All contributors listed in README
- **Release Notes**: Contributors credited in release notes
- **Special Recognition**: Outstanding contributors get special recognition
- **Swag**: Contributors may receive QuantDesk swag

### Types of Recognition
- **Code Contributions**: Code contributors
- **Documentation**: Documentation contributors
- **Community**: Community contributors
- **Security**: Security researchers

## 📋 Release Process

### Release Schedule
- **Major Releases**: Quarterly (breaking changes)
- **Minor Releases**: Monthly (new features)
- **Patch Releases**: Weekly (bug fixes)
- **Hotfixes**: As needed (critical fixes)

### Release Process
1. **Feature Freeze**: Stop adding new features
2. **Testing**: Comprehensive testing phase
3. **Documentation**: Update documentation
4. **Release Notes**: Prepare release notes
5. **Release**: Deploy to production
6. **Monitoring**: Monitor release health

## 🔧 Development Tools

### Recommended Tools
- **IDE**: VS Code with recommended extensions
- **Git**: Git for version control
- **Docker**: Docker for containerization
- **Postman**: Postman for API testing
- **Chrome DevTools**: Browser debugging

### VS Code Extensions
- **TypeScript**: TypeScript support
- **ESLint**: Code linting
- **Prettier**: Code formatting
- **GitLens**: Git integration
- **Thunder Client**: API testing

## 📞 Getting Help

### Support Channels
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions
- **Discord**: For real-time community support
- **Email**: For private or sensitive issues

### Documentation
- **README**: Start with the main README
- **API Docs**: Check API documentation
- **Examples**: Look at code examples
- **Wiki**: Check the project wiki

## 📄 License

By contributing to QuantDesk, you agree that your contributions will be licensed under the Apache License 2.0.

## 🙏 Thank You

Thank you for contributing to QuantDesk! Your contributions help make the platform better for everyone.

---

**Last Updated**: January 27, 2025  
**Version**: 1.0

For questions about contributing, contact contributors@quantdesk.com
