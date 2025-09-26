# Contributing to QuantDesk

Thank you for your interest in contributing to QuantDesk! We welcome contributions from the community and appreciate your help in making QuantDesk the premier Bloomberg Terminal for Crypto.

## 🤝 **How to Contribute**

### 1. **Report Issues**
- Use [GitHub Issues](https://github.com/quantdesk/quantdesk/issues) to report bugs
- Include detailed steps to reproduce the issue
- Provide system information and error logs

### 2. **Suggest Features**
- Use [GitHub Discussions](https://github.com/quantdesk/quantdesk/discussions) for feature requests
- Describe the use case and expected behavior
- Consider the impact on existing users

### 3. **Submit Code**
- Fork the repository
- Create a feature branch
- Make your changes
- Submit a pull request

## 🚀 **Getting Started**

### Prerequisites
- Node.js 18+
- Git
- Solana CLI tools
- Basic knowledge of TypeScript/JavaScript

### Development Setup

1. **Fork and Clone**
```bash
git clone https://github.com/your-username/quantdesk.git
cd quantdesk
```

2. **Install Dependencies**
```bash
# Backend
cd backend
npm install

# Frontend
cd ../frontend
npm install
```

3. **Environment Setup**
```bash
# Copy environment files
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

4. **Start Development**
```bash
# Backend
cd backend
npm run dev

# Frontend (new terminal)
cd frontend
npm run dev
```

## 📋 **Contribution Guidelines**

### Code Style

- **TypeScript**: Use TypeScript for all new code
- **ESLint**: Follow our ESLint configuration
- **Prettier**: Use Prettier for code formatting
- **Naming**: Use descriptive variable and function names

### Commit Messages

Use conventional commit format:
```
type(scope): description

feat(api): add new order endpoint
fix(ui): resolve wallet connection issue
docs(readme): update installation guide
```

### Pull Request Process

1. **Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make Changes**
- Write clean, well-documented code
- Add tests for new functionality
- Update documentation as needed

3. **Test Your Changes**
```bash
# Backend tests
cd backend
npm test

# Frontend tests
cd ../frontend
npm test
```

4. **Submit Pull Request**
- Provide clear description of changes
- Link to related issues
- Include screenshots for UI changes

## 🎯 **Areas for Contribution**

### Frontend Development
- **UI Components**: Improve existing components or create new ones
- **User Experience**: Enhance the trading interface
- **Mobile Responsiveness**: Optimize for mobile devices
- **Accessibility**: Improve accessibility features

### Backend Development
- **API Endpoints**: Add new endpoints or improve existing ones
- **Performance**: Optimize database queries and API responses
- **Security**: Enhance security measures
- **Monitoring**: Improve logging and monitoring

### Smart Contracts
- **New Features**: Implement new trading features
- **Gas Optimization**: Reduce transaction costs
- **Security Audits**: Review and improve security
- **Testing**: Add comprehensive test coverage

### Documentation
- **API Documentation**: Improve API documentation
- **User Guides**: Create tutorials and guides
- **Code Comments**: Add inline documentation
- **Architecture Docs**: Document system architecture

### Testing
- **Unit Tests**: Add tests for new functionality
- **Integration Tests**: Test API endpoints
- **E2E Tests**: Test complete user workflows
- **Performance Tests**: Test system performance

## 🔍 **Code Review Process**

### What We Look For

1. **Functionality**
   - Does the code work as intended?
   - Are edge cases handled properly?
   - Is error handling implemented?

2. **Code Quality**
   - Is the code readable and maintainable?
   - Are best practices followed?
   - Is the code properly documented?

3. **Testing**
   - Are tests included for new functionality?
   - Do existing tests still pass?
   - Is test coverage adequate?

4. **Security**
   - Are there any security vulnerabilities?
   - Is input validation implemented?
   - Are sensitive data handled properly?

### Review Timeline

- **Initial Review**: Within 48 hours
- **Follow-up Reviews**: Within 24 hours
- **Final Approval**: Within 1 week

## 🐛 **Bug Reports**

### Before Reporting

1. **Check Existing Issues**: Search for similar issues
2. **Test Latest Version**: Ensure you're using the latest code
3. **Gather Information**: Collect relevant details

### Bug Report Template

```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g. Windows 10]
- Browser: [e.g. Chrome 91]
- Node.js: [e.g. 18.0.0]
- QuantDesk Version: [e.g. 1.0.0]

**Additional Context**
Any other context about the problem.
```

## 💡 **Feature Requests**

### Before Requesting

1. **Check Roadmap**: Review our planned features
2. **Search Discussions**: Look for similar requests
3. **Consider Impact**: Think about user impact

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature.

**Use Case**
Describe the use case and why this feature is needed.

**Proposed Solution**
Describe your proposed solution.

**Alternatives**
Describe any alternative solutions you've considered.

**Additional Context**
Any other context about the feature request.
```

## 🏆 **Recognition**

### Contributors

We recognize contributors in several ways:
- **GitHub Contributors**: Listed in our contributors section
- **Release Notes**: Mentioned in release notes
- **Community**: Recognized in our community channels
- **Swag**: QuantDesk merchandise for significant contributions

### Types of Contributions

- **Code**: Bug fixes, features, improvements
- **Documentation**: Guides, tutorials, API docs
- **Testing**: Test cases, bug reports
- **Community**: Help others, answer questions
- **Design**: UI/UX improvements, mockups

## 📞 **Getting Help**

### Community Channels

- **Discord**: [QuantDesk Discord](https://discord.gg/quantdesk)
- **GitHub Discussions**: [GitHub Discussions](https://github.com/quantdesk/quantdesk/discussions)
- **Email**: contributors@quantdesk.com

### Mentorship

- **New Contributors**: We provide mentorship for new contributors
- **Code Review**: Experienced developers review your code
- **Guidance**: Help with architecture and best practices

## 📄 **License**

By contributing to QuantDesk, you agree that your contributions will be licensed under the MIT License.

## 🙏 **Thank You**

Thank you for contributing to QuantDesk! Your contributions help make decentralized trading more accessible and powerful for everyone.

---

**Happy Contributing!** 🚀

*Together, we're building the Bloomberg Terminal for Crypto.*