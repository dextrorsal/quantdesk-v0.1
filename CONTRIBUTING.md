# 👥 Contributor Guide

## 🎯 **Welcome Contributors!**

Thank you for your interest in contributing to QuantDesk! This guide will help you get started and understand our development workflow.

## 🚀 **Quick Start**

### **1. Prerequisites**
- Node.js 18+ and npm
- Git
- Docker (optional, for containerized development)
- VS Code or your preferred editor

### **2. Setup**
```bash
# Clone the repository
git clone https://github.com/dextrorsal/quantdesk.git
cd quantdesk

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Start development servers
npm run dev
```

### **3. Project Structure**
- `frontend/` - React trading interface
- `backend/` - Node.js API server
- `admin-dashboard/` - Admin management interface
- `MIKEY-AI/` - AI trading agent
- `contracts/` - Solana smart contracts
- `docs/` - Comprehensive documentation

## 🔧 **Development Workflow**

### **Branch Strategy**
- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - Feature development branches
- `hotfix/*` - Critical bug fixes

### **Making Changes**
1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make changes**: Follow coding standards
3. **Test your changes**: `npm test`
4. **Commit**: Use conventional commit messages
5. **Push**: `git push origin feature/your-feature-name`
6. **Create PR**: Submit a pull request

### **Code Standards**
- **TypeScript**: Use for all new code
- **ESLint**: Follow existing linting rules
- **Prettier**: Use for code formatting
- **Tests**: Write tests for new functionality
- **Documentation**: Update relevant docs

## 📝 **Commit Message Format**

Use conventional commits:
```
feat: add new trading strategy
fix: resolve API authentication issue
docs: update README with new features
test: add unit tests for trading logic
refactor: improve code organization
```

## 🧪 **Testing**

### **Running Tests**
```bash
# All tests
npm test

# Frontend tests
npm run test:frontend

# Backend tests
npm run test:backend

# AI agent tests
npm run test:ai
```

### **Test Coverage**
- Aim for 80%+ test coverage
- Write unit tests for new functions
- Add integration tests for API endpoints
- Test error handling and edge cases

## 📚 **Documentation**

### **Documentation Structure**
- `docs/development/` - Development guides
- `docs/api/` - API documentation
- `docs/architecture/` - System design
- `docs/contributing/` - This guide
- `docs/guides/` - General guides

### **Writing Documentation**
- Use clear, concise language
- Include code examples
- Update docs when making changes
- Follow markdown best practices

## 🔒 **Security**

### **Security Guidelines**
- Never commit secrets or API keys
- Use environment variables for configuration
- Follow secure coding practices
- Report security issues privately

### **Security Checklist**
- [ ] No hardcoded secrets
- [ ] Input validation implemented
- [ ] Error handling doesn't expose sensitive info
- [ ] Dependencies are up to date
- [ ] Security tests pass

## 🐛 **Bug Reports**

### **Reporting Bugs**
1. Check existing issues first
2. Use the bug report template
3. Include steps to reproduce
4. Provide error messages and logs
5. Include system information

### **Bug Report Template**
```markdown
**Bug Description**
Brief description of the bug

**Steps to Reproduce**
1. Go to...
2. Click on...
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10]
- Browser: [e.g., Chrome 91]
- Node.js: [e.g., 18.0.0]

**Additional Context**
Any other relevant information
```

## 💡 **Feature Requests**

### **Suggesting Features**
1. Check existing feature requests
2. Use the feature request template
3. Explain the use case
4. Provide implementation ideas if possible

### **Feature Request Template**
```markdown
**Feature Description**
Brief description of the feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other ways to solve this problem

**Additional Context**
Any other relevant information
```

## 🤝 **Code Review Process**

### **Review Checklist**
- [ ] Code follows project standards
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No security issues
- [ ] Performance considerations
- [ ] Error handling is appropriate

### **Review Guidelines**
- Be constructive and respectful
- Focus on the code, not the person
- Suggest improvements, don't just criticize
- Approve when ready, request changes when needed

## 📞 **Getting Help**

### **Communication Channels**
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Pull Requests**: Code review and discussion

### **Resources**
- **Documentation**: Check `docs/` directory
- **Examples**: See `examples/` directory
- **API Docs**: Available in `docs/api/`

## 🎉 **Recognition**

Contributors are recognized in:
- README.md contributors section
- Release notes for significant contributions
- GitHub contributor statistics

## 📋 **Contributor Checklist**

Before submitting:
- [ ] Code follows project standards
- [ ] Tests pass and coverage is adequate
- [ ] Documentation is updated
- [ ] No security issues
- [ ] Commit messages are clear
- [ ] Pull request description is complete

---

**Thank you for contributing to QuantDesk!** 🚀

*Together, we're building the future of decentralized trading.*
