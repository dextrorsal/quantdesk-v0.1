# 🤖 AI Assistant Workflow Guide

## 🎯 **For AI Assistants Working on QuantDesk**

This guide helps AI assistants understand the project structure and contribute effectively.

## 📁 **Project Overview**

QuantDesk is a comprehensive trading platform with:
- **Frontend**: React-based trading interface
- **Backend**: Node.js API server
- **AI**: Trading agent with LLM integration (MIKEY-AI)
- **Smart Contracts**: Solana blockchain integration
- **Admin Dashboard**: Management interface

## 🔍 **Key Areas for AI Assistance**

### **1. Code Development**
- **Frontend**: `frontend/` - React components, hooks, context
- **Backend**: `backend/` - API routes, services, middleware
- **AI Agent**: `MIKEY-AI/` - Trading logic, LLM integration
- **Contracts**: `contracts/` - Solana smart contracts

### **2. Documentation**
- **Development**: `docs/development/` - Setup guides, coding standards
- **API**: `docs/api/` - API documentation and examples
- **Architecture**: `docs/architecture/` - System design and patterns
- **Operations**: `docs/operations/` - Deployment and monitoring

### **3. Testing & Quality**
- **Tests**: `tests/` - Unit tests, integration tests
- **Scripts**: `scripts/` - Development and deployment scripts
- **CI/CD**: `.github/workflows/` - Automated testing and deployment

## 🛠️ **Common AI Tasks**

### **Code Analysis**
```bash
# Analyze project structure
find . -name "*.ts" -o -name "*.tsx" -o -name "*.js" | head -20

# Check for issues
npm run lint
npm run test
```

### **Documentation Updates**
- Update `README.md` for new features
- Add examples to `docs/examples/`
- Update API docs in `docs/api/`

### **Code Generation**
- Follow existing patterns in each directory
- Use TypeScript for type safety
- Follow React best practices for frontend
- Use Node.js best practices for backend

## 📋 **AI Workflow Checklist**

1. **Understand Context**: Read relevant documentation first
2. **Analyze Existing Code**: Study similar implementations
3. **Follow Patterns**: Maintain consistency with existing code
4. **Test Changes**: Ensure tests pass and new functionality works
5. **Update Documentation**: Keep docs current with changes
6. **Consider Security**: Review for security implications

## 🚨 **Important Notes**

- **Environment**: Always check `.env` files for configuration
- **Dependencies**: Update `package.json` when adding new packages
- **Types**: Use TypeScript interfaces for better type safety
- **Testing**: Write tests for new functionality
- **Documentation**: Update relevant docs when making changes

## 🔗 **Useful Commands**

```bash
# Start development servers
npm run dev              # Start all services
npm run dev:frontend     # Start frontend only
npm run dev:backend      # Start backend only
npm run dev:admin        # Start admin dashboard

# Testing
npm test                 # Run all tests
npm run test:frontend    # Test frontend
npm run test:backend     # Test backend

# Documentation
npm run docs:build       # Build documentation
npm run docs:serve       # Serve documentation locally
```

## 📚 **Learning Resources**

- **React**: [React Documentation](https://react.dev/)
- **Node.js**: [Node.js Documentation](https://nodejs.org/docs/)
- **TypeScript**: [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- **Solana**: [Solana Documentation](https://docs.solana.com/)

---

*This guide helps AI assistants contribute effectively to the QuantDesk project while maintaining code quality and consistency.*
