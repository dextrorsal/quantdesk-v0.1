# Monorepo Workflow Guide

## Overview

This project is a large monorepo containing multiple services and components. This guide explains how to work effectively with this structure in Cursor and provides best practices for development workflows.

## Project Structure

Our monorepo contains:
- **Frontend** (`frontend/`) - Main user interface
- **Backend** (`backend/`) - API server and core logic
- **Admin Dashboard** (`admin-dashboard/`) - Administrative interface
- **Docs Site** (`docs-site/`) - Documentation website
- **Data Pipeline** (`data-ingestion/`) - Data processing and ingestion
- **Smart Contracts** (`contracts/`) - Blockchain contracts
- **SDK** (`sdk/`) - TypeScript SDK
- **Integration** (`integration/`) - Third-party integrations
- **Scripts** (`scripts/`) - Utility and deployment scripts

## Working in Cursor

### When to Open Root vs Subfolder

**Open at Root (`/home/dex/Desktop/quantdesk/`) when:**
- You're unsure where something lives
- You need cross-service context and AI assistance
- Working on multiple services simultaneously
- Making architectural changes
- Updating shared dependencies
- Running global searches or refactors
- Working on documentation that spans multiple services

**Open in Subfolder when:**
- Doing focused work on a single service
- Want faster, cleaner search within that service
- Working on service-specific features only

### Recommended Workflow

1. **Start at Root**: Open the entire project in Cursor for maximum context
2. **Use Working Set**: Keep only relevant files/folders in your Working Set
3. **Navigate to Service Folders**: Use Cursor's file explorer to dive into specific services
4. **Use Global Search**: Leverage Cursor's search across the entire monorepo

## Service Management

### Starting Services

Only start the services you need for your current work:

```bash
# Frontend development
./frontend/start-frontend.sh

# Backend API development
./backend/start-backend.sh

# Admin dashboard work
./admin-dashboard/start-admin.sh

# Documentation site
./start-docs.sh
# OR
cd docs-site && python serve.py

# Data pipeline
./data-ingestion/start-pipeline.sh
```

### Stopping Services

```bash
# Stop specific services
./frontend/kill-frontend.sh
./backend/kill-backend.sh
./admin-dashboard/kill-admin.sh
./kill-docs.sh
./data-ingestion/stop-pipeline.sh
```

## Development Workflows

### Frontend Development
- **Location**: `frontend/`
- **Start**: `./frontend/start-frontend.sh`
- **Focus**: UI components, user flows, client-side logic
- **Dependencies**: Backend API (can mock if needed)

### Backend Development
- **Location**: `backend/`
- **Start**: `./backend/start-backend.sh`
- **Focus**: APIs, business logic, database operations
- **Dependencies**: Database, external APIs

### Admin Dashboard Development
- **Location**: `admin-dashboard/`
- **Start**: `./admin-dashboard/start-admin.sh`
- **Focus**: Administrative features, monitoring, management tools
- **Dependencies**: Backend API

### Data Pipeline Development
- **Location**: `data-ingestion/`
- **Start**: `./data-ingestion/start-pipeline.sh`
- **Focus**: Data processing, ingestion, ETL operations
- **Dependencies**: External data sources, database

### Documentation Work
- **Location**: `docs/` and `docs-site/`
- **Start**: `./start-docs.sh` or `cd docs-site && python serve.py`
- **Focus**: Documentation updates, guides, API docs

## Best Practices

### 1. Service Isolation
- Each service should be independently runnable
- Use environment variables for configuration
- Keep service-specific dependencies isolated

### 2. Shared Resources
- **Types/Interfaces**: Define in one place, import everywhere
- **Common Utilities**: Use shared libraries
- **Configuration**: Use `.env.example` files per service

### 3. Development Environment
- Run only necessary services locally
- Use mocks or remote services for dependencies
- Keep terminal sessions organized by service

### 4. Code Organization
- Keep service boundaries clear
- Avoid circular dependencies
- Use consistent naming conventions

## Cursor-Specific Tips

### Search and Navigation
- Use `Cmd/Ctrl + P` for quick file navigation
- Use `Cmd/Ctrl + Shift + F` for global search
- Use `Cmd/Ctrl + T` for symbol search across the entire repo
- Use "Find References" to see cross-service impacts

### AI Assistance
- Cursor's AI works best with full repo context
- Use inline AI for service-specific questions
- Ask for architectural guidance with full context

### Working Set Management
- Keep Working Set focused on current task
- Add/remove folders as you switch between services
- Use Working Set to limit AI context when needed

## Troubleshooting

### Common Issues

**Service won't start:**
1. Check if port is already in use
2. Verify environment variables are set
3. Check service-specific logs

**Cross-service communication issues:**
1. Verify service URLs and ports
2. Check network connectivity
3. Review API contracts

**Build failures:**
1. Check dependencies are installed
2. Verify TypeScript configurations
3. Review service-specific build scripts

### Getting Help

1. Check service-specific README files
2. Review logs in service directories
3. Use Cursor's AI with full context
4. Check project documentation in `docs/`

## Future Improvements

### Recommended Enhancements

1. **Root Orchestration Scripts**
   ```bash
   # Add to root Makefile or package.json scripts
   make dev-frontend
   make dev-backend
   make dev-admin
   make dev-docs
   make dev-data
   ```

2. **Workspace Configuration**
   - Consider adding workspace configuration files
   - Standardize development environment setup

3. **Service Documentation**
   - Ensure each service has clear README
   - Document API contracts and interfaces
   - Add troubleshooting guides per service

4. **Dependency Management**
   - Consider monorepo tools (Turborepo, Nx)
   - Standardize dependency versions
   - Add shared dependency management

## Quick Reference

### Essential Commands
```bash
# Start all services (not recommended for daily work)
./frontend/start-frontend.sh &
./backend/start-backend.sh &
./admin-dashboard/start-admin.sh &

# Stop all services
./frontend/kill-frontend.sh
./backend/kill-backend.sh
./admin-dashboard/kill-admin.sh

# Check running processes
ps aux | grep -E "(node|python)" | grep -v grep
```

### Key Directories
- **Root**: `/home/dex/Desktop/quantdesk/`
- **Frontend**: `frontend/`
- **Backend**: `backend/`
- **Admin**: `admin-dashboard/`
- **Docs**: `docs/` and `docs-site/`
- **Data**: `data-ingestion/`
- **Contracts**: `contracts/`
- **SDK**: `sdk/`

---

*This guide should be updated as the project evolves and new services are added.*
